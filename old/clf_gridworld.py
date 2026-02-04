import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap, flatten_util
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.envs.env import get_example_environment

# ==========================================
# 1. Sampler
# ==========================================
class GridSampler:
    def __init__(self, env_name, num_agents, seed=42):
        self.env = get_example_environment(env_name)
        self.width = self.env.width; self.height = self.env.height
        self.num_actions = self.env.action_space; self.num_agents = num_agents
        self.step_fn = jit(self._make_step_fn()); self.reset_fn = jit(self._make_reset_fn())
        self.key = random.PRNGKey(seed); self.key, subkey = random.split(self.key)
        self.states = self.reset_fn(subkey)
        y, x = jnp.meshgrid(jnp.arange(self.height), jnp.arange(self.width), indexing='ij')
        self.coords = jnp.stack([x.flatten(), y.flatten()], axis=1) / jnp.array([self.width, self.height])
        self.get_idx = jit(vmap(self.env.get_state_representation))
    def _make_step_fn(self):
        n = self.num_agents
        def step(k, s, a): return vmap(lambda k,s,a: self.env.step(k,s,a)[0])(random.split(k, n), s, a)
        return step
    def _make_reset_fn(self):
        n = self.num_agents
        def reset(k): return vmap(self.env.reset)(random.split(k, n))
        return reset
    def sample_batch(self):
        self.key, k1, k2 = random.split(self.key, 3)
        actions = random.randint(k1, (self.num_agents,), 0, self.num_actions)
        next_states = self.step_fn(k2, self.states, actions)
        i, j = self.get_idx(self.states), self.get_idx(next_states)
        self.states = next_states
        return jnp.concatenate([i, j]), jnp.concatenate([j, i])

# ==========================================
# 2. Helpers
# ==========================================
def compute_ground_truth_spectrum(env_name, k):
    """ Returns Smallest k Eigenvalues of L (Positive numbers) """
    env = get_example_environment(env_name)
    h, w = env.height, env.width
    n_states = h * w
    valid_mask = np.ones((h, w), dtype=bool)
    if env.has_obstacles:
        for obs in env.obstacles:
            if obs[0] >= 0: valid_mask[obs[1], obs[0]] = False
    adj = np.zeros((n_states, n_states))
    valid_indices = []
    for y in range(h):
        for x in range(w):
            if not valid_mask[y, x]: continue
            u = y * w + x
            valid_indices.append(u)
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and valid_mask[ny, nx]:
                    v = ny * w + nx; adj[u, v] = 1.0; adj[v, u] = 1.0 
                else: adj[u, u] += 1.0 
    valid_indices = np.array(valid_indices)
    A_sub = adj[np.ix_(valid_indices, valid_indices)]
    D_sub = np.diag(np.sum(A_sub, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_sub) + 1e-10))
    L_norm = np.eye(len(valid_indices)) - D_inv_sqrt @ A_sub @ D_inv_sqrt
    vals, vecs = np.linalg.eigh(L_norm)
    full_vecs = np.zeros((n_states, k)); full_vecs[valid_indices, :] = vecs[:, :k]
    for i in range(k):
        if np.abs(full_vecs[:, i]).max() > 0: full_vecs[:, i] *= np.sign(full_vecs[np.argmax(np.abs(full_vecs[:, i])), i])
    return vals[:k], full_vecs, valid_mask

def compute_cosine_similarity(V_est, V_true, valid_mask_flat):
    v_est_valid = V_est[valid_mask_flat]
    v_true_valid = V_true[valid_mask_flat]
    v_est_norm = v_est_valid / (jnp.linalg.norm(v_est_valid, axis=0, keepdims=True) + 1e-6)
    v_true_norm = v_true_valid / (jnp.linalg.norm(v_true_valid, axis=0, keepdims=True) + 1e-6)
    return jnp.abs(jnp.sum(v_est_norm * v_true_norm, axis=0))

# ==========================================
# 3. Main Update
# ==========================================
def forward(params, x):
    for W, b in params[:-1]: x = jax.nn.leaky_relu(jnp.dot(x, W) + b)
    W_last, b_last = params[-1]
    return jnp.dot(x, W_last) + b_last

@jit
def sontag_alpha(a, b):
    return (a + jnp.sqrt(a**2 + b**2)) / (b + 1e-6)

@jit
def update_step(params, Beta, state, feats_i, feats_j, n_total, batch_size, eta, 
                lambda_clf, beta_lr):
    scale = n_total / batch_size
    
    # Forward
    v_start = forward(params, feats_i)
    v_end   = forward(params, feats_j)
    
    # Gram Matrices
    G = (v_start.T @ v_start) * scale        
    M = (v_start.T @ v_end)   * scale        
    PTP = (v_end.T @ v_end) * scale          
    
    # -----------------------------------------------------------
    # A. Update Beta (Solve Stationarity: LV = V Beta)
    # -----------------------------------------------------------
    def beta_algebraic_loss(b_matrix):
        b_masked = jnp.tril(b_matrix)
        term_VB = jnp.trace(b_masked.T @ G @ b_masked)
        term_L = jnp.trace(G) - 2*jnp.trace(M) + jnp.trace(PTP)
        term_Cross = -2 * jnp.trace(b_masked.T @ (G - M))
        return term_VB + term_L + term_Cross

    grad_beta_alg = grad(beta_algebraic_loss)(Beta)
    grad_beta_alg = jnp.tril(grad_beta_alg)
    new_Beta = Beta - beta_lr * grad_beta_alg

    # -----------------------------------------------------------
    # B. Update Params (Sontag)
    # -----------------------------------------------------------
    def net_loss_fn(p):
        v_s = forward(p, feats_i)
        v_e = forward(p, feats_j)
        g_b = (v_s.T @ v_s) * scale
        m_b = (v_s.T @ v_e) * scale
        C = g_b - jnp.eye(g_b.shape[0])
        b_masked = jnp.tril(new_Beta)
        
        energy = jnp.trace(g_b - m_b)
        constraint_work = jnp.sum(b_masked * C)
        l_nat = energy - constraint_work
        l_err = 0.5 * jnp.sum(C**2)
        
        return l_nat, l_err, C, energy

    grads_nat = grad(lambda p: net_loss_fn(p)[0])(params)
    grads_ctrl = grad(lambda p: net_loss_fn(p)[1])(params)
    
    flat_nat, unravel = flatten_util.ravel_pytree(grads_nat)
    flat_ctrl, _ = flatten_util.ravel_pytree(grads_ctrl)
    
    # Lie Derivatives
    a_raw = -jnp.dot(flat_ctrl, flat_nat)
    b = jnp.sum(flat_ctrl**2)
    
    _, _, C_val, energy_val = net_loss_fn(params)
    V_val = 0.5 * jnp.sum(C_val**2)
    
    a_eff = a_raw + lambda_clf * V_val
    alpha = sontag_alpha(a_eff, b)
    alpha = jax.lax.stop_gradient(alpha)
    
    total_grad = flat_nat + alpha * flat_ctrl
    
    g_norm = jnp.linalg.norm(total_grad)
    scaling = jnp.minimum(1.0, 1.0 / (g_norm + 1e-6))
    final_grads = unravel(total_grad * scaling)
    new_params = jax.tree_util.tree_map(lambda p, g: p - eta * g, params, final_grads)
    
    # Metrics
    metrics = {
        'energy': energy_val, 
        'V_err': V_val,
        'grad_V': jnp.linalg.norm(flat_ctrl),
        'f_eff': jnp.linalg.norm(flat_nat),
        'alpha': alpha,
        'a': a_raw,
        'b': b,
        'norms': jnp.diag(G),
        'beta_full': jnp.tril(new_Beta), # Pass full matrix
        'beta_loss': beta_algebraic_loss(new_Beta)
    }
    return new_params, new_Beta, metrics

# ==========================================
# 4. Main
# ==========================================
def main():
    env_name = "room4"
    k = 4
    batch_size = 256
    steps = 3000
    eta = 1e-3
    beta_lr = 1e-3
    lambda_clf = 0.1
    
    sampler = GridSampler(env_name, num_agents=batch_size)
    true_vals_L, true_vecs, valid_mask = compute_ground_truth_spectrum(env_name, k)
    
    print(f"Ground Truth Laplacian Eigenvalues: {true_vals_L}")
    
    layers = [2, 64, 64, k] 
    key = random.PRNGKey(42)
    params = []
    keys = random.split(key, len(layers))
    for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
        w_key, b_key = random.split(keys[i])
        n = max(in_dim, out_dim)
        W = random.normal(w_key, (in_dim, out_dim)) * 0.1
        b = jnp.zeros(out_dim)
        params.append((W, b))

    Beta = jnp.eye(k) * 0.1 
    state = {}
    
    # Track full beta matrix history
    history = {k: [] for k in ['energy', 'V_err', 'grad_V', 'f_eff', 'alpha', 'a', 'b', 'sim', 'beta_loss', 'norms']}
    beta_history = [] 

    print("Training Hybrid (Physics: Minimize L, Beta=Evals(L))...")
    
    for i in range(steps):
        idx_i, idx_j = sampler.sample_batch()
        feats_i = sampler.coords[idx_i]
        feats_j = sampler.coords[idx_j]
        
        params, Beta, metrics = update_step(
            params, Beta, state, feats_i, feats_j, 
            sampler.width * sampler.height, batch_size, eta,
            lambda_clf, beta_lr
        )
        
        for key in history:
            if key != 'sim':
                history[key].append(metrics[key])
        
        # Save full Beta matrix
        beta_history.append(metrics['beta_full'])
        
        if i % 100 == 0:
            V_full = forward(params, sampler.coords)
            sims = compute_cosine_similarity(V_full, true_vecs, valid_mask.flatten())
            history['sim'].append(sims)
            
            # Diagnostic for print
            beta_d = jnp.diag(metrics['beta_full'])
            diff = jnp.mean(jnp.abs(jnp.sort(beta_d) - np.sort(true_vals_L)))
            print(f"Step {i:05d} | Err: {metrics['V_err']:.1e} | BetaDiff: {diff:.3f} | Sim: {jnp.mean(sims):.3f}")

    # ==========================================
    # VISUALS
    # ==========================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    axes[0].plot(history['V_err']); axes[0].set_yscale('log'); axes[0].set_title("Constraint Violation")
    axes[1].plot(history['a'], label='a'); axes[1].plot(history['b'], label='b'); axes[1].set_yscale('symlog', linthresh=1e-6); axes[1].legend(); axes[1].set_title("Lie Derivatives")
    axes[2].plot(history['alpha']); axes[2].set_yscale('log'); axes[2].set_title("Alpha")
    axes[3].plot(history['f_eff'], label='Natural'); axes[3].plot(history['grad_V'], label='Control'); axes[3].set_yscale('log'); axes[3].legend(); axes[3].set_title("Gradient Norms")
    
    # Beta Evolution Plot
    beta_hist_np = np.array(beta_history) # Shape (T, k, k)
    # Indices for lower triangle
    rows, cols = np.tril_indices(k)
    
    for r, c in zip(rows, cols):
        label = f'$\\beta_{{{r}{c}}}$'
        style = '-' if r == c else '--' # Solid for diag, dashed for off-diag
        axes[4].plot(beta_hist_np[:, r, c], style, label=label)
    axes[4].set_title("Beta Evolution (10 Values)"); axes[4].legend(fontsize='small', ncol=2)
    
    axes[5].plot(history['energy']); axes[5].set_title("Dirichlet Energy (Minimize)")
    axes[6].plot(history['norms']); axes[6].set_title("Vector Norms"); axes[6].set_ylim(0, 2)
    
    sims = np.array(history['sim'])
    xs = np.arange(0, steps, 100)
    for idx in range(k): axes[7].plot(xs, sims[:, idx], label=f'v_{idx}')
    axes[7].set_title("Cosine Similarity"); axes[7].legend()
    
    final_beta_diag = np.diag(beta_hist_np[-1])
    axes[8].plot(np.sort(final_beta_diag), 'bo-', label='Learned Beta Diag')
    axes[8].plot(np.sort(true_vals_L), 'rx--', label='True L')
    axes[8].legend(); axes[8].set_title("Spectrum (L)")
    
    axes[9].plot(history['beta_loss']); axes[9].set_yscale('log'); axes[9].set_title("Beta Loss")
    
    plt.tight_layout(); plt.show()
    
    # Maps
    all_vecs = forward(params, sampler.coords)
    learned_maps = all_vecs.reshape(sampler.height, sampler.width, k)
    true_maps = true_vecs.reshape(sampler.height, sampler.width, k)
    plt.figure(figsize=(15, 6))
    for i in range(k):
        plt.subplot(2, k, i+1)
        img_l = np.array(learned_maps[:,:,i]); img_l[~valid_mask] = np.nan
        sim_sign = np.sign(np.nansum(img_l * np.array(true_maps[:,:,i])))
        plt.imshow(sim_sign*img_l, cmap='jet'); plt.axis('off'); plt.title(f"Learned {i}")
        plt.subplot(2, k, i+1+k)
        img_t = np.array(true_maps[:,:,i]); img_t[~valid_mask] = np.nan
        plt.imshow(img_t, cmap='jet'); plt.axis('off'); plt.title(f"Truth {i}")
    plt.show()

if __name__ == "__main__":
    main()