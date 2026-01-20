import jax
import jax.numpy as jnp
from jax import random, grad, jit
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt

# ==========================================
# 1. Problem Setup (The Physics)
# ==========================================
def generate_problem(n, k, key):
    k1, k2, k3 = random.split(key, 3)
    
    # A: Symmetric matrix with distinct eigenvalues
    A_root = random.normal(k1, (n, n))
    A = A_root @ A_root.T
    
    # B, C: Defining the Manifold V.T @ B @ V = C
    B = jnp.eye(n) 
    C = jnp.eye(k)
    
    # N: The "Symmetry Breaker" weights.
    # CHANGE 1: Descending Order (k, k-1, ..., 1)
    # Minimizing Trace(V.T A V N) pairs the Smallest Eigenvalue of A 
    # with the Largest Weight of N.
    # By putting the Largest Weight at index 0, we force Col 0 -> Smallest Eig.
    N = jnp.diag(jnp.arange(k, 0, -1, dtype=jnp.float32)) / k
    
    return A, B, C, N

# ==========================================
# 2. The Neural Network (The Generator)
# ==========================================
def init_params(layers, key):
    params = []
    keys = random.split(key, len(layers))
    for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
        w_key, b_key = random.split(keys[i])
        W = random.normal(w_key, (in_dim, out_dim)) * 0.1
        b = jnp.zeros(out_dim)
        params.append((W, b))
    return params

def forward(params, x):
    # Simple MLP
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, W) + b)
    # Final layer
    W_last, b_last = params[-1]
    return jnp.dot(x, W_last) + b_last

# ==========================================
# 3. The Potentials (Lyapunov & Objective)
# ==========================================
def lyapunov_func(params, input_seed, B, C):
    V_flat = forward(params, input_seed)
    V = V_flat.reshape(-1, C.shape[0])
    constraint_err = V.T @ B @ V - C
    return 0.5 * jnp.sum(constraint_err**2)

def objective_func(params, input_seed, A, N):
    V_flat = forward(params, input_seed)
    V = V_flat.reshape(-1, N.shape[0])
    # Weighted trace minimization
    return jnp.trace(V.T @ A @ V @ N)

# ==========================================
# 4. The CLF-QP Controller logic
# ==========================================
@jit
def update_step(params, input_seed, A, B, C, N, eta, lambda_decay):
    
    # Calculate Values
    val_clf = lyapunov_func(params, input_seed, B, C)
    
    # Calculate Gradients
    grads_clf = grad(lyapunov_func)(params, input_seed, B, C)
    grads_obj = grad(objective_func)(params, input_seed, A, N)
    
    # Flatten
    g_clf_flat, unravel_fn = ravel_pytree(grads_clf)
    g_obj_flat, _          = ravel_pytree(grads_obj)
    
    # Natural drift (Descent direction)
    f_flat = -g_obj_flat
    
    # Lie Derivatives
    Lf_V = jnp.dot(g_clf_flat, f_flat)
    Lg_V_sq = jnp.sum(g_clf_flat**2)
    
    # QP Logic
    psi_0 = Lf_V + lambda_decay * val_clf
    
    def active_control(_):
        scale = psi_0 / (Lg_V_sq + 1e-8)
        return -scale * g_clf_flat

    def inactive_control(_):
        return jnp.zeros_like(g_clf_flat)
        
    u_flat = jax.lax.cond(psi_0 > 0, active_control, inactive_control, operand=None)
    
    # Apply Updates
    total_update_flat = f_flat + u_flat
    update_tree = unravel_fn(total_update_flat)
    
    new_params = jax.tree_util.tree_map(lambda p, u: p + eta * u, params, update_tree)
    
    # CHANGE 2: Estimate Eigenvalues for plotting
    # (Re-run forward pass to get current V)
    V_flat = forward(params, input_seed)
    V = V_flat.reshape(-1, N.shape[0])
    # Rayleigh Quotient approximation
    est_eigvals = jnp.diag(V.T @ A @ V)
    
    return new_params, val_clf, objective_func(params, input_seed, A, N), est_eigvals

# ==========================================
# 5. Helper: Cosine Similarity
# ==========================================
def compute_cosine_similarities(V_est, V_true):
    V_est_norm = V_est / jnp.clip(jnp.linalg.norm(V_est, axis=0, keepdims=True), 1e-8)
    V_true_norm = V_true / jnp.clip(jnp.linalg.norm(V_true, axis=0, keepdims=True), 1e-8)
    dot_products = jnp.sum(V_est_norm * V_true_norm, axis=0)
    return jnp.abs(dot_products)

# ==========================================
# 6. Simulation & Visualization
# ==========================================
def main():
    # Parameters (YOUR original settings)
    n, k = 50, 3      
    steps = 200000    
    eta = 3e-4        
    lambda_decay = 1.0 
    
    key = random.PRNGKey(42)
    A, B, C, N = generate_problem(n, k, key)
    
    # Ground Truth
    print("Computing Ground Truth...")
    eigvals, eigvecs = jnp.linalg.eigh(A)
    # eigh returns sorted ascending (min is index 0)
    true_vals = eigvals[:k]
    true_vecs = eigvecs[:, :k]
    print(f"True Bottom-{k} Eigenvalues: {true_vals}")
    
    # Network Init
    latent_dim = 16
    input_seed = random.normal(key, (latent_dim,))
    layers = [latent_dim, 64, n*k] 
    params = init_params(layers, key)

    # Logging
    history_clf = []
    history_obj = []
    history_sim = [] 
    history_eig = [] # New log for eigenvalues
    
    print("\nStarting CLF-Guided Flow...")
    
    for i in range(steps):
        # CHANGE 2: Unpack the extra return value (v_eigs)
        params, v_clf, v_obj, v_eigs = update_step(params, input_seed, A, B, C, N, eta, lambda_decay)
        
        history_clf.append(v_clf)
        history_obj.append(v_obj)
        history_eig.append(v_eigs)
        
        # Log Cosine Similarity
        if i % 1000 == 0:
            V_flat = forward(params, input_seed)
            V_curr = V_flat.reshape(n, k)
            sims = compute_cosine_similarities(V_curr, true_vecs)
            history_sim.append(sims)

        if i % 5000 == 0:
            avg_sim = jnp.mean(history_sim[-1]) if history_sim else 0.0
            print(f"Step {i:06d} | CLF Error: {v_clf:.2e} | Est. Eigs: {v_eigs} | Avg Cos Sim: {avg_sim:.4f}")

    # ==========================================
    # Visualization (Updated to 2x2 grid)
    # ==========================================
    history_sim = jnp.array(history_sim)
    history_eig = jnp.array(history_eig)
    sim_steps = jnp.arange(0, steps, 1000)
    
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"CLF-QP Eigen-Discovery (n={n}, k={k})", fontsize=16)

    # Plot 1: Constraint Violation
    plt.subplot(2, 2, 1)
    plt.plot(history_clf, color='tab:red')
    plt.yscale('log')
    plt.title("Constraint Violation (Log Scale)")
    plt.ylabel("$V_{clf}(x)$")
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # Plot 2: Objective Function
    plt.subplot(2, 2, 2)
    plt.plot(history_obj, color='tab:blue', label='Estimated Trace')
    plt.title("Weighted Trace Objective")
    plt.grid(True)

    # Plot 3: Eigenvalue Estimation (New)
    plt.subplot(2, 2, 3)
    colors = ['r', 'g', 'b', 'orange']
    for idx in range(k):
        c = colors[idx % len(colors)]
        plt.plot(history_eig[:, idx], color=c, alpha=0.6, label=f'Est $\lambda_{idx+1}$')
        plt.axhline(true_vals[idx], color=c, linestyle='--')
    plt.title("Eigenvalue Convergence")
    plt.ylabel("Rayleigh Quotient")
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot 4: Cosine Similarity
    plt.subplot(2, 2, 4)
    for idx in range(k):
        c = colors[idx % len(colors)]
        plt.plot(sim_steps, history_sim[:, idx], color=c, label=f'Vec {idx+1}')
    plt.title("Cosine Similarity (1-to-1)")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Final Stats
    V_final = forward(params, input_seed).reshape(n, k)
    final_sims = compute_cosine_similarities(V_final, true_vecs)
    print(f"\nFinal Cosine Similarities per Vector: {final_sims}")

if __name__ == "__main__":
    main()