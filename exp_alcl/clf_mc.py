import jax
import jax.numpy as jnp
from jax import random, grad, jit
import matplotlib.pyplot as plt

# ==========================================
# 1. Problem Setup (Symmetric P)
# ==========================================
def generate_ring_graph(n, k, key):
    """
    Generates a Ring Graph (Cycle) with slight random noise.
    This creates a clean, smooth spectrum (no huge gaps).
    """
    key, subkey = random.split(key)
    
    # 1. Create Ring Adjacency (2-Regular Graph)
    # Connect i to (i+1)%n and (i-1)%n
    row_idx = jnp.arange(n)
    col_right = (row_idx + 1) % n
    col_left = (row_idx - 1) % n
    
    # Build Adjacency W
    W = jnp.zeros((n, n))
    W = W.at[row_idx, col_right].set(1.0)
    W = W.at[row_idx, col_left].set(1.0)
    
    # 2. Add slight noise (Sparse edges) to break perfect symmetry
    # This prevents repeated eigenvalues
    noise = random.uniform(subkey, (n, n))
    noise = (noise + noise.T) / 2
    mask = noise > 0.96 # Keep only top 4% of noise edges
    W = W + mask * noise * 0.5
    
    # 3. Sinkhorn Normalization (Make Doubly Stochastic)
    # Since W is already nearly regular, this converges instantly.
    for _ in range(10):
        W = W / jnp.sum(W, axis=1, keepdims=True)
        W = W / jnp.sum(W, axis=0, keepdims=True)
        
    P = (W + W.T) / 2
    
    # Manifold
    B = jnp.eye(n)
    C = jnp.eye(k)
    
    # Descending Weights (Forces Index 0 -> Smallest Eig)
    weights = jnp.arange(k, 0, -1, dtype=jnp.float32) / k
    N = jnp.diag(weights)
    
    return P, B, C, N

def generate_symmetric_P(n, k, key):
    """
    Generates a Symmetric Stochastic Matrix P.
    (Corresponds to a Random Walk on a Regular Graph).
    """
    key, subkey = random.split(key)
    
    # 1. Generate Random Symmetric Matrix W
    W = random.uniform(subkey, (n, n))
    W = (W + W.T) / 2
    W = jnp.exp(W) # Make positive
    
    # 2. Sinkhorn-Knopp to make it Doubly Stochastic (Symmetric & Rows sum to 1)
    # Since W is symmetric, alternating norm preserves symmetry in the limit.
    for _ in range(20):
        W = W / jnp.sum(W, axis=1, keepdims=True)
        W = W / jnp.sum(W, axis=0, keepdims=True)
        
    P = (W + W.T) / 2 # Ensure exact symmetry numerically
    
    # Manifold
    B = jnp.eye(n)
    C = jnp.eye(k)
    
    # Weights
    weights = jnp.arange(k, 0, -1, dtype=jnp.float32) / k
    N = jnp.diag(weights)
    
    return P, B, C, N

def sample_transitions(P, batch_size, key):
    n = P.shape[0]
    k1, k2 = random.split(key)
    
    # Uniform start nodes
    start_nodes = random.randint(k1, (batch_size,), 0, n)
    
    # Sample neighbors
    probs = P[start_nodes] 
    end_nodes = random.categorical(k2, jnp.log(probs + 1e-10))
    
    return start_nodes, end_nodes

# ==========================================
# 2. Network
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

def forward(params, inputs):
    x = inputs
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, W) + b)
    W_last, b_last = params[-1]
    return jnp.dot(x, W_last) + b_last

# ==========================================
# 3. Transition-Based Update (No Corrections)
# ==========================================
@jit
def update_step_transitions(params, state, idx_i, idx_j, feats_i, feats_j, N, n_total, eta, lambda_decay):
    
    batch_size = idx_i.shape[0]
    scale = n_total / batch_size
    
    # --- 1. Forward Pass ---
    V_i = forward(params, feats_i)
    V_j = forward(params, feats_j)
    
    # --- 2. Estimators ---
    # G_inst: V.T @ V (Start nodes)
    G_inst = (V_i.T @ V_i) * scale
    
    # M_inst: V.T @ P @ V (Transitions)
    # Since P is symmetric, no correction needed.
    M_inst = (V_i.T @ V_j) * scale
    
    # --- 3. EMA Update ---
    momentum = 0.9
    G_avg = momentum * state['G_avg'] + (1 - momentum) * G_inst
    M_avg = momentum * state['M_avg'] + (1 - momentum) * M_inst
    
    # --- 4. Control Alpha ---
    diff_G = G_avg - jnp.eye(G_avg.shape[0])
    
    # Drift = -2 Trace( (G-I)(G-M)N )
    drift_term = -2 * jnp.trace( diff_G @ (G_avg - M_avg) @ N )
    
    norm_sq = jnp.trace( diff_G @ diff_G @ G_avg )
    val_clf = 0.5 * jnp.sum(diff_G**2)
    
    psi_0 = drift_term + lambda_decay * val_clf
    alpha = jnp.maximum(0.0, psi_0) / (norm_sq + 1e-6)
    alpha = jax.lax.stop_gradient(alpha)
    
    # --- 5. Gradient Step ---
    def loss_fn(p):
        v_start = forward(p, feats_i)
        v_end   = forward(p, feats_j)
        
        g_b = (v_start.T @ v_start) * scale
        m_b = (v_start.T @ v_end)   * scale
        
        l_obj = jnp.trace( (g_b - m_b) @ N )
        l_clf = 0.5 * jnp.sum( (g_b - jnp.eye(g_b.shape[0]))**2 )
        
        return l_obj + alpha * l_clf

    grads = grad(loss_fn)(params)
    new_params = jax.tree_util.tree_map(lambda p, g: p - eta * g, params, grads)
    
    # Obj value for plotting
    obj_val = jnp.trace( (G_avg - M_avg) @ N )
    
    return new_params, {'G_avg': G_avg, 'M_avg': M_avg}, val_clf, obj_val, alpha

# ==========================================
# 4. Main
# ==========================================
def main():
    n, k = 50, 3
    batch_size = 32
    steps = 200000 
    eta = 1e-3        
    lambda_decay = 1.0 
    
    key = random.PRNGKey(42)
    
    # 1. Generate Symmetric P
    P, B, C, N = generate_ring_graph(n, k, key)
    
    # 2. Ground Truth 
    print("Computing Ground Truth...")
    # L = I - P
    L = jnp.eye(n) - P
    vals, vecs = jnp.linalg.eigh(L)
    true_vals = vals[:k]
    true_vecs = vecs[:, :k]
    print(f"True Eigenvalues (I-P): {true_vals}")
    
    # Features
    node_features = jax.random.normal(key, (n, 32))
    
    # Init
    layers = [32, 64, k] 
    params = init_params(layers, key)
    
    # --- Init State properly with one batch ---
    idx_i, idx_j = sample_transitions(P, batch_size, key)
    scale = n / batch_size
    v_i = forward(params, node_features[idx_i])
    v_j = forward(params, node_features[idx_j])
    state = {
        'G_avg': (v_i.T @ v_i) * scale,
        'M_avg': (v_i.T @ v_j) * scale
    }
    # ------------------------------------------

    hist_clf, hist_obj, hist_sim = [], [], []
    
    print("\nStarting Symmetric Stochastic Flow...")
    
    for i in range(steps):
        key, subkey = random.split(key)
        
        idx_i, idx_j = sample_transitions(P, batch_size, subkey)
        
        params, state, v_clf, v_obj, alpha = update_step_transitions(
            params, state, idx_i, idx_j, 
            node_features[idx_i], node_features[idx_j], 
            N, n, eta, lambda_decay
        )
        
        hist_clf.append(v_clf)
        hist_obj.append(v_obj)
        
        if i % 1000 == 0:
            V_full = forward(params, node_features)
            
            V_est_n = V_full / jnp.linalg.norm(V_full, axis=0, keepdims=True)
            V_true_n = true_vecs / jnp.linalg.norm(true_vecs, axis=0, keepdims=True)
            sims = jnp.abs(jnp.sum(V_est_n * V_true_n, axis=0))
            hist_sim.append(sims)
            
            est_eigs = 1.0 - jnp.sort(jnp.diag(state['M_avg']))
            
            print(f"Step {i:05d} | CLF: {v_clf:.1e} | Obj: {v_obj:.2f} | Sim: {jnp.mean(sims):.3f}")

    hist_sim = jnp.array(hist_sim)
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.plot(hist_clf); plt.yscale('log'); plt.title("Constraint Error")
    plt.subplot(1,3,2); plt.plot(hist_obj); plt.title("Objective")
    plt.subplot(1,3,3)
    for idx in range(k): plt.plot(hist_sim[:, idx], label=f'Vec {idx}')
    plt.legend(); plt.title("Cosine Similarity")
    plt.show()

if __name__ == "__main__":
    main()