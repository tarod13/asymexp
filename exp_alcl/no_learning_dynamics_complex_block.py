import numpy as np
import scipy.linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_graph_laplacian(n, delta):
    P = np.random.rand(n, n)**10
    P -= np.min(P)
    P /= np.sum(P, axis=1, keepdims=True)
    # rot = np.roll(np.eye(n), 1, axis=1) - np.eye(n)
    rot = np.zeros((n, n))
    L = np.eye(n)*(1+delta) - P + 0.5*rot
    return L

def init_blocks(n, k_blocks):
    X = np.random.randn(n, k_blocks, 2)
    Y = np.random.randn(n, k_blocks, 2)
    return X, Y

def solve_min_norm_block(grad, drift, epsilon=1e-8):
    grad_norm_sq = np.sum(grad**2)
    if grad_norm_sq < epsilon or drift <= 0:
        return np.zeros_like(grad), 0.0
    gamma = drift / grad_norm_sq
    u = -gamma * grad
    return u, gamma

def control_whitening(block, f_block, lambda_white):
    G = block.T @ block - np.eye(2)
    V = 0.5 * np.sum(G**2)
    grad = 2 * (block @ G)
    drift = np.sum(grad * f_block)
    total_drift = drift + lambda_white * V
    u, gamma = solve_min_norm_block(grad, total_drift)
    return u, V, gamma

def control_orth_block(target_X, prev_Y_all, f_target, lambda_orth):
    if prev_Y_all.shape[1] == 0:
        return np.zeros_like(target_X), 0.0, 0.0
    E = prev_Y_all.T @ target_X
    V = 0.5 * np.sum(E**2)
    grad = prev_Y_all @ E
    drift = np.sum(grad * f_target)
    total_drift = drift + lambda_orth * V
    u, gamma = solve_min_norm_block(grad, total_drift)
    return u, V, gamma

def dynamics_block(L, X, Y, lambda_white, lambda_orth):
    n, k_blocks, _ = X.shape
    FX = np.zeros_like(X)
    FY = np.zeros_like(Y)
    aux = []
    
    for k in range(k_blocks):
        xk = X[:, k, :]
        yk = Y[:, k, :]
        
        # Coupling Matrix M
        Cross = yk.T @ xk
        InvCross = np.linalg.inv(Cross + 1e-6 * np.eye(2))
        Mk = InvCross @ (yk.T @ (L @ xk))
        
        # Flow
        fx_bar = -L @ xk + xk @ Mk
        fy_bar = -L.T @ yk + yk @ Mk.T
        
        # Controls
        ux_w, Vx, bx = control_whitening(xk, fx_bar, lambda_white)
        uy_w, Vy, by = control_whitening(yk, fy_bar, lambda_white)
        
        if k > 0:
            prev_Y = Y[:, :k, :].reshape(n, -1)
            prev_X = X[:, :k, :].reshape(n, -1)
            ux_o, Vox, box = control_orth_block(xk, prev_Y, fx_bar, lambda_orth)
            uy_o, Voy, boy = control_orth_block(yk, prev_X, fy_bar, lambda_orth)
        else:
            ux_o, Vox, box = 0, 0, 0
            uy_o, Voy, boy = 0, 0, 0
            
        FX[:, k, :] = fx_bar + ux_w + ux_o
        FY[:, k, :] = fy_bar + uy_w + uy_o
        
        # Eigenvalues of the block for logging
        eigs_M = np.linalg.eigvals(Mk)
        eigs_M = eigs_M[np.argsort(eigs_M.real)]
        
        aux.append({
            'eigs': eigs_M,
            'err_white': Vx, 'err_orth': Vox,
            'gain_white': bx, 'gain_orth': box
        })
        
    return (FX, FY), aux

def step(L, X, Y, lr, lambda_white, lambda_orth):
    (FX, FY), aux = dynamics_block(L, X, Y, lambda_white, lambda_orth)
    X_new = X + lr * FX
    Y_new = Y + lr * FY
    return (X_new, Y_new), aux

def simulate_dynamics(L, X, Y, lr, lambda_white, lambda_orth, n_steps):
    n, k_blocks, _ = X.shape
    
    # History storage
    t_hist = []
    eigs_hist = []
    err_white_hist = []
    err_orth_hist = []
    gain_white_hist = []
    gain_orth_hist = []
    block_sim_hist = [] # Subspace similarity
    
    # Ground truth for similarity check (pre-computed)
    eigvals, eigvecs_left, eigvecs_right = scipy.linalg.eig(L, left=True, right=True)
    idx = np.argsort(eigvals.real)
    vecs_right = eigvecs_right[:, idx]
    
    for i in tqdm(range(n_steps)):
        (X, Y), aux = step(L, X, Y, lr, lambda_white, lambda_orth)
        
        if i % 10 == 0:
            t_hist.append(i)
            
            # Extract metrics per block
            curr_eigs = []
            curr_white = []
            curr_orth = []
            curr_gain_w = []
            curr_gain_o = []
            curr_sim = []
            
            for k in range(k_blocks):
                a = aux[k]
                curr_eigs.extend(a['eigs'])
                curr_white.append(a['err_white'])
                curr_orth.append(a['err_orth'])
                curr_gain_w.append(a['gain_white'])
                curr_gain_o.append(a['gain_orth'])
                
                # Subspace Similarity:
                # Project learned block X_k onto True Subspace V_{2k, 2k+1}
                # Metric: || V_true^T * orthonormal(X_k) ||_F / sqrt(2)
                # If perfect match, this is 1.0.
                
                # Get true subspace for this block
                V_true = vecs_right[:, 2*k : 2*k+2]
                
                # Orthonormalize X_k slightly for accurate metric (it should be white already)
                X_k = X[:, k, :]
                Q_k, _ = np.linalg.qr(X_k)
                
                # Projection magnitude
                proj = V_true.conj().T @ Q_k
                sim = np.linalg.norm(proj) / np.sqrt(2)
                curr_sim.append(sim)

            eigs_hist.append(curr_eigs)
            err_white_hist.append(curr_white)
            err_orth_hist.append(curr_orth)
            gain_white_hist.append(curr_gain_w)
            gain_orth_hist.append(curr_gain_o)
            block_sim_hist.append(curr_sim)

    # Convert to arrays
    return (t_hist, np.array(eigs_hist), np.array(err_white_hist), 
            np.array(err_orth_hist), np.array(gain_white_hist), 
            np.array(gain_orth_hist), np.array(block_sim_hist))

def visualize_metrics(t, eigs_h, white_h, orth_h, gain_w_h, gain_o_h, sim_h, target_vals):    
    k_blocks = white_h.shape[1]
    
    plt.figure(figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, k_blocks))
    
    # 1. Whitening Errors (Internal Orthonormality)
    plt.subplot(2, 3, 1)
    for k in range(k_blocks):
        plt.plot(t, white_h[:, k], color=colors[k], label=f'Block {k}')
    plt.title('Whitening Error ($X^T X - I$)')
    plt.yscale('log')
    plt.legend()

    # 2. Orthogonality Errors (Deflation)
    plt.subplot(2, 3, 2)
    for k in range(1, k_blocks): # Block 0 has no constraint
        plt.plot(t, orth_h[:, k], color=colors[k], label=f'Block {k}')
    plt.title('Deflation Error ($Y_{prev}^T X$)')
    plt.yscale('log')
    plt.legend()

    # 3. Eigenvalues (Real Parts)
    plt.subplot(2, 3, 3)
    # We have 2*k_blocks eigenvalues
    for i in range(2 * k_blocks):
        block_idx = i // 2
        plt.plot(t, eigs_h[:, i].real, color=colors[block_idx], alpha=0.6)
        plt.axhline(target_vals[i].real, color=colors[block_idx], linestyle='--', alpha=0.3)
    plt.title('Eigenvalues (Real)')
    
    # 4. Subspace Similarity (Convergence)
    plt.subplot(2, 3, 4)
    for k in range(k_blocks):
        plt.plot(t, sim_h[:, k], color=colors[k], label=f'Block {k}')
    plt.title('Subspace Overlap (1.0 = Perfect)')
    plt.ylim([0, 1.05])
    plt.legend()

    # 5. Control Gains (Effort)
    plt.subplot(2, 3, 5)
    for k in range(k_blocks):
        plt.plot(t, gain_w_h[:, k], color=colors[k], linestyle='-', label=f'W-{k}')
        if k > 0:
            plt.plot(t, gain_o_h[:, k], color=colors[k], linestyle=':', label=f'O-{k}')
    plt.title('Barrier Gains ($\gamma$)')
    plt.yscale('log')
    plt.legend(fontsize='small', ncol=2)

    # 6. Eigenvalues (Imag Parts)
    plt.subplot(2, 3, 6)
    for i in range(2 * k_blocks):
        block_idx = i // 2
        plt.plot(t, eigs_h[:, i].imag, color=colors[block_idx], alpha=0.6)
        plt.axhline(target_vals[i].imag, color=colors[block_idx], linestyle='--', alpha=0.3)
    plt.title('Eigenvalues (Imaginary)')

    plt.tight_layout()
    plt.show()

def main(n, k_blocks, n_steps):
    delta = 0.1
    lr = 0.0003
    lambda_white = 10.0
    lambda_orth = 20.0
    
    L = generate_graph_laplacian(n, delta)
    
    # Get targets for plotting
    true_vals = scipy.linalg.eigvals(L)
    true_vals = true_vals[np.argsort(true_vals.real)]
    target_vals = true_vals[:2*k_blocks]
    print("Target Eigenvalues:", target_vals)
    
    X, Y = init_blocks(n, k_blocks)
    
    results = simulate_dynamics(L, X, Y, lr, lambda_white, lambda_orth, n_steps)
    
    visualize_metrics(*results, target_vals)

if __name__ == "__main__":
    main(n=30, k_blocks=4, n_steps=100000)