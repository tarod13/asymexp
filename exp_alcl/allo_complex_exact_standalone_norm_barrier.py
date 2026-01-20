"""
Standalone test of exact gradient dynamics for learning complex eigenvectors.
Modified to include normalization barrier: (<x,x> - 1)^2
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

def create_complex_laplacian(n: int, seed: int = 42) -> np.ndarray:
    """Create a Laplacian that has complex eigenvalues for testing."""
    np.random.seed(seed)
    
    # Create asymmetric transition matrix
    P = np.abs(np.random.randn(n, n)) + 0.1
    P = P + 0.5 * np.random.randn(n, n)
    P = np.maximum(P, 0.01)
    P = P / P.sum(axis=1, keepdims=True)

    gamma = 0.2
    # Successor representation
    SR = np.linalg.inv(np.eye(n) - gamma * P)
    # Non-symmetric Laplacian: Λ in the derivation
    L = 1.1 * np.eye(n) - (1 - gamma) * P @ SR
    
    return L

def compute_eigendecomposition(L: np.ndarray, k: int = None):
    """Compute ground truth eigendecomposition for validation."""
    eigenvalues, right_eigenvectors = np.linalg.eig(L)
    
    # Sort by real part
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    right_eigenvectors = right_eigenvectors[:, idx]
    
    # Compute left eigenvectors (from L^T)
    _, left_eigenvectors = np.linalg.eig(L.T)
    left_eigenvectors = left_eigenvectors[:, idx]
    
    # Normalize for biorthogonality: <u_i, v_j> = δ_ij
    for i in range(len(eigenvalues)):
        scale = left_eigenvectors[:, i].conj() @ right_eigenvectors[:, i]
        if np.abs(scale) > 1e-10:
            left_eigenvectors[:, i] = left_eigenvectors[:, i] / np.sqrt(np.abs(scale))
            right_eigenvectors[:, i] = right_eigenvectors[:, i] / np.sqrt(np.abs(scale))
            
    if k is not None:
        eigenvalues = eigenvalues[:k]
        left_eigenvectors = left_eigenvectors[:, :k]
        right_eigenvectors = right_eigenvectors[:, :k]
        
    return {
        'eigenvalues': eigenvalues,
        'left': left_eigenvectors,
        'right': right_eigenvectors,
    }

def compute_exact_gradients(
    X_real: np.ndarray,      # Re(x)
    X_imag: np.ndarray,      # Im(x)
    Y_real: np.ndarray,      # Re(y)
    Y_imag: np.ndarray,      # Im(y)
    epsilon_x: np.ndarray,   # ε^x: Duals for Re(h^x)
    epsilon_y: np.ndarray,   # ε^y: Duals for Re(h^y)
    zeta_x: np.ndarray,      # ζ^x: Duals for Im(h^x)
    zeta_y: np.ndarray,      # ζ^y: Duals for Im(h^y)
    L: np.ndarray,           # Matrix Λ
    D: np.ndarray,           # Matrix D = diag(ρ)
    b: float,                # Barrier coefficient
) -> Dict[str, np.ndarray]:
    """
    Compute exact gradients ∇_Θ including the new normalization barrier.
    """
    n_states, d = X_real.shape
    D_diag = np.diag(D)

    # Initialize gradient accumulators
    g_X_real = np.zeros_like(X_real)
    g_X_imag = np.zeros_like(X_imag)
    g_Y_real = np.zeros_like(Y_real)
    g_Y_imag = np.zeros_like(Y_imag)

    # Precompute weighted vectors D*y and D*x
    DY_real = D_diag @ Y_real
    DY_imag = D_diag @ Y_imag
    DX_real = D_diag @ X_real
    DX_imag = D_diag @ X_imag

    # -------------------------------------------------------------------
    # 1. Constraint Errors (Bi-orthogonality)
    #    h_{jk}^x = <x_j, y_k>_ρ - δ_{jk}
    # -------------------------------------------------------------------
    # Re(h) = a^x a^y + b^x b^y - δ
    h_x_real = X_real.T @ DY_real + X_imag.T @ DY_imag - np.eye(d)
    # Im(h) = a^x b^y - b^x a^y
    h_x_imag = X_real.T @ DY_imag - X_imag.T @ DY_real

    # Symmetry: h_{jk}^y corresponds to <x_k, y_j> (indices flipped)
    h_y_real = h_x_real.T
    h_y_imag = h_x_imag.T

    # Apply causal mask (j <= k) for the triangular system
    mask = np.tril(np.ones((d, d)))
    h_x_real = h_x_real * mask
    h_x_imag = h_x_imag * mask
    h_y_real = h_y_real * mask
    h_y_imag = h_y_imag * mask

    # Dual gradients are exactly the constraint errors
    g_epsilon_x = h_x_real.copy()
    g_epsilon_y = h_y_real.copy()
    g_zeta_x = h_x_imag.copy()
    g_zeta_y = h_y_imag.copy()

    # Precompute Linear Operator terms: D Λ
    DL = D_diag @ L
    LTD = L.T @ D_diag

    # -------------------------------------------------------------------
    # 2. Primal Gradients (Eigensystem)
    # -------------------------------------------------------------------
    norm_errors_x = []
    norm_errors_y = []

    for i in range(d):
        x_i_real = X_real[:, i]
        x_i_imag = X_imag[:, i]
        y_i_real = Y_real[:, i]
        y_i_imag = Y_imag[:, i]

        # --- L_graph Term ---
        # Calculate inner product: Re(<x_i, Λ y_i>_ρ)
        D_Ly_i_real = DL @ y_i_real
        D_Ly_i_imag = DL @ y_i_imag
        
        inner_prod_real = x_i_real @ D_Ly_i_real + x_i_imag @ D_Ly_i_imag
        
        # Derivation uses sign(Re(...)) for absolute value loss
        sign_term = np.sign(inner_prod_real)
        
        # ∇_x L_graph = sign * D Λ y
        g_xi_real = sign_term * D_Ly_i_real
        g_xi_imag = sign_term * D_Ly_i_imag

        # ∇_y L_graph = sign * Λ^T D x
        LTD_x_i_real = LTD @ x_i_real
        LTD_x_i_imag = LTD @ x_i_imag
        
        g_yi_real = sign_term * LTD_x_i_real
        g_yi_imag = sign_term * LTD_x_i_imag

        # --- L_norm Term (NEW) ---
        # L_norm = b/2 * (<x, x> - 1)^2
        # Gradient = 2 * b * (<x, x> - 1) * D x
        
        # For X
        norm_sq_x = x_i_real @ DX_real[:, i] + x_i_imag @ DX_imag[:, i]
        norm_err_x = norm_sq_x - 1.0
        norm_errors_x.append(norm_err_x)
        
        g_xi_real += 2 * b * norm_err_x * DX_real[:, i]
        g_xi_imag += 2 * b * norm_err_x * DX_imag[:, i]

        # For Y
        norm_sq_y = y_i_real @ DY_real[:, i] + y_i_imag @ DY_imag[:, i]
        norm_err_y = norm_sq_y - 1.0
        norm_errors_y.append(norm_err_y)
        
        g_yi_real += 2 * b * norm_err_y * DY_real[:, i]
        g_yi_imag += 2 * b * norm_err_y * DY_imag[:, i]

        # --- L_bi-orth and L_barrier (Existing Constraints) ---
        for k in range(i + 1):
            # Real part of coeff: ε - b * Re(h)
            coef_real = epsilon_x[i, k] - b * h_x_real[i, k]
            # Imag part of coeff: -ζ + b * Im(h)
            coef_imag = -zeta_x[i, k] + b * h_x_imag[i, k]

            # Term: - Coeff * (D y_k)
            D_y_k_real = DY_real[:, k]
            D_y_k_imag = DY_imag[:, k]

            term_real = -coef_real * D_y_k_real + coef_imag * D_y_k_imag
            term_imag = -coef_real * D_y_k_imag - coef_imag * D_y_k_real

            g_xi_real += term_real
            g_xi_imag += term_imag

        for k in range(i + 1):
            # Real part of coeff: ε - b * Re(h)
            coef_real = epsilon_y[i, k] - b * h_y_real[i, k]
            # Imag part of coeff: ζ - b * Im(h)
            coef_imag = zeta_y[i, k] - b * h_y_imag[i, k]

            # Term: - Coeff * (D x_k)
            D_x_k_real = DX_real[:, k]
            D_x_k_imag = DX_imag[:, k]

            term_real = -coef_real * D_x_k_real + coef_imag * D_x_k_imag
            term_imag = -coef_real * D_x_k_imag - coef_imag * D_x_k_real

            g_yi_real += term_real
            g_yi_imag += term_imag

        g_X_real[:, i] = g_xi_real
        g_X_imag[:, i] = g_xi_imag
        g_Y_real[:, i] = g_yi_real
        g_Y_imag[:, i] = g_yi_imag

    return {
        'g_X_real': g_X_real,
        'g_X_imag': g_X_imag,
        'g_Y_real': g_Y_real,
        'g_Y_imag': g_Y_imag,
        'g_epsilon_x': g_epsilon_x,
        'g_epsilon_y': g_epsilon_y,
        'g_zeta_x': g_zeta_x,
        'g_zeta_y': g_zeta_y,
        'h_x_real': h_x_real,
        'h_x_imag': h_x_imag,
        'h_y_real': h_y_real,
        'h_y_imag': h_y_imag,
        'norm_err_x': np.array(norm_errors_x),
        'norm_err_y': np.array(norm_errors_y),
    }

def compute_cosine_similarity(learned: np.ndarray, gt: np.ndarray, D: np.ndarray = None) -> float:
    """Compute cosine similarity between complex vectors."""
    # Normalize by largest magnitude component
    magnitudes = np.abs(learned)
    max_idx = np.argmax(magnitudes)
    scale = learned[max_idx]
    if np.abs(scale) > 1e-10:
        learned = learned / scale
        
    if D is not None:
        sqrt_D = np.sqrt(D)
        learned = learned * sqrt_D
        gt = gt * sqrt_D
        
    inner = np.abs(np.vdot(learned, gt))
    norm_l = np.linalg.norm(learned)
    norm_gt = np.linalg.norm(gt)
    
    return inner / (norm_l * norm_gt + 1e-10)

def run_experiment(
    n_states: int = 20,
    d: int = 1,
    learning_rate: float = 0.01,
    barrier_coef: float = 4.0,
    num_steps: int = 5000,
    log_freq: int = 100,
    seed: int = 42,
):
    """Run the exact gradient dynamics simulation."""
    print(f"=" * 80)
    print(f"Testing exact gradient dynamics with Norm Barrier")
    print(f"  n_states={n_states}, d={d}, lr={learning_rate}, b={barrier_coef}")
    print(f"=" * 80)

    L = create_complex_laplacian(n_states, seed=seed)
    D = np.ones(n_states) / n_states

    # Ground truth
    eigendecomp = compute_eigendecomposition(L, k=d)
    gt_vals = eigendecomp['eigenvalues']
    gt_left = eigendecomp['left']
    gt_right = eigendecomp['right']

    print(f"\nGround truth eigenvalues:")
    for i in range(d):
        print(f"  λ_{i}: {np.real(gt_vals[i]):.6f} + {np.imag(gt_vals[i]):.6f}i")

    # Initialize Primal Variables
    np.random.seed(seed + 1)
    init_scale = 0.1
    X_real = init_scale * np.random.randn(n_states, d)
    X_imag = init_scale * np.random.randn(n_states, d)
    Y_real = init_scale * np.random.randn(n_states, d)
    Y_imag = init_scale * np.random.randn(n_states, d)

    # Initialize Dual Variables (ε and ζ)
    epsilon_x = np.zeros((d, d))
    epsilon_y = np.zeros((d, d))
    zeta_x = np.zeros((d, d))
    zeta_y = np.zeros((d, d))

    mask = np.tril(np.ones((d, d)))

    print(f"\nStarting optimization...")
    print(f"{'Step':>8} {'Constr Err':>12} {'Norm Err':>12} {'Grad Norm':>12} {'Left Sim':>10} {'Right Sim':>10} {'λ Est Real':>12}")
    print("-" * 90)

    history = []

    for step in range(num_steps):
        grads = compute_exact_gradients(
            X_real, X_imag, Y_real, Y_imag,
            epsilon_x, epsilon_y, zeta_x, zeta_y,
            L, D, barrier_coef
        )

        # Update Primal: Gradient Descent
        X_real -= learning_rate * grads['g_X_real']
        X_imag -= learning_rate * grads['g_X_imag']
        Y_real -= learning_rate * grads['g_Y_real']
        Y_imag -= learning_rate * grads['g_Y_imag']

        # Update Dual: Gradient Ascent (maximization)
        epsilon_x += learning_rate * grads['g_epsilon_x']
        epsilon_y += learning_rate * grads['g_epsilon_y']
        zeta_x += learning_rate * grads['g_zeta_x']
        zeta_y += learning_rate * grads['g_zeta_y']

        # Mask Duals
        epsilon_x *= mask; epsilon_y *= mask
        zeta_x *= mask;    zeta_y *= mask

        if step % log_freq == 0:
            constraint_err = (np.abs(grads['h_x_real']).sum() + np.abs(grads['h_x_imag']).sum() +
                              np.abs(grads['h_y_real']).sum() + np.abs(grads['h_y_imag']).sum())
            
            norm_err = np.sum(np.abs(grads['norm_err_x'])) + np.sum(np.abs(grads['norm_err_y']))

            grad_norm = np.sqrt(
                np.sum(grads['g_X_real']**2) + np.sum(grads['g_X_imag']**2) +
                np.sum(grads['g_Y_real']**2) + np.sum(grads['g_Y_imag']**2)
            )

            # Metrics
            left_sims = []
            right_sims = []
            ev_ests_real = []
            
            for i in range(d):
                l_vec = X_real[:, i] + 1j * X_imag[:, i]
                r_vec = Y_real[:, i] + 1j * Y_imag[:, i]
                
                left_sims.append(compute_cosine_similarity(l_vec, gt_left[:, i], D))
                right_sims.append(compute_cosine_similarity(r_vec, gt_right[:, i], None))
                
                ev_ests_real.append(-0.5 * (epsilon_x[i, i] + epsilon_y[i, i]))

            avg_l = np.mean(left_sims)
            avg_r = np.mean(right_sims)

            history.append({
                'step': step, 'constraint_error': constraint_err, 'norm_error': norm_err,
                'grad_norm': grad_norm, 'left_sim': avg_l, 'right_sim': avg_r
            })

            print(f"{step:>8} {constraint_err:>12.6f} {norm_err:>12.6f} {grad_norm:>12.6f} {avg_l:>10.4f} {avg_r:>10.4f} {ev_ests_real[0]:>12.6f}")

    print(f"\nFinal Result (Step {num_steps}):")
    print(f"  Constraint Error: {history[-1]['constraint_error']:.6f}")
    print(f"  Norm Error: {history[-1]['norm_error']:.6f}")
    print(f"  Left Sim: {history[-1]['left_sim']:.4f}")
    
    converged = history[-1]['constraint_error'] < 0.1 and history[-1]['left_sim'] > 0.9
    print(f"  Convergence: {'SUCCESS' if converged else 'FAILED'}")
    return history, converged

def get_flattened_state(X_r, X_i, Y_r, Y_i, ep_x, ep_y, ze_x, ze_y):
    """Flatten all parameters into a single state vector Theta."""
    return np.concatenate([
        X_r.ravel(), X_i.ravel(), 
        Y_r.ravel(), Y_i.ravel(),
        ep_x.ravel(), ep_y.ravel(),
        ze_x.ravel(), ze_y.ravel()
    ])

def reshape_state(Theta, n, d):
    """Reshape flat Theta back into component matrices."""
    # Sizes
    size_vec = n * d
    size_dual = d * d
    
    # Indices
    i = 0
    X_r = Theta[i : i+size_vec].reshape(n, d); i += size_vec
    X_i = Theta[i : i+size_vec].reshape(n, d); i += size_vec
    Y_r = Theta[i : i+size_vec].reshape(n, d); i += size_vec
    Y_i = Theta[i : i+size_vec].reshape(n, d); i += size_vec
    
    ep_x = Theta[i : i+size_dual].reshape(d, d); i += size_dual
    ep_y = Theta[i : i+size_dual].reshape(d, d); i += size_dual
    ze_x = Theta[i : i+size_dual].reshape(d, d); i += size_dual
    ze_y = Theta[i : i+size_dual].reshape(d, d); i += size_dual
    
    return X_r, X_i, Y_r, Y_i, ep_x, ep_y, ze_x, ze_y

def compute_update_vector(Theta, L, D, b, n, d, mask):
    """
    Computes the update vector F(Theta) such that Theta_{t+1} = Theta_t + F(Theta).
    F consists of (-grad_primal) and (+grad_dual).
    """
    X_r, X_i, Y_r, Y_i, ep_x, ep_y, ze_x, ze_y = reshape_state(Theta, n, d)
    
    # Get gradients
    grads = compute_exact_gradients(
        X_r, X_i, Y_r, Y_i, ep_x, ep_y, ze_x, ze_y, L, D, b
    )
    
    # Primal updates (Gradient Descent: -grad)
    dX_r = -grads['g_X_real']
    dX_i = -grads['g_X_imag']
    dY_r = -grads['g_Y_real']
    dY_i = -grads['g_Y_imag']
    
    # Dual updates (Gradient Ascent: +grad)
    # Note: We must apply the mask to the gradient updates to match the code logic
    dep_x = grads['g_epsilon_x'] * mask
    dep_y = grads['g_epsilon_y'] * mask
    dze_x = grads['g_zeta_x'] * mask
    dze_y = grads['g_zeta_y'] * mask
    
    return np.concatenate([
        dX_r.ravel(), dX_i.ravel(),
        dY_r.ravel(), dY_i.ravel(),
        dep_x.ravel(), dep_y.ravel(),
        dze_x.ravel(), dze_y.ravel()
    ])

def analyze_stability(n_states=10, d=1, b=4.0, seed=42):
    print(f"\nStability Analysis (n={n_states}, d={d}, b={b})")
    
    # 1. Setup Environment
    L = create_complex_laplacian(n_states, seed=seed)
    D = np.ones(n_states) / n_states
    
    # 2. Get Ground Truth (Fixed Point)
    ed = compute_eigendecomposition(L, k=d)
    vals = ed['eigenvalues']
    
    # Primal Fixed Point
    left_u = ed['left']
    left_x = left_u / D[:, None] # Metric correction
    
    X_r_star = np.real(left_x)
    X_i_star = np.imag(left_x)
    
    Y_r_star = np.real(ed['right'])
    Y_i_star = np.imag(ed['right'])
    
    # Dual Fixed Point
    ep_x_star = np.zeros((d,d))
    ep_y_star = np.zeros((d,d))
    ze_x_star = np.zeros((d,d))
    ze_y_star = np.zeros((d,d))
    
    for k in range(d):
        re_val = np.real(vals[k])
        im_val = np.imag(vals[k])
        sign_re = np.sign(re_val) if np.abs(re_val) > 1e-9 else 1.0

        ep_x_star[k,k] = sign_re * re_val
        ep_y_star[k,k] = sign_re * re_val

        ze_x_star[k,k] = -sign_re * im_val
        ze_y_star[k,k] = -sign_re * im_val

    mask = np.tril(np.ones((d, d)))
    
    Theta_star = get_flattened_state(
        X_r_star, X_i_star, Y_r_star, Y_i_star, 
        ep_x_star, ep_y_star, ze_x_star, ze_y_star
    )
    
    # 3. Compute Jacobian numerically
    epsilon = 1e-6
    N = len(Theta_star)
    Jacobian = np.zeros((N, N))
    
    print(f"  Computing Jacobian ({N}x{N})...")
    
    F_star = compute_update_vector(Theta_star, L, D, b, n_states, d, mask)
    
    for j in range(N):
        Theta_perturb = Theta_star.copy()
        Theta_perturb[j] += epsilon
        F_perturb = compute_update_vector(Theta_perturb, L, D, b, n_states, d, mask)
        
        Jacobian[:, j] = (F_perturb - F_star) / epsilon
        
    # 4. Compute Eigenvalues
    evals_J = np.linalg.eigvals(Jacobian)
    
    # 5. Analysis
    max_real = np.max(np.real(evals_J))
    print(f"  Max Real Part of Jacobian Eigenvalues: {max_real:.6f}")
    
    if max_real > 0:
        print("  >> UNSTABLE (Continuous time divergence)")
    elif max_real == 0:
        print("  >> MARGINAL (Oscillations likely)")
    else:
        print("  >> STABLE")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(np.real(evals_J), np.imag(evals_J), alpha=0.6)
    plt.axvline(0, color='r', linestyle='--', alpha=0.3)
    plt.axhline(0, color='r', linestyle='--', alpha=0.3)
    plt.title(f"Jacobian Eigenvalues (b={b})\nMax Re: {max_real:.4f}")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    print("\n" + "=" * 70)
    print("  EXACT GRADIENT DYNAMICS TEST (WITH NORM BARRIER)")
    print("=" * 70)

    # Use smaller learning rate because 'sign' gradient is more aggressive
    run_experiment(n_states=10, d=1, learning_rate=0.0005, num_steps=600000, log_freq=5000, barrier_coef=4.0)

    # Stability Analysis
    # Test with current settings
    analyze_stability(b=4.0)
    
    # Test with higher barrier
    analyze_stability(b=20.0)

if __name__ == "__main__":
    main()