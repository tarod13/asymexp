"""
Standalone test of Complex ALLO with Norm Constraints.

Dynamical System:
    Maximizes: Re(<x, L_adj x>_ρ) + Re(<y, L y>_ρ)
    Constraints:
        1. Norm: <x, x>_ρ = 1, <y, y>_ρ = 1 (Duals: α_x, α_y)
        2. Bi-orth: <x_j, y_k>_ρ = 0 for j < k (Duals: ε, ζ)

This implementation follows the provided "Stability analysis" derivation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def create_complex_laplacian(n: int, seed: int = 42) -> np.ndarray:
    """Create a Laplacian that has complex eigenvalues."""
    np.random.seed(seed)
    # Create asymmetric transition matrix
    P = np.abs(np.random.randn(n, n)) + 0.1
    P = P + 0.5 * np.random.randn(n, n)
    P = np.maximum(P, 0.01)
    P = P / P.sum(axis=1, keepdims=True)

    gamma = 0.2
    # Successor representation
    SR = np.linalg.inv(np.eye(n) - gamma * P)
    # Non-symmetric Laplacian
    L = 1.1 * np.eye(n) - (1 - gamma) * P @ SR
    return L

def compute_eigendecomposition(L: np.ndarray, k: int = None):
    """Compute ground truth eigendecomposition."""
    eigenvalues, right_eigenvectors = np.linalg.eig(L)
    
    # Sort by real part
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    right_eigenvectors = right_eigenvectors[:, idx]
    
    # Compute left eigenvectors (from L^T)
    _, left_eigenvectors = np.linalg.eig(L.T)
    left_eigenvectors = left_eigenvectors[:, idx]
    
    # Biorthogonalize
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

def compute_gradients(
    X_real: np.ndarray,      # Re(x)
    X_imag: np.ndarray,      # Im(x)
    Y_real: np.ndarray,      # Re(y)
    Y_imag: np.ndarray,      # Im(y)
    alpha_x: np.ndarray,     # α^x: Norm duals (vector size d)
    alpha_y: np.ndarray,     # α^y: Norm duals (vector size d)
    epsilon_x: np.ndarray,   # ε^x: Bi-orth duals (strictly lower tri)
    epsilon_y: np.ndarray,   # ε^y: Bi-orth duals (strictly lower tri)
    zeta_x: np.ndarray,      # ζ^x: Bi-orth duals (strictly lower tri)
    zeta_y: np.ndarray,      # ζ^y: Bi-orth duals (strictly lower tri)
    L: np.ndarray,           # Matrix Λ
    D: np.ndarray,           # Matrix D = diag(ρ)
    b: float,                # Barrier coefficient
) -> Dict[str, np.ndarray]:
    """
    Compute gradients based on the new derivation.
    """
    n_states, d = X_real.shape
    D_diag = np.diag(D)

    # Initialize gradient accumulators
    g_X_real = np.zeros_like(X_real)
    g_X_imag = np.zeros_like(X_imag)
    g_Y_real = np.zeros_like(Y_real)
    g_Y_imag = np.zeros_like(Y_imag)

    # Precompute weighted vectors
    DX_real = D_diag @ X_real
    DX_imag = D_diag @ X_imag
    DY_real = D_diag @ Y_real
    DY_imag = D_diag @ Y_imag

    # 1. Calculate Errors
    
    # Norm Errors: h_x = <x, x> - 1
    # <x, x> = x_real^T D x_real + x_imag^T D x_imag
    norm_sq_x = np.sum(X_real * DX_real + X_imag * DX_imag, axis=0)
    h_x = norm_sq_x - 1.0
    
    norm_sq_y = np.sum(Y_real * DY_real + Y_imag * DY_imag, axis=0)
    h_y = norm_sq_y - 1.0

    # Bi-orthogonality Errors: h_{jk}^x = <x_j, y_k>_ρ (No -delta, handled by strictly lower tri)
    # Re(h)
    h_orth_real = X_real.T @ DY_real + X_imag.T @ DY_imag
    # Im(h)
    h_orth_imag = X_real.T @ DY_imag - X_imag.T @ DY_real

    # Strictly lower triangular mask (j < k is irrelevant for gradients due to summation k < i)
    # The gradient loops iterate k < i.
    
    # Precompute Linear Operator terms
    # For X: D L^T x (since derivation uses L_adj)
    # L_adj = L.T
    DLT = D_diag @ L.T
    DL = D_diag @ L

    # 2. Compute Primal Gradients
    for i in range(d):
        # Current vectors
        xi_r, xi_im = X_real[:, i], X_imag[:, i]
        yi_r, yi_im = Y_real[:, i], Y_imag[:, i]
        
        # --- Graph Term ---
        # ∇_x = D L_adj x
        g_xi_r = DLT @ xi_r
        g_xi_im = DLT @ xi_im
        
        # ∇_y = D L y
        g_yi_r = DL @ yi_r
        g_yi_im = DL @ yi_im
        
        # --- Norm Term ---
        # - (α - b*h) D x
        coef_norm_x = alpha_x[i] - b * h_x[i]
        g_xi_r -= coef_norm_x * DX_real[:, i]
        g_xi_im -= coef_norm_x * DX_imag[:, i]
        
        coef_norm_y = alpha_y[i] - b * h_y[i]
        g_yi_r -= coef_norm_y * DY_real[:, i]
        g_yi_im -= coef_norm_y * DY_imag[:, i]

        # --- Bi-orthogonality Term ---
        # Summation over k < i
        for k in range(i):
            # For X: - (ε_{ik} - iζ_{ik} - b * h_{ik}_bar) D y_k
            # h_bar = h_real - i*h_imag
            # Coeff = (ε - b*h_real) + i(-ζ + b*h_imag)
            
            c_xr = epsilon_x[i, k] - b * h_orth_real[i, k]
            c_xi = -zeta_x[i, k] + b * h_orth_imag[i, k]
            
            # Complex mul: (cr + i ci) * (yr + i yi)
            # Re: cr*yr - ci*yi
            # Im: cr*yi + ci*yr
            # We subtract this whole term
            term_xr = c_xr * DY_real[:, k] - c_xi * DY_imag[:, k]
            term_xi = c_xr * DY_imag[:, k] + c_xi * DY_real[:, k]
            
            g_xi_r -= term_xr
            g_xi_im -= term_xi
            
            # For Y: - (ε_{ik} + iζ_{ik} - b * h_{ik}) D x_k
            # Coeff = (ε - b*h_real) + i(ζ - b*h_imag)
            # Note: The latex uses h_{ik}^y, which is <x_k, y_i>.
            # Our matrix h_orth is <x_row, y_col>.
            # h_{ik}^y is h_orth[k, i].
            # BUT the summation is over k < i. The derivation uses indices slightly differently.
            # Let's trust the derivation's structure:
            # ∇y_i uses x_k for k < i.
            
            # Using symmetry from derivation:
            # h_{ik} in derivation usually refers to the constraint between current i and previous k.
            # We use h_orth[k, i] which is <x_k, y_i>.
            
            h_y_ik_r = h_orth_real[k, i]
            h_y_ik_im = h_orth_imag[k, i]
            
            c_yr = epsilon_y[i, k] - b * h_y_ik_r
            c_yi = zeta_y[i, k] - b * h_y_ik_im
            
            term_yr = c_yr * DX_real[:, k] - c_yi * DX_imag[:, k]
            term_yi = c_yr * DX_imag[:, k] + c_yi * DX_real[:, k]
            
            g_yi_r -= term_yr
            g_yi_im -= term_yi

        g_X_real[:, i] = g_xi_r
        g_X_imag[:, i] = g_xi_im
        g_Y_real[:, i] = g_yi_r
        g_Y_imag[:, i] = g_yi_im

    # 3. Compute Dual Gradients
    # These are exactly the constraint errors
    g_alpha_x = h_x
    g_alpha_y = h_y
    
    # Strictly lower triangular mask
    mask = np.tril(np.ones((d, d)), k=-1)
    
    # For epsilon/zeta x: error is h_{ik}^x = <x_i, y_k>
    g_epsilon_x = h_orth_real * mask
    g_zeta_x = h_orth_imag * mask
    
    # For epsilon/zeta y: error is h_{ik}^y = <x_k, y_i> -> Transpose of h_orth
    g_epsilon_y = h_orth_real.T * mask
    g_zeta_y = h_orth_imag.T * mask

    return {
        'g_X_real': g_X_real, 'g_X_imag': g_X_imag,
        'g_Y_real': g_Y_real, 'g_Y_imag': g_Y_imag,
        'g_alpha_x': g_alpha_x, 'g_alpha_y': g_alpha_y,
        'g_epsilon_x': g_epsilon_x, 'g_epsilon_y': g_epsilon_y,
        'g_zeta_x': g_zeta_x, 'g_zeta_y': g_zeta_y,
        'h_x': h_x, 'h_y': h_y
    }

def get_flattened_state(X_r, X_i, Y_r, Y_i, ax, ay, ex, ey, zx, zy):
    return np.concatenate([
        X_r.ravel(), X_i.ravel(), Y_r.ravel(), Y_i.ravel(),
        ax.ravel(), ay.ravel(),
        ex.ravel(), ey.ravel(), zx.ravel(), zy.ravel()
    ])

def reshape_state(Theta, n, d):
    size_vec = n * d
    size_dual_vec = d
    size_dual_mat = d * d
    
    i = 0
    X_r = Theta[i:i+size_vec].reshape(n, d); i+=size_vec
    X_i = Theta[i:i+size_vec].reshape(n, d); i+=size_vec
    Y_r = Theta[i:i+size_vec].reshape(n, d); i+=size_vec
    Y_i = Theta[i:i+size_vec].reshape(n, d); i+=size_vec
    
    ax = Theta[i:i+size_dual_vec]; i+=size_dual_vec
    ay = Theta[i:i+size_dual_vec]; i+=size_dual_vec
    
    ex = Theta[i:i+size_dual_mat].reshape(d, d); i+=size_dual_mat
    ey = Theta[i:i+size_dual_mat].reshape(d, d); i+=size_dual_mat
    zx = Theta[i:i+size_dual_mat].reshape(d, d); i+=size_dual_mat
    zy = Theta[i:i+size_dual_mat].reshape(d, d); i+=size_dual_mat
    
    return X_r, X_i, Y_r, Y_i, ax, ay, ex, ey, zx, zy

def compute_update_vector(Theta, L, D, b, n, d):
    vars = reshape_state(Theta, n, d)
    grads = compute_gradients(*vars, L, D, b)
    
    # Primal Ascent (Maximize Objective)
    # The derivation defines g_a^x as gradient of L.
    # Dynamical system: Theta[t+1] = Theta[t] + alpha * Gradient
    # Note: The derivation says Theta[t] - alpha * g.
    # BUT, g for x is ascent on Graph, descent on constraints.
    # Let's check signs.
    # L_graph is positive. We want to maximize it. So we add Gradient.
    # L_norm is -alpha*h. We want to minimize error.
    # The derivation g_x = D L x - alpha D x.
    # If we use +g_x, we move in direction of eigenvectors. Correct.
    
    dx_r = grads['g_X_real']
    dx_i = grads['g_X_imag']
    dy_r = grads['g_Y_real']
    dy_i = grads['g_Y_imag']
    
    # Dual Ascent (Maximize L w.r.t duals) -> Minimize constraints
    # L_norm = - alpha * h.
    # dL/dAlpha = -h.
    # We want to Update Alpha to maximize L? No, duals enforce constraints.
    # Standard saddle point: Max_primal Min_dual L.
    # Update dual: dual = dual + lr * dL/dDual.
    # If Min dual, dual = dual - lr * dL/dDual.
    # Derivation says: - alpha * g.
    # And defines g_dual = constraint_error.
    # If we do - alpha * h, we are descending the error.
    
    # Let's use the provided update rule: - alpha * g_Theta.
    # Where g_Theta for duals is -Constraint Error.
    # So Update = - alpha * (-Error) = + alpha * Error.
    # This is standard Gradient Ascent on duals (if maximizing).
    
    # Using +Error for dual update
    dax = grads['g_alpha_x']
    day = grads['g_alpha_y']
    dex = grads['g_epsilon_x']
    dey = grads['g_epsilon_y']
    dzx = grads['g_zeta_x']
    dzy = grads['g_zeta_y']
    
    return np.concatenate([
        dx_r.ravel(), dx_i.ravel(), dy_r.ravel(), dy_i.ravel(),
        dax.ravel(), day.ravel(),
        dex.ravel(), dey.ravel(), dzx.ravel(), dzy.ravel()
    ])

def run_experiment(n_states=10, d=2, steps=20000, lr=0.01, b=4.0):
    print(f"\nExperiment: n={n_states}, d={d}, lr={lr}, b={b}")
    L = create_complex_laplacian(n_states)
    D = np.ones(n_states)/n_states
    
    # Ground truth
    gt = compute_eigendecomposition(L, k=d)
    print("GT Eigenvalues:", gt['eigenvalues'])
    
    # Init
    init_s = 0.1
    X_r = np.random.randn(n_states, d) * init_s
    X_i = np.random.randn(n_states, d) * init_s
    Y_r = np.random.randn(n_states, d) * init_s
    Y_i = np.random.randn(n_states, d) * init_s
    
    ax = np.zeros(d); ay = np.zeros(d)
    ex = np.zeros((d,d)); ey = np.zeros((d,d))
    zx = np.zeros((d,d)); zy = np.zeros((d,d))
    
    theta = get_flattened_state(X_r, X_i, Y_r, Y_i, ax, ay, ex, ey, zx, zy)
    
    history = []
    
    for t in range(steps):
        update = compute_update_vector(theta, L, D, b, n_states, d)
        
        # Apply update (Gradient Ascent direction as derived)
        theta += lr * update
        
        if t % 1000 == 0:
            current = reshape_state(theta, n_states, d)
            grads = compute_gradients(*current, L, D, b)
            
            # Metrics
            h_norm = np.mean(np.abs(grads['h_x']) + np.abs(grads['h_y']))
            h_orth = np.sum(np.abs(grads['g_epsilon_x'])) # Only lower tri
            
            # Similarity
            sims = []
            for i in range(d):
                u = current[0][:, i] + 1j * current[1][:, i]
                # Normalize u for cosine sim
                u = u / (np.linalg.norm(u) + 1e-9)
                gt_u = gt['left'][:, i]
                # Cosine sim magnitude
                sim = np.abs(np.vdot(u, gt_u)) / (np.linalg.norm(gt_u) + 1e-9)
                sims.append(sim)
            
            print(f"Step {t}: NormErr={h_norm:.4f}, OrthErr={h_orth:.4f}, LeftSim={np.mean(sims):.4f}")

def analyze_stability(n_states=10, d=1, b=4.0):
    print(f"\nStability Analysis (Jacobean) n={n_states}, d={d}, b={b}")
    L = create_complex_laplacian(n_states)
    D = np.ones(n_states)/n_states
    
    gt = compute_eigendecomposition(L, k=d)
    val = gt['eigenvalues'][0]
    
    # Construct Fixed Point (approximate)
    # Note: If eigenvalue is complex, alpha cannot satisfy real constraint stationary
    # D L^T x = alpha D x.
    # If x is eigenvector, L^T x = conj(val) * x.
    # So alpha should be conj(val). But alpha is real.
    # This implies the system rotates if val is complex.
    
    # We set alpha to Real part of eigenvalue
    alpha_val = np.real(val)
    
    # Primal
    # x must be scaled such that <x,x>_D = 1
    # compute_eigendecomposition returns u such that u^H v = 1.
    # But here we want <x,x>_D = 1.
    u = gt['left'][:, 0]
    # We need x = u / D to match standard forms, or just normalize u w.r.t D
    norm_u = np.sqrt(u.conj().T @ np.diag(D) @ u)
    x = u / norm_u
    
    v = gt['right'][:, 0]
    norm_v = np.sqrt(v.conj().T @ np.diag(D) @ v)
    y = v / norm_v
    
    X_r = np.real(x).reshape(-1, 1)
    X_i = np.imag(x).reshape(-1, 1)
    Y_r = np.real(y).reshape(-1, 1)
    Y_i = np.imag(y).reshape(-1, 1)
    
    ax = np.array([alpha_val])
    ay = np.array([alpha_val])
    
    # Duals for orth are 0 for d=1
    ex = np.zeros((1,1)); ey = np.zeros((1,1))
    zx = np.zeros((1,1)); zy = np.zeros((1,1))
    
    theta_star = get_flattened_state(X_r, X_i, Y_r, Y_i, ax, ay, ex, ey, zx, zy)
    
    # Check if it is actually a fixed point (Update should be 0)
    F0 = compute_update_vector(theta_star, L, D, b, n_states, d)
    print(f"  Update vector norm at GT: {np.linalg.norm(F0):.6f}")
    if np.linalg.norm(F0) > 1e-3:
        print("  [INFO] Ground truth is NOT a stationary point (Expected for Complex Eigenvalues).")
        print("   The system is likely in a limit cycle (rotation).")
        print("   Jacobian eigenvalues will reflect the rotation frequency.")

    # Compute Jacobian
    eps = 1e-6
    N = len(theta_star)
    J = np.zeros((N, N))
    
    print("  Computing Jacobian...")
    for i in range(N):
        t_p = theta_star.copy()
        t_p[i] += eps
        F_p = compute_update_vector(t_p, L, D, b, n_states, d)
        J[:, i] = (F_p - F0) / eps
        
    evals = np.linalg.eigvals(J)
    print(f"  Max Real Part: {np.max(np.real(evals)):.6f}")
    
    plt.figure()
    plt.scatter(np.real(evals), np.imag(evals))
    plt.title(f"Jacobian Eigenvalues (b={b})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_experiment(n_states=10, d=2, b=10.0)
    analyze_stability(n_states=10, d=1, b=10.0)