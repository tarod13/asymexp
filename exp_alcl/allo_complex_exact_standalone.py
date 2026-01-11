"""
Standalone test of exact gradient dynamics for learning complex eigenvectors.

This is a minimal implementation to verify that the exact gradient formulas
from Complex_ALLO.pdf converge correctly, without dependencies on the full codebase.

Usage:
    python3 allo_complex_exact_standalone.py

The gradient formulas (from PDF Section 1.2):

For left eigenvector x_i:
  g_{x_i} = Re(<x̄_i, Ly_i>_ρ) * D*L*ȳ_i - Σ_{k=1}^{i} (α_{x,ik} + iβ_{x,ik} - b*h_{x,ik}) * D*ȳ_k

For right eigenvector y_i:
  g_{y_i} = Re(<x̄_i, Ly_i>_ρ) * L^T*D*x̄_i - Σ_{k=1}^{i} (α_{y,ik} + iβ_{y,ik} - b*h_{y,ik}) * D*x̄_k

For dual variables:
  g_{α_{x,jk}} = Re(h_{x,jk})
  g_{α_{y,jk}} = Re(h_{y,jk})
  g_{β_{x,jk}} = Im(h_{x,jk})
  g_{β_{y,jk}} = Im(h_{y,jk})

Update rule: Θ[t+1] = Θ[t] - α * g_Θ(Θ[t])
(Duals use negative gradient = gradient ascent)
"""

import numpy as np
from typing import Dict, Tuple


def create_random_nonsymmetric_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Create a random non-symmetric matrix with real eigenvalues for testing."""
    np.random.seed(seed)
    # Create a random stochastic-ish matrix
    P = np.abs(np.random.randn(n, n))
    P = P / P.sum(axis=1, keepdims=True)

    # Compute Laplacian-like: L = I - P
    L = np.eye(n) - 0.5 * P
    return L


def create_complex_laplacian(n: int, seed: int = 42) -> np.ndarray:
    """Create a Laplacian that has complex eigenvalues."""
    np.random.seed(seed)

    # Create asymmetric transition matrix
    P = np.abs(np.random.randn(n, n)) + 0.1
    # Make it more asymmetric
    P = P + 0.5 * np.random.randn(n, n)
    P = np.maximum(P, 0.01)  # Keep positive
    P = P / P.sum(axis=1, keepdims=True)  # Row-normalize

    gamma = 0.2
    # Successor representation
    SR = np.linalg.inv(np.eye(n) - gamma * P)
    # Non-symmetric Laplacian
    L = 1.1 * np.eye(n) - (1 - gamma) * P @ SR

    return L


def compute_eigendecomposition(L: np.ndarray, k: int = None):
    """Compute eigendecomposition of matrix L."""
    eigenvalues, right_eigenvectors = np.linalg.eig(L)

    # Sort by real part (ascending)
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    right_eigenvectors = right_eigenvectors[:, idx]

    # Compute left eigenvectors (from L^T)
    _, left_eigenvectors = np.linalg.eig(L.T)
    # Match left to right eigenvectors
    left_eigenvectors = left_eigenvectors[:, idx]

    # Normalize to have biorthogonality
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
    X_real: np.ndarray,  # (n_states, d) - left eigenvector real parts
    X_imag: np.ndarray,  # (n_states, d) - left eigenvector imag parts
    Y_real: np.ndarray,  # (n_states, d) - right eigenvector real parts
    Y_imag: np.ndarray,  # (n_states, d) - right eigenvector imag parts
    alpha_x: np.ndarray,  # (d, d) lower triangular - duals for Re(h_x)
    alpha_y: np.ndarray,  # (d, d) lower triangular - duals for Re(h_y)
    beta_x: np.ndarray,   # (d, d) lower triangular - duals for Im(h_x)
    beta_y: np.ndarray,   # (d, d) lower triangular - duals for Im(h_y)
    L: np.ndarray,        # (n_states, n_states) - Laplacian matrix
    D: np.ndarray,        # (n_states,) - sampling distribution (diagonal)
    b: float,             # barrier coefficient
) -> Dict[str, np.ndarray]:
    """
    Compute exact gradients according to the theory in Complex_ALLO.pdf.
    """
    n_states, d = X_real.shape
    D_diag = np.diag(D)  # Convert to diagonal matrix

    # Initialize gradient arrays
    g_X_real = np.zeros_like(X_real)
    g_X_imag = np.zeros_like(X_imag)
    g_Y_real = np.zeros_like(Y_real)
    g_Y_imag = np.zeros_like(Y_imag)

    # Precompute D @ Y and D @ X
    DY_real = D_diag @ Y_real
    DY_imag = D_diag @ Y_imag
    DX_real = D_diag @ X_real
    DX_imag = D_diag @ X_imag

    # Compute all constraint errors h_{x,jk} and h_{y,jk}
    # h_{x,jk} = <x̄_j, [[y_k]]>_ρ - δ_{jk} = x_j^T D conj(y_k) - δ_{jk}

    # For h_x: x_j^T D conj(y_k)
    # Real: x_real^T D y_real + x_imag^T D y_imag
    # Imag: -x_real^T D y_imag + x_imag^T D y_real
    h_x_real = X_real.T @ DY_real + X_imag.T @ DY_imag - np.eye(d)
    h_x_imag = -X_real.T @ DY_imag + X_imag.T @ DY_real

    # For h_y: [[x_k]]^T D y_j = conj(x_k)^T D y_j
    # Real: x_real^T D y_real + x_imag^T D y_imag
    # Imag: x_real^T D y_imag - x_imag^T D y_real
    h_y_real = X_real.T @ DY_real + X_imag.T @ DY_imag - np.eye(d)
    h_y_imag = X_real.T @ DY_imag - X_imag.T @ DY_real

    # Apply lower triangular mask (we only constrain j >= k)
    tril_mask = np.tril(np.ones((d, d)))
    h_x_real = h_x_real * tril_mask
    h_x_imag = h_x_imag * tril_mask
    h_y_real = h_y_real * tril_mask
    h_y_imag = h_y_imag * tril_mask

    # Dual gradients are just the constraint errors
    g_alpha_x = h_x_real.copy()
    g_alpha_y = h_y_real.copy()
    g_beta_x = h_x_imag.copy()
    g_beta_y = h_y_imag.copy()

    # Precompute matrices for efficiency
    DL = D_diag @ L
    LTD = L.T @ D_diag

    # Compute gradients for each eigenvector
    for i in range(d):
        x_i_real = X_real[:, i]
        x_i_imag = X_imag[:, i]
        y_i_real = Y_real[:, i]
        y_i_imag = Y_imag[:, i]

        # Compute Ly_i
        Ly_i_real = L @ y_i_real
        Ly_i_imag = L @ y_i_imag

        # Compute Re(<x̄_i, Ly_i>_ρ) = x_i^T D conj(Ly_i)
        # = x_real^T D Ly_real + x_imag^T D Ly_imag
        D_Ly_i_real = D * Ly_i_real
        D_Ly_i_imag = D * Ly_i_imag
        spectral_product_real = x_i_real @ D_Ly_i_real + x_i_imag @ D_Ly_i_imag

        # First term for x_i: Re(<x̄_i, Ly_i>_ρ) * D*L*ȳ_i
        # ȳ_i = y_i_real - i*y_i_imag (conjugate)
        DL_ybar_i_real = DL @ y_i_real
        DL_ybar_i_imag = -DL @ y_i_imag  # Conjugate

        g_xi_real = spectral_product_real * DL_ybar_i_real
        g_xi_imag = spectral_product_real * DL_ybar_i_imag

        # First term for y_i: Re(<x̄_i, Ly_i>_ρ) * L^T*D*x̄_i
        # x̄_i = x_i_real - i*x_i_imag (conjugate)
        LTD_xbar_i_real = LTD @ x_i_real
        LTD_xbar_i_imag = -LTD @ x_i_imag  # Conjugate

        g_yi_real = spectral_product_real * LTD_xbar_i_real
        g_yi_imag = spectral_product_real * LTD_xbar_i_imag

        # Second term: -Σ_{k=1}^{i} (α_{x,ik} + iβ_{x,ik} - b*h_{x,ik}) * D*ȳ_k
        for k in range(i + 1):
            # Complex coefficient
            coef_real = alpha_x[i, k] - b * h_x_real[i, k]
            coef_imag = beta_x[i, k] - b * h_x_imag[i, k]

            # D*ȳ_k
            D_ybar_k_real = DY_real[:, k]
            D_ybar_k_imag = -DY_imag[:, k]  # Conjugate

            # Complex multiplication
            term_real = coef_real * D_ybar_k_real - coef_imag * D_ybar_k_imag
            term_imag = coef_real * D_ybar_k_imag + coef_imag * D_ybar_k_real

            g_xi_real = g_xi_real - term_real
            g_xi_imag = g_xi_imag - term_imag

        # For y_i: -Σ_{k=1}^{i} (α_{y,ik} + iβ_{y,ik} - b*h_{y,ik}) * D*x̄_k
        for k in range(i + 1):
            coef_real = alpha_y[i, k] - b * h_y_real[i, k]
            coef_imag = beta_y[i, k] - b * h_y_imag[i, k]

            D_xbar_k_real = DX_real[:, k]
            D_xbar_k_imag = -DX_imag[:, k]  # Conjugate

            term_real = coef_real * D_xbar_k_real - coef_imag * D_xbar_k_imag
            term_imag = coef_real * D_xbar_k_imag + coef_imag * D_xbar_k_real

            g_yi_real = g_yi_real - term_real
            g_yi_imag = g_yi_imag - term_imag

        g_X_real[:, i] = g_xi_real
        g_X_imag[:, i] = g_xi_imag
        g_Y_real[:, i] = g_yi_real
        g_Y_imag[:, i] = g_yi_imag

    return {
        'g_X_real': g_X_real,
        'g_X_imag': g_X_imag,
        'g_Y_real': g_Y_real,
        'g_Y_imag': g_Y_imag,
        'g_alpha_x': g_alpha_x,
        'g_alpha_y': g_alpha_y,
        'g_beta_x': g_beta_x,
        'g_beta_y': g_beta_y,
        'h_x_real': h_x_real,
        'h_x_imag': h_x_imag,
        'h_y_real': h_y_real,
        'h_y_imag': h_y_imag,
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

    inner = np.abs(np.vdot(learned, gt))  # Complex inner product
    norm_l = np.linalg.norm(learned)
    norm_gt = np.linalg.norm(gt)

    return inner / (norm_l * norm_gt + 1e-10)


def run_experiment(
    n_states: int = 20,
    d: int = 1,  # Number of eigenvectors
    learning_rate: float = 0.01,
    barrier_coef: float = 4.0,
    num_steps: int = 5000,
    log_freq: int = 100,
    seed: int = 42,
):
    """Run the exact gradient descent experiment."""
    print(f"=" * 60)
    print(f"Testing exact gradient dynamics")
    print(f"  n_states={n_states}, d={d}, lr={learning_rate}, b={barrier_coef}")
    print(f"=" * 60)

    # Create test matrix
    L = create_complex_laplacian(n_states, seed=seed)

    # Create uniform sampling distribution
    D = np.ones(n_states) / n_states

    # Compute ground truth eigendecomposition
    eigendecomp = compute_eigendecomposition(L, k=d)
    gt_eigenvalues = eigendecomp['eigenvalues']
    gt_left = eigendecomp['left']
    gt_right = eigendecomp['right']

    print(f"\nGround truth eigenvalues:")
    for i in range(d):
        print(f"  λ_{i}: {np.real(gt_eigenvalues[i]):.6f} + {np.imag(gt_eigenvalues[i]):.6f}i")

    # Check if eigenvalues are complex
    has_complex = np.any(np.abs(np.imag(gt_eigenvalues)) > 1e-6)
    print(f"  Has complex eigenvalues: {has_complex}")

    # Initialize parameters
    np.random.seed(seed + 1)
    init_scale = 0.1

    X_real = init_scale * np.random.randn(n_states, d)
    X_imag = init_scale * np.random.randn(n_states, d)
    Y_real = init_scale * np.random.randn(n_states, d)
    Y_imag = init_scale * np.random.randn(n_states, d)

    alpha_x = np.zeros((d, d))
    alpha_y = np.zeros((d, d))
    beta_x = np.zeros((d, d))
    beta_y = np.zeros((d, d))

    # Lower triangular mask
    tril_mask = np.tril(np.ones((d, d)))

    print(f"\nStarting optimization...")
    print(f"{'Step':>8} {'Constr Err':>12} {'Grad Norm':>12} {'Left Sim':>10} {'Right Sim':>10} {'λ Est Real':>12}")
    print("-" * 70)

    history = []

    for step in range(num_steps):
        # Compute gradients
        grads = compute_exact_gradients(
            X_real, X_imag, Y_real, Y_imag,
            alpha_x, alpha_y, beta_x, beta_y,
            L, D, barrier_coef
        )

        # Update primal variables (gradient descent)
        X_real = X_real - learning_rate * grads['g_X_real']
        X_imag = X_imag - learning_rate * grads['g_X_imag']
        Y_real = Y_real - learning_rate * grads['g_Y_real']
        Y_imag = Y_imag - learning_rate * grads['g_Y_imag']

        # Update dual variables (gradient ascent)
        alpha_x = alpha_x + learning_rate * grads['g_alpha_x']
        alpha_y = alpha_y + learning_rate * grads['g_alpha_y']
        beta_x = beta_x + learning_rate * grads['g_beta_x']
        beta_y = beta_y + learning_rate * grads['g_beta_y']

        # Apply lower triangular mask
        alpha_x = alpha_x * tril_mask
        alpha_y = alpha_y * tril_mask
        beta_x = beta_x * tril_mask
        beta_y = beta_y * tril_mask

        # Logging
        if step % log_freq == 0:
            # Constraint error
            constraint_error = (np.abs(grads['h_x_real']).sum() + np.abs(grads['h_x_imag']).sum() +
                              np.abs(grads['h_y_real']).sum() + np.abs(grads['h_y_imag']).sum())

            # Gradient norm
            grad_norm = np.sqrt(
                np.sum(grads['g_X_real']**2) + np.sum(grads['g_X_imag']**2) +
                np.sum(grads['g_Y_real']**2) + np.sum(grads['g_Y_imag']**2)
            )

            # Cosine similarities
            left_sims = []
            right_sims = []
            eigenvalue_ests_real = []
            eigenvalue_ests_imag = []

            for i in range(d):
                learned_left = X_real[:, i] + 1j * X_imag[:, i]
                learned_right = Y_real[:, i] + 1j * Y_imag[:, i]

                left_sim = compute_cosine_similarity(learned_left, gt_left[:, i], D)
                right_sim = compute_cosine_similarity(learned_right, gt_right[:, i], None)

                left_sims.append(left_sim)
                right_sims.append(right_sim)

                # Eigenvalue estimate from diagonal duals
                ev_est_real = -0.5 * (alpha_x[i, i] + alpha_y[i, i])
                ev_est_imag = -0.5 * (beta_x[i, i] + beta_y[i, i])
                eigenvalue_ests_real.append(ev_est_real)
                eigenvalue_ests_imag.append(ev_est_imag)

            avg_left_sim = np.mean(left_sims)
            avg_right_sim = np.mean(right_sims)

            history.append({
                'step': step,
                'constraint_error': constraint_error,
                'grad_norm': grad_norm,
                'left_sim': avg_left_sim,
                'right_sim': avg_right_sim,
                'eigenvalue_est_real': eigenvalue_ests_real[0] if d > 0 else 0,
                'eigenvalue_est_imag': eigenvalue_ests_imag[0] if d > 0 else 0,
            })

            print(f"{step:>8} {constraint_error:>12.6f} {grad_norm:>12.6f} {avg_left_sim:>10.4f} {avg_right_sim:>10.4f} {eigenvalue_ests_real[0]:>12.6f}")

    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"  Constraint error: {history[-1]['constraint_error']:.6f}")
    print(f"  Gradient norm: {history[-1]['grad_norm']:.6f}")
    print(f"  Left cosine sim: {history[-1]['left_sim']:.4f}")
    print(f"  Right cosine sim: {history[-1]['right_sim']:.4f}")

    for i in range(d):
        print(f"  Eigenvector {i}:")
        ev_est_real = -0.5 * (alpha_x[i, i] + alpha_y[i, i])
        ev_est_imag = -0.5 * (beta_x[i, i] + beta_y[i, i])
        print(f"    λ_est: {ev_est_real:.6f} + {ev_est_imag:.6f}i")
        print(f"    λ_gt:  {np.real(gt_eigenvalues[i]):.6f} + {np.imag(gt_eigenvalues[i]):.6f}i")

    # Check convergence
    converged = history[-1]['constraint_error'] < 0.1 and history[-1]['left_sim'] > 0.9
    print(f"\n  Convergence: {'SUCCESS' if converged else 'FAILED'}")

    return history, converged


def main():
    print("\n" + "=" * 70)
    print("  EXACT GRADIENT DYNAMICS TEST FOR COMPLEX EIGENVECTOR LEARNING")
    print("=" * 70)

    # Test 1: Single eigenvector with small matrix
    print("\n\n>>> TEST 1: Single eigenvector (d=1), small matrix (n=10)")
    history1, converged1 = run_experiment(
        n_states=10,
        d=1,
        learning_rate=0.01,
        barrier_coef=4.0,
        num_steps=3000,
        log_freq=200,
    )

    # Test 2: Single eigenvector with larger matrix
    print("\n\n>>> TEST 2: Single eigenvector (d=1), larger matrix (n=30)")
    history2, converged2 = run_experiment(
        n_states=30,
        d=1,
        learning_rate=0.01,
        barrier_coef=4.0,
        num_steps=5000,
        log_freq=500,
    )

    # Test 3: Multiple eigenvectors
    print("\n\n>>> TEST 3: Multiple eigenvectors (d=3), medium matrix (n=20)")
    history3, converged3 = run_experiment(
        n_states=20,
        d=3,
        learning_rate=0.005,
        barrier_coef=4.0,
        num_steps=10000,
        log_freq=1000,
    )

    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (d=1, n=10): {'PASSED' if converged1 else 'FAILED'}")
    print(f"  Test 2 (d=1, n=30): {'PASSED' if converged2 else 'FAILED'}")
    print(f"  Test 3 (d=3, n=20): {'PASSED' if converged3 else 'FAILED'}")


if __name__ == "__main__":
    main()
