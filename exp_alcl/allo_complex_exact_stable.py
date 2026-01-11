"""
Stabilized exact gradient dynamics for learning complex eigenvectors.

Key modifications from the raw theory implementation:
1. Eigenvector normalization after each step (prevents magnitude explosion)
2. Separate learning rates for primal and dual variables
3. Gradient clipping for stability
4. Option to disable the spectral objective term initially

The gradient formulas (from PDF Section 1.2) are unchanged, but we add stabilization.
"""

import numpy as np
from typing import Dict, Tuple
import argparse


def create_complex_laplacian(n: int, seed: int = 42, asymmetry: float = 0.3) -> np.ndarray:
    """Create a Laplacian that has complex eigenvalues."""
    np.random.seed(seed)

    # Create asymmetric transition matrix
    P = np.abs(np.random.randn(n, n)) + 0.1
    P = P + asymmetry * np.random.randn(n, n)  # Add asymmetry
    P = np.maximum(P, 0.01)
    P = P / P.sum(axis=1, keepdims=True)

    gamma = 0.2
    SR = np.linalg.inv(np.eye(n) - gamma * P)
    L = 1.1 * np.eye(n) - (1 - gamma) * P @ SR

    return L


def compute_eigendecomposition(L: np.ndarray, k: int = None):
    """Compute eigendecomposition of matrix L."""
    eigenvalues, right_eigenvectors = np.linalg.eig(L)

    # Sort by real part (ascending)
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    right_eigenvectors = right_eigenvectors[:, idx]

    # Compute left eigenvectors
    _, left_eigenvectors = np.linalg.eig(L.T)
    left_eigenvectors = left_eigenvectors[:, idx]

    # Normalize for biorthogonality
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


def normalize_eigenvectors(X_real, X_imag, Y_real, Y_imag, D):
    """
    Normalize eigenvector pairs to have unit norm under D-weighted inner product.
    This prevents magnitude explosion during training.
    """
    n, d = X_real.shape
    D_diag = np.diag(D)

    for i in range(d):
        # Normalize X (left eigenvector) under D-norm
        x = X_real[:, i] + 1j * X_imag[:, i]
        x_norm = np.sqrt(np.real(x.conj() @ D_diag @ x))
        if x_norm > 1e-10:
            X_real[:, i] = X_real[:, i] / x_norm
            X_imag[:, i] = X_imag[:, i] / x_norm

        # Normalize Y (right eigenvector) under standard norm
        y = Y_real[:, i] + 1j * Y_imag[:, i]
        y_norm = np.linalg.norm(y)
        if y_norm > 1e-10:
            Y_real[:, i] = Y_real[:, i] / y_norm
            Y_imag[:, i] = Y_imag[:, i] / y_norm

    return X_real, X_imag, Y_real, Y_imag


def compute_constraint_errors(X_real, X_imag, Y_real, Y_imag, D):
    """Compute biorthogonality constraint errors h_x and h_y."""
    n, d = X_real.shape
    D_diag = np.diag(D)

    DY_real = D_diag @ Y_real
    DY_imag = D_diag @ Y_imag

    # h_x: <x̄_j, [[y_k]]>_ρ - δ_{jk}
    h_x_real = X_real.T @ DY_real + X_imag.T @ DY_imag - np.eye(d)
    h_x_imag = -X_real.T @ DY_imag + X_imag.T @ DY_real

    # h_y: same structure but conjugate differently
    h_y_real = X_real.T @ DY_real + X_imag.T @ DY_imag - np.eye(d)
    h_y_imag = X_real.T @ DY_imag - X_imag.T @ DY_real

    # Lower triangular
    tril_mask = np.tril(np.ones((d, d)))
    h_x_real *= tril_mask
    h_x_imag *= tril_mask
    h_y_real *= tril_mask
    h_y_imag *= tril_mask

    return h_x_real, h_x_imag, h_y_real, h_y_imag


def compute_exact_gradients(
    X_real, X_imag, Y_real, Y_imag,
    alpha_x, alpha_y, beta_x, beta_y,
    L, D, b,
    use_spectral_term: bool = True,
    clip_grad: float = None,
):
    """Compute exact gradients with optional spectral term and gradient clipping."""
    n_states, d = X_real.shape
    D_diag = np.diag(D)

    g_X_real = np.zeros_like(X_real)
    g_X_imag = np.zeros_like(X_imag)
    g_Y_real = np.zeros_like(Y_real)
    g_Y_imag = np.zeros_like(Y_imag)

    # Precompute
    DY_real = D_diag @ Y_real
    DY_imag = D_diag @ Y_imag
    DX_real = D_diag @ X_real
    DX_imag = D_diag @ X_imag

    # Constraint errors
    h_x_real, h_x_imag, h_y_real, h_y_imag = compute_constraint_errors(
        X_real, X_imag, Y_real, Y_imag, D
    )

    # Dual gradients
    g_alpha_x = h_x_real.copy()
    g_alpha_y = h_y_real.copy()
    g_beta_x = h_x_imag.copy()
    g_beta_y = h_y_imag.copy()

    DL = D_diag @ L
    LTD = L.T @ D_diag

    for i in range(d):
        x_i_real = X_real[:, i]
        x_i_imag = X_imag[:, i]
        y_i_real = Y_real[:, i]
        y_i_imag = Y_imag[:, i]

        # Spectral product
        Ly_i_real = L @ y_i_real
        Ly_i_imag = L @ y_i_imag
        D_Ly_i_real = D * Ly_i_real
        D_Ly_i_imag = D * Ly_i_imag
        spectral_product_real = x_i_real @ D_Ly_i_real + x_i_imag @ D_Ly_i_imag

        # Initialize gradients
        g_xi_real = np.zeros(n_states)
        g_xi_imag = np.zeros(n_states)
        g_yi_real = np.zeros(n_states)
        g_yi_imag = np.zeros(n_states)

        # First term (spectral objective) - can be disabled for initial stabilization
        if use_spectral_term:
            DL_ybar_i_real = DL @ y_i_real
            DL_ybar_i_imag = -DL @ y_i_imag

            g_xi_real = spectral_product_real * DL_ybar_i_real
            g_xi_imag = spectral_product_real * DL_ybar_i_imag

            LTD_xbar_i_real = LTD @ x_i_real
            LTD_xbar_i_imag = -LTD @ x_i_imag

            g_yi_real = spectral_product_real * LTD_xbar_i_real
            g_yi_imag = spectral_product_real * LTD_xbar_i_imag

        # Constraint terms
        for k in range(i + 1):
            coef_real = alpha_x[i, k] - b * h_x_real[i, k]
            coef_imag = beta_x[i, k] - b * h_x_imag[i, k]

            D_ybar_k_real = DY_real[:, k]
            D_ybar_k_imag = -DY_imag[:, k]

            term_real = coef_real * D_ybar_k_real - coef_imag * D_ybar_k_imag
            term_imag = coef_real * D_ybar_k_imag + coef_imag * D_ybar_k_real

            g_xi_real = g_xi_real - term_real
            g_xi_imag = g_xi_imag - term_imag

        for k in range(i + 1):
            coef_real = alpha_y[i, k] - b * h_y_real[i, k]
            coef_imag = beta_y[i, k] - b * h_y_imag[i, k]

            D_xbar_k_real = DX_real[:, k]
            D_xbar_k_imag = -DX_imag[:, k]

            term_real = coef_real * D_xbar_k_real - coef_imag * D_xbar_k_imag
            term_imag = coef_real * D_xbar_k_imag + coef_imag * D_xbar_k_real

            g_yi_real = g_yi_real - term_real
            g_yi_imag = g_yi_imag - term_imag

        g_X_real[:, i] = g_xi_real
        g_X_imag[:, i] = g_xi_imag
        g_Y_real[:, i] = g_yi_real
        g_Y_imag[:, i] = g_yi_imag

    # Gradient clipping
    if clip_grad is not None:
        grad_norm = np.sqrt(
            np.sum(g_X_real**2) + np.sum(g_X_imag**2) +
            np.sum(g_Y_real**2) + np.sum(g_Y_imag**2)
        )
        if grad_norm > clip_grad:
            scale = clip_grad / grad_norm
            g_X_real *= scale
            g_X_imag *= scale
            g_Y_real *= scale
            g_Y_imag *= scale

    return {
        'g_X_real': g_X_real, 'g_X_imag': g_X_imag,
        'g_Y_real': g_Y_real, 'g_Y_imag': g_Y_imag,
        'g_alpha_x': g_alpha_x, 'g_alpha_y': g_alpha_y,
        'g_beta_x': g_beta_x, 'g_beta_y': g_beta_y,
        'h_x_real': h_x_real, 'h_x_imag': h_x_imag,
        'h_y_real': h_y_real, 'h_y_imag': h_y_imag,
        'spectral_product': spectral_product_real if d == 1 else 0,
    }


def compute_cosine_similarity(learned: np.ndarray, gt: np.ndarray, D: np.ndarray = None) -> float:
    """Compute cosine similarity between complex vectors."""
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
    lr_primal: float = 0.001,
    lr_dual: float = 0.01,
    barrier_coef: float = 4.0,
    num_steps: int = 10000,
    log_freq: int = 500,
    normalize_every: int = 10,
    use_spectral_term: bool = True,
    spectral_warmup: int = 0,
    clip_grad: float = 1.0,
    seed: int = 42,
):
    """Run stabilized experiment."""
    print(f"=" * 70)
    print(f"Stabilized exact gradient dynamics")
    print(f"  n_states={n_states}, d={d}")
    print(f"  lr_primal={lr_primal}, lr_dual={lr_dual}, b={barrier_coef}")
    print(f"  normalize_every={normalize_every}, clip_grad={clip_grad}")
    print(f"  spectral_warmup={spectral_warmup}")
    print(f"=" * 70)

    L = create_complex_laplacian(n_states, seed=seed)
    D = np.ones(n_states) / n_states

    eigendecomp = compute_eigendecomposition(L, k=d)
    gt_eigenvalues = eigendecomp['eigenvalues']
    gt_left = eigendecomp['left']
    gt_right = eigendecomp['right']

    print(f"\nGround truth eigenvalues:")
    for i in range(d):
        print(f"  λ_{i}: {np.real(gt_eigenvalues[i]):.6f} + {np.imag(gt_eigenvalues[i]):.6f}i")

    # Initialize
    np.random.seed(seed + 1)
    init_scale = 0.5

    X_real = init_scale * np.random.randn(n_states, d)
    X_imag = init_scale * np.random.randn(n_states, d)
    Y_real = init_scale * np.random.randn(n_states, d)
    Y_imag = init_scale * np.random.randn(n_states, d)

    # Normalize initially
    X_real, X_imag, Y_real, Y_imag = normalize_eigenvectors(X_real, X_imag, Y_real, Y_imag, D)

    alpha_x = np.zeros((d, d))
    alpha_y = np.zeros((d, d))
    beta_x = np.zeros((d, d))
    beta_y = np.zeros((d, d))

    tril_mask = np.tril(np.ones((d, d)))

    print(f"\n{'Step':>8} {'Constr Err':>12} {'Grad X':>10} {'Left Sim':>10} {'Right Sim':>10} {'λ Est':>15}")
    print("-" * 80)

    history = []

    for step in range(num_steps):
        # Decide whether to use spectral term
        use_spec = use_spectral_term and (step >= spectral_warmup)

        grads = compute_exact_gradients(
            X_real, X_imag, Y_real, Y_imag,
            alpha_x, alpha_y, beta_x, beta_y,
            L, D, barrier_coef,
            use_spectral_term=use_spec,
            clip_grad=clip_grad,
        )

        # Update primal (gradient descent)
        X_real = X_real - lr_primal * grads['g_X_real']
        X_imag = X_imag - lr_primal * grads['g_X_imag']
        Y_real = Y_real - lr_primal * grads['g_Y_real']
        Y_imag = Y_imag - lr_primal * grads['g_Y_imag']

        # Update dual (gradient ascent)
        alpha_x = alpha_x + lr_dual * grads['g_alpha_x']
        alpha_y = alpha_y + lr_dual * grads['g_alpha_y']
        beta_x = beta_x + lr_dual * grads['g_beta_x']
        beta_y = beta_y + lr_dual * grads['g_beta_y']

        # Apply masks
        alpha_x *= tril_mask
        alpha_y *= tril_mask
        beta_x *= tril_mask
        beta_y *= tril_mask

        # Normalize eigenvectors periodically
        if (step + 1) % normalize_every == 0:
            X_real, X_imag, Y_real, Y_imag = normalize_eigenvectors(X_real, X_imag, Y_real, Y_imag, D)

        # Logging
        if step % log_freq == 0:
            constraint_error = (np.abs(grads['h_x_real']).sum() + np.abs(grads['h_x_imag']).sum() +
                              np.abs(grads['h_y_real']).sum() + np.abs(grads['h_y_imag']).sum())

            grad_norm = np.sqrt(
                np.sum(grads['g_X_real']**2) + np.sum(grads['g_X_imag']**2) +
                np.sum(grads['g_Y_real']**2) + np.sum(grads['g_Y_imag']**2)
            )

            left_sims = []
            right_sims = []
            ev_ests = []

            for i in range(d):
                learned_left = X_real[:, i] + 1j * X_imag[:, i]
                learned_right = Y_real[:, i] + 1j * Y_imag[:, i]

                left_sim = compute_cosine_similarity(learned_left, gt_left[:, i], D)
                right_sim = compute_cosine_similarity(learned_right, gt_right[:, i], None)

                left_sims.append(left_sim)
                right_sims.append(right_sim)

                ev_est = -0.5 * (alpha_x[i, i] + alpha_y[i, i]) + 1j * (-0.5 * (beta_x[i, i] + beta_y[i, i]))
                ev_ests.append(ev_est)

            avg_left_sim = np.mean(left_sims)
            avg_right_sim = np.mean(right_sims)

            history.append({
                'step': step,
                'constraint_error': constraint_error,
                'grad_norm': grad_norm,
                'left_sim': avg_left_sim,
                'right_sim': avg_right_sim,
                'eigenvalue_est': ev_ests[0] if d > 0 else 0,
            })

            ev_str = f"{np.real(ev_ests[0]):.4f}+{np.imag(ev_ests[0]):.4f}i" if d > 0 else "N/A"
            print(f"{step:>8} {constraint_error:>12.6f} {grad_norm:>10.4f} {avg_left_sim:>10.4f} {avg_right_sim:>10.4f} {ev_str:>15}")

    print("\n" + "=" * 70)
    print("Final Results:")
    print(f"  Constraint error: {history[-1]['constraint_error']:.6f}")
    print(f"  Left cosine sim: {history[-1]['left_sim']:.4f}")
    print(f"  Right cosine sim: {history[-1]['right_sim']:.4f}")

    for i in range(d):
        ev_est = -0.5 * (alpha_x[i, i] + alpha_y[i, i]) + 1j * (-0.5 * (beta_x[i, i] + beta_y[i, i]))
        print(f"  Eigenvector {i}:")
        print(f"    λ_est: {np.real(ev_est):.6f} + {np.imag(ev_est):.6f}i")
        print(f"    λ_gt:  {np.real(gt_eigenvalues[i]):.6f} + {np.imag(gt_eigenvalues[i]):.6f}i")

    # Check convergence
    converged = history[-1]['constraint_error'] < 0.5 and history[-1]['left_sim'] > 0.8
    print(f"\n  Convergence: {'SUCCESS' if converged else 'FAILED'}")

    return history, converged


def main():
    print("\n" + "=" * 70)
    print("  STABILIZED EXACT GRADIENT DYNAMICS TEST")
    print("=" * 70)

    # Test 1: No spectral term (pure constraint satisfaction)
    print("\n\n>>> TEST 1: Pure constraint satisfaction (no spectral term)")
    history1, conv1 = run_experiment(
        n_states=15,
        d=1,
        lr_primal=0.01,
        lr_dual=0.1,
        barrier_coef=4.0,
        num_steps=5000,
        log_freq=500,
        normalize_every=5,
        use_spectral_term=False,
        clip_grad=1.0,
    )

    # Test 2: With spectral term but warmup
    print("\n\n>>> TEST 2: With spectral term + warmup")
    history2, conv2 = run_experiment(
        n_states=15,
        d=1,
        lr_primal=0.001,
        lr_dual=0.05,
        barrier_coef=4.0,
        num_steps=10000,
        log_freq=1000,
        normalize_every=5,
        use_spectral_term=True,
        spectral_warmup=2000,
        clip_grad=0.5,
    )

    # Test 3: Multiple eigenvectors
    print("\n\n>>> TEST 3: Multiple eigenvectors (d=2)")
    history3, conv3 = run_experiment(
        n_states=20,
        d=2,
        lr_primal=0.001,
        lr_dual=0.05,
        barrier_coef=4.0,
        num_steps=15000,
        log_freq=1500,
        normalize_every=5,
        use_spectral_term=True,
        spectral_warmup=3000,
        clip_grad=0.5,
    )

    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (no spectral): {'PASSED' if conv1 else 'FAILED'}")
    print(f"  Test 2 (with warmup): {'PASSED' if conv2 else 'FAILED'}")
    print(f"  Test 3 (d=2):         {'PASSED' if conv3 else 'FAILED'}")


if __name__ == "__main__":
    main()
