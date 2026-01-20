"""
Exact gradient dynamics v2 - without normalization, with careful learning rate tuning.

Key insight: The biorthogonality constraint <x̄, [[y]]>_D = 1 doesn't require
unit norm eigenvectors. Normalizing them breaks the dynamics.

Instead, we use:
1. Very small learning rates
2. Dual variable clipping (to prevent unbounded growth)
3. No eigenvector normalization (let them find their own scale)
"""

import numpy as np
from typing import Dict


def create_simple_laplacian(n: int, seed: int = 42) -> np.ndarray:
    """Create a simple non-symmetric Laplacian."""
    np.random.seed(seed)

    # Simple asymmetric random matrix
    A = np.abs(np.random.randn(n, n))
    A = A + 0.2 * np.random.randn(n, n)  # Add asymmetry
    A = np.maximum(A, 0.01)

    # Make row-stochastic
    P = A / A.sum(axis=1, keepdims=True)

    # Laplacian: L = I - P
    L = np.eye(n) - 0.9 * P

    return L


def compute_eigendecomposition(L: np.ndarray, k: int = None):
    """Compute eigendecomposition."""
    eigenvalues, right_eigenvectors = np.linalg.eig(L)

    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    right_eigenvectors = right_eigenvectors[:, idx]

    _, left_eigenvectors = np.linalg.eig(L.T)
    left_eigenvectors = left_eigenvectors[:, idx]

    # Biorthonormalize
    for i in range(len(eigenvalues)):
        scale = left_eigenvectors[:, i].conj() @ right_eigenvectors[:, i]
        if np.abs(scale) > 1e-10:
            left_eigenvectors[:, i] = left_eigenvectors[:, i] / np.sqrt(np.abs(scale))
            right_eigenvectors[:, i] = right_eigenvectors[:, i] / np.sqrt(np.abs(scale))

    if k is not None:
        eigenvalues = eigenvalues[:k]
        left_eigenvectors = left_eigenvectors[:, :k]
        right_eigenvectors = right_eigenvectors[:, :k]

    return {'eigenvalues': eigenvalues, 'left': left_eigenvectors, 'right': right_eigenvectors}


def compute_gradients_single_eigenvector(
    x_real, x_imag,  # (n,) - left eigenvector
    y_real, y_imag,  # (n,) - right eigenvector
    alpha, beta,     # scalars - duals for Re and Im constraints
    L, D, b,
    use_spectral: bool = True,
):
    """
    Compute gradients for a single eigenvector pair.

    The theory specifies for the first (i=0) eigenvector:
    - Constraint: h = <x̄, [[y]]>_D - 1 = x^T D conj(y) - 1
    - Primal grad for x: g_x = Re(<x̄, Ly>_D) * D*L*ȳ - (α + iβ - b*h) * D*ȳ
    - Primal grad for y: g_y = Re(<x̄, Ly>_D) * L^T*D*x̄ - (α + iβ - b*h) * D*x̄
    - Dual grad: g_α = Re(h), g_β = Im(h)
    """
    n = len(x_real)
    D_diag = np.diag(D)

    # Compute constraint error h = <x̄, [[y]]>_D - 1
    # [[y]] = conj(y) = y_real - i*y_imag
    # <x̄, [[y]]> = conj(x)^T D conj(y) = (x_real - i*x_imag)^T D (y_real - i*y_imag)
    Dy_real = D * y_real
    Dy_imag = D * y_imag

    # Inner product: (x_real - i*x_imag) . (Dy_real - i*Dy_imag)
    # Real: x_real.Dy_real + x_imag.Dy_imag
    # Imag: -x_real.Dy_imag + x_imag.Dy_real
    h_real = np.dot(x_real, Dy_real) + np.dot(x_imag, Dy_imag) - 1.0
    h_imag = -np.dot(x_real, Dy_imag) + np.dot(x_imag, Dy_real)

    # Dual gradients (simple)
    g_alpha = h_real
    g_beta = h_imag

    # Compute spectral product Re(<x̄, Ly>_D)
    Ly_real = L @ y_real
    Ly_imag = L @ y_imag
    DLy_real = D * Ly_real
    DLy_imag = D * Ly_imag

    # <x̄, Ly> = (x_real - i*x_imag)^T D (Ly_real + i*Ly_imag)
    # Real: x_real.DLy_real + x_imag.DLy_imag
    spectral_real = np.dot(x_real, DLy_real) + np.dot(x_imag, DLy_imag)

    # Primal gradients
    # First term (spectral objective): Re(<x̄, Ly>) * D*L*ȳ
    DL = D_diag @ L
    LTD = L.T @ D_diag

    # D*L*ȳ = D @ L @ (y_real - i*y_imag)
    DL_ybar_real = DL @ y_real
    DL_ybar_imag = -DL @ y_imag

    # L^T*D*x̄ = L^T @ D @ (x_real - i*x_imag)
    LTD_xbar_real = LTD @ x_real
    LTD_xbar_imag = -LTD @ x_imag

    if use_spectral:
        g_x_real = spectral_real * DL_ybar_real
        g_x_imag = spectral_real * DL_ybar_imag
        g_y_real = spectral_real * LTD_xbar_real
        g_y_imag = spectral_real * LTD_xbar_imag
    else:
        g_x_real = np.zeros(n)
        g_x_imag = np.zeros(n)
        g_y_real = np.zeros(n)
        g_y_imag = np.zeros(n)

    # Second term (constraint): -(α + iβ - b*h) * D*ȳ
    # D*ȳ = D @ (y_real - i*y_imag)
    D_ybar_real = D * y_real
    D_ybar_imag = -D * y_imag

    coef_real = alpha - b * h_real
    coef_imag = beta - b * h_imag

    # Complex mult: (coef_real + i*coef_imag) * (D_ybar_real + i*D_ybar_imag)
    constraint_x_real = coef_real * D_ybar_real - coef_imag * D_ybar_imag
    constraint_x_imag = coef_real * D_ybar_imag + coef_imag * D_ybar_real

    g_x_real = g_x_real - constraint_x_real
    g_x_imag = g_x_imag - constraint_x_imag

    # Same for y: -(α + iβ - b*h) * D*x̄
    D_xbar_real = D * x_real
    D_xbar_imag = -D * x_imag

    constraint_y_real = coef_real * D_xbar_real - coef_imag * D_xbar_imag
    constraint_y_imag = coef_real * D_xbar_imag + coef_imag * D_xbar_real

    g_y_real = g_y_real - constraint_y_real
    g_y_imag = g_y_imag - constraint_y_imag

    return {
        'g_x_real': g_x_real, 'g_x_imag': g_x_imag,
        'g_y_real': g_y_real, 'g_y_imag': g_y_imag,
        'g_alpha': g_alpha, 'g_beta': g_beta,
        'h_real': h_real, 'h_imag': h_imag,
        'spectral_real': spectral_real,
    }


def compute_cosine_similarity(learned: np.ndarray, gt: np.ndarray) -> float:
    """Compute cosine similarity."""
    # Phase normalize
    mag = np.abs(learned)
    if mag.max() > 1e-10:
        max_idx = np.argmax(mag)
        learned = learned / learned[max_idx]

    inner = np.abs(np.vdot(learned, gt))
    return inner / (np.linalg.norm(learned) * np.linalg.norm(gt) + 1e-10)


def run_single_eigenvector_test(
    n_states: int = 10,
    lr_primal: float = 0.0001,
    lr_dual: float = 0.001,
    barrier_coef: float = 4.0,
    num_steps: int = 20000,
    log_freq: int = 1000,
    use_spectral: bool = False,  # Start without spectral term
    dual_clip: float = 100.0,
    seed: int = 42,
):
    """Test learning a single eigenvector."""
    print(f"\n{'='*60}")
    print(f"Single eigenvector test")
    print(f"  n={n_states}, lr_primal={lr_primal}, lr_dual={lr_dual}, b={barrier_coef}")
    print(f"  use_spectral={use_spectral}, dual_clip={dual_clip}")
    print(f"{'='*60}")

    L = create_simple_laplacian(n_states, seed)
    D = np.ones(n_states) / n_states

    eigen = compute_eigendecomposition(L, k=1)
    gt_ev = eigen['eigenvalues'][0]
    gt_left = eigen['left'][:, 0]
    gt_right = eigen['right'][:, 0]

    print(f"\nGround truth: λ = {np.real(gt_ev):.6f} + {np.imag(gt_ev):.6f}i")

    # Initialize close to ground truth for testing
    np.random.seed(seed + 1)
    init_scale = 1.0

    x_real = init_scale * np.random.randn(n_states)
    x_imag = init_scale * np.random.randn(n_states)
    y_real = init_scale * np.random.randn(n_states)
    y_imag = init_scale * np.random.randn(n_states)

    alpha = 0.0
    beta = 0.0

    print(f"\n{'Step':>8} {'|h|':>10} {'α':>10} {'|g_x|':>10} {'L sim':>8} {'R sim':>8}")
    print("-" * 60)

    for step in range(num_steps):
        grads = compute_gradients_single_eigenvector(
            x_real, x_imag, y_real, y_imag,
            alpha, beta, L, D, barrier_coef,
            use_spectral=use_spectral,
        )

        # Update primal (gradient descent)
        x_real = x_real - lr_primal * grads['g_x_real']
        x_imag = x_imag - lr_primal * grads['g_x_imag']
        y_real = y_real - lr_primal * grads['g_y_real']
        y_imag = y_imag - lr_primal * grads['g_y_imag']

        # Update dual (gradient ascent)
        alpha = alpha + lr_dual * grads['g_alpha']
        beta = beta + lr_dual * grads['g_beta']

        # Clip duals
        alpha = np.clip(alpha, -dual_clip, dual_clip)
        beta = np.clip(beta, -dual_clip, dual_clip)

        if step % log_freq == 0:
            h_mag = np.sqrt(grads['h_real']**2 + grads['h_imag']**2)
            g_x_norm = np.sqrt(np.sum(grads['g_x_real']**2) + np.sum(grads['g_x_imag']**2))

            learned_left = x_real + 1j * x_imag
            learned_right = y_real + 1j * y_imag

            left_sim = compute_cosine_similarity(learned_left, gt_left)
            right_sim = compute_cosine_similarity(learned_right, gt_right)

            print(f"{step:>8} {h_mag:>10.6f} {alpha:>10.4f} {g_x_norm:>10.6f} {left_sim:>8.4f} {right_sim:>8.4f}")

    print(f"\nFinal constraint error |h|: {np.sqrt(grads['h_real']**2 + grads['h_imag']**2):.6f}")
    print(f"Final duals: α={alpha:.4f}, β={beta:.4f}")
    print(f"Eigenvalue estimate: {-alpha:.6f} + {-beta:.6f}i")
    print(f"Ground truth:        {np.real(gt_ev):.6f} + {np.imag(gt_ev):.6f}i")

    return grads['h_real']**2 + grads['h_imag']**2 < 0.1


def run_initialized_test(
    n_states: int = 10,
    seed: int = 42,
):
    """Test with initialization close to ground truth."""
    print(f"\n{'='*60}")
    print(f"Test with initialization near ground truth")
    print(f"{'='*60}")

    L = create_simple_laplacian(n_states, seed)
    D = np.ones(n_states) / n_states

    eigen = compute_eigendecomposition(L, k=1)
    gt_ev = eigen['eigenvalues'][0]
    gt_left = eigen['left'][:, 0]
    gt_right = eigen['right'][:, 0]

    print(f"Ground truth eigenvalue: {gt_ev}")

    # Initialize VERY close to ground truth
    np.random.seed(seed + 1)
    noise_scale = 0.01

    x_real = np.real(gt_left) + noise_scale * np.random.randn(n_states)
    x_imag = np.imag(gt_left) + noise_scale * np.random.randn(n_states)
    y_real = np.real(gt_right) + noise_scale * np.random.randn(n_states)
    y_imag = np.imag(gt_right) + noise_scale * np.random.randn(n_states)

    alpha = -np.real(gt_ev) + noise_scale * np.random.randn()
    beta = -np.imag(gt_ev) + noise_scale * np.random.randn()

    # Check initial constraint
    h_real = np.dot(x_real, D * y_real) + np.dot(x_imag, D * y_imag) - 1.0
    h_imag = -np.dot(x_real, D * y_imag) + np.dot(x_imag, D * y_real)
    print(f"Initial constraint error: h = {h_real:.6f} + {h_imag:.6f}i")

    # Run a few steps
    lr_primal = 0.001
    lr_dual = 0.01
    b = 4.0

    print(f"\nRunning dynamics with lr_primal={lr_primal}, lr_dual={lr_dual}, b={b}")
    print(f"{'Step':>6} {'h_real':>10} {'h_imag':>10} {'α':>10} {'β':>10}")
    print("-" * 50)

    for step in range(1000):
        grads = compute_gradients_single_eigenvector(
            x_real, x_imag, y_real, y_imag,
            alpha, beta, L, D, b,
            use_spectral=True,
        )

        x_real = x_real - lr_primal * grads['g_x_real']
        x_imag = x_imag - lr_primal * grads['g_x_imag']
        y_real = y_real - lr_primal * grads['g_y_real']
        y_imag = y_imag - lr_primal * grads['g_y_imag']

        alpha = alpha + lr_dual * grads['g_alpha']
        beta = beta + lr_dual * grads['g_beta']

        if step % 100 == 0:
            print(f"{step:>6} {grads['h_real']:>10.6f} {grads['h_imag']:>10.6f} {alpha:>10.4f} {beta:>10.4f}")

    print(f"\nFinal eigenvalue estimate: {-alpha:.6f} + {-beta:.6f}i")
    print(f"Ground truth eigenvalue:   {np.real(gt_ev):.6f} + {np.imag(gt_ev):.6f}i")

    # Check cosine similarity
    learned_left = x_real + 1j * x_imag
    learned_right = y_real + 1j * y_imag
    print(f"Left cosine sim: {compute_cosine_similarity(learned_left, gt_left):.4f}")
    print(f"Right cosine sim: {compute_cosine_similarity(learned_right, gt_right):.4f}")


def main():
    print("=" * 70)
    print("  EXACT GRADIENT DYNAMICS V2 - NO NORMALIZATION")
    print("=" * 70)

    # Test 1: Initialize near ground truth to verify dynamics are correct
    run_initialized_test(n_states=10, seed=42)

    # Test 2: Random initialization with constraint-only objective
    print("\n\n>>> TEST: Random init, constraint only")
    run_single_eigenvector_test(
        n_states=10,
        lr_primal=0.001,
        lr_dual=0.01,
        barrier_coef=4.0,
        num_steps=10000,
        log_freq=1000,
        use_spectral=False,
        dual_clip=10.0,
    )

    # Test 3: Random init with spectral term
    print("\n\n>>> TEST: Random init, with spectral term")
    run_single_eigenvector_test(
        n_states=10,
        lr_primal=0.0001,
        lr_dual=0.001,
        barrier_coef=4.0,
        num_steps=20000,
        log_freq=2000,
        use_spectral=True,
        dual_clip=10.0,
    )


if __name__ == "__main__":
    main()
