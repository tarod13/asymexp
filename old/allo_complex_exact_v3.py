"""
Exact gradient dynamics v3 - with proper biorthonormalization verification.

Key finding: The numpy eigenvectors need proper biorthonormalization to satisfy
<x̄, [[y]]>_D = 1. Let's verify this first.
"""

import numpy as np


def create_simple_laplacian(n: int, seed: int = 42) -> np.ndarray:
    """Create a simple non-symmetric Laplacian."""
    np.random.seed(seed)
    A = np.abs(np.random.randn(n, n)) + 0.1 + 0.2 * np.random.randn(n, n)
    A = np.maximum(A, 0.01)
    P = A / A.sum(axis=1, keepdims=True)
    L = np.eye(n) - 0.9 * P
    return L


def verify_eigenvectors(L: np.ndarray, D: np.ndarray):
    """Verify eigendecomposition and biorthonormality."""
    n = L.shape[0]

    print("=" * 60)
    print("Verifying eigendecomposition and biorthonormality")
    print("=" * 60)

    # Compute right eigenvectors
    eigenvalues, V_right = np.linalg.eig(L)

    # Sort by real part
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    V_right = V_right[:, idx]

    # Compute left eigenvectors (eigenvectors of L^T)
    _, V_left_T = np.linalg.eig(L.T)
    # Need to match left to right eigenvectors

    # For each right eigenvector, find the corresponding left eigenvector
    V_left = np.zeros_like(V_left_T, dtype=complex)

    for i in range(n):
        # The left eigenvector for eigenvalue λ_i satisfies L^T @ x = λ_i @ x
        # Find which column of V_left_T has eigenvalue closest to eigenvalues[i]
        eigenvalues_T, _ = np.linalg.eig(L.T)
        best_j = np.argmin(np.abs(eigenvalues_T - eigenvalues[i]))

        # Actually, let's compute left eigenvectors more carefully
        # Left eigenvectors are rows of V^{-1} where L = V Λ V^{-1}
        pass

    # Better approach: Left eigenvectors are columns of V^{-H} (conjugate transpose inverse)
    try:
        V_inv = np.linalg.inv(V_right)
        V_left = V_inv.conj().T  # Columns of V^{-H}
    except:
        print("Warning: Could not invert V_right")
        V_left = V_left_T[:, idx]

    print(f"\nFirst 3 eigenvalues:")
    for i in range(min(3, n)):
        print(f"  λ_{i} = {eigenvalues[i]:.6f}")

    # Verify L @ y = λ @ y
    print(f"\nVerifying L @ y = λ * y:")
    for i in range(min(3, n)):
        y = V_right[:, i]
        Ly = L @ y
        lambda_y = eigenvalues[i] * y
        error = np.linalg.norm(Ly - lambda_y)
        print(f"  Eigenvector {i}: ||L @ y - λ * y|| = {error:.2e}")

    # Verify L^T @ x = λ * x
    print(f"\nVerifying x^T @ L = λ * x^T (left eigenvector):")
    for i in range(min(3, n)):
        x = V_left[:, i]
        xTL = x.conj() @ L
        lambda_xT = eigenvalues[i].conj() * x.conj()
        error = np.linalg.norm(xTL - lambda_xT)
        print(f"  Eigenvector {i}: ||x^T @ L - λ* @ x^T|| = {error:.2e}")

    # Check biorthogonality: <x_i, y_j> = δ_{ij}
    print(f"\nBiorthogonality matrix <x_i, y_j> (standard inner product):")
    biorth_std = V_left.conj().T @ V_right
    print("  First 3x3 block:")
    for i in range(min(3, n)):
        row = [f"{biorth_std[i,j]:.4f}" for j in range(min(3, n))]
        print(f"    {row}")

    # Check biorthogonality with D weighting: <x_i, y_j>_D
    print(f"\nBiorthogonality matrix <x_i, y_j>_D (D-weighted inner product):")
    D_diag = np.diag(D)
    biorth_D = V_left.conj().T @ D_diag @ V_right
    print("  First 3x3 block:")
    for i in range(min(3, n)):
        row = [f"{biorth_D[i,j]:.4f}" for j in range(min(3, n))]
        print(f"    {row}")

    # The constraint in the theory is <x̄, [[y]]> where x̄ = conj(x) and [[y]] = conj(y)
    # So we need: conj(x)^T @ D @ conj(y) = x^T @ D @ y* = ...
    # Actually let's compute h = <x̄, [[y]]>_D - 1 = x^T D y* - 1
    print(f"\nConstraint error h = <x̄_i, [[y_i]]>_D - 1 (theory definition):")
    for i in range(min(3, n)):
        x = V_left[:, i]
        y = V_right[:, i]
        # <x̄, [[y]]> = conj(x)^T @ D @ conj(y) = x.conj().T @ D @ y.conj()
        inner = x.conj() @ D_diag @ y.conj()
        h = inner - 1.0
        print(f"  Eigenvector {i}: <x̄, [[y]]>_D = {inner:.6f}, h = {h:.6f}")

    # The standard biorthogonality is x^H @ y = x.conj().T @ y
    print(f"\nStandard biorthogonality <x^H, y> (what numpy gives):")
    for i in range(min(3, n)):
        x = V_left[:, i]
        y = V_right[:, i]
        inner = x.conj() @ y
        print(f"  Eigenvector {i}: <x^H, y> = {inner:.6f}")

    return eigenvalues, V_left, V_right


def biorthonormalize_for_theory(V_left, V_right, D):
    """
    Normalize eigenvectors so that <x̄_i, [[y_j]]>_D = δ_{ij}

    The theory constraint: x_i^T @ D @ y_j* = δ_{ij}
    Equivalently: conj(x_i)^T @ D @ conj(y_j) = δ_{ij}

    If standard biorthogonality holds: x_i^H @ y_j = δ_{ij}
    i.e., conj(x_i)^T @ y_j = δ_{ij}

    We need: conj(x_i)^T @ D @ conj(y_j) = δ_{ij}
    """
    n, k = V_right.shape
    D_diag = np.diag(D)

    X = V_left.copy()
    Y = V_right.copy()

    for i in range(k):
        # Current inner product: conj(x_i)^T @ D @ conj(y_i)
        inner = X[:, i].conj() @ D_diag @ Y[:, i].conj()

        # Scale to make it 1
        if np.abs(inner) > 1e-10:
            scale = np.sqrt(np.abs(inner))
            X[:, i] = X[:, i] / scale
            Y[:, i] = Y[:, i] / scale
            # Now: inner = conj(x_i/s)^T @ D @ conj(y_i/s) = inner / s^2 = 1

    return X, Y


def test_gradient_dynamics(
    L: np.ndarray,
    D: np.ndarray,
    x_init: np.ndarray,  # Complex left eigenvector
    y_init: np.ndarray,  # Complex right eigenvector
    gt_eigenvalue: complex,
    num_steps: int = 5000,
    lr_primal: float = 0.001,
    lr_dual: float = 0.01,
    barrier_coef: float = 4.0,
):
    """
    Test gradient dynamics starting from given initialization.
    """
    print(f"\n{'='*60}")
    print(f"Gradient dynamics test")
    print(f"  lr_primal={lr_primal}, lr_dual={lr_dual}, b={barrier_coef}")
    print(f"{'='*60}")

    n = L.shape[0]
    D_diag = np.diag(D)

    # Split into real/imag
    x_real = np.real(x_init).copy()
    x_imag = np.imag(x_init).copy()
    y_real = np.real(y_init).copy()
    y_imag = np.imag(y_init).copy()

    alpha = 0.0
    beta = 0.0

    # Verify initial constraint
    inner = (x_real - 1j * x_imag) @ D_diag @ (y_real - 1j * y_imag)
    h_init = inner - 1.0
    print(f"\nInitial <x̄, [[y]]>_D = {inner:.6f}")
    print(f"Initial constraint error h = {h_init:.6f}")

    print(f"\n{'Step':>6} {'Re(h)':>10} {'Im(h)':>10} {'α':>10} {'β':>10} {'|g_x|':>10}")
    print("-" * 60)

    for step in range(num_steps):
        # Compute constraint error h = <x̄, [[y]]>_D - 1
        # <x̄, [[y]]> = (x_real - i*x_imag)^T @ D @ (y_real - i*y_imag)
        Dy_real = D * y_real
        Dy_imag = D * y_imag

        h_real = np.dot(x_real, Dy_real) + np.dot(x_imag, Dy_imag) - 1.0
        h_imag = -np.dot(x_real, Dy_imag) + np.dot(x_imag, Dy_real)

        # Spectral product Re(<x̄, Ly>_D)
        Ly_real = L @ y_real
        Ly_imag = L @ y_imag
        DLy_real = D * Ly_real
        DLy_imag = D * Ly_imag

        spectral_real = np.dot(x_real, DLy_real) + np.dot(x_imag, DLy_imag)

        # Gradient for x: g_x = spectral * D*L*ȳ - (α + iβ - b*h) * D*ȳ
        DL = D_diag @ L
        DL_ybar_real = DL @ y_real
        DL_ybar_imag = -DL @ y_imag

        coef_real = alpha - barrier_coef * h_real
        coef_imag = beta - barrier_coef * h_imag

        D_ybar_real = D * y_real
        D_ybar_imag = -D * y_imag

        g_x_real = spectral_real * DL_ybar_real - (coef_real * D_ybar_real - coef_imag * D_ybar_imag)
        g_x_imag = spectral_real * DL_ybar_imag - (coef_real * D_ybar_imag + coef_imag * D_ybar_real)

        # Gradient for y
        LTD = L.T @ D_diag
        LTD_xbar_real = LTD @ x_real
        LTD_xbar_imag = -LTD @ x_imag

        D_xbar_real = D * x_real
        D_xbar_imag = -D * x_imag

        g_y_real = spectral_real * LTD_xbar_real - (coef_real * D_xbar_real - coef_imag * D_xbar_imag)
        g_y_imag = spectral_real * LTD_xbar_imag - (coef_real * D_xbar_imag + coef_imag * D_xbar_real)

        # Update primal (gradient descent)
        x_real = x_real - lr_primal * g_x_real
        x_imag = x_imag - lr_primal * g_x_imag
        y_real = y_real - lr_primal * g_y_real
        y_imag = y_imag - lr_primal * g_y_imag

        # Update dual (gradient ascent)
        alpha = alpha + lr_dual * h_real
        beta = beta + lr_dual * h_imag

        g_x_norm = np.sqrt(np.sum(g_x_real**2) + np.sum(g_x_imag**2))

        if step % 500 == 0:
            print(f"{step:>6} {h_real:>10.6f} {h_imag:>10.6f} {alpha:>10.4f} {beta:>10.4f} {g_x_norm:>10.4f}")

    print(f"\nFinal constraint error: h = {h_real:.6f} + {h_imag:.6f}i")
    print(f"Final duals: α = {alpha:.4f}, β = {beta:.4f}")
    print(f"Eigenvalue estimate: λ = {-alpha:.6f} + {-beta:.6f}i")
    print(f"Ground truth:        λ = {np.real(gt_eigenvalue):.6f} + {np.imag(gt_eigenvalue):.6f}i")


def main():
    print("=" * 70)
    print("  EXACT GRADIENT DYNAMICS V3 - PROPER BIORTHONORMALIZATION")
    print("=" * 70)

    # Create test matrix
    n = 10
    L = create_simple_laplacian(n, seed=42)
    D = np.ones(n) / n

    # Verify eigendecomposition
    eigenvalues, V_left, V_right = verify_eigenvectors(L, D)

    # Biorthonormalize for the theory constraint
    print("\n" + "=" * 60)
    print("After biorthonormalization for theory constraint:")
    X, Y = biorthonormalize_for_theory(V_left, V_right, D)

    D_diag = np.diag(D)
    print(f"\nVerifying <x̄_i, [[y_i]]>_D = 1:")
    for i in range(min(3, n)):
        inner = X[:, i].conj() @ D_diag @ Y[:, i].conj()
        print(f"  Eigenvector {i}: {inner:.6f}")

    # Test gradient dynamics from biorthonormalized eigenvectors
    print("\n\n>>> TEST 1: Starting from properly biorthonormalized eigenvectors")
    test_gradient_dynamics(
        L, D,
        x_init=X[:, 0],
        y_init=Y[:, 0],
        gt_eigenvalue=eigenvalues[0],
        num_steps=3000,
        lr_primal=0.001,
        lr_dual=0.01,
        barrier_coef=4.0,
    )

    # Test with small perturbation
    print("\n\n>>> TEST 2: Starting with small perturbation from biorthonormalized")
    np.random.seed(123)
    noise = 0.01
    x_perturbed = X[:, 0] + noise * (np.random.randn(n) + 1j * np.random.randn(n))
    y_perturbed = Y[:, 0] + noise * (np.random.randn(n) + 1j * np.random.randn(n))

    test_gradient_dynamics(
        L, D,
        x_init=x_perturbed,
        y_init=y_perturbed,
        gt_eigenvalue=eigenvalues[0],
        num_steps=3000,
        lr_primal=0.001,
        lr_dual=0.01,
        barrier_coef=4.0,
    )

    # Test from random initialization
    print("\n\n>>> TEST 3: Starting from random initialization")
    np.random.seed(456)
    x_random = np.random.randn(n) + 1j * np.random.randn(n)
    y_random = np.random.randn(n) + 1j * np.random.randn(n)

    test_gradient_dynamics(
        L, D,
        x_init=x_random,
        y_init=y_random,
        gt_eigenvalue=eigenvalues[0],
        num_steps=5000,
        lr_primal=0.0001,
        lr_dual=0.001,
        barrier_coef=4.0,
    )


if __name__ == "__main__":
    main()
