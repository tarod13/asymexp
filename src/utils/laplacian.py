import jax
import jax.numpy as jnp
from typing import Dict, Optional


def get_transition_matrix(
    transition_counts: jnp.ndarray,
    make_stochastic: bool = True,
) -> jnp.ndarray:
    """
    Build transition matrix from counts (non-symmetrized).

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]
        make_stochastic: Whether to row-normalize to get proper transition probabilities
    Returns:
        Transition matrix of shape [num_states, num_states]
    """
    # Sum over actions if needed
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)  # [num_states, num_states]
    else:
        transition_matrix = transition_counts

    if make_stochastic:
        row_sums = jnp.sum(transition_matrix, axis=1, keepdims=True)
        transition_matrix = jnp.where(
            row_sums > 0, 
            transition_matrix / row_sums, 
            jnp.zeros_like(transition_matrix)  # Avoid division by zero; rows with zero counts remain zero
        )

    return transition_matrix

def compute_successor_representation(
    transition_matrix: jnp.ndarray,
    gamma: float,
    max_horizon: int = None,
) -> jnp.ndarray:
    """
    Compute the successor representation SR_γ = (I - γP)^(-1) or finite-horizon version.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor
        max_horizon: If provided, compute finite-horizon SR_γ^(T) = Σ_{k=0}^{T} γ^k P^k
                     Otherwise, compute infinite-horizon SR_γ = (I - γP)^(-1)

    Returns:
        Successor representation of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    if max_horizon is None:
        # Infinite-horizon: SR_γ = (I - γP)^(-1)
        sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix)
    else:
        # Finite-horizon: SR_γ^(T) = Σ_{k=0}^{T} γ^k P^k
        # Using closed form: SR_γ^(T) = (I - γP)^{-1}(I - γ^{T+1}P^{T+1})
        # Compute P^{T+1} using built-in matrix power (uses efficient algorithm)
        P_power_T_plus_1 = jnp.linalg.matrix_power(
            transition_matrix, max_horizon + 1
        )

        # Apply finite geometric series formula (left multiplication is mathematically correct)
        gamma_power = gamma ** (max_horizon + 1)
        sr_matrix = jnp.linalg.solve(
            identity - gamma * transition_matrix, 
            identity - gamma_power * P_power_T_plus_1,
        )

    return sr_matrix


def compute_sampling_distribution(
    transition_counts: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the empirical sampling distribution from transition counts.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]

    Returns:
        Diagonal matrix D where D_{ii} is the empirical sampling probability of state i
    """
    # Sum over actions and next states to get visit counts per state
    if len(transition_counts.shape) == 3:
        # Sum over actions and next states
        state_visit_counts = jnp.sum(transition_counts, axis=(1, 2))
    else:
        # Sum over next states
        state_visit_counts = jnp.sum(transition_counts, axis=1)

    # Normalize to get probabilities
    total_visits = jnp.sum(state_visit_counts)
    sampling_probs = state_visit_counts / total_visits

    # Create diagonal matrix
    D = jnp.diag(sampling_probs)

    return D


def compute_laplacian(
    transition_matrix: jnp.ndarray,
    gamma: float,
    delta: float = 0.0,
    max_horizon: int = None,
) -> jnp.ndarray:
    """
    Compute the non-symmetric Laplacian L = (1+δ)I - (1-γ)P·SR_γ.

    This definition matches what the episodic replay buffer approximates with
    geometric sampling (k >= 1). The transition matrix P applied to SR_γ gives
    the expected discounted future occupancy starting from the next state.

    The δ parameter shifts eigenvalues away from zero, improving numerical stability.
    With δ > 0, the smallest eigenvalue is δ instead of 0.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor used in successor representation
        delta: Eigenvalue shift parameter (default 0.0). Recommended: 0.1 for stability.
        max_horizon: If provided, use finite-horizon SR_γ^(T) instead of infinite-horizon

    Returns:
        Non-symmetric Laplacian of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    # Compute successor representation (infinite or finite-horizon)
    sr_matrix = compute_successor_representation(transition_matrix, gamma, max_horizon)

    # Compute Laplacian: L = (1+δ)I - (1-γ)P·SR_γ
    laplacian = (1 + delta) * identity - (1 - gamma) * (transition_matrix @ sr_matrix)

    return laplacian


def get_nonsymmetrized_transition_matrix(
    transition_counts: jnp.ndarray,
    smoothing: float = 1e-5,
    normalize: bool = True
) -> jnp.ndarray:
    """
    Build transition matrix from counts WITHOUT symmetrization.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]
        smoothing: Small value to add for numerical stability
        normalize: Whether to row-normalize to get proper transition probabilities

    Returns:
        Transition matrix of shape [num_states, num_states]
    """
    # Sum over actions if needed
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)  # [num_states, num_states]
    else:
        transition_matrix = transition_counts

    # Add smoothing factor
    transition_matrix = transition_matrix + smoothing

    # Row-normalize if requested
    if normalize:
        row_sums = jnp.sum(transition_matrix, axis=1, keepdims=True)
        transition_matrix = transition_matrix / jnp.maximum(row_sums, 1e-10)

    return transition_matrix


def compute_eigendecomposition(
    transition_matrix: jnp.ndarray,
    k: Optional[int] = None,
    sort_by_magnitude: bool = True,
    ascending: bool = False,
    sym_eig: bool = False,
    D: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """
    Compute eigendecomposition of a matrix.

    For non-symmetric matrices, eigenvalues and eigenvectors can be complex.
    Returns both left and right eigenvectors.

    Args:
        transition_matrix: Shape [num_states, num_states]
        k: Number of eigenvalues/vectors to keep (None = keep all)
        sort_by_magnitude: If True, sort by magnitude of eigenvalues
        ascending: If True, sort in ascending order (smallest first, for Laplacians).
                   If False, sort in descending order (largest first, for transition matrices).
        sym_eig: If True, use eigh (assumes symmetric matrix); imaginary parts are set to zero.
        D: Optional diagonal weight matrix of shape [num_states, num_states].
           When provided, left eigenvectors are computed as the right eigenvectors of the
           D-adjoint D^{-1} L^T D instead of L^T.  When None (default), the standard
           transpose L^T is used (equivalent to D = Identity).

    Returns:
        Dictionary containing:
            - eigenvalues: Shape [k] (complex)
            - right_eigenvectors: Shape [num_states, k] (complex, column vectors)
            - left_eigenvectors: Shape [num_states, k] (complex, column vectors)
            - eigenvalues_real: Real part of eigenvalues
            - eigenvalues_imag: Imaginary part of eigenvalues (zeros when sym_eig=True)
            - right_eigenvectors_real: Real part of right eigenvectors [num_states, k]
            - right_eigenvectors_imag: Imaginary part of right eigenvectors [num_states, k] (zeros when sym_eig=True)
            - left_eigenvectors_real: Real part of left eigenvectors [num_states, k]
            - left_eigenvectors_imag: Imaginary part of left eigenvectors [num_states, k] (zeros when sym_eig=True)
    """
    if sym_eig:
        # eigh assumes symmetry; returns real eigenvalues sorted ascending
        eigenvalues, right_eigenvectors = jnp.linalg.eigh(transition_matrix)
        left_eigenvectors = right_eigenvectors

        # Sort by magnitude if requested (eigh already sorts ascending by value)
        if sort_by_magnitude:
            magnitudes = jnp.abs(eigenvalues)
            if ascending:
                sorted_indices = jnp.argsort(magnitudes)
            else:
                sorted_indices = jnp.argsort(-magnitudes)
            eigenvalues = eigenvalues[sorted_indices]
            right_eigenvectors = right_eigenvectors[:, sorted_indices]
            left_eigenvectors = left_eigenvectors[:, sorted_indices]

        if k is not None:
            eigenvalues = eigenvalues[:k]
            right_eigenvectors = right_eigenvectors[:, :k]
            left_eigenvectors = left_eigenvectors[:, :k]

        zeros = jnp.zeros_like(eigenvalues)
        zeros_vecs = jnp.zeros_like(right_eigenvectors)
        return {
            "eigenvalues": eigenvalues.astype(jnp.complex64),
            "right_eigenvectors": right_eigenvectors.astype(jnp.complex64),
            "left_eigenvectors": left_eigenvectors.astype(jnp.complex64),
            "eigenvalues_real": eigenvalues,
            "eigenvalues_imag": zeros,
            "right_eigenvectors_real": right_eigenvectors,
            "right_eigenvectors_imag": zeros_vecs,
            "left_eigenvectors_real": left_eigenvectors,
            "left_eigenvectors_imag": zeros_vecs,
        }

    # Compute right eigendecomposition.
    # jnp.linalg.eig returns (eigenvalues, right_eigenvectors).
    eigenvalues, right_eigenvectors = jnp.linalg.eig(transition_matrix)

    # Compute the adjoint matrix and its right eigenvectors.
    # When D is None the adjoint is L^T (standard left eigenvectors).
    # When D is provided the adjoint is D^{-1} L^T D, whose right eigenvectors
    # are the left eigenvectors of L in the D-weighted inner product.
    if D is not None:
        d = jnp.diag(D)
        D_inv = jnp.diag(1.0 / d)
        adjoint_matrix = D_inv @ transition_matrix.T @ D
    else:
        adjoint_matrix = transition_matrix.T
        d = None

    _, left_eigenvectors = jnp.linalg.eig(adjoint_matrix)

    # Normalize adjoint eigenvector columns by their D-weighted norm before
    # pairing.  This makes the matching scale-invariant (directional only),
    # preventing large-norm vectors from dominating the argmax.
    # D-norm:  ||w||_D = sqrt( w^H D w ).  When D is None this reduces to the
    # standard Euclidean norm.
    if d is not None:
        # einsum computes diag of (left^H diag(d) left), i.e. per-column D-norms²
        norms = jnp.sqrt(
            jnp.einsum('i,ij,ij->j', d, jnp.conj(left_eigenvectors), left_eigenvectors).real
        )
    else:
        norms = jnp.sqrt(jnp.sum(jnp.abs(left_eigenvectors) ** 2, axis=0))

    left_normalized = left_eigenvectors / jnp.maximum(norms, 1e-12)[None, :]

    # Match each right eigenvector to the best adjoint eigenvector via the
    # D-weighted inner product of the *normalized* adjoint columns.
    # cross_products[i, j] ≈ (w_i / ||w_i||_D)^H D r_j.
    if d is not None:
        cross_products_norm = jnp.einsum(
            'ij,i,ik->jk', jnp.conj(left_normalized), d, right_eigenvectors
        )
    else:
        cross_products_norm = jnp.einsum(
            'ij,ik->jk', jnp.conj(left_normalized), right_eigenvectors
        )

    # For each right eigenvector (column j) find the best matching adjoint
    # eigenvector (row i) — scale-free thanks to normalization above.
    best_left_indices = jnp.argmax(jnp.abs(cross_products_norm), axis=0)

    # Reorder the *un-normalized* adjoint eigenvectors to align with the right
    # eigenvectors, then apply biorthogonality normalization using the
    # D-weighted inner product of the original (un-normalized) vectors.
    left_eigenvectors = left_eigenvectors[:, best_left_indices]

    if d is not None:
        dot_products = jnp.einsum(
            'ij,i,ij->j', jnp.conj(left_eigenvectors), d, right_eigenvectors
        )
    else:
        dot_products = jnp.einsum(
            'ij,ij->j', jnp.conj(left_eigenvectors), right_eigenvectors
        )

    left_eigenvectors = left_eigenvectors / dot_products[None, :]

    # Sort by magnitude of right eigenvalues if requested.
    if sort_by_magnitude:
        magnitudes = jnp.abs(eigenvalues)
        if ascending:
            sorted_indices = jnp.argsort(magnitudes)  # Ascending order (smallest first)
        else:
            sorted_indices = jnp.argsort(-magnitudes)  # Descending order (largest first)
        eigenvalues = eigenvalues[sorted_indices]
        right_eigenvectors = right_eigenvectors[:, sorted_indices]
        left_eigenvectors = left_eigenvectors[:, sorted_indices]

    # Keep only top k if specified.
    if k is not None:
        eigenvalues = eigenvalues[:k]
        right_eigenvectors = right_eigenvectors[:, :k]
        left_eigenvectors = left_eigenvectors[:, :k]

    # Separate into real and imaginary parts.
    eigenvalues_real = jnp.real(eigenvalues)
    eigenvalues_imag = jnp.imag(eigenvalues)
    right_eigenvectors_real = jnp.real(right_eigenvectors)
    right_eigenvectors_imag = jnp.imag(right_eigenvectors)
    left_eigenvectors_real = jnp.real(left_eigenvectors)
    left_eigenvectors_imag = jnp.imag(left_eigenvectors)

    return {
        "eigenvalues": eigenvalues,
        "right_eigenvectors": right_eigenvectors,
        "left_eigenvectors": left_eigenvectors,
        "eigenvalues_real": eigenvalues_real,
        "eigenvalues_imag": eigenvalues_imag,
        "right_eigenvectors_real": right_eigenvectors_real,
        "right_eigenvectors_imag": right_eigenvectors_imag,
        "left_eigenvectors_real": left_eigenvectors_real,
        "left_eigenvectors_imag": left_eigenvectors_imag,
    }


def compute_eigendecomposition_from_counts(
    transition_counts: jnp.ndarray,
    k: Optional[int] = None,
    smoothing: float = 1e-5,
    normalize: bool = True,
    sort_by_magnitude: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Convenience function: build non-symmetrized transition matrix and compute eigendecomposition.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]
        k: Number of top eigenvalues/vectors to keep
        smoothing: Numerical stability parameter
        normalize: Whether to row-normalize
        sort_by_magnitude: Sort eigenvalues by magnitude

    Returns:
        Dictionary with eigendecomposition results (see compute_eigendecomposition)
    """
    transition_matrix = get_nonsymmetrized_transition_matrix(
        transition_counts,
        smoothing=smoothing,
        normalize=normalize
    )

    return compute_eigendecomposition(
        transition_matrix,
        k=k,
        sort_by_magnitude=sort_by_magnitude
    )


def analyze_eigenvalue_spectrum(eigendecomposition: Dict[str, jnp.ndarray]) -> Dict[str, any]:
    """
    Analyze the eigenvalue spectrum properties.

    Args:
        eigendecomposition: Output from compute_eigendecomposition

    Returns:
        Dictionary with analysis metrics:
            - num_real: Number of purely real eigenvalues
            - num_complex: Number of complex eigenvalues
            - max_imaginary_component: Maximum imaginary component magnitude
            - mean_imaginary_component: Mean imaginary component magnitude
            - eigenvalue_magnitudes: Magnitudes of all eigenvalues
    """
    eigenvalues = eigendecomposition["eigenvalues"]
    eigenvalues_imag = eigendecomposition["eigenvalues_imag"]

    # Count real vs complex eigenvalues (threshold for numerical noise)
    imag_threshold = 1e-8
    is_real = jnp.abs(eigenvalues_imag) < imag_threshold
    num_real = jnp.sum(is_real)
    num_complex = len(eigenvalues) - num_real

    # Analyze imaginary components
    max_imag = jnp.max(jnp.abs(eigenvalues_imag))
    mean_imag = jnp.mean(jnp.abs(eigenvalues_imag))

    # Eigenvalue magnitudes
    magnitudes = jnp.abs(eigenvalues)

    return {
        "num_real": int(num_real),
        "num_complex": int(num_complex),
        "max_imaginary_component": float(max_imag),
        "mean_imaginary_component": float(mean_imag),
        "eigenvalue_magnitudes": magnitudes,
    }


def compute_eigendecomposition_batched(
    batched_transition_counts: jnp.ndarray,
    k: Optional[int] = None,
    smoothing: float = 1e-5,
    normalize: bool = True,
    sort_by_magnitude: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Compute eigendecomposition for multiple environments in parallel using vmap.

    Args:
        batched_transition_counts: Shape [num_envs, num_states, num_actions, num_states]
                                   or [num_envs, num_states, num_states]
        k: Number of top eigenvalues/vectors to keep (None = keep all)
        smoothing: Numerical stability parameter
        normalize: Whether to row-normalize
        sort_by_magnitude: Sort eigenvalues by magnitude

    Returns:
        Dictionary containing batched results:
            - eigenvalues: Shape [num_envs, k] (complex)
            - eigenvectors: Shape [num_envs, num_states, k] (complex)
            - eigenvalues_real: [num_envs, k]
            - eigenvalues_imag: [num_envs, k]
            - eigenvectors_real: [num_envs, num_states, k]
            - eigenvectors_imag: [num_envs, num_states, k]
    """
    def process_one_env(env_transition_counts):
        """Process single environment."""
        # Build transition matrix
        transition_matrix = get_nonsymmetrized_transition_matrix(
            env_transition_counts,
            smoothing=smoothing,
            normalize=normalize
        )

        # Compute eigendecomposition
        eigendecomp = compute_eigendecomposition(
            transition_matrix,
            k=k,
            sort_by_magnitude=sort_by_magnitude
        )

        return eigendecomp

    # Vmap over all environments
    # We need to handle the dictionary outputs
    batched_eigendecomp = jax.vmap(process_one_env)(batched_transition_counts)

    return batched_eigendecomp