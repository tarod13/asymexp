from typing import Dict
import numpy as np
import jax.numpy as jnp

def compute_complex_cosine_similarities_with_conjugate_skipping(
    learned_real: jnp.ndarray,
    learned_imag: jnp.ndarray,
    gt_real: jnp.ndarray,
    gt_imag: jnp.ndarray,
    eigenvalues_real: jnp.ndarray,
    eigenvalues_imag: jnp.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute absolute value of real part of complex cosine similarity between learned and ground truth.

    Uses standard complex inner product with conjugate:
    For complex vectors u = u_real + i·u_imag, v = v_real + i·v_imag:
        <u, v> = conj(u)^T v = (u_real^T v_real + u_imag^T v_imag) + i(u_real^T v_imag - u_imag^T v_real)
        ||u|| = sqrt(u_real^T u_real + u_imag^T u_imag)
        cos(θ) = <u, v> / (||u|| ||v||)
        Result: |cos(θ)|

    Args:
        learned_real: Learned eigenvector real parts [num_states, num_eigenvectors]
        learned_imag: Learned eigenvector imaginary parts [num_states, num_eigenvectors]
        gt_real: Ground truth eigenvector real parts [num_states, num_eigenvectors]
        gt_imag: Ground truth eigenvector imaginary parts [num_states, num_eigenvectors]
        eigenvalues_real: Real parts of eigenvalues corresponding to eigenvectors [num_eigenvectors]
        eigenvalues_imag: Imaginary parts of eigenvalues corresponding to eigenvectors [num_eigenvectors]
        prefix: Prefix for metric names (e.g., "left_" or "right_")

    Returns:
        Dictionary containing:
            - {prefix}cosine_sim_{i}: Absolute value of real part of cosine similarity for component i
            - {prefix}cosine_sim_avg: Average across all components
    """
    num_components = learned_real.shape[1]

    similarities = {}
    cosine_sims = []

    j = 0
    for i in range(num_components):

        # Real eigenvalue: single real eigenvector
        u_real = learned_real[:, i]
        u_imag = learned_imag[:, i]
        v_real = gt_real[:, j]
        v_imag = gt_imag[:, j]

        # Standard complex inner product (with conjugate): <u, v> = conj(u)^T v
        # Real part: u_real^T v_real + u_imag^T v_imag
        # Imag part: u_real^T v_imag - u_imag^T v_real
        inner_real = jnp.dot(u_real, v_real) + jnp.dot(u_imag, v_imag)
        inner_imag = jnp.dot(u_real, v_imag) - jnp.dot(u_imag, v_real)

        # Conjugate inner product: <u, conj(v)> = conj(u)^T conj(v)
        inner_real_conj = jnp.dot(u_real, v_real) - jnp.dot(u_imag, v_imag)
        inner_imag_conj = -jnp.dot(u_real, v_imag) - jnp.dot(u_imag, v_real)

        # Magnitudes: ||u|| = sqrt(u_real^T u_real + u_imag^T u_imag)
        u_norm = jnp.sqrt(jnp.dot(u_real, u_real) + jnp.dot(u_imag, u_imag))
        v_norm = jnp.sqrt(jnp.dot(v_real, v_real) + jnp.dot(v_imag, v_imag))

        # Complex cosine similarity
        cos_real = inner_real / (u_norm * v_norm + 1e-10)
        cos_imag = inner_imag / (u_norm * v_norm + 1e-10)

        cos_real_conj = inner_real_conj / (u_norm * v_norm + 1e-10)
        cos_imag_conj = inner_imag_conj / (u_norm * v_norm + 1e-10)

        # Take absolute value of real part
        abs_cos_real = (cos_real**2 + cos_imag**2)**0.5
        abs_cos_real_conj = (cos_real_conj**2 + cos_imag_conj**2)**0.5

        sim = float(max(abs_cos_real, abs_cos_real_conj))
        similarities[f'{prefix}cosine_sim_{j}'] = sim
        cosine_sims.append(sim)

        is_real_eigval = eigenvalues_imag[j] < 1e-8
        if is_real_eigval:
            j += 1
        else:
            # Skip the next component as it's part of the complex pair
            j += 2

            
    # Average across all components
    similarities[f'{prefix}cosine_sim_avg'] = float(np.mean(cosine_sims))

    return similarities


def compute_complex_cosine_similarities(
    learned_real: jnp.ndarray,
    learned_imag: jnp.ndarray,
    gt_real: jnp.ndarray,
    gt_imag: jnp.ndarray,
    eigenvalues_real: jnp.ndarray,
    eigenvalues_imag: jnp.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute absolute value of real part of complex cosine similarity between learned and ground truth.

    Uses standard complex inner product with conjugate:
    For complex vectors u = u_real + i·u_imag, v = v_real + i·v_imag:
        <u, v> = conj(u)^T v = (u_real^T v_real + u_imag^T v_imag) + i(u_real^T v_imag - u_imag^T v_real)
        ||u|| = sqrt(u_real^T u_real + u_imag^T u_imag)
        cos(θ) = <u, v> / (||u|| ||v||)
        Result: |cos(θ)|

    Args:
        learned_real: Learned eigenvector real parts [num_states, num_eigenvectors]
        learned_imag: Learned eigenvector imaginary parts [num_states, num_eigenvectors]
        gt_real: Ground truth eigenvector real parts [num_states, num_eigenvectors]
        gt_imag: Ground truth eigenvector imaginary parts [num_states, num_eigenvectors]
        eigenvalues_real: Real parts of eigenvalues corresponding to eigenvectors [num_eigenvectors]
        eigenvalues_imag: Imaginary parts of eigenvalues corresponding to eigenvectors [num_eigenvectors]
        prefix: Prefix for metric names (e.g., "left_" or "right_")

    Returns:
        Dictionary containing:
            - {prefix}cosine_sim_{i}: Absolute value of real part of cosine similarity for component i
            - {prefix}cosine_sim_avg: Average across all components
    """
    num_components = learned_real.shape[1]

    similarities = {}
    cosine_sims = []

    j =0
    while j < num_components:
        is_real_eigval = eigenvalues_imag[j] < 1e-8
        if is_real_eigval:
            # Real eigenvalue: single real eigenvector
            u_real = learned_real[:, j]
            u_imag = learned_imag[:, j]
            v_real = gt_real[:, j]
            v_imag = gt_imag[:, j]

            # Standard complex inner product (with conjugate): <u, v> = conj(u)^T v
            # Real part: u_real^T v_real + u_imag^T v_imag
            # Imag part: u_real^T v_imag - u_imag^T v_real
            inner_real = jnp.dot(u_real, v_real) + jnp.dot(u_imag, v_imag)
            inner_imag = jnp.dot(u_real, v_imag) - jnp.dot(u_imag, v_real)

            # Magnitudes: ||u|| = sqrt(u_real^T u_real + u_imag^T u_imag)
            u_norm = jnp.sqrt(jnp.dot(u_real, u_real) + jnp.dot(u_imag, u_imag))
            v_norm = jnp.sqrt(jnp.dot(v_real, v_real) + jnp.dot(v_imag, v_imag))

            # Complex cosine similarity
            cos_real = inner_real / (u_norm * v_norm + 1e-10)
            cos_imag = inner_imag / (u_norm * v_norm + 1e-10)

            # Take absolute value of real part
            abs_cos_real = (cos_real**2 + cos_imag**2)**0.5

            sim = float(abs_cos_real)
            similarities[f'{prefix}cosine_sim_{j}'] = sim
            cosine_sims.append(sim)

            j += 1

        else:
            # Extract complex vectors
            u_real = learned_real[:, j:j+2]
            u_imag = learned_imag[:, j:j+2]
            v_real = gt_real[:, j:j+2]
            v_imag = gt_imag[:, j:j+2]

            # Standard complex inner product (with conjugate): <u, v> = conj(u)^T v
            # Real part: u_real^T v_real + u_imag^T v_imag
            # Imag part: u_real^T v_imag - u_imag^T v_real
            inner_real = jnp.einsum('ij,ik->jk', u_real, v_real) + jnp.einsum('ij,ik->jk', u_imag, v_imag)
            inner_imag = jnp.einsum('ij,ik->jk', u_real, v_imag) - jnp.einsum('ij,ik->jk', u_imag, v_real)

            # Magnitudes: ||u|| = sqrt(u_real^T u_real + u_imag^T u_imag)
            u_norm = jnp.sqrt(jnp.einsum('ij,ij->j', u_real, u_real) + jnp.einsum('ij,ij->j', u_imag, u_imag)).reshape(-1, 1)
            v_norm = jnp.sqrt(jnp.einsum('ij,ij->j', v_real, v_real) + jnp.einsum('ij,ij->j', v_imag, v_imag)).reshape(1, -1)

            # Complex cosine similarity
            cos_real = inner_real / (u_norm * v_norm + 1e-10)
            cos_imag = inner_imag / (u_norm * v_norm + 1e-10)

            # Take absolute value of real part
            abs_cos_real = (cos_real**2 + cos_imag**2).sum(-1)**0.5

            similarities[f'{prefix}cosine_sim_{j}'] = float(abs_cos_real[0])
            similarities[f'{prefix}cosine_sim_{j+1}'] = float(abs_cos_real[1])
            cosine_sims.append(float(abs_cos_real[0]))
            cosine_sims.append(float(abs_cos_real[1]))

            j += 2

    # Average across all components
    similarities[f'{prefix}cosine_sim_avg'] = float(np.mean(cosine_sims))

    return similarities


def normalize_eigenvectors_for_comparison(
    left_real: jnp.ndarray,
    left_imag: jnp.ndarray,
    right_real: jnp.ndarray,
    right_imag: jnp.ndarray,
    sampling_probs: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Apply normalization transformations to eigenvectors for proper comparison.

    The learned left eigenvectors correspond to eigenvectors of the adjoint with respect
    to the inner product determined by the replay buffer state distribution D.

    Normalization procedure:
    1. For RIGHT eigenvectors:
       - Find component with largest magnitude
       - Divide by that component (fixes arbitrary phase)
       - Normalize to unit norm

    2. For LEFT eigenvectors:
       - Multiply by the largest component from corresponding right eigenvector
       - Multiply by the norm of the corresponding right eigenvector (before normalization)
       - Scale each entry by the stationary state distribution (to convert from adjoint)

    This ensures proper comparison between learned and ground truth eigenvectors while
    accounting for:
    - Arbitrary complex scaling freedom
    - Adjoint vs. true left eigenvector relationship

    Args:
        left_real: Left eigenvector real parts [num_states, num_eigenvectors]
        left_imag: Left eigenvector imaginary parts [num_states, num_eigenvectors]
        right_real: Right eigenvector real parts [num_states, num_eigenvectors]
        right_imag: Right eigenvector imaginary parts [num_states, num_eigenvectors]
        sampling_probs: State distribution probabilities [num_states]

    Returns:
        Dictionary containing:
            - 'left_real': Normalized left eigenvector real parts
            - 'left_imag': Normalized left eigenvector imaginary parts
            - 'right_real': Normalized right eigenvector real parts
            - 'right_imag': Normalized right eigenvector imaginary parts
    """
    num_components = right_real.shape[1]
    num_states = right_real.shape[0]

    # Normalize eigenvectors
    normalized_right_real = jnp.zeros_like(right_real)
    normalized_right_imag = jnp.zeros_like(right_imag)
    normalized_left_real = jnp.zeros_like(left_real)
    normalized_left_imag = jnp.zeros_like(left_imag)

    for i in range(num_components):
        # Step 1: Process right eigenvectors
        right_r = right_real[:, i]
        right_i = right_imag[:, i]

        # Find component with largest magnitude
        magnitudes = jnp.sqrt(right_r**2 + right_i**2)
        max_idx = jnp.argmax(magnitudes)
        max_component_real = right_r[max_idx]
        max_component_imag = right_i[max_idx]

        # Complex division: divide by max component
        # (a + bi) / (c + di) = [(ac + bd) + i(bc - ad)] / (c^2 + d^2)
        denom = max_component_real**2 + max_component_imag**2 + 1e-10
        scaled_right_real = (right_r * max_component_real + right_i * max_component_imag) / denom
        scaled_right_imag = (right_i * max_component_real - right_r * max_component_imag) / denom

        # Normalize to unit norm
        right_norm = jnp.sqrt(jnp.sum(scaled_right_real**2 + scaled_right_imag**2))
        normalized_right_real = normalized_right_real.at[:, i].set(scaled_right_real / (right_norm + 1e-10))
        normalized_right_imag = normalized_right_imag.at[:, i].set(scaled_right_imag / (right_norm + 1e-10))

        # Step 2: Process left eigenvectors (complementary normalization)
        left_r = left_real[:, i]
        left_i = left_imag[:, i]

        # Compute the original norm of the right eigenvector (before normalization)
        original_right_norm = jnp.sqrt(jnp.sum(right_r**2 + right_i**2))

        # Multiply by max component (conjugate for complex multiplication)
        # (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        scaled_left_real = (left_r * max_component_real - left_i * max_component_imag)
        scaled_left_imag = (left_r * max_component_imag + left_i * max_component_real)

        # Multiply by the original norm of the right eigenvector
        scaled_left_real = scaled_left_real * original_right_norm
        scaled_left_imag = scaled_left_imag * original_right_norm

        # Scale each entry by the stationary state distribution
        # This converts from adjoint eigenvectors to true left eigenvectors
        scaled_left_real = scaled_left_real * sampling_probs
        scaled_left_imag = scaled_left_imag * sampling_probs

        normalized_left_real = normalized_left_real.at[:, i].set(scaled_left_real)
        normalized_left_imag = normalized_left_imag.at[:, i].set(scaled_left_imag)

    return {
        'left_real': normalized_left_real,
        'left_imag': normalized_left_imag,
        'right_real': normalized_right_real,
        'right_imag': normalized_right_imag,
    }


def compute_complex_cosine_similarities_with_normalization(
    learned_left_real: jnp.ndarray,
    learned_left_imag: jnp.ndarray,
    learned_right_real: jnp.ndarray,
    learned_right_imag: jnp.ndarray,
    gt_left_real: jnp.ndarray,
    gt_left_imag: jnp.ndarray,
    gt_right_real: jnp.ndarray,
    gt_right_imag: jnp.ndarray,
    gt_eigenvalues_real: jnp.ndarray,
    gt_eigenvalues_imag: jnp.ndarray,
    sampling_probs: jnp.ndarray,
    skip_conjugates: bool = False,
) -> Dict[str, float]:
    """
    Compute cosine similarities with proper normalization for adjoint eigenvectors.

    The learned left eigenvectors (psi) correspond to complex conjugates of eigenvectors
    of the adjoint with respect to the inner product determined by the replay buffer
    state distribution D.

    Normalization procedure:
    1. For RIGHT eigenvectors:
       - Find component with largest magnitude
       - Divide by that component (fixes arbitrary phase)
       - Normalize to unit norm

    2. For LEFT eigenvectors:
       - Multiply by the largest component from corresponding right eigenvector
       - Multiply by the norm of the corresponding right eigenvector (before normalization)
       - Scale each entry by the stationary state distribution (to convert from adjoint)

    Important: Ground truth eigenvectors are already normalized and should NOT be
    normalized again. We only normalize the learned eigenvectors.

    Also: Since learned psi are complex conjugates of adjoint eigenvectors, we use
    the complex conjugate of learned psi when computing cosine similarities.

    Args:
        learned_left_real: Learned left eigenvector real parts [num_states, num_learned]
        learned_left_imag: Learned left eigenvector imaginary parts [num_states, num_learned]
        learned_right_real: Learned right eigenvector real parts [num_states, num_learned]
        learned_right_imag: Learned right eigenvector imaginary parts [num_states, num_learned]
        gt_left_real: Ground truth left eigenvector real parts [num_states, num_gt]
        gt_left_imag: Ground truth left eigenvector imaginary parts [num_states, num_gt]
        gt_right_real: Ground truth right eigenvector real parts [num_states, num_gt]
        gt_right_imag: Ground truth right eigenvector imaginary parts [num_states, num_gt]
        gt_eigenvalues_real: Real parts of ground truth eigenvalues [num_gt]
        gt_eigenvalues_imag: Imaginary parts of ground truth eigenvalues [num_gt]
        sampling_probs: State distribution probabilities [num_states]
        skip_conjugates: If True, use conjugate skipping when matching learned to ground truth.
            This is needed when conjugate eigenvectors are not directly learned but skipped.
            Ground truth should have more eigenvectors than learned (e.g., 2x) to accommodate.

    Returns:
        Dictionary containing cosine similarities for left and right eigenvectors
    """
    # Normalize learned eigenvectors only (ground truth is already normalized)
    learned_normalized = normalize_eigenvectors_for_comparison(
        left_real=learned_left_real,
        left_imag=learned_left_imag,
        right_real=learned_right_real,
        right_imag=learned_right_imag,
        sampling_probs=sampling_probs
    )

    # Select the appropriate cosine similarity function
    if skip_conjugates:
        cosine_sim_fn = compute_complex_cosine_similarities_with_conjugate_skipping
    else:
        cosine_sim_fn = compute_complex_cosine_similarities

    # For left eigenvectors
    left_sims = cosine_sim_fn(
        learned_normalized['left_real'], learned_normalized['left_imag'],
        gt_left_real, gt_left_imag,
        gt_eigenvalues_real, gt_eigenvalues_imag,
        prefix="left_"
    )

    # For right eigenvectors
    right_sims = cosine_sim_fn(
        learned_normalized['right_real'], learned_normalized['right_imag'],
        gt_right_real, gt_right_imag,
        gt_eigenvalues_real, gt_eigenvalues_imag,
        prefix="right_"
    )

    # Combine results
    result = {}
    result.update(left_sims)
    result.update(right_sims)

    return result


def enforce_conjugate_pairs(
    left_real: jnp.ndarray,
    left_imag: jnp.ndarray,
    right_real: jnp.ndarray,
    right_imag: jnp.ndarray,
    eigenvalues_real: jnp.ndarray,
    eigenvalues_imag: jnp.ndarray,
    imag_threshold: float = 1e-6,
    conjugate_tolerance: float = 0.5,
):
    """
    Enforce conjugate structure for consecutive eigenvector pairs.

    For learned eigenvectors, imperfections mean that pairs (k, k+1) that should
    be complex conjugates don't perfectly satisfy v_{k+1} = conj(v_k). This
    function detects such pairs and enforces exact conjugation by averaging:

    For a detected conjugate pair at indices k, k+1:
        - Real parts: (real_k + real_{k+1}) / 2  (should be identical)
        - Imag parts: (imag_k - imag_{k+1}) / 2 for k, negated for k+1

    Detection: consecutive eigenvalues with |imag| > imag_threshold and
    imaginary parts that are approximately opposite (relative error < conjugate_tolerance).

    Args:
        left_real, left_imag: Left eigenvectors [num_states, num_eigenvectors]
        right_real, right_imag: Right eigenvectors [num_states, num_eigenvectors]
        eigenvalues_real, eigenvalues_imag: Eigenvalues [num_eigenvectors]
        imag_threshold: Minimum |imag(eigenvalue)| to consider a pair complex
        conjugate_tolerance: Maximum relative sum |imag_k + imag_{k+1}| / max(|imag_k|, |imag_{k+1}|)

    Returns:
        Tuple of (left_real, left_imag, right_real, right_imag,
                  eigenvalues_real, eigenvalues_imag) with conjugate pairs enforced
    """
    n_eig = eigenvalues_real.shape[0]

    out_left_real = jnp.array(left_real)
    out_left_imag = jnp.array(left_imag)
    out_right_real = jnp.array(right_real)
    out_right_imag = jnp.array(right_imag)
    out_eig_real = jnp.array(eigenvalues_real)
    out_eig_imag = jnp.array(eigenvalues_imag)

    k = 0
    while k < n_eig - 1:
        imag_k = float(eigenvalues_imag[k])
        imag_k1 = float(eigenvalues_imag[k + 1])

        max_abs_imag = max(abs(imag_k), abs(imag_k1))

        # Both must have non-negligible imaginary parts
        if max_abs_imag > imag_threshold:
            # Check if imaginary parts are approximately opposite
            sum_imag = abs(imag_k + imag_k1)
            if sum_imag < conjugate_tolerance * max_abs_imag:
                # Enforce conjugate structure for eigenvalues
                avg_eig_real = (float(eigenvalues_real[k]) + float(eigenvalues_real[k + 1])) / 2.0
                avg_eig_imag = (imag_k - imag_k1) / 2.0
                out_eig_real = out_eig_real.at[k].set(avg_eig_real)
                out_eig_real = out_eig_real.at[k + 1].set(avg_eig_real)
                out_eig_imag = out_eig_imag.at[k].set(avg_eig_imag)
                out_eig_imag = out_eig_imag.at[k + 1].set(-avg_eig_imag)

                # Enforce conjugate structure for right eigenvectors
                avg_right_real = (right_real[:, k] + right_real[:, k + 1]) / 2.0
                avg_right_imag = (right_imag[:, k] - right_imag[:, k + 1]) / 2.0
                out_right_real = out_right_real.at[:, k].set(avg_right_real)
                out_right_real = out_right_real.at[:, k + 1].set(avg_right_real)
                out_right_imag = out_right_imag.at[:, k].set(avg_right_imag)
                out_right_imag = out_right_imag.at[:, k + 1].set(-avg_right_imag)

                # Enforce conjugate structure for left eigenvectors
                avg_left_real = (left_real[:, k] + left_real[:, k + 1]) / 2.0
                avg_left_imag = (left_imag[:, k] - left_imag[:, k + 1]) / 2.0
                out_left_real = out_left_real.at[:, k].set(avg_left_real)
                out_left_real = out_left_real.at[:, k + 1].set(avg_left_real)
                out_left_imag = out_left_imag.at[:, k].set(avg_left_imag)
                out_left_imag = out_left_imag.at[:, k + 1].set(-avg_left_imag)

                k += 2
                continue

        k += 1

    return (out_left_real, out_left_imag, out_right_real, out_right_imag,
            out_eig_real, out_eig_imag)


def compute_hitting_times_from_eigenvectors(
    left_real: jnp.ndarray,
    left_imag: jnp.ndarray,
    right_real: jnp.ndarray,
    right_imag: jnp.ndarray,
    eigenvalues_real: jnp.ndarray,
    eigenvalues_imag: jnp.ndarray,
    gamma: float = None,
    delta: float = 0.0,
    enforce_conjugates: bool = True,
) -> jnp.ndarray:
    """
    Compute hitting times from eigenvector decomposition.

    The hitting time formula is:
        h(i,j) = Σ_{k>=1} [1/(1-λ_P_k)] * (1/π_j) * ψ_jk * (φ_jk - φ_ik)

    where λ_P are eigenvalues of the transition matrix P, ψ/φ are left/right
    eigenvectors, and π is the stationary distribution (left eigenvector 0).

    When eigenvalues come from the Laplacian L = (1+δ)I - (1-γ)P·SR_γ (i.e.
    gamma is provided), they are converted to transition matrix eigenvalues:
        λ_P = (1 + δ - λ_L) / (1 + γδ - γλ_L)
    The eigenvectors are shared between P and L so no conversion is needed.

    For learned eigenvectors, conjugate pairs may be imperfect. When
    enforce_conjugates=True, consecutive pairs are averaged to enforce exact
    conjugate structure before computing hitting times.

    Args:
        left_real: Left eigenvector real parts [num_states, num_eigenvectors]
        left_imag: Left eigenvector imaginary parts [num_states, num_eigenvectors]
        right_real: Right eigenvector real parts [num_states, num_eigenvectors]
        right_imag: Right eigenvector imaginary parts [num_states, num_eigenvectors]
        eigenvalues_real: Real parts of eigenvalues [num_eigenvectors]
        eigenvalues_imag: Imaginary parts of eigenvalues [num_eigenvectors]
        gamma: Discount factor. If provided, eigenvalues are treated as Laplacian
            eigenvalues and converted to transition matrix eigenvalues.
            If None, eigenvalues are assumed to already be transition matrix eigenvalues.
        delta: Eigenvalue shift parameter used in Laplacian construction (default 0.0).
        enforce_conjugates: If True, enforce conjugate structure for consecutive
            pairs before computing hitting times.

    Returns:
        Hitting times matrix of shape [num_states, num_states]
    """
    if enforce_conjugates:
        left_real, left_imag, right_real, right_imag, \
            eigenvalues_real, eigenvalues_imag = enforce_conjugate_pairs(
                left_real, left_imag, right_real, right_imag,
                eigenvalues_real, eigenvalues_imag,
            )

    left = left_real + 1j * left_imag
    right = right_real + 1j * right_imag
    eigenvalues = eigenvalues_real + 1j * eigenvalues_imag

    # Convert Laplacian eigenvalues to transition matrix eigenvalues if needed
    if gamma is not None:
        # λ_P = (1 + δ - λ_L) / (1 + γδ - γλ_L)
        eigenvalues = (1.0 + delta - eigenvalues) / (1.0 + gamma * delta - gamma * eigenvalues)

    # Stationary distribution from first left eigenvector
    stationary = left[:, 0]

    # Effective horizon weights: 1/(1-λ_P) for k >= 1
    mode_weights = 1.0 / (1.0 - eigenvalues[1:])  # [num_eigenvectors-1]

    # Pairwise differences of right eigenvectors: φ_jk - φ_ik
    # pairwise_diff[j, i, k] = right[j, k] - right[i, k]
    pairwise_diff = (
        jnp.expand_dims(right[:, 1:], axis=1)
        - jnp.expand_dims(right[:, 1:], axis=0)
    )  # [num_states, num_states, num_eigenvectors-1]

    # h(i,j) = Σ_k mode_weights[k] * (1/π_j) * ψ_jk * (φ_jk - φ_ik)
    hitting_times = jnp.real(jnp.einsum(
        'k,j,jk,jik->ij',
        mode_weights,
        1.0 / stationary,
        left[:, 1:],
        pairwise_diff,
    ))  # [num_states, num_states]

    return hitting_times