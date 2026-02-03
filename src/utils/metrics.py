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
        abs_cos_real = (cos_real**2 + cos_imag**2).sum(-1)**0.5
        abs_cos_real_conj = (cos_real_conj**2 + cos_imag_conj**2).sum(-1)**0.5

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
            abs_cos_real = (cos_real**2 + cos_imag**2).sum(-1)**0.5

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


def compute_hitting_times_from_eigenvectors(
    left_real: jnp.ndarray,
    left_imag: jnp.ndarray,
    right_real: jnp.ndarray,
    right_imag: jnp.ndarray,
    eigenvalues_real: jnp.ndarray,
    eigenvalues_imag: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute 

    Args:
        right_real: Learned right eigenvector real parts [num_states, num_eigenvectors]
        right_imag: Learned right eigenvector imaginary parts [num_states, num_eigenvectors]
        left_real: Ground truth left eigenvector real parts [num_states, num_eigenvectors]
        left_imag: Ground truth left eigenvector imaginary parts [num_states, num_eigenvectors]
        eigenvalues_real: Real parts of eigenvalues corresponding to eigenvectors [num_eigenvectors]
        eigenvalues_imag: Imaginary parts of eigenvalues corresponding to eigenvectors [num_eigenvectors]

    Returns:
        Hitting times matrix of shape [num_states, num_states]
    """
    left = left_real + 1j * left_imag
    right = right_real + 1j * right_imag

    differences = right[:, jnp.newaxis, 1:] - right[jnp.newaxis, :, 1:]  # [num_states, num_states, num_eigenvectors]
    weighted_left = left[:, 1:] / left[:, 0:1]  # [num_states, num_eigenvectors-1]
    weighted_differences = weighted_left[:, jnp.newaxis, :] * differences  # [num_states, num_states, num_eigenvectors-1]
    eigenvalues = eigenvalues_real + 1j * eigenvalues_imag
    eigenvector_weights = 1.0 / (1.0 - eigenvalues[1:]) # [num_eigenvectors-1]
    hitting_times = jnp.einsum(
        'ijk,k->ij', jnp.real(weighted_differences), eigenvector_weights,
    )  # [num_states, num_states]

    return hitting_times