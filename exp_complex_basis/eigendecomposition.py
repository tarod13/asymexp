"""
Eigendecomposition utilities for non-symmetrized dynamics matrices.

This module provides functions to compute eigendecomposition of transition matrices
without symmetrization, preserving the potentially complex eigenvalues and eigenvectors.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Optional
import numpy as np


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
    sort_by_magnitude: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Compute eigendecomposition of a non-symmetric matrix.

    For non-symmetric matrices, eigenvalues and eigenvectors can be complex.
    Returns both eigenvalues and eigenvectors (right eigenvectors by default).

    Args:
        transition_matrix: Shape [num_states, num_states], non-symmetric
        k: Number of top eigenvalues/vectors to keep (None = keep all)
        sort_by_magnitude: If True, sort by magnitude of eigenvalues (descending)

    Returns:
        Dictionary containing:
            - eigenvalues: Shape [k] (complex)
            - eigenvectors: Shape [num_states, k] (complex, column vectors)
            - eigenvalues_real: Real part of eigenvalues
            - eigenvalues_imag: Imaginary part of eigenvalues
            - eigenvectors_real: Real part of eigenvectors [num_states, k]
            - eigenvectors_imag: Imaginary part of eigenvectors [num_states, k]
    """
    # Compute eigendecomposition
    # jnp.linalg.eig returns (eigenvalues, right_eigenvectors)
    eigenvalues, eigenvectors = jnp.linalg.eig(transition_matrix)

    # Sort by magnitude if requested
    if sort_by_magnitude:
        magnitudes = jnp.abs(eigenvalues)
        sorted_indices = jnp.argsort(-magnitudes)  # Descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

    # Keep only top k if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    # Separate into real and imaginary parts
    eigenvalues_real = jnp.real(eigenvalues)
    eigenvalues_imag = jnp.imag(eigenvalues)
    eigenvectors_real = jnp.real(eigenvectors)
    eigenvectors_imag = jnp.imag(eigenvectors)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "eigenvalues_real": eigenvalues_real,
        "eigenvalues_imag": eigenvalues_imag,
        "eigenvectors_real": eigenvectors_real,
        "eigenvectors_imag": eigenvectors_imag,
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
