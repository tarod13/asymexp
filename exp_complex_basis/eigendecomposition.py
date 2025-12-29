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
    sort_by_magnitude: bool = True,
    ascending: bool = False
) -> Dict[str, jnp.ndarray]:
    """
    Compute eigendecomposition of a non-symmetric matrix.

    For non-symmetric matrices, eigenvalues and eigenvectors can be complex.
    Returns both left and right eigenvectors.

    Args:
        transition_matrix: Shape [num_states, num_states], non-symmetric
        k: Number of eigenvalues/vectors to keep (None = keep all)
        sort_by_magnitude: If True, sort by magnitude of eigenvalues
        ascending: If True, sort in ascending order (smallest first, for Laplacians).
                   If False, sort in descending order (largest first, for transition matrices).

    Returns:
        Dictionary containing:
            - eigenvalues: Shape [k] (complex)
            - right_eigenvectors: Shape [num_states, k] (complex, column vectors)
            - left_eigenvectors: Shape [num_states, k] (complex, column vectors)
            - eigenvalues_real: Real part of eigenvalues
            - eigenvalues_imag: Imaginary part of eigenvalues
            - right_eigenvectors_real: Real part of right eigenvectors [num_states, k]
            - right_eigenvectors_imag: Imaginary part of right eigenvectors [num_states, k]
            - left_eigenvectors_real: Real part of left eigenvectors [num_states, k]
            - left_eigenvectors_imag: Imaginary part of left eigenvectors [num_states, k]
    """
    # Compute right eigendecomposition
    # jnp.linalg.eig returns (eigenvalues, right_eigenvectors)
    eigenvalues, right_eigenvectors = jnp.linalg.eig(transition_matrix)

    # Compute left eigendecomposition via transpose
    # For left eigenvectors: v^T A = lambda v^T => A^T v = lambda v
    eigenvalues_left, left_eigenvectors = jnp.linalg.eig(transition_matrix.T)

    # Match left eigenvectors to right eigenvectors
    cross_products = jnp.einsum('ij,ik->jk', left_eigenvectors, right_eigenvectors)
    
    # For each right eigenvector (column j), find the best matching left eigenvector (row i)
    best_left_indices = jnp.argmax(jnp.abs(cross_products), axis=0)
    
    # Reorder left eigenvectors to match right eigenvectors
    left_eigenvectors = left_eigenvectors[:, best_left_indices]
    
    # Normalize left eigenvectors
    dot_products = cross_products[best_left_indices, jnp.arange(cross_products.shape[1])]
    
    # Scale left vectors by 1/dot_product
    left_eigenvectors = left_eigenvectors / dot_products[None, :]

    # Sort by magnitude if requested
    if sort_by_magnitude:
        magnitudes = jnp.abs(eigenvalues)
        if ascending:
            sorted_indices = jnp.argsort(magnitudes)  # Ascending order (smallest first)
        else:
            sorted_indices = jnp.argsort(-magnitudes)  # Descending order (largest first)
        eigenvalues = eigenvalues[sorted_indices]
        right_eigenvectors = right_eigenvectors[:, sorted_indices]
        left_eigenvectors = left_eigenvectors[:, sorted_indices]

    # Keep only top k if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        right_eigenvectors = right_eigenvectors[:, :k]
        left_eigenvectors = left_eigenvectors[:, :k]

    # Separate into real and imaginary parts
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
