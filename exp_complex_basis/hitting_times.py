"""
Hitting times utilities for non-symmetrized dynamics matrices.

This module provides functions to compute hitting times of 
transition matrices without symmetrization.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Optional
import numpy as np


def compute_hitting_times(
    eigendecomposition: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute hitting times from complex eigenvectors.

    Args:
        eigendecomposition: Output from compute_eigendecomposition function.

    Returns:
        Matrix of shape [num_states, num_states]
    """
    left_eigenvectors = eigendecomposition["left_eigenvectors"]
    right_eigenvectors = eigendecomposition["right_eigenvectors"]
    eigenvalues = eigendecomposition["eigenvalues"]

    # The stationary distribution is given by the first left eigenvector
    stationary_distribution = jnp.real(left_eigenvectors[:, 0] / jnp.sum(left_eigenvectors[:, 0]))
    
    # Effective horizons for each mode, coming from the fundamental matrix Z 
    # calculation, in terms of the eigen-decomposition of P. 
    mode_effective_horizons = 1 / (1 - eigenvalues[1:])  # Exclude first eigenvalue (1.0)

    pairwise_differences = (
        jnp.expand_dims(right_eigenvectors[:, 1:], axis=1) 
        - jnp.expand_dims(right_eigenvectors[:, 1:], axis=0)
    )  # Shape [num_states, num_states, num_eigenvectors-1]

    hitting_times = jnp.einsum(
        'k,j,jk,jik->ij',
        mode_effective_horizons,
        1 / stationary_distribution,
        left_eigenvectors[:, 1:],
        pairwise_differences,
    )  # Shape [num_states, num_states]

    hitting_times_imaginary_error = (jnp.imag(hitting_times)**2).sum() / (hitting_times.shape[0]**2)
    hitting_times = jnp.real(hitting_times)

    return hitting_times, hitting_times_imaginary_error


def compute_hitting_times_batched(
    batched_eigendecomposition: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute hitting times for a batch of eigendecompositions.

    Args:
        batched_eigendecomposition: Dict with keys 'left_eigenvectors', 
            'right_eigenvectors', 'eigenvalues', each of shape 
            [batch_size, ...] corresponding to multiple environments.
    Returns:
        hitting_times: Shape [batch_size, num_states, num_states]
        hitting_times_imaginary_error: Shape [batch_size]
    """
    def compute_single_hitting_times(eigendecomposition):
        return compute_hitting_times(eigendecomposition)

    hitting_times, hitting_times_imaginary_error = jax.vmap(
        compute_single_hitting_times
    )(batched_eigendecomposition)

    return hitting_times, hitting_times_imaginary_error