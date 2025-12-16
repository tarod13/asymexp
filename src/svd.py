"""
SVD decomposition utilities for transition matrices.
"""

from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=1)
def transition_svd_jax(transition_matrix, k: int = 10):
    """
    Compute truncated SVD of a transition matrix using JAX.

    Args:
        transition_matrix: Square matrix of transition probabilities (n x n)
        k: Number of singular values/vectors to keep (must be static for JIT)

    Returns:
        U: Left singular vectors (n x k)
        s: Singular values (k,)
        Vh: Right singular vectors transposed (k x n)
    """
    # Compute full SVD
    U_full, s_full, Vh_full = jnp.linalg.svd(transition_matrix, full_matrices=False)

    # Truncate to top k components
    U = U_full[:, :k]
    s = s_full[:k]
    Vh = Vh_full[:k, :]

    return U, s, Vh
