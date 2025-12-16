"""
Experiment package for complex eigendecomposition analysis of non-symmetrized dynamics matrices.

This package provides tools to:
1. Compute eigendecomposition of non-symmetrized transition matrices
2. Analyze distances between states in eigenspace (real and imaginary components)
3. Compare eigenspace distances with actual environment distances
"""

from .eigendecomposition import (
    compute_eigendecomposition,
    get_nonsymmetrized_transition_matrix,
)

from .distance_analysis import (
    compute_eigenspace_distances,
    compute_environment_distances,
    compare_distances,
)

__all__ = [
    "compute_eigendecomposition",
    "get_nonsymmetrized_transition_matrix",
    "compute_eigenspace_distances",
    "compute_environment_distances",
    "compare_distances",
]
