"""
Experiment package for complex eigendecomposition analysis of non-symmetrized dynamics matrices.

This package provides tools to:
1. Compute eigendecomposition of non-symmetrized transition matrices
2. Analyze distances between states in eigenspace (real and imaginary components)
3. Compare eigenspace distances with actual environment distances
4. Handle multiple environments with batched processing and aggregation
"""

from .eigendecomposition import (
    compute_eigendecomposition,
    compute_eigendecomposition_batched,
    get_nonsymmetrized_transition_matrix,
    analyze_eigenvalue_spectrum,
)

from .distance_analysis import (
    compute_eigenspace_distances,
    compute_environment_distances,
    compare_distances,
    analyze_distance_relationships,
    analyze_distance_relationships_batched,
    aggregate_multi_environment_results,
)

__all__ = [
    "compute_eigendecomposition",
    "compute_eigendecomposition_batched",
    "get_nonsymmetrized_transition_matrix",
    "analyze_eigenvalue_spectrum",
    "compute_eigenspace_distances",
    "compute_environment_distances",
    "compare_distances",
    "analyze_distance_relationships",
    "analyze_distance_relationships_batched",
    "aggregate_multi_environment_results",
]
