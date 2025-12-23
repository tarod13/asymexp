"""
Experiment package for complex eigendecomposition analysis of non-symmetrized dynamics matrices.

This package provides tools to:
1. Compute eigendecomposition of non-symmetrized transition matrices
2. Analyze distances between states in eigenspace (real and imaginary components)
3. Visualize eigenspace distances overlaid on grid environments
4. Handle multiple environments with batched processing
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
)

from .distance_visualization import (
    visualize_distance_comparison_for_state,
    visualize_distance_heatmaps,
    visualize_multiple_example_states,
    create_distance_visualization_report,
)

from .grid_visualization import (
    visualize_distances_on_grid,
    compare_distances_on_grid,
    visualize_multiple_states_on_grid,
    visualize_grid_with_portals,
    create_grid_visualization_report,
)

__all__ = [
    "compute_eigendecomposition",
    "compute_eigendecomposition_batched",
    "get_nonsymmetrized_transition_matrix",
    "analyze_eigenvalue_spectrum",
    "compute_eigenspace_distances",
    "compute_environment_distances",
    "visualize_distance_comparison_for_state",
    "visualize_distance_heatmaps",
    "visualize_multiple_example_states",
    "create_distance_visualization_report",
    "visualize_distances_on_grid",
    "compare_distances_on_grid",
    "visualize_multiple_states_on_grid",
    "visualize_grid_with_portals",
    "create_grid_visualization_report",
]
