"""
Experiment package for complex eigendecomposition analysis of non-symmetrized dynamics matrices.

This package provides tools to:
1. Compute eigendecomposition of non-symmetrized transition matrices (both left and right eigenvectors)
2. Separate real and imaginary components of eigenvectors
3. Visualize eigenvector values overlaid on grid environments
4. Handle multiple environments with batched processing
"""

from .eigenvector_visualization import (
    visualize_eigenvector_on_grid,
    visualize_eigenvector_components,
    visualize_multiple_eigenvectors,
    visualize_left_right_eigenvectors,
    create_eigenvector_visualization_report,
)

__all__ = [
    "visualize_eigenvector_on_grid",
    "visualize_eigenvector_components",
    "visualize_multiple_eigenvectors",
    "visualize_left_right_eigenvectors",
    "create_eigenvector_visualization_report",
]
