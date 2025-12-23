"""
Grid-based visualization utilities for eigendecomposition distance analysis.

This module provides functions to visualize distances overlaid on the actual
grid environment, making it easy to see spatial patterns.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from pathlib import Path


def visualize_distances_on_grid(
    source_state_idx: int,
    distances: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    title: str = "Distances on Grid",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    show_values: bool = False
) -> plt.Axes:
    """
    Visualize distances from a source state overlaid on the grid.

    Args:
        source_state_idx: Canonical index of source state
        distances: Distance vector from source to all canonical states
        canonical_states: Array mapping canonical indices to full state space
        grid_width: Width of grid
        grid_height: Height of grid
        obstacles: List of (x, y) obstacle positions
        title: Plot title
        cmap: Colormap name
        ax: Optional matplotlib axes
        show_values: Whether to show distance values in cells

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Create grid of distances
    distance_grid = np.full((grid_height, grid_width), np.nan)

    # Map canonical indices back to grid positions
    for canonical_idx, dist in enumerate(distances):
        full_state_idx = canonical_states[canonical_idx]
        y = int(full_state_idx) // grid_width
        x = int(full_state_idx) % grid_width
        if y < grid_height and x < grid_width:
            distance_grid[y, x] = dist

    # Plot distances as heatmap
    im = ax.imshow(distance_grid, cmap=cmap, origin='upper', interpolation='nearest')

    # Mark obstacles
    if obstacles is not None:
        for obs_x, obs_y in obstacles:
            rect = patches.Rectangle(
                (obs_x - 0.5, obs_y - 0.5), 1, 1,
                linewidth=1, edgecolor='gray', facecolor='black', alpha=0.9
            )
            ax.add_patch(rect)

    # Mark source state
    source_full_idx = canonical_states[source_state_idx]
    source_y = int(source_full_idx) // grid_width
    source_x = int(source_full_idx) % grid_width
    ax.scatter(source_x, source_y, color='red', s=300, marker='*',
               edgecolors='white', linewidths=2, zorder=10, label='Source')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Distance')

    # Show values if requested
    if show_values:
        for y in range(grid_height):
            for x in range(grid_width):
                if not np.isnan(distance_grid[y, x]):
                    text = ax.text(x, y, f'{distance_grid[y, x]:.1f}',
                                 ha="center", va="center", color="white", fontsize=8)

    ax.legend()

    return ax


def compare_distances_on_grid(
    source_state_idx: int,
    eigenspace_distances: Dict[str, jnp.ndarray],
    environment_distances: Dict[str, jnp.ndarray],
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
):
    """
    Compare eigenspace and environment distances on the grid.

    Args:
        source_state_idx: Canonical index of source state
        eigenspace_distances: Dict with 'distances_real', 'distances_imag', 'distances_combined'
        environment_distances: Dict with 'euclidean', 'manhattan'
        canonical_states: Array mapping canonical to full state space
        grid_width: Width of grid
        grid_height: Height of grid
        obstacles: List of obstacle positions
        portals: Optional portal configuration
        figsize: Figure size
        save_path: Optional path to save
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Get distances from source state
    eigen_real = np.array(eigenspace_distances['distances_real'][source_state_idx, :])
    eigen_imag = np.array(eigenspace_distances['distances_imag'][source_state_idx, :])
    eigen_combined = np.array(eigenspace_distances['distances_combined'][source_state_idx, :])
    euclidean = np.array(environment_distances['euclidean'][source_state_idx, :])
    manhattan = np.array(environment_distances['manhattan'][source_state_idx, :])

    # Source position in full state space
    source_full_idx = canonical_states[source_state_idx]
    source_y = int(source_full_idx) // grid_width
    source_x = int(source_full_idx) % grid_width

    # Plot each distance type
    visualize_distances_on_grid(
        source_state_idx, eigen_real, canonical_states, grid_width, grid_height, obstacles,
        title=f'Eigenspace (Real) from State ({source_x}, {source_y})',
        cmap='Blues', ax=axes[0, 0]
    )

    visualize_distances_on_grid(
        source_state_idx, eigen_imag, canonical_states, grid_width, grid_height, obstacles,
        title=f'Eigenspace (Imaginary) from State ({source_x}, {source_y})',
        cmap='Reds', ax=axes[0, 1]
    )

    visualize_distances_on_grid(
        source_state_idx, eigen_combined, canonical_states, grid_width, grid_height, obstacles,
        title=f'Eigenspace (Combined) from State ({source_x}, {source_y})',
        cmap='Purples', ax=axes[0, 2]
    )

    visualize_distances_on_grid(
        source_state_idx, euclidean, canonical_states, grid_width, grid_height, obstacles,
        title=f'Euclidean Distance from State ({source_x}, {source_y})',
        cmap='Greens', ax=axes[1, 0]
    )

    visualize_distances_on_grid(
        source_state_idx, manhattan, canonical_states, grid_width, grid_height, obstacles,
        title=f'Manhattan Distance from State ({source_x}, {source_y})',
        cmap='Oranges', ax=axes[1, 1]
    )

    # Difference plot: Combined eigenspace - Euclidean
    difference = eigen_combined - euclidean
    visualize_distances_on_grid(
        source_state_idx, difference, canonical_states, grid_width, grid_height, obstacles,
        title=f'Difference (Combined - Euclidean)',
        cmap='RdBu_r', ax=axes[1, 2]
    )

    # Add portals to all subplots if provided
    if portals is not None:
        action_offsets = {0: (0, -0.25), 1: (0.25, 0), 2: (0, 0.25), 3: (-0.25, 0)}
        for ax in axes.flat:
            for (source_idx, action), dest_idx in portals.items():
                source_y_p = source_idx // grid_width
                source_x_p = source_idx % grid_width
                dest_y_p = dest_idx // grid_width
                dest_x_p = dest_idx % grid_width
                dx, dy = action_offsets.get(action, (0, 0))
                ax.arrow(
                    source_x_p + dx, source_y_p + dy,
                    dest_x_p - source_x_p - 2*dx, dest_y_p - source_y_p - 2*dy,
                    head_width=0.15, head_length=0.15,
                    fc='yellow', ec='yellow', alpha=0.7, linewidth=1.5, zorder=5
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid comparison to {save_path}")

    return fig


def visualize_multiple_states_on_grid(
    source_state_indices: List[int],
    eigenspace_distances_combined: jnp.ndarray,
    environment_distances_euclidean: jnp.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    figsize: Tuple[int, int] = (24, 10),
    save_path: Optional[str] = None
):
    """
    Visualize distances from multiple source states on the grid.

    Args:
        source_state_indices: List of canonical source state indices
        eigenspace_distances_combined: Combined eigenspace distance matrix
        environment_distances_euclidean: Euclidean distance matrix
        canonical_states: Array mapping canonical to full state space
        grid_width: Grid width
        grid_height: Grid height
        obstacles: Obstacle positions
        portals: Optional portal configuration
        figsize: Figure size
        save_path: Optional save path
    """
    num_sources = len(source_state_indices)
    # Use 2 rows, multiple columns for better aspect ratio
    ncols = min(5, num_sources)
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    action_offsets = {0: (0, -0.25), 1: (0.25, 0), 2: (0, 0.25), 3: (-0.25, 0)}

    for idx, source_idx in enumerate(source_state_indices[:ncols]):
        eigen_dists = np.array(eigenspace_distances_combined[source_idx, :])
        eucl_dists = np.array(environment_distances_euclidean[source_idx, :])

        source_full_idx = canonical_states[source_idx]
        source_y = int(source_full_idx) // grid_width
        source_x = int(source_full_idx) % grid_width

        # Eigenspace distances (top row)
        visualize_distances_on_grid(
            source_idx, eigen_dists, canonical_states, grid_width, grid_height, obstacles,
            title=f'State ({source_x}, {source_y}): Eigenspace',
            cmap='Purples', ax=axes[0, idx]
        )

        # Euclidean distances (bottom row)
        visualize_distances_on_grid(
            source_idx, eucl_dists, canonical_states, grid_width, grid_height, obstacles,
            title=f'State ({source_x}, {source_y}): Euclidean',
            cmap='Greens', ax=axes[1, idx]
        )

        # Add portals if provided
        if portals is not None:
            for ax in [axes[0, idx], axes[1, idx]]:
                for (source_idx_p, action), dest_idx in portals.items():
                    source_y_p = source_idx_p // grid_width
                    source_x_p = source_idx_p % grid_width
                    dest_y_p = dest_idx // grid_width
                    dest_x_p = dest_idx % grid_width
                    dx, dy = action_offsets.get(action, (0, 0))
                    ax.arrow(
                        source_x_p + dx, source_y_p + dy,
                        dest_x_p - source_x_p - 2*dx, dest_y_p - source_y_p - 2*dy,
                        head_width=0.15, head_length=0.15,
                        fc='yellow', ec='yellow', alpha=0.7, linewidth=1.5, zorder=5
                    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-state grid visualization to {save_path}")

    return fig


def visualize_grid_with_portals(
    grid_width: int,
    grid_height: int,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    canonical_states: Optional[jnp.ndarray] = None,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None
):
    """
    Visualize the grid environment with obstacles and portals.

    Args:
        grid_width: Grid width
        grid_height: Grid height
        obstacles: List of (x, y) obstacle positions
        portals: Dict mapping (state_idx, action) -> dest_state_idx
        canonical_states: Array of canonical state indices
        figsize: Figure size
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw grid
    for x in range(grid_width):
        for y in range(grid_height):
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                linewidth=1, edgecolor='gray', facecolor='white', alpha=0.3
            )
            ax.add_patch(rect)

    # Draw obstacles
    if obstacles is not None:
        for obs_x, obs_y in obstacles:
            rect = patches.Rectangle(
                (obs_x - 0.5, obs_y - 0.5), 1, 1,
                linewidth=1, edgecolor='gray', facecolor='black', alpha=0.9
            )
            ax.add_patch(rect)

    # Draw portals
    if portals is not None:
        action_offsets = {
            0: (0, -0.3),   # Up
            1: (0.3, 0),    # Right
            2: (0, 0.3),    # Down
            3: (-0.3, 0)    # Left
        }

        for (source_idx, action), dest_idx in portals.items():
            # Convert to grid coordinates
            source_y = source_idx // grid_width
            source_x = source_idx % grid_width
            dest_y = dest_idx // grid_width
            dest_x = dest_idx % grid_width

            # Add offset based on action
            dx, dy = action_offsets.get(action, (0, 0))

            # Draw arrow
            ax.arrow(
                source_x + dx, source_y + dy,
                dest_x - source_x - 2*dx, dest_y - source_y - 2*dy,
                head_width=0.2, head_length=0.2,
                fc='red', ec='red', alpha=0.6, linewidth=2
            )

    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Grid Environment with Portals')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid visualization to {save_path}")

    return fig


def create_grid_visualization_report(
    eigenspace_distances: Dict[str, jnp.ndarray],
    environment_distances: Dict[str, jnp.ndarray],
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    output_dir: str = "exp_complex_basis/results/grid_visualizations",
    num_example_states: int = 5
):
    """
    Create a complete grid-based visualization report.

    Args:
        eigenspace_distances: Dict with eigenspace distance matrices
        environment_distances: Dict with environment distance matrices
        canonical_states: Array mapping canonical to full state space
        grid_width: Grid width
        grid_height: Grid height
        obstacles: Obstacle positions
        portals: Portal configuration
        output_dir: Output directory
        num_example_states: Number of example states
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating grid-based visualizations...")

    # 1. Visualize grid with portals
    print("  Creating grid environment visualization...")
    visualize_grid_with_portals(
        grid_width, grid_height, obstacles, portals, canonical_states,
        save_path=output_path / "grid_environment.png"
    )
    plt.close()

    # 2. Select example states (evenly spaced)
    num_states = eigenspace_distances['distances_combined'].shape[0]
    example_indices = np.linspace(0, num_states-1, num_example_states, dtype=int)

    # 3. Detailed comparison for each example state
    print(f"  Creating detailed comparisons for {num_example_states} states...")
    for i, state_idx in enumerate(example_indices):
        compare_distances_on_grid(
            source_state_idx=int(state_idx),
            eigenspace_distances=eigenspace_distances,
            environment_distances=environment_distances,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            obstacles=obstacles,
            portals=portals,
            save_path=output_path / f"state_{state_idx}_grid_comparison.png"
        )
        plt.close()

    # 4. Side-by-side comparison for multiple states
    print("  Creating multi-state comparison...")
    visualize_multiple_states_on_grid(
        source_state_indices=list(example_indices),
        eigenspace_distances_combined=eigenspace_distances['distances_combined'],
        environment_distances_euclidean=environment_distances['euclidean'],
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        obstacles=obstacles,
        portals=portals,
        save_path=output_path / "multi_state_comparison.png"
    )
    plt.close()

    print(f"\nGrid visualization report saved to {output_dir}")
