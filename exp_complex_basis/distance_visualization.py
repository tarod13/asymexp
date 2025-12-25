"""
Visualization utilities for eigendecomposition distance analysis.

This module provides functions to visualize distances between states
using eigenspace representations (real and imaginary components) and
compare them with environment distances.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from pathlib import Path


def visualize_distance_comparison_for_state(
    state_idx: int,
    eigenspace_distances_real: jnp.ndarray,
    eigenspace_distances_imag: jnp.ndarray,
    eigenspace_distances_combined: jnp.ndarray,
    environment_distances: Dict[str, jnp.ndarray],
    grid_width: int,
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None
):
    """
    Visualize distances from a specific state to all other states.

    Shows how distances in eigenspace (real, imag, combined) compare
    to actual environment distances (Euclidean, Manhattan).

    Args:
        state_idx: Index of the source state
        eigenspace_distances_real: Distance matrix using real components
        eigenspace_distances_imag: Distance matrix using imaginary components
        eigenspace_distances_combined: Distance matrix using both components
        environment_distances: Dict with 'euclidean', 'manhattan' distance matrices
        grid_width: Width of the grid (for coordinate conversion)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    num_states = eigenspace_distances_real.shape[0]

    # Get distances from this state to all others
    eigen_real_dists = np.array(eigenspace_distances_real[state_idx, :])
    eigen_imag_dists = np.array(eigenspace_distances_imag[state_idx, :])
    eigen_combined_dists = np.array(eigenspace_distances_combined[state_idx, :])
    euclidean_dists = np.array(environment_distances['euclidean'][state_idx, :])
    manhattan_dists = np.array(environment_distances['manhattan'][state_idx, :])

    # Create state coordinates for x-axis
    state_indices = np.arange(num_states)

    # Compute 2D grid positions
    y_coords = state_indices // grid_width
    x_coords = state_indices % grid_width
    source_y = state_idx // grid_width
    source_x = state_idx % grid_width

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: All eigenspace distances vs state index
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(state_indices, eigen_real_dists, alpha=0.5, s=20, label='Real component', c='blue')
    ax1.scatter(state_indices, eigen_imag_dists, alpha=0.5, s=20, label='Imaginary component', c='red')
    ax1.scatter(state_indices, eigen_combined_dists, alpha=0.5, s=20, label='Combined', c='purple')
    ax1.axvline(state_idx, color='black', linestyle='--', alpha=0.3, label=f'Source state {state_idx}')
    ax1.set_xlabel('Target State Index')
    ax1.set_ylabel('Eigenspace Distance')
    ax1.set_title(f'Eigenspace Distances from State {state_idx} ({source_x}, {source_y})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Environment distances vs state index
    ax2 = fig.add_subplot(gs[1, :])
    ax2.scatter(state_indices, euclidean_dists, alpha=0.5, s=20, label='Euclidean', c='green')
    ax2.scatter(state_indices, manhattan_dists, alpha=0.5, s=20, label='Manhattan', c='orange')
    ax2.axvline(state_idx, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Target State Index')
    ax2.set_ylabel('Environment Distance')
    ax2.set_title(f'Environment Distances from State {state_idx}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Real eigenspace vs Euclidean
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.scatter(euclidean_dists, eigen_real_dists, alpha=0.3, s=10)
    ax3.set_xlabel('Euclidean Distance')
    ax3.set_ylabel('Eigenspace Distance (Real)')
    ax3.set_title('Real Component vs Euclidean')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Combined eigenspace vs Euclidean
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(euclidean_dists, eigen_combined_dists, alpha=0.3, s=10)
    ax4.set_xlabel('Euclidean Distance')
    ax4.set_ylabel('Eigenspace Distance (Combined)')
    ax4.set_title('Combined vs Euclidean')
    ax4.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig


def visualize_distance_heatmaps(
    eigenspace_distances: Dict[str, jnp.ndarray],
    environment_distances: Dict[str, jnp.ndarray],
    max_states: Optional[int] = 50,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
):
    """
    Visualize distance matrices as heatmaps.

    Args:
        eigenspace_distances: Dict with 'distances_real', 'distances_imag', 'distances_combined'
        environment_distances: Dict with 'euclidean', 'manhattan', etc.
        max_states: Maximum number of states to show (for readability)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Limit to first max_states if specified
    if max_states is not None:
        slice_idx = slice(0, max_states)
    else:
        slice_idx = slice(None)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Eigenspace distances
    sns.heatmap(
        np.array(eigenspace_distances['distances_real'][slice_idx, slice_idx]),
        ax=axes[0, 0], cmap='viridis', cbar=True, square=True
    )
    axes[0, 0].set_title('Eigenspace: Real Component')
    axes[0, 0].set_xlabel('State')
    axes[0, 0].set_ylabel('State')

    sns.heatmap(
        np.array(eigenspace_distances['distances_imag'][slice_idx, slice_idx]),
        ax=axes[0, 1], cmap='viridis', cbar=True, square=True
    )
    axes[0, 1].set_title('Eigenspace: Imaginary Component')
    axes[0, 1].set_xlabel('State')
    axes[0, 1].set_ylabel('State')

    sns.heatmap(
        np.array(eigenspace_distances['distances_combined'][slice_idx, slice_idx]),
        ax=axes[0, 2], cmap='viridis', cbar=True, square=True
    )
    axes[0, 2].set_title('Eigenspace: Combined')
    axes[0, 2].set_xlabel('State')
    axes[0, 2].set_ylabel('State')

    # Environment distances
    sns.heatmap(
        np.array(environment_distances['euclidean'][slice_idx, slice_idx]),
        ax=axes[1, 0], cmap='plasma', cbar=True, square=True
    )
    axes[1, 0].set_title('Environment: Euclidean')
    axes[1, 0].set_xlabel('State')
    axes[1, 0].set_ylabel('State')

    sns.heatmap(
        np.array(environment_distances['manhattan'][slice_idx, slice_idx]),
        ax=axes[1, 1], cmap='plasma', cbar=True, square=True
    )
    axes[1, 1].set_title('Environment: Manhattan')
    axes[1, 1].set_xlabel('State')
    axes[1, 1].set_ylabel('State')

    # Difference: Combined eigenspace vs Euclidean
    diff = np.array(eigenspace_distances['distances_combined'][slice_idx, slice_idx]) - \
           np.array(environment_distances['euclidean'][slice_idx, slice_idx])
    sns.heatmap(diff, ax=axes[1, 2], cmap='RdBu_r', center=0, cbar=True, square=True)
    axes[1, 2].set_title('Difference: Combined - Euclidean')
    axes[1, 2].set_xlabel('State')
    axes[1, 2].set_ylabel('State')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmaps to {save_path}")

    return fig


def visualize_multiple_example_states(
    example_state_indices: List[int],
    eigenspace_distances_combined: jnp.ndarray,
    environment_distances: Dict[str, jnp.ndarray],
    grid_width: int,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
):
    """
    Visualize distances for multiple example states side by side.

    Args:
        example_state_indices: List of state indices to visualize
        eigenspace_distances_combined: Combined eigenspace distance matrix
        environment_distances: Dict with environment distance matrices
        grid_width: Width of grid
        figsize: Figure size
        save_path: Optional path to save
    """
    num_examples = len(example_state_indices)
    fig, axes = plt.subplots(num_examples, 2, figsize=figsize)

    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for idx, state_idx in enumerate(example_state_indices):
        # Get distances from this state
        eigen_dists = np.array(eigenspace_distances_combined[state_idx, :])
        euclidean_dists = np.array(environment_distances['euclidean'][state_idx, :])

        # Convert to grid coordinates
        source_y = state_idx // grid_width
        source_x = state_idx % grid_width

        # Plot eigenspace distances
        num_states = len(eigen_dists)
        state_indices = np.arange(num_states)

        axes[idx, 0].scatter(state_indices, eigen_dists, alpha=0.4, s=10, c='purple')
        axes[idx, 0].axvline(state_idx, color='red', linestyle='--', alpha=0.5)
        axes[idx, 0].set_ylabel('Eigenspace Distance')
        axes[idx, 0].set_title(f'State {state_idx} ({source_x},{source_y}): Eigenspace')
        axes[idx, 0].grid(True, alpha=0.3)

        # Plot comparison
        axes[idx, 1].scatter(euclidean_dists, eigen_dists, alpha=0.3, s=10)
        axes[idx, 1].set_xlabel('Euclidean Distance')
        axes[idx, 1].set_ylabel('Eigenspace Distance')
        axes[idx, 1].set_title(f'State {state_idx}: Comparison')
        axes[idx, 1].grid(True, alpha=0.3)

        # Add diagonal reference line
        max_dist = max(euclidean_dists.max(), eigen_dists.max())
        axes[idx, 1].plot([0, max_dist], [0, max_dist], 'r--', alpha=0.3)

    axes[-1, 0].set_xlabel('Target State Index')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved example states visualization to {save_path}")

    return fig


def create_distance_visualization_report(
    eigenspace_distances: Dict[str, jnp.ndarray],
    environment_distances: Dict[str, jnp.ndarray],
    grid_width: int,
    output_dir: str = "exp_complex_basis/results/visualizations",
    num_example_states: int = 5
):
    """
    Create a complete visualization report for distance analysis.

    Args:
        eigenspace_distances: Dict with eigenspace distance matrices
        environment_distances: Dict with environment distance matrices
        grid_width: Width of the grid
        output_dir: Directory to save visualizations
        num_example_states: Number of random example states to visualize
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating distance visualizations...")

    # 1. Distance heatmaps
    print("  Creating distance heatmaps...")
    visualize_distance_heatmaps(
        eigenspace_distances,
        environment_distances,
        max_states=50,
        save_path=output_path / "distance_heatmaps.png"
    )
    plt.close()

    # 2. Select example states (evenly spaced)
    num_states = eigenspace_distances['distances_combined'].shape[0]
    example_indices = np.linspace(0, num_states-1, num_example_states, dtype=int)

    print(f"  Visualizing {num_example_states} example states...")
    visualize_multiple_example_states(
        example_state_indices=list(example_indices),
        eigenspace_distances_combined=eigenspace_distances['distances_combined'],
        environment_distances=environment_distances,
        grid_width=grid_width,
        save_path=output_path / "example_states_comparison.png"
    )
    plt.close()

    # 3. Individual detailed views for a few states
    print("  Creating detailed views for specific states...")
    for i, state_idx in enumerate(example_indices[:3]):  # First 3 examples
        visualize_distance_comparison_for_state(
            state_idx=int(state_idx),
            eigenspace_distances_real=eigenspace_distances['distances_real'],
            eigenspace_distances_imag=eigenspace_distances['distances_imag'],
            eigenspace_distances_combined=eigenspace_distances['distances_combined'],
            environment_distances=environment_distances,
            grid_width=grid_width,
            save_path=output_path / f"state_{state_idx}_detailed.png"
        )
        plt.close()

    print(f"\nVisualization report saved to {output_dir}")
