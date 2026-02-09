"""
Visualization utilities for eigendecomposition analysis.

This module provides functions to visualize:
1. Eigenvalue spectrum (magnitude, real/imaginary components)
2. Distance matrices (eigenspace vs environment)
3. Scatter plots comparing distances
4. Eigenvector heatmaps
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from pathlib import Path


def plot_eigenvalue_spectrum(
    eigendecomposition: Dict[str, jnp.ndarray],
    max_k: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """Plot eigenvalue spectrum showing magnitudes and real/imaginary components."""
    eigenvalues = eigendecomposition["eigenvalues"]
    eigenvalues_real = eigendecomposition["eigenvalues_real"]
    eigenvalues_imag = eigendecomposition["eigenvalues_imag"]

    if max_k is not None:
        eigenvalues = eigenvalues[:max_k]
        eigenvalues_real = eigenvalues_real[:max_k]
        eigenvalues_imag = eigenvalues_imag[:max_k]

    magnitudes = jnp.abs(eigenvalues)
    k = len(eigenvalues)
    indices = np.arange(k)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].bar(indices, magnitudes, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Eigenvalue Magnitudes')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(eigenvalues_real, eigenvalues_imag, alpha=0.6, s=50)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Real Component')
    axes[1].set_ylabel('Imaginary Component')
    axes[1].set_title('Eigenvalues in Complex Plane')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    axes[2].plot(indices, eigenvalues_real, 'o-', label='Real', alpha=0.7)
    axes[2].plot(indices, eigenvalues_imag, 's-', label='Imaginary', alpha=0.7)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Eigenvalue Index')
    axes[2].set_ylabel('Component Value')
    axes[2].set_title('Real and Imaginary Components')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvalue spectrum plot to {save_path}")

    return fig


def plot_distance_matrices(
    eigenspace_distances: Dict[str, jnp.ndarray],
    environment_distances: Dict[str, jnp.ndarray],
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None
):
    """Plot heatmaps of distance matrices."""
    n_eigen = len(eigenspace_distances)
    n_env = len(environment_distances)
    n_total = n_eigen + n_env

    n_cols = 3
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    plot_idx = 0

    for name, dist_matrix in eigenspace_distances.items():
        ax = axes[plot_idx]
        dist_array = np.array(dist_matrix)
        sns.heatmap(dist_array, cmap='viridis', ax=ax, cbar=True, square=True)
        ax.set_title(f'Eigenspace: {name}')
        ax.set_xlabel('State')
        ax.set_ylabel('State')
        plot_idx += 1

    for name, dist_matrix in environment_distances.items():
        ax = axes[plot_idx]
        dist_array = np.array(dist_matrix)
        if np.any(np.isinf(dist_array)):
            dist_array = np.where(np.isinf(dist_array), np.nan, dist_array)
        sns.heatmap(dist_array, cmap='plasma', ax=ax, cbar=True, square=True)
        ax.set_title(f'Environment: {name}')
        ax.set_xlabel('State')
        ax.set_ylabel('State')
        plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distance matrices plot to {save_path}")

    return fig


def plot_distance_comparison_scatter(
    eigenspace_distances: jnp.ndarray,
    environment_distances: jnp.ndarray,
    title: str = "Eigenspace vs Environment Distances",
    exclude_diagonal: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
):
    """Scatter plot comparing eigenspace and environment distances."""
    if exclude_diagonal:
        num_states = eigenspace_distances.shape[0]
        mask = np.triu(np.ones((num_states, num_states), dtype=bool), k=1)
        eigen_flat = np.array(eigenspace_distances)[mask]
        env_flat = np.array(environment_distances)[mask]
    else:
        eigen_flat = np.array(eigenspace_distances).flatten()
        env_flat = np.array(environment_distances).flatten()

    finite_mask = np.isfinite(eigen_flat) & np.isfinite(env_flat)
    eigen_flat = eigen_flat[finite_mask]
    env_flat = env_flat[finite_mask]

    correlation = np.corrcoef(eigen_flat, env_flat)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(env_flat, eigen_flat, alpha=0.3, s=10)
    ax.set_xlabel('Environment Distance')
    ax.set_ylabel('Eigenspace Distance')
    ax.set_title(f'{title}\nPearson r = {correlation:.4f}')
    ax.grid(True, alpha=0.3)

    min_val = min(env_flat.min(), eigen_flat.min())
    max_val = max(env_flat.max(), eigen_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distance comparison scatter to {save_path}")

    return fig


def plot_eigenvector_heatmap(
    eigendecomposition: Dict[str, jnp.ndarray],
    max_k: Optional[int] = 20,
    plot_real: bool = True,
    plot_imag: bool = True,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
):
    """Plot heatmap of eigenvectors."""
    n_plots = int(plot_real) + int(plot_imag)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if plot_real:
        eigenvectors_real = np.array(eigendecomposition["eigenvectors_real"])
        if max_k is not None:
            eigenvectors_real = eigenvectors_real[:, :max_k]
        ax = axes[plot_idx]
        sns.heatmap(eigenvectors_real.T, cmap='RdBu_r', center=0, ax=ax, cbar=True)
        ax.set_xlabel('State')
        ax.set_ylabel('Eigenvector Index')
        ax.set_title('Real Components of Eigenvectors')
        plot_idx += 1

    if plot_imag:
        eigenvectors_imag = np.array(eigendecomposition["eigenvectors_imag"])
        if max_k is not None:
            eigenvectors_imag = eigenvectors_imag[:, :max_k]
        ax = axes[plot_idx]
        sns.heatmap(eigenvectors_imag.T, cmap='RdBu_r', center=0, ax=ax, cbar=True)
        ax.set_xlabel('State')
        ax.set_ylabel('Eigenvector Index')
        ax.set_title('Imaginary Components of Eigenvectors')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvector heatmap to {save_path}")

    return fig


def create_full_analysis_report(
    results: Dict,
    output_dir: str = "experiments/01/results",
    max_k_plot: int = 20
):
    """Create a full visualization report from analysis results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualization report...")

    print("  Plotting eigenvalue spectrum...")
    plot_eigenvalue_spectrum(
        results["eigendecomposition"],
        max_k=max_k_plot,
        save_path=output_path / "eigenvalue_spectrum.png"
    )
    plt.close()

    print("  Plotting eigenvector heatmaps...")
    plot_eigenvector_heatmap(
        results["eigendecomposition"],
        max_k=max_k_plot,
        save_path=output_path / "eigenvector_heatmap.png"
    )
    plt.close()

    print("  Plotting distance comparisons...")
    distance_analysis = results["distance_analysis"]

    for k_label, k_results in distance_analysis["eigenspace_comparisons"].items():
        k_dir = output_path / k_label.replace("=", "_")
        k_dir.mkdir(exist_ok=True)

        plot_distance_matrices(
            k_results["distances"],
            distance_analysis["environment_distances"],
            save_path=k_dir / "distance_matrices.png"
        )
        plt.close()

        for env_type in distance_analysis["environment_distances"].keys():
            for component in ["real", "imag", "combined"]:
                plot_distance_comparison_scatter(
                    k_results["distances"][f"distances_{component}"],
                    distance_analysis["environment_distances"][env_type],
                    title=f"{k_label}: {component.capitalize()} vs {env_type.capitalize()}",
                    save_path=k_dir / f"scatter_{component}_vs_{env_type}.png"
                )
                plt.close()

    print(f"\nVisualization report saved to {output_dir}")
