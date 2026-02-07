"""
Visualization utilities for eigenvector analysis.

Core visualization functions (visualize_eigenvector_on_grid, visualize_multiple_eigenvectors)
have been moved to src/utils/plotting.py. They are re-exported here for backward compatibility.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from src.utils.plotting import (
    visualize_eigenvector_on_grid,
    visualize_multiple_eigenvectors,
)


def visualize_eigenvector_components(
    eigenvector_idx: int,
    eigendecomposition: Dict[str, jnp.ndarray],
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    eigenvector_type: str = 'right',
    figsize: Tuple[int, int] = (16, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize both real and imaginary components of an eigenvector.
    """
    if eigenvector_type == 'right':
        real_values = eigendecomposition['right_eigenvectors_real'][:, eigenvector_idx]
        imag_values = eigendecomposition['right_eigenvectors_imag'][:, eigenvector_idx]
    else:
        real_values = eigendecomposition['left_eigenvectors_real'][:, eigenvector_idx]
        imag_values = eigendecomposition['left_eigenvectors_imag'][:, eigenvector_idx]

    eigenvalue = eigendecomposition['eigenvalues'][eigenvector_idx]
    eigenvalue_real = eigendecomposition['eigenvalues_real'][eigenvector_idx]
    eigenvalue_imag = eigendecomposition['eigenvalues_imag'][eigenvector_idx]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    visualize_eigenvector_on_grid(
        eigenvector_idx=eigenvector_idx,
        eigenvector_values=np.array(real_values),
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        title=f'{eigenvector_type.capitalize()} Eigenvector {eigenvector_idx} - Real Component\nλ = {eigenvalue_real:.3f} + {eigenvalue_imag:.3f}i',
        ax=axes[0],
        cmap='RdBu_r',
        show_colorbar=True
    )

    visualize_eigenvector_on_grid(
        eigenvector_idx=eigenvector_idx,
        eigenvector_values=np.array(imag_values),
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        title=f'{eigenvector_type.capitalize()} Eigenvector {eigenvector_idx} - Imaginary Component\n|λ| = {np.abs(eigenvalue):.3f}',
        ax=axes[1],
        cmap='RdBu_r',
        show_colorbar=True
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvector visualization to {save_path}")

    return fig


def visualize_left_right_eigenvectors(
    eigenvector_indices: List[int],
    eigendecomposition: Dict[str, jnp.ndarray],
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    component: str = 'real',
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    wall_color: str = 'gray',
    save_path: Optional[str] = None,
    shared_colorbar: bool = True
) -> plt.Figure:
    """
    Visualize both left and right eigenvectors in the same figure.
    """
    num_eigenvectors = len(eigenvector_indices)

    if ncols is None:
        ncols = min(5, num_eigenvectors)

    num_groups = (num_eigenvectors + ncols - 1) // ncols
    if nrows is None:
        nrows = num_groups * 2

    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    if component == 'real':
        left_eigenvector_matrix = eigendecomposition['left_eigenvectors_real']
        right_eigenvector_matrix = eigendecomposition['right_eigenvectors_real']
    else:
        left_eigenvector_matrix = eigendecomposition['left_eigenvectors_imag']
        right_eigenvector_matrix = eigendecomposition['right_eigenvectors_imag']

    vmin = None
    vmax = None
    if shared_colorbar:
        all_left_values = [left_eigenvector_matrix[:, idx] for idx in eigenvector_indices]
        all_right_values = [right_eigenvector_matrix[:, idx] for idx in eigenvector_indices]
        all_values = np.concatenate([np.concatenate(all_left_values), np.concatenate(all_right_values)])
        vmin = np.min(all_values)
        vmax = np.max(all_values)

    for idx, eigenvec_idx in enumerate(eigenvector_indices):
        group_idx = idx // ncols
        col_idx = idx % ncols

        left_row = group_idx * 2
        right_row = group_idx * 2 + 1

        eigenvalue = eigendecomposition['eigenvalues'][eigenvec_idx]

        left_values = left_eigenvector_matrix[:, eigenvec_idx]
        right_values = right_eigenvector_matrix[:, eigenvec_idx]

        visualize_eigenvector_on_grid(
            eigenvector_idx=eigenvec_idx,
            eigenvector_values=np.array(left_values),
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            title=f'Left Eigvec {eigenvec_idx} ({component})\nλ = {np.abs(eigenvalue):.3f}',
            ax=axes[left_row, col_idx],
            cmap='RdBu_r',
            show_colorbar=not shared_colorbar,
            wall_color=wall_color,
            vmin=vmin,
            vmax=vmax
        )

        visualize_eigenvector_on_grid(
            eigenvector_idx=eigenvec_idx,
            eigenvector_values=np.array(right_values),
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            title=f'Right Eigvec {eigenvec_idx} ({component})\nλ = {np.abs(eigenvalue):.3f}',
            ax=axes[right_row, col_idx],
            cmap='RdBu_r',
            show_colorbar=not shared_colorbar,
            wall_color=wall_color,
            vmin=vmin,
            vmax=vmax
        )

    for group_idx in range(num_groups):
        start_idx = group_idx * ncols
        end_idx = min(start_idx + ncols, num_eigenvectors)
        num_in_group = end_idx - start_idx

        left_row = group_idx * 2
        right_row = group_idx * 2 + 1

        for col_idx in range(num_in_group, ncols):
            axes[left_row, col_idx].axis('off')
            axes[right_row, col_idx].axis('off')

    plt.tight_layout()

    if shared_colorbar:
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Eigenvector Value', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved left-right eigenvectors visualization to {save_path}")

    return fig


def create_eigenvector_visualization_report(
    eigendecomposition: Dict[str, jnp.ndarray],
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    output_dir: str = "exp_complex_basis/results/visualizations",
    num_eigenvectors: int = 6,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    wall_color: str = 'gray'
):
    """
    Create a complete visualization report for eigenvector analysis.
    Generates both shared-scale and independent-scale plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating eigenvector visualizations...")

    num_available = eigendecomposition['eigenvalues'].shape[0]
    num_to_visualize = min(num_eigenvectors, num_available)
    
    # Helper to generate both shared and independent plots
    def generate_versions(func, name_base, **kwargs):
        # 1. Shared
        func(
            save_path=output_path / f"{name_base}_shared_scale.png",
            shared_colorbar=True,
            **kwargs
        )
        plt.close()
        # 2. Independent
        func(
            save_path=output_path / f"{name_base}_independent_scale.png",
            shared_colorbar=False,
            **kwargs
        )
        plt.close()

    # 1. Visualize top eigenvectors - right, real component
    print(f"  Visualizing top {num_to_visualize} right eigenvectors (real)...")
    generate_versions(
        visualize_multiple_eigenvectors,
        "right_eigenvectors_real",
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        eigenvector_type='right',
        component='real',
        nrows=nrows,
        ncols=ncols,
        wall_color=wall_color
    )

    # 2. Visualize top eigenvectors - right, imaginary component
    print(f"  Visualizing top {num_to_visualize} right eigenvectors (imaginary)...")
    generate_versions(
        visualize_multiple_eigenvectors,
        "right_eigenvectors_imag",
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        eigenvector_type='right',
        component='imag',
        nrows=nrows,
        ncols=ncols,
        wall_color=wall_color
    )

    # 3. Visualize top eigenvectors - left, real component
    print(f"  Visualizing top {num_to_visualize} left eigenvectors (real)...")
    generate_versions(
        visualize_multiple_eigenvectors,
        "left_eigenvectors_real",
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        eigenvector_type='left',
        component='real',
        nrows=nrows,
        ncols=ncols,
        wall_color=wall_color
    )

    # 4. Visualize top eigenvectors - left, imaginary component
    print(f"  Visualizing top {num_to_visualize} left eigenvectors (imaginary)...")
    generate_versions(
        visualize_multiple_eigenvectors,
        "left_eigenvectors_imag",
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        eigenvector_type='left',
        component='imag',
        nrows=nrows,
        ncols=ncols,
        wall_color=wall_color
    )

    # 5. Combined left-right visualizations (real component)
    print(f"  Visualizing left and right eigenvectors side-by-side (real)...")
    generate_versions(
        visualize_left_right_eigenvectors,
        "left_right_eigenvectors_real",
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        component='real',
        wall_color=wall_color
    )

    # 6. Combined left-right visualizations (imaginary component)
    print(f"  Visualizing left and right eigenvectors side-by-side (imaginary)...")
    generate_versions(
        visualize_left_right_eigenvectors,
        "left_right_eigenvectors_imag",
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        component='imag',
        wall_color=wall_color
    )

    # 7. Detailed views for first few eigenvectors (both components) - Only independent scale needed
    print("  Creating detailed views for specific eigenvectors...")
    for i in range(min(3, num_to_visualize)):
        visualize_eigenvector_components(
            eigenvector_idx=i,
            eigendecomposition=eigendecomposition,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            eigenvector_type='right',
            save_path=output_path / f"right_eigenvector_{i}_detailed.png"
        )
        plt.close()

        visualize_eigenvector_components(
            eigenvector_idx=i,
            eigendecomposition=eigendecomposition,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            eigenvector_type='left',
            save_path=output_path / f"left_eigenvector_{i}_detailed.png"
        )
        plt.close()

    print(f"\nEigenvector visualization report saved to {output_dir}")