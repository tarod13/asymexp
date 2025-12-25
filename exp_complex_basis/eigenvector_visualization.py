"""
Visualization utilities for eigenvector analysis.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional, Tuple, List
from pathlib import Path


def visualize_eigenvector_on_grid(
    eigenvector_idx: int,
    eigenvector_values: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'RdBu_r',
    show_colorbar: bool = True,
    wall_color: str = 'gray',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Axes:
    """
    Visualize a single eigenvector's values overlaid on the grid.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Create grid for eigenvector values
    eigenvector_grid = np.full((grid_height, grid_width), np.nan)

    # Map canonical states to full grid positions
    for canonical_idx, value in enumerate(eigenvector_values):
        full_state_idx = canonical_states[canonical_idx]
        y = int(full_state_idx) // grid_width
        x = int(full_state_idx) % grid_width
        eigenvector_grid[y, x] = value

    # Plot eigenvector values with grid alignment
    # Set up colormap to handle NaN values (walls) with the specified wall_color
    import matplotlib.cm as cm

    current_cmap = cm.get_cmap(cmap).copy()
    current_cmap.set_bad(color=wall_color)

    im = ax.imshow(
        eigenvector_grid,
        cmap=current_cmap,
        origin='upper',
        interpolation='nearest',
        extent=[-0.5, grid_width - 0.5, grid_height - 0.5, -0.5],
        vmin=vmin,
        vmax=vmax
    )

    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Eigenvector Value', fraction=0.046, pad=0.04)

    # Add grid lines
    for i in range(grid_height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for j in range(grid_width + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

    # Add portals/doors if provided
    if portals is not None and len(portals) > 0:
        # Door rectangle dimensions
        rect_thickness = 0.15
        rect_width = 0.7

        for (source_idx, action), dest_idx in portals.items():
            source_y = source_idx // grid_width
            source_x = source_idx % grid_width
            margin = 0.02

            if action == 0:  # Up
                rect_x, rect_y = source_x - rect_width / 2, source_y - 0.5 - rect_thickness / 2
                rect_w, rect_h = rect_width, rect_thickness
                triangle = mpatches.Polygon([
                    (source_x, source_y - 0.5 - rect_thickness / 2 + margin),
                    (source_x - (rect_thickness - 2 * margin) / 2, source_y - 0.5 + rect_thickness / 2 - margin),
                    (source_x + (rect_thickness - 2 * margin) / 2, source_y - 0.5 + rect_thickness / 2 - margin)
                ], facecolor='white', edgecolor='none', zorder=11)
            elif action == 1:  # Right
                rect_x, rect_y = source_x + 0.5 - rect_thickness / 2, source_y - rect_width / 2
                rect_w, rect_h = rect_thickness, rect_width
                triangle = mpatches.Polygon([
                    (source_x + 0.5 + rect_thickness / 2 - margin, source_y),
                    (source_x + 0.5 - rect_thickness / 2 + margin, source_y - (rect_thickness - 2 * margin) / 2),
                    (source_x + 0.5 - rect_thickness / 2 + margin, source_y + (rect_thickness - 2 * margin) / 2)
                ], facecolor='white', edgecolor='none', zorder=11)
            elif action == 2:  # Down
                rect_x, rect_y = source_x - rect_width / 2, source_y + 0.5 - rect_thickness / 2
                rect_w, rect_h = rect_width, rect_thickness
                triangle = mpatches.Polygon([
                    (source_x, source_y + 0.5 + rect_thickness / 2 - margin),
                    (source_x - (rect_thickness - 2 * margin) / 2, source_y + 0.5 - rect_thickness / 2 + margin),
                    (source_x + (rect_thickness - 2 * margin) / 2, source_y + 0.5 - rect_thickness / 2 + margin)
                ], facecolor='white', edgecolor='none', zorder=11)
            else:  # Left
                rect_x, rect_y = source_x - 0.5 - rect_thickness / 2, source_y - rect_width / 2
                rect_w, rect_h = rect_thickness, rect_width
                triangle = mpatches.Polygon([
                    (source_x - 0.5 - rect_thickness / 2 + margin, source_y),
                    (source_x - 0.5 + rect_thickness / 2 - margin, source_y - (rect_thickness - 2 * margin) / 2),
                    (source_x - 0.5 + rect_thickness / 2 - margin, source_y + (rect_thickness - 2 * margin) / 2)
                ], facecolor='white', edgecolor='none', zorder=11)

            rect = mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=0, edgecolor='none', facecolor='black', zorder=10)
            ax.add_patch(rect)
            ax.add_patch(triangle)

    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(grid_height - 0.5, -0.5)
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))

    return ax


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


def visualize_multiple_eigenvectors(
    eigenvector_indices: List[int],
    eigendecomposition: Dict[str, jnp.ndarray],
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    eigenvector_type: str = 'right',
    component: str = 'real',
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    wall_color: str = 'gray',
    save_path: Optional[str] = None,
    shared_colorbar: bool = True
) -> plt.Figure:
    """
    Visualize multiple eigenvectors side by side.
    
    Args:
        shared_colorbar: If True, use a single global color scale and one colorbar.
                         If False, each plot is scaled independently.
    """
    num_eigenvectors = len(eigenvector_indices)

    if ncols is None:
        ncols = min(5, num_eigenvectors)
    if nrows is None:
        nrows = (num_eigenvectors + ncols - 1) // ncols

    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    if eigenvector_type == 'right':
        if component == 'real':
            eigenvector_matrix = eigendecomposition['right_eigenvectors_real']
        else:
            eigenvector_matrix = eigendecomposition['right_eigenvectors_imag']
    else:
        if component == 'real':
            eigenvector_matrix = eigendecomposition['left_eigenvectors_real']
        else:
            eigenvector_matrix = eigendecomposition['left_eigenvectors_imag']

    vmin = None
    vmax = None
    if shared_colorbar:
        all_values = np.concatenate([eigenvector_matrix[:, idx] for idx in eigenvector_indices])
        vmin = np.min(all_values)
        vmax = np.max(all_values)

    for plot_idx, eigenvec_idx in enumerate(eigenvector_indices):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row, col]

        eigenvector_values = eigenvector_matrix[:, eigenvec_idx]
        eigenvalue = eigendecomposition['eigenvalues'][eigenvec_idx]

        visualize_eigenvector_on_grid(
            eigenvector_idx=eigenvec_idx,
            eigenvector_values=np.array(eigenvector_values),
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            title=f'{eigenvector_type.capitalize()} Eigvec {eigenvec_idx} ({component})\nλ = {np.abs(eigenvalue):.3f}',
            ax=ax,
            cmap='RdBu_r',
            show_colorbar=not shared_colorbar,
            wall_color=wall_color,
            vmin=vmin,
            vmax=vmax
        )

    for plot_idx in range(num_eigenvectors, nrows * ncols):
        row = plot_idx // ncols
        col = plot_idx % ncols
        axes[row, col].axis('off')

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
        print(f"Saved multiple eigenvectors visualization to {save_path}")

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