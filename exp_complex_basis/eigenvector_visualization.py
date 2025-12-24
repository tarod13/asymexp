"""
Visualization utilities for eigenvector analysis.

This module provides functions to visualize eigenvector values on grid environments,
showing both real and imaginary components for left and right eigenvectors.
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
) -> plt.Axes:
    """
    Visualize a single eigenvector's values overlaid on the grid.

    Args:
        eigenvector_idx: Index of the eigenvector to visualize
        eigenvector_values: Values of the eigenvector [num_states]
        canonical_states: Mapping from canonical indices to full state indices [num_states]
        grid_width: Width of the grid
        grid_height: Height of the grid
        portals: Optional dict mapping (source_state_idx, action) -> dest_state_idx
        title: Optional title for the plot
        ax: Optional axes to plot on
        cmap: Colormap to use
        show_colorbar: Whether to show the colorbar
        wall_color: Color for wall/obstacle cells (default: 'gray')

    Returns:
        Matplotlib axes object
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
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as cm

    current_cmap = cm.get_cmap(cmap).copy()
    current_cmap.set_bad(color=wall_color)

    im = ax.imshow(
        eigenvector_grid,
        cmap=current_cmap,
        origin='upper',
        interpolation='nearest',
        extent=[-0.5, grid_width - 0.5, grid_height - 0.5, -0.5]
    )

    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Eigenvector Value')

    # Add grid lines
    for i in range(grid_height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for j in range(grid_width + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

    # Add portals/doors if provided
    if portals is not None and len(portals) > 0:
        # Door rectangle dimensions
        rect_thickness = 0.15  # Thin dimension
        rect_width = 0.7  # Wide dimension (parallel to action)

        for (source_idx, action), dest_idx in portals.items():
            # Convert state indices to grid coordinates
            source_y = source_idx // grid_width
            source_x = source_idx % grid_width

            # Action mapping: 0=up, 1=right, 2=down, 3=left
            # Determine rectangle position and dimensions based on action
            if action == 0:  # Up - rectangle on top edge
                rect_x = source_x - rect_width / 2
                rect_y = source_y - 0.5 - rect_thickness / 2
                rect_w = rect_width
                rect_h = rect_thickness
                arrow_start = (source_x, source_y - 0.5 + rect_thickness / 4)
                arrow_end = (source_x, source_y - 0.5 - rect_thickness / 4)
            elif action == 1:  # Right - rectangle on right edge
                rect_x = source_x + 0.5 - rect_thickness / 2
                rect_y = source_y - rect_width / 2
                rect_w = rect_thickness
                rect_h = rect_width
                arrow_start = (source_x + 0.5 - rect_thickness / 4, source_y)
                arrow_end = (source_x + 0.5 + rect_thickness / 4, source_y)
            elif action == 2:  # Down - rectangle on bottom edge
                rect_x = source_x - rect_width / 2
                rect_y = source_y + 0.5 - rect_thickness / 2
                rect_w = rect_width
                rect_h = rect_thickness
                arrow_start = (source_x, source_y + 0.5 - rect_thickness / 4)
                arrow_end = (source_x, source_y + 0.5 + rect_thickness / 4)
            else:  # Left - rectangle on left edge
                rect_x = source_x - 0.5 - rect_thickness / 2
                rect_y = source_y - rect_width / 2
                rect_w = rect_thickness
                rect_h = rect_width
                arrow_start = (source_x - 0.5 + rect_thickness / 4, source_y)
                arrow_end = (source_x - 0.5 - rect_thickness / 4, source_y)

            # Draw black rectangle
            rect = mpatches.Rectangle(
                (rect_x, rect_y), rect_w, rect_h,
                linewidth=0,
                edgecolor='none',
                facecolor='black',
                zorder=10
            )
            ax.add_patch(rect)

            # Draw white arrow inside rectangle
            ax.annotate('',
                       xy=arrow_end,
                       xytext=arrow_start,
                       arrowprops=dict(
                           arrowstyle='->',
                           lw=2,
                           color='white',
                           shrinkA=0,
                           shrinkB=0,
                           zorder=11
                       ))

    # Set limits and labels
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(grid_height - 0.5, -0.5)
    ax.set_aspect('equal')

    # Set title
    if title is not None:
        ax.set_title(title, fontsize=12)

    # Set tick labels
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

    Args:
        eigenvector_idx: Index of the eigenvector to visualize
        eigendecomposition: Dictionary containing eigendecomposition results
        canonical_states: Mapping from canonical to full state indices
        grid_width: Width of the grid
        grid_height: Height of the grid
        portals: Optional portal dictionary
        eigenvector_type: 'right' or 'left'
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Get eigenvector components
    if eigenvector_type == 'right':
        real_values = eigendecomposition['right_eigenvectors_real'][:, eigenvector_idx]
        imag_values = eigendecomposition['right_eigenvectors_imag'][:, eigenvector_idx]
    else:
        real_values = eigendecomposition['left_eigenvectors_real'][:, eigenvector_idx]
        imag_values = eigendecomposition['left_eigenvectors_imag'][:, eigenvector_idx]

    # Get eigenvalue info
    eigenvalue = eigendecomposition['eigenvalues'][eigenvector_idx]
    eigenvalue_real = eigendecomposition['eigenvalues_real'][eigenvector_idx]
    eigenvalue_imag = eigendecomposition['eigenvalues_imag'][eigenvector_idx]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot real component
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

    # Plot imaginary component
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
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize multiple eigenvectors side by side.

    Args:
        eigenvector_indices: List of eigenvector indices to visualize
        eigendecomposition: Dictionary containing eigendecomposition results
        canonical_states: Mapping from canonical to full state indices
        grid_width: Width of the grid
        grid_height: Height of the grid
        portals: Optional portal dictionary
        eigenvector_type: 'right' or 'left'
        component: 'real' or 'imag'
        nrows: Number of rows (if None, computed automatically)
        ncols: Number of columns (if None, computed automatically)
        figsize: Figure size (if None, computed based on nrows and ncols)
        wall_color: Color for wall/obstacle cells (default: 'gray')
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    num_eigenvectors = len(eigenvector_indices)

    # Compute nrows and ncols if not provided
    if ncols is None:
        ncols = min(5, num_eigenvectors)
    if nrows is None:
        nrows = (num_eigenvectors + ncols - 1) // ncols

    # Compute figsize if not provided
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes if needed
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Get eigenvector values
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

    # Plot each eigenvector
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
            show_colorbar=False,
            wall_color=wall_color
        )

    # Hide unused subplots
    for plot_idx in range(num_eigenvectors, nrows * ncols):
        row = plot_idx // ncols
        col = plot_idx % ncols
        axes[row, col].axis('off')

    plt.tight_layout()

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
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize both left and right eigenvectors in the same figure.

    Each row shows a specific eigenvector index, with left eigenvector
    in the left column and right eigenvector in the right column.

    Args:
        eigenvector_indices: List of eigenvector indices to visualize
        eigendecomposition: Dictionary containing eigendecomposition results
        canonical_states: Mapping from canonical to full state indices
        grid_width: Width of the grid
        grid_height: Height of the grid
        portals: Optional portal dictionary
        component: 'real' or 'imag'
        nrows: Number of rows (if None, set to len(eigenvector_indices))
        ncols: Number of columns (if None, set to 2 for left/right)
        figsize: Figure size (if None, computed based on nrows and ncols)
        wall_color: Color for wall/obstacle cells (default: 'gray')
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    num_eigenvectors = len(eigenvector_indices)

    # Set default layout: each row is one eigenvector, 2 columns for left/right
    if nrows is None:
        nrows = num_eigenvectors
    if ncols is None:
        ncols = 2  # Left and Right

    # Compute figsize if not provided
    if figsize is None:
        figsize = (ncols * 5, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Get eigenvector matrices
    if component == 'real':
        left_eigenvector_matrix = eigendecomposition['left_eigenvectors_real']
        right_eigenvector_matrix = eigendecomposition['right_eigenvectors_real']
    else:
        left_eigenvector_matrix = eigendecomposition['left_eigenvectors_imag']
        right_eigenvector_matrix = eigendecomposition['right_eigenvectors_imag']

    # Plot each eigenvector
    for row_idx, eigenvec_idx in enumerate(eigenvector_indices):
        if row_idx >= nrows:
            break

        eigenvalue = eigendecomposition['eigenvalues'][eigenvec_idx]

        # Plot left eigenvector
        left_values = left_eigenvector_matrix[:, eigenvec_idx]
        visualize_eigenvector_on_grid(
            eigenvector_idx=eigenvec_idx,
            eigenvector_values=np.array(left_values),
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            title=f'Left Eigvec {eigenvec_idx} ({component})\nλ = {np.abs(eigenvalue):.3f}',
            ax=axes[row_idx, 0],
            cmap='RdBu_r',
            show_colorbar=False,
            wall_color=wall_color
        )

        # Plot right eigenvector
        right_values = right_eigenvector_matrix[:, eigenvec_idx]
        visualize_eigenvector_on_grid(
            eigenvector_idx=eigenvec_idx,
            eigenvector_values=np.array(right_values),
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            title=f'Right Eigvec {eigenvec_idx} ({component})\nλ = {np.abs(eigenvalue):.3f}',
            ax=axes[row_idx, 1],
            cmap='RdBu_r',
            show_colorbar=False,
            wall_color=wall_color
        )

    # Hide unused subplots
    for row_idx in range(num_eigenvectors, nrows):
        for col_idx in range(ncols):
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout()

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

    Args:
        eigendecomposition: Dictionary with eigendecomposition results
        canonical_states: Mapping from canonical to full state indices
        grid_width: Width of the grid
        grid_height: Height of the grid
        portals: Optional portal dictionary
        output_dir: Directory to save visualizations
        num_eigenvectors: Number of top eigenvectors to visualize
        nrows: Number of rows for grid layout (None = auto)
        ncols: Number of columns for grid layout (None = auto)
        wall_color: Color for wall/obstacle cells (default: 'gray')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating eigenvector visualizations...")

    # Determine how many eigenvectors we have
    num_available = eigendecomposition['eigenvalues'].shape[0]
    num_to_visualize = min(num_eigenvectors, num_available)

    # 1. Visualize top eigenvectors - right, real component
    print(f"  Visualizing top {num_to_visualize} right eigenvectors (real)...")
    visualize_multiple_eigenvectors(
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
        wall_color=wall_color,
        save_path=output_path / "right_eigenvectors_real.png"
    )
    plt.close()

    # 2. Visualize top eigenvectors - right, imaginary component
    print(f"  Visualizing top {num_to_visualize} right eigenvectors (imaginary)...")
    visualize_multiple_eigenvectors(
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
        wall_color=wall_color,
        save_path=output_path / "right_eigenvectors_imag.png"
    )
    plt.close()

    # 3. Visualize top eigenvectors - left, real component
    print(f"  Visualizing top {num_to_visualize} left eigenvectors (real)...")
    visualize_multiple_eigenvectors(
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
        wall_color=wall_color,
        save_path=output_path / "left_eigenvectors_real.png"
    )
    plt.close()

    # 4. Visualize top eigenvectors - left, imaginary component
    print(f"  Visualizing top {num_to_visualize} left eigenvectors (imaginary)...")
    visualize_multiple_eigenvectors(
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
        wall_color=wall_color,
        save_path=output_path / "left_eigenvectors_imag.png"
    )
    plt.close()

    # 5. Combined left-right visualizations (real component)
    print(f"  Visualizing left and right eigenvectors side-by-side (real)...")
    visualize_left_right_eigenvectors(
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        component='real',
        wall_color=wall_color,
        save_path=output_path / "left_right_eigenvectors_real.png"
    )
    plt.close()

    # 6. Combined left-right visualizations (imaginary component)
    print(f"  Visualizing left and right eigenvectors side-by-side (imaginary)...")
    visualize_left_right_eigenvectors(
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        component='imag',
        wall_color=wall_color,
        save_path=output_path / "left_right_eigenvectors_imag.png"
    )
    plt.close()

    # 7. Detailed views for first few eigenvectors (both components)
    print("  Creating detailed views for specific eigenvectors...")
    for i in range(min(3, num_to_visualize)):
        # Right eigenvector
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

        # Left eigenvector
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
