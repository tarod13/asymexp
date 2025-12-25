"""
Visualization utilities for hitting time analysis.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional, Tuple, List
from pathlib import Path


def visualize_hitting_time_on_grid(
    hitting_time_values: jnp.ndarray,
    center_state_idx: int,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    mode: str = 'target',  # 'target' or 'source'
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    show_colorbar: bool = True,
    wall_color: str = 'gray',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
) -> plt.Axes:
    """
    Visualize hitting times overlaid on the grid.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Transform values if log_scale is requested
    if log_scale:
        # Clip to 0 to prevent NaNs from small negative errors
        safe_values = np.maximum(hitting_time_values, 0)
        values_to_plot = np.log1p(safe_values)
    else:
        values_to_plot = hitting_time_values

    # Create grid for hitting time values
    ht_grid = np.full((grid_height, grid_width), np.nan)

    # Map canonical states to full grid positions
    for canonical_idx, value in enumerate(values_to_plot):
        full_state_idx = canonical_states[canonical_idx]
        y = int(full_state_idx) // grid_width
        x = int(full_state_idx) % grid_width
        ht_grid[y, x] = value

    # Set up colormap
    import matplotlib.cm as cm
    current_cmap = cm.get_cmap(cmap).copy()
    current_cmap.set_bad(color=wall_color)

    im = ax.imshow(
        ht_grid,
        cmap=current_cmap,
        origin='upper',
        interpolation='nearest',
        extent=[-0.5, grid_width - 0.5, grid_height - 0.5, -0.5],
        vmin=vmin,
        vmax=vmax
    )

    # Mark the center state
    center_full_idx = canonical_states[center_state_idx]
    c_y = int(center_full_idx) // grid_width
    c_x = int(center_full_idx) % grid_width
    
    if mode == 'target':
        # Star for Target
        ax.scatter(c_x, c_y, c='red', marker='*', s=250, edgecolors='white', linewidth=1.5, zorder=20, label='Target')
    else:
        # Circle for Source
        ax.scatter(c_x, c_y, c='cyan', marker='o', s=150, edgecolors='black', linewidth=1.5, zorder=20, label='Source')

    # Add colorbar
    if show_colorbar:
        label = 'Log(Expected Steps + 1)' if log_scale else 'Expected Steps'
        plt.colorbar(im, ax=ax, label=label)

    # Add grid lines
    for i in range(grid_height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for j in range(grid_width + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

    # Add portals/doors if provided
    if portals is not None and len(portals) > 0:
        rect_thickness = 0.15
        rect_width = 0.7
        
        for (source_idx, action), dest_idx in portals.items():
            source_y = source_idx // grid_width
            source_x = source_idx % grid_width
            margin = 0.02
            
            # Action mapping: 0=up, 1=right, 2=down, 3=left
            if action == 0:  # Up
                rect_x, rect_y = source_x - rect_width/2, source_y - 0.5 - rect_thickness/2
                rect_w, rect_h = rect_width, rect_thickness
                triangle = mpatches.Polygon([
                    (source_x, source_y - 0.5 - rect_thickness/2 + margin),
                    (source_x - (rect_thickness-2*margin)/2, source_y - 0.5 + rect_thickness/2 - margin),
                    (source_x + (rect_thickness-2*margin)/2, source_y - 0.5 + rect_thickness/2 - margin)
                ], facecolor='white', edgecolor='none', zorder=11)
            elif action == 1:  # Right
                rect_x, rect_y = source_x + 0.5 - rect_thickness/2, source_y - rect_width/2
                rect_w, rect_h = rect_thickness, rect_width
                triangle = mpatches.Polygon([
                    (source_x + 0.5 + rect_thickness/2 - margin, source_y),
                    (source_x + 0.5 - rect_thickness/2 + margin, source_y - (rect_thickness-2*margin)/2),
                    (source_x + 0.5 - rect_thickness/2 + margin, source_y + (rect_thickness-2*margin)/2)
                ], facecolor='white', edgecolor='none', zorder=11)
            elif action == 2:  # Down
                rect_x, rect_y = source_x - rect_width/2, source_y + 0.5 - rect_thickness/2
                rect_w, rect_h = rect_width, rect_thickness
                triangle = mpatches.Polygon([
                    (source_x, source_y + 0.5 + rect_thickness/2 - margin),
                    (source_x - (rect_thickness-2*margin)/2, source_y + 0.5 - rect_thickness/2 + margin),
                    (source_x + (rect_thickness-2*margin)/2, source_y + 0.5 - rect_thickness/2 + margin)
                ], facecolor='white', edgecolor='none', zorder=11)
            else:  # Left
                rect_x, rect_y = source_x - 0.5 - rect_thickness/2, source_y - rect_width/2
                rect_w, rect_h = rect_thickness, rect_width
                triangle = mpatches.Polygon([
                    (source_x - 0.5 - rect_thickness/2 + margin, source_y),
                    (source_x - 0.5 + rect_thickness/2 - margin, source_y - (rect_thickness-2*margin)/2),
                    (source_x - 0.5 + rect_thickness/2 - margin, source_y + (rect_thickness-2*margin)/2)
                ], facecolor='white', edgecolor='none', zorder=11)

            ax.add_patch(mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=0, edgecolor='none', facecolor='black', zorder=10))
            ax.add_patch(triangle)

    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(grid_height - 0.5, -0.5)
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title, fontsize=10)
    
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def visualize_source_vs_target_hitting_times(
    state_indices: List[int],
    hitting_time_matrix: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    ncols: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    wall_color: str = 'gray',
    save_path: Optional[str] = None,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Visualize hitting times for states acting as Targets (columns) vs Sources (rows).
    """
    num_states = len(state_indices)
    
    # Calculate grid dimensions
    num_logical_rows = (num_states + ncols - 1) // ncols
    nrows = num_logical_rows * 2

    if figsize is None:
        figsize = (ncols * 3, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Reshape axes to be 2D array [nrows, ncols] even if single row/col
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Compute global color scales
    # Clip to 0 for safety in visualization scale
    safe_matrix = np.maximum(hitting_time_matrix, 0)
    
    if log_scale:
        max_val = np.log1p(np.max(safe_matrix))
    else:
        max_val = np.max(safe_matrix)
    
    if np.isnan(max_val):
        max_val = np.nanmax(np.log1p(safe_matrix) if log_scale else safe_matrix)

    vmin = 0
    vmax = max_val

    for idx, state_idx in enumerate(state_indices):
        r_logical = idx // ncols
        c = idx % ncols
        
        # Row for Target View (Even rows: 0, 2, 4...)
        ax_target = axes[r_logical * 2, c]
        
        # Row for Source View (Odd rows: 1, 3, 5...)
        ax_source = axes[r_logical * 2 + 1, c]

        # 1. Target View: Column of H (Time TO state_idx)
        times_to_state = hitting_time_matrix[:, state_idx]
        
        visualize_hitting_time_on_grid(
            hitting_time_values=times_to_state,
            center_state_idx=state_idx,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            mode='target',
            portals=portals,
            title=f'State {state_idx}\n(Target View: Times TO here)',
            ax=ax_target,
            cmap='viridis',
            show_colorbar=False,
            wall_color=wall_color,
            vmin=vmin,
            vmax=vmax,
            log_scale=log_scale
        )

        # 2. Source View: Row of H (Time FROM state_idx)
        times_from_state = hitting_time_matrix[state_idx, :]
        
        visualize_hitting_time_on_grid(
            hitting_time_values=times_from_state,
            center_state_idx=state_idx,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            mode='source',
            portals=portals,
            title=f'State {state_idx}\n(Source View: Times FROM here)',
            ax=ax_source,
            cmap='viridis',
            show_colorbar=False,
            wall_color=wall_color,
            vmin=vmin,
            vmax=vmax,
            log_scale=log_scale
        )

    # Hide unused axes
    for idx in range(num_states, num_logical_rows * ncols):
        r_logical = idx // ncols
        c = idx % ncols
        axes[r_logical * 2, c].axis('off')
        axes[r_logical * 2 + 1, c].axis('off')

    # Apply tight layout first to organize the subplots
    plt.tight_layout()
    
    # Adjust the right margin to create space for the colorbar
    fig.subplots_adjust(right=0.9)
    
    # Add a global colorbar on the right side
    # Coordinates are [left, bottom, width, height] in figure relative coords
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    
    cbar_label = 'Log(Expected Steps + 1)' if log_scale else 'Expected Steps'
    cbar.set_label(cbar_label, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved source-vs-target hitting times visualization to {save_path}")

    return fig


def create_hitting_time_visualization_report(
    hitting_time_matrix: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    output_dir: str = "exp_complex_basis/results/visualizations",
    num_targets: int = 6,
    target_indices: Optional[List[int]] = None,
    ncols: int = 6,
    wall_color: str = 'gray',
    log_scale: bool = False,
):
    """
    Create a complete visualization report for hitting times.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating hitting time visualizations...")

    num_states = hitting_time_matrix.shape[0]

    if target_indices is None:
        target_indices = np.linspace(0, num_states - 1, num_targets, dtype=int).tolist()
    
    print(f"  Visualizing source/target comparisons for {len(target_indices)} states...")
    
    # Use the new source vs target visualization
    filename = f"hitting_times_asymmetry{'_log' if log_scale else ''}.png"
    
    visualize_source_vs_target_hitting_times(
        state_indices=target_indices,
        hitting_time_matrix=hitting_time_matrix,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        ncols=ncols,
        wall_color=wall_color,
        save_path=output_path / filename,
        log_scale=log_scale
    )
    plt.close()

    print(f"\nHitting time visualization report saved to {output_dir}")