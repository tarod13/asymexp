import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Optional, Tuple, List



def plot_learning_curves_one(metrics_history: list, save_path: str):
    """
    Plot comprehensive learning curves for single-eigenvector training.

    Visualizes losses, Lyapunov functions, eigenvalue estimates, and other metrics.

    Args:
        metrics_history: List of metric dictionaries from training
        save_path: Path to save the plot
    """
    if not metrics_history:
        print("Warning: metrics_history is empty, skipping learning curves plot")
        return

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Training Metrics (CLF-Based Complex Eigenvector Learning)', fontsize=16)

    steps = [m['gradient_step'] for m in metrics_history]

    # Row 1: Main losses
    # Plot 1: Total loss
    if 'total_loss' in metrics_history[0]:
        axes[0, 0].plot(steps, [m['total_loss'] for m in metrics_history])
        axes[0, 0].set_xlabel('Gradient Step')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Graph loss
    if 'graph_loss' in metrics_history[0]:
        axes[0, 1].plot(steps, [m['graph_loss'] for m in metrics_history], label='Graph')
        if 'clf_loss' in metrics_history[0]:
            axes[0, 1].plot(steps, [m['clf_loss'] for m in metrics_history], label='CLF')
        axes[0, 1].set_xlabel('Gradient Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Graph & CLF Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Gradient norm
    if 'grad_norm' in metrics_history[0]:
        axes[0, 2].plot(steps, [m['grad_norm'] for m in metrics_history])
        axes[0, 2].set_xlabel('Gradient Step')
        axes[0, 2].set_ylabel('Gradient Norm')
        axes[0, 2].set_title('Gradient Norm')
        axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Lyapunov functions and norms
    # Plot 4: Lyapunov functions
    lyapunov_keys = ['V_x_norm', 'V_y_norm', 'V_xy_phase']
    lyapunov_labels = ['V_x (norm)', 'V_y (norm)', 'V_xy (phase)']
    axes[1, 0].set_title('Lyapunov Functions')
    for key, label in zip(lyapunov_keys, lyapunov_labels):
        if key in metrics_history[0]:
            axes[1, 0].plot(steps, [m[key] for m in metrics_history], label=label)
    axes[1, 0].set_xlabel('Gradient Step')
    axes[1, 0].set_ylabel('V')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Plot 5: Norms
    if 'norm_x_sq' in metrics_history[0]:
        axes[1, 1].plot(steps, [m['norm_x_sq'] for m in metrics_history], label='||x||²')
    if 'norm_y_sq' in metrics_history[0]:
        axes[1, 1].plot(steps, [m['norm_y_sq'] for m in metrics_history], label='||y||²')
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target=1')
    axes[1, 1].set_xlabel('Gradient Step')
    axes[1, 1].set_ylabel('Squared Norm')
    axes[1, 1].set_title('Squared Norms (Target=1)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Barrier (CLF control effort)
    if 'barrier' in metrics_history[0]:
        barrier_values = [float(np.mean(m['barrier'])) if hasattr(m['barrier'], '__len__') else float(m['barrier'])
                         for m in metrics_history]
        axes[1, 2].plot(steps, barrier_values)
        axes[1, 2].set_xlabel('Gradient Step')
        axes[1, 2].set_ylabel('Barrier (avg)')
        axes[1, 2].set_title('CLF Barrier Term')
        axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Eigenvalue estimates and cosine similarities
    # Plot 7: Eigenvalue estimates (real and imaginary)
    axes[2, 0].set_title('Eigenvalue Estimates')
    if 'lambda_x_real' in metrics_history[0]:
        axes[2, 0].plot(steps, [m['lambda_x_real'] for m in metrics_history], label='λ_x (real)', linestyle='-')
        axes[2, 0].plot(steps, [m['lambda_x_imag'] for m in metrics_history], label='λ_x (imag)', linestyle='--')
    if 'lambda_y_real' in metrics_history[0]:
        axes[2, 0].plot(steps, [m['lambda_y_real'] for m in metrics_history], label='λ_y (real)', linestyle='-', alpha=0.7)
        axes[2, 0].plot(steps, [m['lambda_y_imag'] for m in metrics_history], label='λ_y (imag)', linestyle='--', alpha=0.7)
    axes[2, 0].set_xlabel('Gradient Step')
    axes[2, 0].set_ylabel('Eigenvalue')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Cosine similarities
    axes[2, 1].set_title('Cosine Similarities')
    if 'left_cosine_sim_avg' in metrics_history[0]:
        axes[2, 1].plot(steps, [m['left_cosine_sim_avg'] for m in metrics_history], label='Left (ψ)', linewidth=2)
    if 'right_cosine_sim_avg' in metrics_history[0]:
        axes[2, 1].plot(steps, [m['right_cosine_sim_avg'] for m in metrics_history], label='Right (φ)', linewidth=2)
    axes[2, 1].set_xlabel('Gradient Step')
    axes[2, 1].set_ylabel('|Cosine Similarity|')
    axes[2, 1].set_ylim([-0.05, 1.05])
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 9: Component-wise graph losses
    axes[2, 2].set_title('Graph Loss Components')
    component_keys = ['graph_loss_x_real', 'graph_loss_x_imag', 'graph_loss_y_real', 'graph_loss_y_imag']
    component_labels = ['x_real', 'x_imag', 'y_real', 'y_imag']
    for key, label in zip(component_keys, component_labels):
        if key in metrics_history[0]:
            axes[2, 2].plot(steps, [m[key] for m in metrics_history], label=label, alpha=0.7)
    axes[2, 2].set_xlabel('Gradient Step')
    axes[2, 2].set_ylabel('Loss')
    axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {save_path}")


def plot_cosine_similarity_evolution(metrics_history: list, save_path: str):
    """
    Plot the evolution of cosine similarities between learned and ground truth eigenvectors.

    Args:
        metrics_history: List of metric dictionaries containing cosine similarity values
        save_path: Path to save the plot
    """
    if not metrics_history:
        print("Warning: metrics_history is empty, skipping cosine similarity plot")
        return

    steps = [m['gradient_step'] for m in metrics_history]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    has_left = 'left_cosine_sim_avg' in metrics_history[0]
    has_right = 'right_cosine_sim_avg' in metrics_history[0]

    if has_left:
        left_values = [m['left_cosine_sim_avg'] for m in metrics_history]
        ax.plot(steps, left_values, label='Left Eigenvector (ψ)',
                color='blue', linewidth=2.5, linestyle='-',
                marker='o', markersize=3, markevery=max(1, len(steps)//20))

    if has_right:
        right_values = [m['right_cosine_sim_avg'] for m in metrics_history]
        ax.plot(steps, right_values, label='Right Eigenvector (φ)',
                color='red', linewidth=2.5, linestyle='-',
                marker='s', markersize=3, markevery=max(1, len(steps)//20))

    if has_left and has_right:
        avg_values = [(left_values[i] + right_values[i]) / 2 for i in range(len(steps))]
        ax.plot(steps, avg_values, label='Average',
                color='green', linewidth=2.0, linestyle='--', alpha=0.7)

    ax.set_xlabel('Gradient Step', fontsize=12)
    ax.set_ylabel('|Complex Cosine Similarity|', fontsize=12)
    ax.set_title('Evolution of Cosine Similarity with Ground Truth\n(Single Eigenvector Learning)', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cosine similarity evolution plot saved to {save_path}")


def plot_eigenvector_comparison(
    learned_left_real: np.ndarray,
    learned_left_imag: np.ndarray,
    learned_right_real: np.ndarray,
    learned_right_imag: np.ndarray,
    gt_left_real: np.ndarray,
    gt_left_imag: np.ndarray,
    gt_right_real: np.ndarray,
    gt_right_imag: np.ndarray,
    normalized_left_real: np.ndarray,
    normalized_left_imag: np.ndarray,
    normalized_right_real: np.ndarray,
    normalized_right_imag: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    save_dir: str,
    door_markers: dict = None,
):
    """
    Create comparison plots between ground truth and learned eigenvectors.

    Generates separate plots for each eigenvector pair:
    - Right eigenvector (real and imaginary parts)
    - Left eigenvector (real and imaginary parts)

    Each plot shows: Ground Truth | Raw Learned | Normalized Learned

    Args:
        learned_*: Raw learned eigenvector components [num_states, num_eigenvector_pairs]
        gt_*: Ground truth eigenvector components [num_states, num_eigenvector_pairs]
        normalized_*: Normalized learned eigenvector components [num_states, num_eigenvector_pairs]
        canonical_states: Array of canonical state indices
        grid_width, grid_height: Grid dimensions
        save_dir: Directory to save plots
        door_markers: Optional door/portal markers for visualization
    """
    from pathlib import Path
    save_dir = Path(save_dir)

    # Get number of eigenvector pairs
    num_eigenvector_pairs = learned_right_real.shape[1]

    # Helper function to create a comparison plot
    def create_comparison_plot(gt_vals, learned_vals, normalized_vals, title_prefix, save_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Create value grids
        for ax, vals, title in zip(axes, [gt_vals, learned_vals, normalized_vals],
                                   ['Ground Truth', 'Raw Learned', 'Normalized Learned']):
            # Create grid with NaN for obstacles
            grid_values = np.full((grid_height, grid_width), np.nan)

            for canon_idx, full_state_idx in enumerate(canonical_states):
                y = int(full_state_idx) // grid_width
                x = int(full_state_idx) % grid_width
                grid_values[y, x] = float(vals[canon_idx])

            # Plot heatmap
            vmax = max(abs(np.nanmin(grid_values)), abs(np.nanmax(grid_values)))
            vmin = -vmax
            im = ax.imshow(grid_values, cmap='RdBu_r', interpolation='nearest',
                          origin='upper', vmin=vmin, vmax=vmax)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add grid lines
            ax.set_xticks(np.arange(grid_width) - 0.5, minor=True)
            ax.set_yticks(np.arange(grid_height) - 0.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

            # Add door markers if provided
            if door_markers:
                for (state, action), next_state in door_markers.items():
                    y = state // grid_width
                    x = state % grid_width
                    next_y = next_state // grid_width
                    next_x = next_state % grid_width
                    dx = next_x - x
                    dy = next_y - y
                    ax.arrow(x, y, dx * 0.3, dy * 0.3,
                            head_width=0.2, head_length=0.15,
                            fc='green', ec='green', linewidth=2, alpha=0.7)

            ax.set_title(f'{title}', fontsize=12)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        fig.suptitle(f'{title_prefix}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_name}")

    print("Generating eigenvector comparison plots...")

    # Generate plots for each eigenvector pair
    for i in range(num_eigenvector_pairs):
        suffix = f"_{i}" if num_eigenvector_pairs > 1 else ""

        # Right eigenvector - Real part
        create_comparison_plot(
            gt_right_real[:, i], learned_right_real[:, i], normalized_right_real[:, i],
            f'Right Eigenvector {i} (φ) - Real Part', f'comparison_right_real{suffix}.png'
        )

        # Right eigenvector - Imaginary part
        create_comparison_plot(
            gt_right_imag[:, i], learned_right_imag[:, i], normalized_right_imag[:, i],
            f'Right Eigenvector {i} (φ) - Imaginary Part', f'comparison_right_imag{suffix}.png'
        )

        # Left eigenvector - Real part
        create_comparison_plot(
            gt_left_real[:, i], learned_left_real[:, i], normalized_left_real[:, i],
            f'Left Eigenvector {i} (ψ) - Real Part', f'comparison_left_real{suffix}.png'
        )

        # Left eigenvector - Imaginary part
        create_comparison_plot(
            gt_left_imag[:, i], learned_left_imag[:, i], normalized_left_imag[:, i],
            f'Left Eigenvector {i} (ψ) - Imaginary Part', f'comparison_left_imag{suffix}.png'
        )

    print("Eigenvector comparison plots complete.")


def plot_auxiliary_metrics(metrics_history: list, save_path: str):
    """
    Plot auxiliary metrics evolution: eigenvalue estimates, norms, and phase.

    Args:
        metrics_history: List of metric dictionaries
        save_path: Path to save the plot
    """
    if not metrics_history:
        print("Warning: metrics_history is empty, skipping auxiliary metrics plot")
        return

    steps = [m['gradient_step'] for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Auxiliary Metrics Evolution', fontsize=14)

    # Plot 1: Eigenvalue estimates (real part)
    ax = axes[0, 0]
    if 'lambda_x_real' in metrics_history[0]:
        ax.plot(steps, [m['lambda_x_real'] for m in metrics_history], label='λ_x (real)', color='blue')
    if 'lambda_y_real' in metrics_history[0]:
        ax.plot(steps, [m['lambda_y_real'] for m in metrics_history], label='λ_y (real)', color='red')
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Eigenvalue (Real Part)')
    ax.set_title('Eigenvalue Estimates - Real Part')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Eigenvalue estimates (imaginary part)
    ax = axes[0, 1]
    if 'lambda_x_imag' in metrics_history[0]:
        ax.plot(steps, [m['lambda_x_imag'] for m in metrics_history], label='λ_x (imag)', color='blue')
    if 'lambda_y_imag' in metrics_history[0]:
        ax.plot(steps, [m['lambda_y_imag'] for m in metrics_history], label='λ_y (imag)', color='red')
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Eigenvalue (Imag Part)')
    ax.set_title('Eigenvalue Estimates - Imaginary Part')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Squared norms
    ax = axes[1, 0]
    if 'norm_x_sq' in metrics_history[0]:
        ax.plot(steps, [m['norm_x_sq'] for m in metrics_history], label='||x||²', color='blue')
    if 'norm_y_sq' in metrics_history[0]:
        ax.plot(steps, [m['norm_y_sq'] for m in metrics_history], label='||y||²', color='red')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target = 1')
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Squared Norm')
    ax.set_title('Squared Norms (should converge to 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Lyapunov functions (log scale)
    ax = axes[1, 1]
    if 'V_x_norm' in metrics_history[0]:
        ax.plot(steps, [m['V_x_norm'] for m in metrics_history], label='V_x (norm constraint)')
    if 'V_y_norm' in metrics_history[0]:
        ax.plot(steps, [m['V_y_norm'] for m in metrics_history], label='V_y (norm constraint)')
    if 'V_xy_phase' in metrics_history[0]:
        ax.plot(steps, [m['V_xy_phase'] for m in metrics_history], label='V_xy (phase constraint)')
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Lyapunov Function V')
    ax.set_title('Lyapunov Functions (should decay)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Auxiliary metrics plot saved to {save_path}")


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

        # Handle case where all values are the same (e.g., all zeros for imaginary parts)
        if vmin == vmax:
            # Set a small symmetric range around the constant value
            if abs(vmin) < 1e-10:
                # If essentially zero, use symmetric range around zero
                vmin = -0.1
                vmax = 0.1
            else:
                # Otherwise, use ±10% of the value
                delta = abs(vmin) * 0.1
                vmin = vmin - delta
                vmax = vmax + delta

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

    # Ensure values are real (hitting times should be real after conjugate enforcement)
    hitting_time_values = np.real(hitting_time_values)

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
        label = 'Log(Steps+1)' if log_scale else 'Steps'
        # Shrink colorbar to fit better next to subplot
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

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
    shared_colorbar: bool = True,
) -> plt.Figure:
    """
    Visualize hitting times for states acting as Targets (columns) vs Sources (rows).

    Args:
        state_indices: List of state indices to visualize
        hitting_time_matrix: Matrix [num_states, num_states]
        canonical_states: State mapping
        grid_width: Grid width
        grid_height: Grid height
        portals: Portal/Door dict
        ncols: Number of states per row
        figsize: Figure size
        wall_color: Color for walls
        save_path: Path to save
        log_scale: Whether to plot log(values + 1)
        shared_colorbar: If True, all plots share the same color scale and one colorbar.
                         If False, each plot is scaled independently with its own colorbar.

    Returns:
        Matplotlib figure
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

    # Compute global color scales if shared
    # Ensure the matrix is real (conjugate enforcement should handle this,
    # but take real part as a safety net for matplotlib compatibility)
    safe_matrix = np.real(np.maximum(hitting_time_matrix, 0))

    vmin = None
    vmax = None

    if shared_colorbar:
        vmin = 0.0
        if log_scale:
            max_val = float(np.log1p(np.max(safe_matrix)))
        else:
            max_val = float(np.max(safe_matrix))

        if np.isnan(max_val):
            max_val = float(np.nanmax(np.log1p(safe_matrix) if log_scale else safe_matrix))
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
            show_colorbar=not shared_colorbar,
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
            show_colorbar=not shared_colorbar,
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

    plt.tight_layout()

    # Add global colorbar if shared
    if shared_colorbar:
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
    Visualize both real and imaginary components of an eigenvector side-by-side.
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
        title=f'{eigenvector_type.capitalize()} Eigenvector {eigenvector_idx} - Real Component\n\u03bb = {eigenvalue_real:.3f} + {eigenvalue_imag:.3f}i',
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
        title=f'{eigenvector_type.capitalize()} Eigenvector {eigenvector_idx} - Imaginary Component\n|\u03bb| = {np.abs(eigenvalue):.3f}',
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
            title=f'Left Eigvec {eigenvec_idx} ({component})\n\u03bb = {np.abs(eigenvalue):.3f}',
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
            title=f'Right Eigvec {eigenvec_idx} ({component})\n\u03bb = {np.abs(eigenvalue):.3f}',
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
    output_dir: str = "results/visualizations",
    num_eigenvectors: int = 6,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    wall_color: str = 'gray'
):
    """
    Create a complete visualization report for eigenvector analysis.
    Generates both shared-scale and independent-scale plots for all combinations
    of left/right eigenvectors and real/imaginary components.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating eigenvector visualizations...")

    num_available = eigendecomposition['eigenvalues'].shape[0]
    num_to_visualize = min(num_eigenvectors, num_available)

    def generate_versions(func, name_base, **kwargs):
        func(
            save_path=output_path / f"{name_base}_shared_scale.png",
            shared_colorbar=True,
            **kwargs
        )
        plt.close()
        func(
            save_path=output_path / f"{name_base}_independent_scale.png",
            shared_colorbar=False,
            **kwargs
        )
        plt.close()

    common_kwargs = dict(
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        nrows=nrows,
        ncols=ncols,
        wall_color=wall_color
    )

    print(f"  Visualizing top {num_to_visualize} right eigenvectors (real)...")
    generate_versions(visualize_multiple_eigenvectors, "right_eigenvectors_real",
                      eigenvector_type='right', component='real', **common_kwargs)

    print(f"  Visualizing top {num_to_visualize} right eigenvectors (imaginary)...")
    generate_versions(visualize_multiple_eigenvectors, "right_eigenvectors_imag",
                      eigenvector_type='right', component='imag', **common_kwargs)

    print(f"  Visualizing top {num_to_visualize} left eigenvectors (real)...")
    generate_versions(visualize_multiple_eigenvectors, "left_eigenvectors_real",
                      eigenvector_type='left', component='real', **common_kwargs)

    print(f"  Visualizing top {num_to_visualize} left eigenvectors (imaginary)...")
    generate_versions(visualize_multiple_eigenvectors, "left_eigenvectors_imag",
                      eigenvector_type='left', component='imag', **common_kwargs)

    lr_kwargs = dict(
        eigenvector_indices=list(range(num_to_visualize)),
        eigendecomposition=eigendecomposition,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        wall_color=wall_color
    )

    print(f"  Visualizing left and right eigenvectors side-by-side (real)...")
    generate_versions(visualize_left_right_eigenvectors, "left_right_eigenvectors_real",
                      component='real', **lr_kwargs)

    print(f"  Visualizing left and right eigenvectors side-by-side (imaginary)...")
    generate_versions(visualize_left_right_eigenvectors, "left_right_eigenvectors_imag",
                      component='imag', **lr_kwargs)

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


def create_hitting_time_visualization_report(
    hitting_time_matrix: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    output_dir: str = "results/visualizations",
    num_targets: int = 6,
    target_indices: Optional[List[int]] = None,
    ncols: int = 6,
    wall_color: str = 'gray',
    log_scale: bool = False,
):
    """
    Create a complete visualization report for hitting times.
    Generates both shared-scale and independent-scale versions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating hitting time visualizations...")

    num_states = hitting_time_matrix.shape[0]

    if target_indices is None:
        target_indices = np.linspace(0, num_states - 1, num_targets, dtype=int).tolist()

    print(f"  Visualizing source/target comparisons for {len(target_indices)} states...")

    common_kwargs = dict(
        state_indices=target_indices,
        hitting_time_matrix=hitting_time_matrix,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        ncols=ncols,
        wall_color=wall_color,
        log_scale=log_scale,
    )

    filename_shared = f"hitting_times_asymmetry{'_log' if log_scale else ''}_shared_scale.png"
    visualize_source_vs_target_hitting_times(
        save_path=output_path / filename_shared,
        shared_colorbar=True,
        **common_kwargs
    )
    plt.close()

    filename_independent = f"hitting_times_asymmetry{'_log' if log_scale else ''}_independent_scale.png"
    visualize_source_vs_target_hitting_times(
        save_path=output_path / filename_independent,
        shared_colorbar=False,
        **common_kwargs
    )
    plt.close()

    print(f"\nHitting time visualization reports saved to {output_dir}")