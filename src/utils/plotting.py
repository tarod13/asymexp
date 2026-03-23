import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, List


def _draw_portal_tile_overlays(ax, portal_sources, portal_ends, grid_width):
    """Overlay colored circles on portal source (blue) and end (orange) states.

    Colors match the classic Portal game aesthetic:
      - Portal start-states: electric blue (#009FE3 / #005FA3)
      - Portal end-states:   vivid orange  (#FF6600 / #CC4400)

    Drawn at high zorder so they always render above the heatmap.
    """
    for state in (portal_sources or []):
        y, x = divmod(state, grid_width)
        ax.add_patch(mpatches.Circle(
            (x, y), radius=0.4,
            facecolor='#009FE3', edgecolor='#005FA3',
            alpha=0.6, linewidth=1.5, zorder=15,
        ))
    for state in (portal_ends or []):
        y, x = divmod(state, grid_width)
        ax.add_patch(mpatches.Circle(
            (x, y), radius=0.4,
            facecolor='#FF6600', edgecolor='#CC4400',
            alpha=0.6, linewidth=1.5, zorder=15,
        ))



def _draw_door_markers(ax, portals, grid_width):
    """Draw black rect + white triangle on each portal/door cell edge."""
    if not portals:
        return
    rect_thickness = 0.15
    rect_width = 0.7
    for (source_idx, action), dest_idx in portals.items():
        sy = source_idx // grid_width
        sx = source_idx % grid_width
        m = 0.02
        if action == 0:  # Up
            rx, ry = sx - rect_width / 2, sy - 0.5 - rect_thickness / 2
            rw, rh = rect_width, rect_thickness
            tri = mpatches.Polygon([
                (sx, sy - 0.5 - rect_thickness / 2 + m),
                (sx - (rect_thickness - 2*m) / 2, sy - 0.5 + rect_thickness / 2 - m),
                (sx + (rect_thickness - 2*m) / 2, sy - 0.5 + rect_thickness / 2 - m),
            ], facecolor='white', edgecolor='none', zorder=11)
        elif action == 1:  # Right
            rx, ry = sx + 0.5 - rect_thickness / 2, sy - rect_width / 2
            rw, rh = rect_thickness, rect_width
            tri = mpatches.Polygon([
                (sx + 0.5 + rect_thickness / 2 - m, sy),
                (sx + 0.5 - rect_thickness / 2 + m, sy - (rect_thickness - 2*m) / 2),
                (sx + 0.5 - rect_thickness / 2 + m, sy + (rect_thickness - 2*m) / 2),
            ], facecolor='white', edgecolor='none', zorder=11)
        elif action == 2:  # Down
            rx, ry = sx - rect_width / 2, sy + 0.5 - rect_thickness / 2
            rw, rh = rect_width, rect_thickness
            tri = mpatches.Polygon([
                (sx, sy + 0.5 + rect_thickness / 2 - m),
                (sx - (rect_thickness - 2*m) / 2, sy + 0.5 - rect_thickness / 2 + m),
                (sx + (rect_thickness - 2*m) / 2, sy + 0.5 - rect_thickness / 2 + m),
            ], facecolor='white', edgecolor='none', zorder=11)
        else:  # Left
            rx, ry = sx - 0.5 - rect_thickness / 2, sy - rect_width / 2
            rw, rh = rect_thickness, rect_width
            tri = mpatches.Polygon([
                (sx - 0.5 - rect_thickness / 2 + m, sy),
                (sx - 0.5 + rect_thickness / 2 - m, sy - (rect_thickness - 2*m) / 2),
                (sx - 0.5 + rect_thickness / 2 - m, sy + (rect_thickness - 2*m) / 2),
            ], facecolor='white', edgecolor='none', zorder=11)
        ax.add_patch(mpatches.Rectangle((rx, ry), rw, rh, linewidth=0, edgecolor='none', facecolor='black', zorder=10))
        ax.add_patch(tri)


def plot_learning_curves_one(
    metrics_history: list,
    save_path: str,
    gt_eigenvalues_real=None,
    gt_eigenvalues_imag=None,
    delta: float = 0.0,
):
    """
    Plot comprehensive learning curves for single-eigenvector training.

    Visualizes losses, Lyapunov functions, eigenvalue estimates, ROC metrics,
    and cosine similarities.

    Args:
        metrics_history: List of metric dictionaries from training
        save_path: Path to save the plot
        gt_eigenvalues_real: GT Laplacian eigenvalue real parts — used to draw
            horizontal reference lines in the eigenvalue subplot (converted to
            kernel space: λ_M = (1+δ) − λ_L).
        gt_eigenvalues_imag: GT Laplacian eigenvalue imaginary parts.
        delta: Discount parameter used in the kernel transformation.
    """
    if not metrics_history:
        print("Warning: metrics_history is empty, skipping learning curves plot")
        return

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
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
    # Plot 7: Eigenvalue estimates (real and imaginary) — keyed as eigenvalue_{k}_real/imag
    axes[2, 0].set_title('Eigenvalue Estimates (kernel/SR space)')
    _eig_indices = sorted(set(
        int(key.split('_')[1])
        for m in metrics_history
        for key in m
        if key.startswith('eigenvalue_') and key.endswith('_real')
    ))
    _colors = plt.cm.tab10.colors
    for _ki in _eig_indices:
        _c = _colors[_ki % len(_colors)]
        axes[2, 0].plot(
            steps, [m.get(f'eigenvalue_{_ki}_real', float('nan')) for m in metrics_history],
            label=f'λ_{_ki} (real)', linestyle='-', color=_c,
        )
        axes[2, 0].plot(
            steps, [m.get(f'eigenvalue_{_ki}_imag', float('nan')) for m in metrics_history],
            label=f'λ_{_ki} (imag)', linestyle='--', color=_c, alpha=0.7,
        )
    # GT reference lines: convert Laplacian → kernel space (λ_M = (1+δ) − λ_L)
    if gt_eigenvalues_real is not None:
        _gt_r = np.asarray(gt_eigenvalues_real)
        _gt_i = np.asarray(gt_eigenvalues_imag) if gt_eigenvalues_imag is not None else np.zeros_like(_gt_r)
        for _ki in range(min(len(_gt_r), len(_eig_indices) if _eig_indices else len(_gt_r))):
            _c = _colors[_ki % len(_colors)]
            _ref_real = float((1.0 + delta) - _gt_r[_ki])
            _ref_imag = float(-_gt_i[_ki])
            axes[2, 0].axhline(_ref_real, color=_c, linestyle='-',  linewidth=1.0, alpha=0.4)
            axes[2, 0].axhline(_ref_imag, color=_c, linestyle='--', linewidth=1.0, alpha=0.4)
    if any('avg_eigenvalue_error' in m for m in metrics_history):
        ax2 = axes[2, 0].twinx()
        ax2.plot(
            steps, [m.get('avg_eigenvalue_error', float('nan')) for m in metrics_history],
            label='Avg error', color='black', linestyle=':', linewidth=1.5,
        )
        ax2.set_ylabel('Avg |λ error|', fontsize=8)
        ax2.legend(fontsize=7, loc='upper right')
    axes[2, 0].set_xlabel('Gradient Step')
    axes[2, 0].set_ylabel('Eigenvalue')
    axes[2, 0].legend(fontsize=7, loc='upper left')
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

    # Row 4: ROC metrics
    # Plot 10: Hitting-time rank-order correlations (goal and source)
    _roc_keys = {
        'avg_goal_roc':              ('Goal ROC (trunc. GT)',       'blue',   '-'),
        'avg_source_roc':            ('Source ROC (trunc. GT)',      'red',    '-'),
        'full_ideal_avg_goal_roc':   ('Goal ROC (full ideal GT)',    'blue',   '--'),
        'full_ideal_avg_source_roc': ('Source ROC (full ideal GT)',  'red',    '--'),
    }
    _has_roc = any(k in metrics_history[0] for k in _roc_keys)
    if _has_roc:
        for _key, (_lbl, _col, _ls) in _roc_keys.items():
            if _key in metrics_history[0]:
                axes[3, 0].plot(
                    steps, [m.get(_key, float('nan')) for m in metrics_history],
                    label=_lbl, color=_col, linestyle=_ls, linewidth=1.8,
                )
        axes[3, 0].set_xlabel('Gradient Step')
        axes[3, 0].set_ylabel('Spearman ρ')
        axes[3, 0].set_title('Hitting-Time Rank Correlations')
        axes[3, 0].set_ylim([-0.05, 1.05])
        axes[3, 0].legend(fontsize=8)
        axes[3, 0].grid(True, alpha=0.3)
    else:
        axes[3, 0].set_visible(False)

    axes[3, 1].set_visible(False)
    axes[3, 2].set_visible(False)

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


def plot_cosine_similarity_per_eigenvector(metrics_history: list, save_path: str):
    """Plot the per-eigenvector cosine similarity evolution for right and left separately.

    Creates a figure with two side-by-side panels (right φ | left ψ).  Each
    panel shows one thin line per eigenvector (colour-coded by index) plus a
    bold dashed line for the overall average.

    Args:
        metrics_history: List of metric dicts from training.
        save_path: Path to save the PNG.
    """
    if not metrics_history:
        print("Warning: metrics_history is empty, skipping per-eigenvector cosine plot")
        return

    steps = [m['gradient_step'] for m in metrics_history]

    # Detect how many individual similarity keys exist for each side
    def _collect_per_eig(prefix):
        indices = sorted(set(
            int(k[len(prefix):])
            for m in metrics_history
            for k in m
            if k.startswith(prefix) and k[len(prefix):].isdigit()
        ))
        return indices

    right_indices = _collect_per_eig('right_cosine_sim_')
    left_indices  = _collect_per_eig('left_cosine_sim_')

    has_right = bool(right_indices)
    has_left  = bool(left_indices)

    if not has_right and not has_left:
        print("Warning: no per-eigenvector cosine similarity keys found, skipping plot")
        return

    ncols = (1 if has_right else 0) + (1 if has_left else 0)
    fig, axes_row = plt.subplots(1, ncols, figsize=(7 * ncols, 6), squeeze=False)
    axes_row = axes_row[0]
    col = 0

    palette = plt.cm.tab10.colors

    for side, indices, avg_key, sym, ax in [
        ('Right (φ)', right_indices, 'right_cosine_sim_avg', 'φ', axes_row[col] if has_right else None),
        ('Left (ψ)',  left_indices,  'left_cosine_sim_avg',  'ψ', axes_row[col + (1 if has_right else 0)] if has_left else None),
    ]:
        if ax is None:
            continue
        for idx in indices:
            key = f'{avg_key[:avg_key.rfind("_avg")]}_{idx}'
            vals = [m.get(key, float('nan')) for m in metrics_history]
            ax.plot(steps, vals, color=palette[idx % len(palette)],
                    linewidth=1.0, alpha=0.75, label=f'{sym}{idx}')
        if avg_key in metrics_history[0]:
            avg_vals = [m[avg_key] for m in metrics_history]
            ax.plot(steps, avg_vals, color='black', linewidth=2.2,
                    linestyle='--', label='Average', zorder=10)
        ax.set_xlabel('Gradient Step', fontsize=11)
        ax.set_ylabel('|Cosine Similarity|', fontsize=11)
        ax.set_title(f'{side} — Per-Eigenvector Cosine Similarity', fontsize=12)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=8, ncol=max(1, len(indices) // 8 + 1), loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-eigenvector cosine similarity plot saved to {save_path}")


def _plot_eigvec_grid(
    data_real: np.ndarray,
    data_imag: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    title: str,
    portals=None,
    portal_sources=None,
    portal_ends=None,
    wall_color: str = 'gray',
    max_cols: int = 12,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """4 rows (Real, Imaginary, Magnitude, Phase) × up to max_cols columns (one per eigenvector).

    Each subplot has its own colorbar. Row labels on the left; column titles on top.
    """
    import matplotlib.cm as cm
    from pathlib import Path

    num_eigenvectors = min(data_real.shape[1], max_cols)
    row_labels = ['Real', 'Imaginary', 'Magnitude', 'Phase']
    cmaps     = ['RdBu_r', 'RdBu_r', 'viridis', 'hsv']

    magnitude = np.sqrt(data_real ** 2 + data_imag ** 2)
    phase     = np.arctan2(data_imag, data_real)
    components = [data_real, data_imag, magnitude, phase]

    fig, axes = plt.subplots(4, num_eigenvectors, figsize=(num_eigenvectors * 2.5, 4 * 2.5))
    if num_eigenvectors == 1:
        axes = axes.reshape(4, 1)

    for row_idx, (comp_data, cmap_name, row_label) in enumerate(zip(components, cmaps, row_labels)):
        current_cmap = cm.get_cmap(cmap_name).copy()
        current_cmap.set_bad(color=wall_color)

        for col_idx in range(num_eigenvectors):
            ax = axes[row_idx, col_idx]
            vals = comp_data[:, col_idx]

            grid = np.full((grid_height, grid_width), np.nan)
            for canon_idx, full_state_idx in enumerate(canonical_states):
                y = int(full_state_idx) // grid_width
                x = int(full_state_idx) % grid_width
                grid[y, x] = float(vals[canon_idx])

            if cmap_name == 'RdBu_r':
                vabs = max(abs(np.nanmin(grid)), abs(np.nanmax(grid))) if not np.all(np.isnan(grid)) else 1.0
                vmin, vmax = -vabs, vabs
            elif cmap_name == 'hsv':
                vmin, vmax = -np.pi, np.pi
            else:  # viridis / magnitude
                vmin = 0.0
                vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0

            im = ax.imshow(
                grid, cmap=current_cmap, origin='upper', interpolation='nearest',
                extent=[-0.5, grid_width - 0.5, grid_height - 0.5, -0.5],
                vmin=vmin, vmax=vmax,
            )

            for i in range(grid_height + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.3)
            for j in range(grid_width + 1):
                ax.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.3)

            _draw_door_markers(ax, portals, grid_width)
            _draw_portal_tile_overlays(ax, portal_sources, portal_ends, grid_width)

            ax.set_xlim(-0.5, grid_width - 0.5)
            ax.set_ylim(grid_height - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row_idx == 0:
                ax.set_title(f'φ{col_idx}', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {Path(save_path).name}")

    return fig


def plot_potential_vs_value(
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    potential_per_seed: np.ndarray,
    value_per_seed: np.ndarray,
    goal_per_seed: np.ndarray,
    cond_name: str = "",
    portals=None,
    portal_sources=None,
    portal_ends=None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """2 × num_seeds figure: row 0 = potential Φ(s), row 1 = value V(s).

    Each column is one seed.  Colorbars are independent per subplot.
    The goal cell is marked with a white star.

    Args:
        canonical_states:    flat grid indices of free states, shape [num_canonical]
        grid_width/height:   environment dimensions
        potential_per_seed:  Φ(s) for each seed, shape [num_seeds, num_canonical]
        value_per_seed:      V(s)=max_a Q(s,a) for each seed, shape [num_seeds, num_canonical]
        goal_per_seed:       canonical state index of each seed's goal, shape [num_seeds]
        cond_name:           figure super-title (condition label)
        portals:             {(src_flat, action): dst_flat} door markers (optional)
        portal_sources/ends: flat state indices for stochastic portal overlays (optional)
        save_path:           if provided, save figure here and close it
    """
    import matplotlib.cm as cm

    num_seeds = len(goal_per_seed)
    row_labels = ["Potential Φ(s)", "Value V(s)"]
    data_rows  = [potential_per_seed, value_per_seed]
    cmap_names = ["plasma", "viridis"]

    fig, axes = plt.subplots(2, num_seeds, figsize=(num_seeds * 3, 6),
                             squeeze=False)

    for row_idx, (row_label, data, cmap_name) in enumerate(
        zip(row_labels, data_rows, cmap_names)
    ):
        cmap = cm.get_cmap(cmap_name).copy()
        cmap.set_bad(color="black")

        for col_idx in range(num_seeds):
            ax   = axes[row_idx, col_idx]
            vals = data[col_idx]

            grid = np.full((grid_height, grid_width), np.nan)
            for canon_idx, full_state_idx in enumerate(canonical_states):
                y = int(full_state_idx) // grid_width
                x = int(full_state_idx) % grid_width
                grid[y, x] = float(vals[canon_idx])

            vmin = float(np.nanmin(grid)) if not np.all(np.isnan(grid)) else 0.0
            vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0

            im = ax.imshow(
                grid, cmap=cmap, origin="upper", interpolation="nearest",
                extent=[-0.5, grid_width - 0.5, grid_height - 0.5, -0.5],
                vmin=vmin, vmax=vmax,
            )

            for i in range(grid_height + 1):
                ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.3)
            for j in range(grid_width + 1):
                ax.axvline(j - 0.5, color="gray", linewidth=0.3, alpha=0.3)

            _draw_door_markers(ax, portals, grid_width)
            _draw_portal_tile_overlays(ax, portal_sources, portal_ends, grid_width)

            # Mark goal with a white star
            goal_flat = int(canonical_states[goal_per_seed[col_idx]])
            goal_y, goal_x = divmod(goal_flat, grid_width)
            ax.plot(goal_x, goal_y, marker="*", markersize=12, color="white",
                    markeredgecolor="black", markeredgewidth=0.8, zorder=20)

            ax.set_xlim(-0.5, grid_width - 0.5)
            ax.set_ylim(grid_height - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row_idx == 0:
                ax.set_title(f"seed {col_idx} (goal {goal_per_seed[col_idx]})",
                             fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9)

    if cond_name:
        fig.suptitle(cond_name, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {Path(save_path).name}")

    return fig


def _plot_eigvec_magnitude_comparison(
    learned_real: np.ndarray,
    learned_imag: np.ndarray,
    gt_real: np.ndarray,
    gt_imag: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    title: str,
    portals=None,
    portal_sources=None,
    portal_ends=None,
    wall_color: str = 'gray',
    max_cols: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Row-pairs (Learned magnitude, GT magnitude) × up to max_cols columns.

    Shared color scale per eigenvector pair. Subtle dashed separator between row-pairs.
    """
    import matplotlib.cm as cm
    from pathlib import Path

    num_eigenvectors = learned_real.shape[1]
    ncols = min(num_eigenvectors, max_cols)
    num_pairs = (num_eigenvectors + ncols - 1) // ncols
    nrows = num_pairs * 2

    learned_mag = np.sqrt(learned_real ** 2 + learned_imag ** 2)
    gt_mag      = np.sqrt(gt_real      ** 2 + gt_imag      ** 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    cmap = cm.get_cmap('viridis').copy()
    cmap.set_bad(color=wall_color)

    for idx in range(num_eigenvectors):
        pair_row = idx // ncols
        col_idx  = idx  % ncols
        ax_l = axes[pair_row * 2,     col_idx]
        ax_g = axes[pair_row * 2 + 1, col_idx]

        def make_grid(vals):
            g = np.full((grid_height, grid_width), np.nan)
            for ci, fsi in enumerate(canonical_states):
                y = int(fsi) // grid_width
                x = int(fsi) % grid_width
                g[y, x] = float(vals[ci])
            return g

        gl = make_grid(learned_mag[:, idx])
        gg = make_grid(gt_mag[:, idx])

        for ax, grid, row_label in ((ax_l, gl, 'Learned'), (ax_g, gg, 'GT')):
            vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
            im = ax.imshow(
                grid, cmap=cmap, origin='upper', interpolation='nearest',
                extent=[-0.5, grid_width - 0.5, grid_height - 0.5, -0.5],
                vmin=0.0, vmax=vmax,
            )
            for i in range(grid_height + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.3)
            for j in range(grid_width + 1):
                ax.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.3)

            _draw_door_markers(ax, portals, grid_width)
            _draw_portal_tile_overlays(ax, portal_sources, portal_ends, grid_width)

            ax.set_xlim(-0.5, grid_width - 0.5)
            ax.set_ylim(grid_height - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9)

        # Column title on top row of first pair only
        if pair_row == 0:
            ax_l.set_title(f'φ{idx}', fontsize=9)

    # Hide unused axes in the last (possibly incomplete) pair row
    for idx in range(num_eigenvectors, num_pairs * ncols):
        pair_row = idx // ncols
        col_idx  = idx  % ncols
        axes[pair_row * 2,     col_idx].axis('off')
        axes[pair_row * 2 + 1, col_idx].axis('off')

    plt.tight_layout()

    # Subtle dashed separators between row-pairs
    if num_pairs > 1:
        for r in range(num_pairs - 1):
            y_bottom = axes[r * 2 + 1, 0].get_position().y0
            y_top    = axes[r * 2 + 2, 0].get_position().y1
            fig.add_artist(plt.Line2D(
                [0.01, 0.99], [(y_bottom + y_top) / 2] * 2,
                transform=fig.transFigure,
                color='gray', linewidth=0.8, linestyle='--', alpha=0.4,
            ))

    fig.suptitle(title, fontsize=13, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {Path(save_path).name}")

    return fig


def plot_eigenvector_comparison(
    learned_right_real: np.ndarray,
    learned_right_imag: np.ndarray,
    gt_right_real: np.ndarray,
    gt_right_imag: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    save_dir: str,
    learned_left_real: np.ndarray = None,
    learned_left_imag: np.ndarray = None,
    gt_left_real: np.ndarray = None,
    gt_left_imag: np.ndarray = None,
    portals: dict = None,
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
    wall_color: str = 'gray',
):
    """Eigenvector overview and magnitude-comparison plots.

    Produces up to six PNG files in save_dir:

    Overview (4 rows × up to 12 cols each):
      eigvec_learned_right.png   — learned right eigenvectors  (φ)
      eigvec_gt_right.png        — GT right eigenvectors       (φ)
      eigvec_learned_left.png    — learned left eigenvectors   (ψ)  [if left data provided]
      eigvec_gt_left.png         — GT left eigenvectors        (ψ)  [if left data provided]

    Rows: Real | Imaginary | Magnitude | Phase

    Magnitude comparison (row-pairs × up to 6 cols each):
      eigvec_magnitude_comparison_right.png  — learned vs GT right magnitude
      eigvec_magnitude_comparison_left.png   — learned vs GT left magnitude  [if left data provided]

    Args:
        learned_right_real / learned_right_imag: [num_states, num_eigvecs]
        gt_right_real      / gt_right_imag:      [num_states, num_eigvecs]
        learned_left_real  / learned_left_imag:  [num_states, num_eigvecs] or None
        gt_left_real       / gt_left_imag:       [num_states, num_eigvecs] or None
        canonical_states: Array mapping canonical index → full grid index
        grid_width, grid_height: Grid dimensions
        save_dir: Directory to save PNG files
        portals: Door/portal dict {(source_idx, action): dest_idx}
        portal_sources: Set of portal-source state indices (blue overlay)
        portal_ends:    Set of portal-end   state indices (orange overlay)
        wall_color: Color for NaN (wall) cells
    """
    from pathlib import Path
    save_dir = Path(save_dir)
    print("Generating eigenvector comparison plots...")

    shared = dict(
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        portal_sources=portal_sources,
        portal_ends=portal_ends,
        wall_color=wall_color,
    )

    # --- Overview plots (4 rows: Real, Imaginary, Magnitude, Phase) ---
    _plot_eigvec_grid(
        data_real=learned_right_real, data_imag=learned_right_imag,
        title='Learned Right Eigenvectors (φ)',
        save_path=str(save_dir / 'eigvec_learned_right.png'),
        **shared,
    )
    _plot_eigvec_grid(
        data_real=gt_right_real, data_imag=gt_right_imag,
        title='GT Right Eigenvectors (φ)',
        save_path=str(save_dir / 'eigvec_gt_right.png'),
        **shared,
    )

    # --- Magnitude comparison: right ---
    _plot_eigvec_magnitude_comparison(
        learned_real=learned_right_real, learned_imag=learned_right_imag,
        gt_real=gt_right_real, gt_imag=gt_right_imag,
        title='Right Eigenvector Magnitude — Learned vs GT',
        save_path=str(save_dir / 'eigvec_magnitude_comparison_right.png'),
        **shared,
    )

    if learned_left_real is not None:
        _plot_eigvec_grid(
            data_real=learned_left_real, data_imag=learned_left_imag,
            title='Learned Left Eigenvectors (ψ)',
            save_path=str(save_dir / 'eigvec_learned_left.png'),
            **shared,
        )
        _plot_eigvec_grid(
            data_real=gt_left_real, data_imag=gt_left_imag,
            title='GT Left Eigenvectors (ψ)',
            save_path=str(save_dir / 'eigvec_gt_left.png'),
            **shared,
        )
        _plot_eigvec_magnitude_comparison(
            learned_real=learned_left_real, learned_imag=learned_left_imag,
            gt_real=gt_left_real, gt_imag=gt_left_imag,
            title='Left Eigenvector Magnitude — Learned vs GT',
            save_path=str(save_dir / 'eigvec_magnitude_comparison_left.png'),
            **shared,
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

    # Plot 1: Eigenvalue estimates (real part) — keyed as eigenvalue_{k}_real
    # Search all entries (not just the first) so that runs resumed from older
    # checkpoints that pre-date eigenvalue logging still render correctly.
    ax = axes[0, 0]
    _eig_indices = sorted(set(
        int(key.split('_')[1])
        for m in metrics_history
        for key in m
        if key.startswith('eigenvalue_') and key.endswith('_real')
    ))
    _colors = plt.cm.tab10.colors
    for _ki in _eig_indices:
        ax.plot(
            steps, [m.get(f'eigenvalue_{_ki}_real', float('nan')) for m in metrics_history],
            label=f'λ_{_ki} (real)', color=_colors[_ki % len(_colors)],
        )
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Eigenvalue (Real Part)')
    ax.set_title('Eigenvalue Estimates - Real Part (kernel/SR)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Eigenvalue estimates (imaginary part) + average eigenvalue error
    ax = axes[0, 1]
    for _ki in _eig_indices:
        ax.plot(
            steps, [m.get(f'eigenvalue_{_ki}_imag', float('nan')) for m in metrics_history],
            label=f'λ_{_ki} (imag)', color=_colors[_ki % len(_colors)],
        )
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Eigenvalue (Imag Part)')
    ax.set_title('Eigenvalue Estimates - Imag Part (kernel/SR)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if any('avg_eigenvalue_error' in m for m in metrics_history):
        ax2 = ax.twinx()
        ax2.plot(
            steps, [m.get('avg_eigenvalue_error', float('nan')) for m in metrics_history],
            label='Avg |λ error|', color='black', linestyle=':', linewidth=1.5,
        )
        ax2.set_ylabel('Avg |λ error|', fontsize=8)
        ax2.legend(fontsize=7, loc='upper right')

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
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
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

    # Add door markers (rect+triangle at cell edge) if provided
    if portals is not None and len(portals) > 0:
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

    _draw_portal_tile_overlays(ax, portal_sources, portal_ends, grid_width)

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
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
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
            portal_sources=portal_sources,
            portal_ends=portal_ends,
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
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
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
        values_to_plot = np.log1p(np.maximum(hitting_time_values, 0))
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

    # Add door markers (rect+triangle at cell edge) if provided
    if portals is not None and len(portals) > 0:
        rect_thickness = 0.15
        rect_width = 0.7

        for (source_idx, action), dest_idx in portals.items():
            source_y = source_idx // grid_width
            source_x = source_idx % grid_width
            margin = 0.02

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

    _draw_portal_tile_overlays(ax, portal_sources, portal_ends, grid_width)

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
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
    ncols: int = 6,
    figsize: Optional[Tuple[int, int]] = None,
    wall_color: str = 'gray',
    save_path: Optional[str] = None,
    log_scale: bool = True,
    shared_colorbar: bool = False,
) -> plt.Figure:
    """
    Visualize hitting times for states acting as Targets (goal) vs Sources (start).

    Layout: (num_logical_rows × 2) physical rows × ncols columns, where each
    logical row-pair shows ncols states with:
      - Top row of pair:    Target view — hitting times TO each state (state as goal)
      - Bottom row of pair: Source view — hitting times FROM each state (state as start)

    A subtle dashed separator is drawn between consecutive row-pairs.

    Args:
        state_indices: List of state indices to visualize
        hitting_time_matrix: Matrix [num_states, num_states]
        canonical_states: State mapping
        grid_width: Grid width
        grid_height: Grid height
        portals: Portal/Door dict
        ncols: Number of states per logical row-pair
        figsize: Figure size
        wall_color: Color for walls
        save_path: Path to save
        log_scale: Whether to plot log(values + 1)
        shared_colorbar: If True, all subplots share one colorbar.
                         If False (default), each subplot has its own colorbar.

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
    real_matrix = np.real(hitting_time_matrix)

    vmin = None
    vmax = None

    if shared_colorbar:
        if log_scale:
            max_val = float(np.nanmax(np.log1p(np.maximum(real_matrix, 0))))
            vmin = 0.0
        else:
            max_val = float(np.nanmax(real_matrix))
            vmin = float(np.nanmin(real_matrix))

        if np.isnan(max_val):
            max_val = 0.0
        vmax = max_val

    for idx, state_idx in enumerate(state_indices):
        r_logical = idx // ncols
        c = idx % ncols

        # Top row of pair: Target View (Even physical rows: 0, 2, 4...)
        ax_target = axes[r_logical * 2, c]

        # Bottom row of pair: Source View (Odd physical rows: 1, 3, 5...)
        ax_source = axes[r_logical * 2 + 1, c]

        # Target View: Column of H (Time TO state_idx)
        times_to_state = hitting_time_matrix[:, state_idx]

        visualize_hitting_time_on_grid(
            hitting_time_values=times_to_state,
            center_state_idx=state_idx,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            mode='target',
            portals=portals,
            portal_sources=portal_sources,
            portal_ends=portal_ends,
            title=f'State {state_idx} — Goal\n(Times TO here)',
            ax=ax_target,
            cmap='viridis',
            show_colorbar=not shared_colorbar,
            wall_color=wall_color,
            vmin=vmin,
            vmax=vmax,
            log_scale=log_scale
        )

        # Source View: Row of H (Time FROM state_idx)
        times_from_state = hitting_time_matrix[state_idx, :]

        visualize_hitting_time_on_grid(
            hitting_time_values=times_from_state,
            center_state_idx=state_idx,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            mode='source',
            portals=portals,
            portal_sources=portal_sources,
            portal_ends=portal_ends,
            title=f'State {state_idx} — Start\n(Times FROM here)',
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

    # Draw subtle dashed separators between row-pairs (in figure coordinates,
    # so they must be added after tight_layout has finalised axis positions)
    if num_logical_rows > 1:
        for r_logical in range(num_logical_rows - 1):
            source_row = r_logical * 2 + 1        # bottom physical row of current pair
            next_target_row = r_logical * 2 + 2   # top physical row of next pair
            y_bottom = axes[source_row, 0].get_position().y0
            y_top = axes[next_target_row, 0].get_position().y1
            y_line = (y_bottom + y_top) / 2
            fig.add_artist(plt.Line2D(
                [0.01, 0.99], [y_line, y_line],
                transform=fig.transFigure,
                color='gray', linewidth=0.8, linestyle='--', alpha=0.4,
            ))

    # Add global colorbar if shared
    if shared_colorbar:
        fig.subplots_adjust(right=0.9)
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


# ─── Complex training diagnostics ──────────────────────────────────────────

def plot_complex_learning_curves(metrics_history: list, save_path: str):
    """
    Plot comprehensive learning curves for complex eigenvector training.

    Includes: ALLO loss, graph loss, gradient norm, dual losses, barrier loss,
    orthogonality errors, per-component diagonal errors, and constraint distances.
    """
    if not metrics_history:
        print("Warning: metrics_history is empty, skipping learning curves plot")
        return

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Training Metrics (Comprehensive Diagnostics)', fontsize=16)

    steps = [m['gradient_step'] for m in metrics_history]

    if 'allo' in metrics_history[0]:
        axes[0, 0].plot(steps, [m['allo'] for m in metrics_history])
        axes[0, 0].set_xlabel('Gradient Step')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('ALLO Loss')
        axes[0, 0].grid(True, alpha=0.3)

    if 'graph_loss' in metrics_history[0]:
        axes[0, 1].plot(steps, [m['graph_loss'] for m in metrics_history])
        axes[0, 1].set_xlabel('Gradient Step')
        axes[0, 1].set_ylabel('Graph Loss')
        axes[0, 1].set_title('Graph Drawing Loss')
        axes[0, 1].grid(True, alpha=0.3)

    if 'grad_norm' in metrics_history[0]:
        axes[0, 2].plot(steps, [m['grad_norm'] for m in metrics_history])
        axes[0, 2].set_xlabel('Gradient Step')
        axes[0, 2].set_ylabel('Gradient Norm')
        axes[0, 2].set_title('Gradient Norm')
        axes[0, 2].grid(True, alpha=0.3)

    if 'dual_loss' in metrics_history[0] or 'dual_loss_neg' in metrics_history[0]:
        if 'dual_loss' in metrics_history[0]:
            axes[1, 0].plot(steps, [m['dual_loss'] for m in metrics_history], label='Positive')
        if 'dual_loss_neg' in metrics_history[0]:
            axes[1, 0].plot(steps, [m['dual_loss_neg'] for m in metrics_history], label='Negative')
        axes[1, 0].set_xlabel('Gradient Step')
        axes[1, 0].set_ylabel('Dual Loss')
        axes[1, 0].set_title('Dual Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    if 'barrier_loss' in metrics_history[0]:
        axes[1, 1].plot(steps, [m['barrier_loss'] for m in metrics_history])
        axes[1, 1].set_xlabel('Gradient Step')
        axes[1, 1].set_ylabel('Barrier Loss')
        axes[1, 1].set_title('Barrier Loss (Fixed Coefficient)')
        axes[1, 1].grid(True, alpha=0.3)

    if 'total_error' in metrics_history[0]:
        axes[1, 2].plot(steps, [m['total_error'] for m in metrics_history], label='Total')
    if 'total_norm_error' in metrics_history[0]:
        axes[1, 2].plot(steps, [m['total_norm_error'] for m in metrics_history], label='Norm')
    if 'total_two_component_error' in metrics_history[0]:
        axes[1, 2].plot(steps, [m['total_two_component_error'] for m in metrics_history], label='First 2')
    axes[1, 2].set_xlabel('Gradient Step')
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].set_title('Orthogonality Errors')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    for row_idx, (component, label) in enumerate([
        ('left_real', 'Left Real'), ('left_imag', 'Left Imag'), ('right_real', 'Right Real')
    ]):
        ax = axes[2, row_idx]
        ax.set_title(f'{label} Diagonal Errors')
        for i in range(min(5, 11)):
            key = f'error_{component}_{i}'
            if key in metrics_history[0]:
                ax.plot(steps, [m[key] for m in metrics_history], label=f'ev{i}', alpha=0.7)
        ax.set_xlabel('Gradient Step')
        ax.set_ylabel('Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[3, 0].set_title('Right Imag Diagonal Errors')
    for i in range(min(5, 11)):
        key = f'error_right_imag_{i}'
        if key in metrics_history[0]:
            axes[3, 0].plot(steps, [m[key] for m in metrics_history], label=f'ev{i}', alpha=0.7)
    axes[3, 0].set_xlabel('Gradient Step')
    axes[3, 0].set_ylabel('Error')
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].grid(True, alpha=0.3)

    if 'distance_to_constraint_manifold' in metrics_history[0]:
        axes[3, 1].plot(steps, [m['distance_to_constraint_manifold'] for m in metrics_history])
        axes[3, 1].set_title('Distance to Constraint Manifold')
    else:
        axes[3, 1].text(0.5, 0.5, 'Metric not available', ha='center', va='center', transform=axes[3, 1].transAxes)
        axes[3, 1].set_title('Distance to Constraint Manifold')
    axes[3, 1].set_xlabel('Gradient Step')
    axes[3, 1].set_ylabel('Distance')
    axes[3, 1].grid(True, alpha=0.3)

    if 'distance_to_origin' in metrics_history[0]:
        axes[3, 2].plot(steps, [m['distance_to_origin'] for m in metrics_history])
        axes[3, 2].set_title('Distance to Origin')
    else:
        axes[3, 2].text(0.5, 0.5, 'Metric not available', ha='center', va='center', transform=axes[3, 2].transAxes)
        axes[3, 2].set_title('Distance to Origin')
    axes[3, 2].set_xlabel('Gradient Step')
    axes[3, 2].set_ylabel('Distance')
    axes[3, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive learning curves saved to {save_path}")


def plot_complex_dual_variable_evolution(
    metrics_history, ground_truth_eigenvalues, gamma, save_path,
    num_eigenvectors=11
):
    """
    Plot evolution of complex dual variables (Laplacian eigenvalue estimates)
    vs ground truth eigenvalues, showing real/imag components, magnitude, and errors.
    """
    steps = [m['gradient_step'] for m in metrics_history]
    num_plot = min(num_eigenvectors, len(ground_truth_eigenvalues))

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    colors = plt.cm.tab10(np.linspace(0, 1, num_plot))

    ax1 = axes[0]
    for i in range(num_plot):
        dual_real_key = f'dual_real_{i}'
        dual_imag_key = f'dual_imag_{i}'
        if dual_real_key in metrics_history[0] and dual_imag_key in metrics_history[0]:
            dual_values_real = np.array([m[dual_real_key] for m in metrics_history])
            dual_values_imag = np.array([m[dual_imag_key] for m in metrics_history])
            ax1.plot(steps, dual_values_real, label=f'lambda_{i} (real)',
                     color=colors[i], linewidth=1.5, linestyle='-')
            ax1.plot(steps, dual_values_imag, label=f'lambda_{i} (imag)',
                     color=colors[i], linewidth=1.5, linestyle='--', alpha=0.7)
            gt_value_real = float(ground_truth_eigenvalues[i].real)
            gt_value_imag = float(ground_truth_eigenvalues[i].imag)
            ax1.axhline(y=gt_value_real, color=colors[i], linestyle='-', alpha=0.2, linewidth=2)
            ax1.axhline(y=gt_value_imag, color=colors[i], linestyle='--', alpha=0.2, linewidth=2)
    ax1.set_xlabel('Gradient Step', fontsize=12)
    ax1.set_ylabel('Dual Variable Value', fontsize=12)
    ax1.set_title('Dual Variables: Real (solid) and Imaginary (dashed) Components', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i in range(num_plot):
        dual_real_key = f'dual_real_{i}'
        dual_imag_key = f'dual_imag_{i}'
        if dual_real_key in metrics_history[0] and dual_imag_key in metrics_history[0]:
            dual_values_real = np.array([m[dual_real_key] for m in metrics_history])
            dual_values_imag = np.array([m[dual_imag_key] for m in metrics_history])
            dual_magnitude = np.sqrt(dual_values_real**2 + dual_values_imag**2)
            ax2.plot(steps, dual_magnitude, label=f'|lambda_{i}|', color=colors[i], linewidth=1.5)
            gt_magnitude = np.abs(ground_truth_eigenvalues[i])
            ax2.axhline(y=gt_magnitude, color=colors[i], linestyle='-', alpha=0.3, linewidth=2.5)
    ax2.set_xlabel('Gradient Step', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Magnitude of Complex Dual Variables', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for i in range(num_plot):
        dual_real_key = f'dual_real_{i}'
        dual_imag_key = f'dual_imag_{i}'
        if dual_real_key in metrics_history[0] and dual_imag_key in metrics_history[0]:
            dual_values_real = np.array([m[dual_real_key] for m in metrics_history])
            dual_values_imag = np.array([m[dual_imag_key] for m in metrics_history])
            dual_values_real_scaled = 0.5 * dual_values_real
            dual_values_imag_scaled = 0.5 * dual_values_imag
            gt_value_real = float(ground_truth_eigenvalues[i].real)
            gt_value_imag = float(ground_truth_eigenvalues[i].imag)
            error_real = dual_values_real_scaled - gt_value_real
            error_imag = dual_values_imag_scaled - gt_value_imag
            error_magnitude = np.sqrt(error_real**2 + error_imag**2)
            ax3.plot(steps, error_magnitude, label=f'|error lambda_{i}|',
                     color=colors[i], linewidth=1.5)
    ax3.set_xlabel('Gradient Step', fontsize=12)
    ax3.set_ylabel('Error Magnitude', fontsize=12)
    ax3.set_title('Complex Error Magnitude in Complex Plane', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dual variable evolution plot saved to {save_path}")


def plot_complex_cosine_similarity_evolution(
    metrics_history: list, save_path: str, num_eigenvectors: int = None
):
    """
    Plot evolution of cosine similarities for complex (left/right) eigenvectors.
    """
    steps = [m['gradient_step'] for m in metrics_history]
    has_left = 'left_cosine_sim_avg' in metrics_history[0]
    has_right = 'right_cosine_sim_avg' in metrics_history[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if has_left:
        left_values = [m['left_cosine_sim_avg'] for m in metrics_history]
        ax.plot(steps, left_values, label='Left Eigenvectors',
                color='blue', linewidth=2.5, linestyle='-',
                marker='o', markersize=3, markevery=max(1, len(steps)//20))

    if has_right:
        right_values = [m['right_cosine_sim_avg'] for m in metrics_history]
        ax.plot(steps, right_values, label='Right Eigenvectors',
                color='red', linewidth=2.5, linestyle='-',
                marker='s', markersize=3, markevery=max(1, len(steps)//20))

    if has_left and has_right:
        avg_values = [(left_values[i] + right_values[i]) / 2 for i in range(len(steps))]
        ax.plot(steps, avg_values, label='Average (Left + Right)',
                color='green', linewidth=2.0, linestyle='--', alpha=0.7)

    ax.set_xlabel('Gradient Step', fontsize=12)
    ax.set_ylabel('|Re(Complex Cosine Similarity)|', fontsize=12)
    ax.set_title('Evolution of Complex Cosine Similarity', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cosine similarity evolution plot saved to {save_path}")


def plot_complex_all_duals_evolution(
    metrics_history: list, save_path: str, num_eigenvectors: int = 6
):
    """
    Plot dual variable evolution for diagonal norm constraints (real and imaginary).
    """
    steps = [m['gradient_step'] for m in metrics_history]

    fig, axes = plt.subplots(num_eigenvectors, 2, figsize=(16, 4 * num_eigenvectors))
    fig.suptitle('Dual Variables Evolution (Diagonal Norms)', fontsize=16)

    if num_eigenvectors == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_eigenvectors):
        ax_real = axes[i, 0]
        if f'dual_real_{i}' in metrics_history[0]:
            dual_real = [m[f'dual_real_{i}'] for m in metrics_history]
            ax_real.plot(steps, dual_real, label='Dual (Real)', linewidth=2.0, color='blue')
        ax_real.set_xlabel('Gradient Step')
        ax_real.set_ylabel('Dual Value (Real)')
        ax_real.set_title(f'Eigenvector {i} - Real Part Dual')
        ax_real.legend()
        ax_real.grid(True, alpha=0.3)

        ax_imag = axes[i, 1]
        if f'dual_imag_{i}' in metrics_history[0]:
            dual_imag = [m[f'dual_imag_{i}'] for m in metrics_history]
            ax_imag.plot(steps, dual_imag, label='Dual (Imag)', linewidth=2.0, color='red')
        ax_imag.set_xlabel('Gradient Step')
        ax_imag.set_ylabel('Dual Value (Imag)')
        ax_imag.set_title(f'Eigenvector {i} - Imaginary Part Dual')
        ax_imag.legend()
        ax_imag.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"All duals evolution plot saved to {save_path}")


def plot_sampling_distribution(
    sampling_probs: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    save_path: str,
    portals: dict = None,
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
):
    """Visualize the empirical sampling distribution D on the grid."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    grid_values = np.full((grid_height, grid_width), np.nan)
    for canon_idx, full_state_idx in enumerate(canonical_states):
        y = int(full_state_idx) // grid_width
        x = int(full_state_idx) % grid_width
        grid_values[y, x] = float(sampling_probs[canon_idx])

    im = ax.imshow(grid_values, cmap='viridis', interpolation='nearest', origin='upper')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Sampling Probability', fontsize=12)

    ax.set_xticks(np.arange(grid_width) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_height) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

    if portals:
        for (state, action), next_state in portals.items():
            y = state // grid_width
            x = state % grid_width
            next_y = next_state // grid_width
            next_x = next_state % grid_width
            dx = next_x - x
            dy = next_y - y
            ax.arrow(x, y, dx * 0.3, dy * 0.3,
                     head_width=0.2, head_length=0.15,
                     fc='red', ec='red', linewidth=2, alpha=0.7)

    _draw_portal_tile_overlays(ax, portal_sources, portal_ends, grid_width)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Empirical Sampling Distribution D', fontsize=14, fontweight='bold')

    stats_text = (f'Min: {np.nanmin(grid_values):.6f}\n'
                  f'Max: {np.nanmax(grid_values):.6f}\n'
                  f'Mean: {np.nanmean(grid_values):.6f}\n'
                  f'Std: {np.nanstd(grid_values):.6f}')
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sampling distribution visualization saved to {save_path}")


def plot_roc_heatmap(
    roc_values: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    title: str,
    save_path: str,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
) -> None:
    """
    Plot a per-state Spearman ROC value overlaid onto the maze topology.

    Args:
        roc_values:       1-D array [num_states] of Spearman ρ values (may contain NaN
                          for states where correlation was undefined).
        canonical_states: [num_states, 2] (row, col) grid coordinates.
        grid_width:       Width of the grid in cells.
        grid_height:      Height of the grid in cells.
        title:            Plot title (e.g. 'Goal ROC' or 'Source ROC').
        save_path:        File path for the saved figure.
        portals:          Optional dict of portal/door markers (same format used by
                          visualize_eigenvector_on_grid).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    visualize_eigenvector_on_grid(
        eigenvector_idx=0,
        eigenvector_values=np.array(roc_values, dtype=float),
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        portal_sources=portal_sources,
        portal_ends=portal_ends,
        title=title,
        ax=ax,
        cmap='RdBu_r',
        show_colorbar=True,
        vmin=-1.0,
        vmax=1.0,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC heatmap saved to {save_path}")


def plot_hitting_time_heatmap(
    hitting_times: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    title: str,
    save_path: str,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
) -> None:
    """
    Save a heatmap of the mean hitting time FROM each state (row mean of the
    hitting-time matrix) overlaid on the maze topology.

    The per-state mean hitting time (averaged over all goals) gives an
    interpretable single-number summary of how 'central' or 'reachable'
    each state is.

    Args:
        hitting_times: [num_states, num_states] matrix where entry [i, j] is
                       the expected hitting time from state i to state j.
        canonical_states: 1-D array of canonical state indices [num_states].
        grid_width:       Width of the grid in cells.
        grid_height:      Height of the grid in cells.
        title:            Plot title.
        save_path:        File path for the saved figure.
        portals:          Optional portal/door markers.
    """
    mean_ht = np.nanmean(hitting_times, axis=1)   # [num_states]
    fig, ax = plt.subplots(figsize=(7, 6))
    visualize_eigenvector_on_grid(
        eigenvector_idx=0,
        eigenvector_values=mean_ht,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        portal_sources=portal_sources,
        portal_ends=portal_ends,
        title=title,
        ax=ax,
        cmap='viridis',
        show_colorbar=True,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hitting time heatmap saved to {save_path}")


def plot_full_ideal_summary(
    goal_rocs: np.ndarray,
    source_rocs: np.ndarray,
    canonical_states: np.ndarray,
    grid_width: int,
    grid_height: int,
    save_dir: str,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    portal_sources: Optional[Set[int]] = None,
    portal_ends: Optional[Set[int]] = None,
) -> None:
    """
    Generate and save the two Full Ideal ROC heatmaps:
      1. Goal ROC heatmap  – per-goal  Spearman ρ overlaid on maze.
      2. Source ROC heatmap – per-source Spearman ρ overlaid on maze.

    The Full Ideal GT hitting time visualization is produced separately via
    visualize_source_vs_target_hitting_times (same format as the truncated GT).

    Args:
        goal_rocs:   [N] per-goal   Spearman ρ values (learned vs Full Ideal).
        source_rocs: [N] per-source Spearman ρ values (learned vs Full Ideal).
        canonical_states: 1-D array of canonical state indices [N].
        grid_width, grid_height: Grid dimensions.
        save_dir: Directory in which to save the PNG files.
        portals, portal_sources, portal_ends: Optional topology markers.
    """
    save_dir = Path(save_dir)

    plot_roc_heatmap(
        roc_values=goal_rocs,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        title='Full Ideal Goal ROC (Spearman ρ, learned vs Full Ideal GT)',
        save_path=str(save_dir / 'full_ideal_goal_roc.png'),
        portals=portals,
        portal_sources=portal_sources,
        portal_ends=portal_ends,
    )

    plot_roc_heatmap(
        roc_values=source_rocs,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        title='Full Ideal Source ROC (Spearman ρ, learned vs Full Ideal GT)',
        save_path=str(save_dir / 'full_ideal_source_roc.png'),
        portals=portals,
        portal_sources=portal_sources,
        portal_ends=portal_ends,
    )


# ── Wind-sweep visualization primitives ──────────────────────────────────────

def align_phase(real: np.ndarray, imag: np.ndarray) -> tuple:
    """Rotate each eigenvector column so its largest-magnitude component is
    real and positive, removing the arbitrary U(1) phase from jnp.linalg.eig."""
    K = real.shape[1]
    out_real = real.copy()
    out_imag = imag.copy()
    for k in range(K):
        cplx    = real[:, k] + 1j * imag[:, k]
        max_idx = np.argmax(np.abs(cplx))
        phase   = np.angle(cplx[max_idx])
        rotated = cplx * np.exp(-1j * phase)
        out_real[:, k] = rotated.real
        out_imag[:, k] = rotated.imag
    return out_real, out_imag


def eigvec_to_grid(values: np.ndarray,
                   canonical_states: np.ndarray,
                   width: int, height: int) -> np.ndarray:
    """Place per-state values on a (height, width) grid; walls stay NaN."""
    grid = np.full((height, width), np.nan)
    for i, s in enumerate(canonical_states):
        grid[int(s) // width, int(s) % width] = values[i]
    return grid


def make_figure(display: np.ndarray,
                wind_values: np.ndarray,
                canonical_states: np.ndarray,
                grid_width: int, grid_height: int,
                eig_indices: list,
                title: str) -> plt.Figure:
    """Build the heatmap grid figure.

    display: (n_wind, num_states, K)
    rows    = eigenvectors (indexed by eig_indices)
    columns = wind values
    """
    n_wind = len(wind_values)
    n_rows = len(eig_indices)

    vabs    = max(float(np.nanmax(np.abs(display[:, :, eig_indices]))), 1e-9)
    hm_norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)

    cell_w  = 0.55
    cell_h  = cell_w * grid_height / grid_width
    cb_w    = 0.25
    cb_gap  = 0.08
    pad_l   = 0.50
    pad_top = 0.35
    pad_bot = 0.25

    fig_w = pad_l + n_wind * cell_w + cb_gap + cb_w + 0.10
    fig_h = pad_top + n_rows * cell_h + pad_bot

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=10, y=1.0)

    hmap_right = 1.0 - (cb_gap + cb_w + 0.10) / fig_w
    gs = gridspec.GridSpec(
        n_rows, n_wind + 1,
        width_ratios=[1.0] * n_wind + [cb_w / cell_w],
        hspace=0.06,
        wspace=0.0,
        left=pad_l / fig_w,
        right=hmap_right,
        top=1.0 - pad_top / fig_h,
        bottom=pad_bot / fig_h,
    )

    for row, k in enumerate(eig_indices):
        eig_data  = display[:, :, k]
        row_label = "ψ₀" if k == 0 else f"ψ{k}"

        for col in range(n_wind):
            ax = fig.add_subplot(gs[row, col])
            grid_img = eigvec_to_grid(
                eig_data[col], canonical_states, grid_width, grid_height
            )
            ax.imshow(
                grid_img,
                cmap="RdBu_r", norm=hm_norm,
                origin="upper", aspect="auto",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"{wind_values[col]:+.2f}", fontsize=4.5, pad=2)

        fig.text(
            pad_l / fig_w - 0.005,
            1.0 - (pad_top + (row + 0.5) * cell_h) / fig_h,
            row_label,
            ha="right", va="center", fontsize=8,
            transform=fig.transFigure,
        )

    cb_left   = hmap_right + cb_gap / fig_w
    cb_bottom = pad_bot / fig_h
    cb_height = 1.0 - (pad_top + pad_bot) / fig_h
    ax_cb = fig.add_axes([cb_left, cb_bottom, cb_w / fig_w, cb_height])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=hm_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.ax.tick_params(labelsize=6)

    return fig


def save_figure(display, wind_values, canonical_states, grid_width, grid_height,
                eig_indices, title, out_path):
    """Save a wind-sweep heatmap grid to *out_path* and close the figure."""
    fig = make_figure(display, wind_values, canonical_states,
                      grid_width, grid_height, eig_indices, title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wind_sweep_grid(sweep_data: list,
                         canonical_states: np.ndarray,
                         grid_width: int,
                         grid_height: int,
                         eig_indices: list,
                         save_dir: str,
                         method: str = "clf"):
    """Generate wind-sweep heatmap figures from a single training run.

    Parameters
    ----------
    sweep_data : list of dicts, one per wind value, each containing:
        wind               – scalar wind value
        gt_right_real      – (S, K) ground-truth right eigenvectors, real part
        gt_right_imag      – (S, K)
        gt_left_real       – (S, K) ground-truth left eigenvectors, real part
        gt_left_imag       – (S, K)
        learned_right_real – (S, K) learned right eigenvectors, real part
        learned_right_imag – (S, K)
    canonical_states : (S,) array of canonical state indices
    grid_width, grid_height : int
    eig_indices : list of column indices to display (rows in the figure)
    save_dir : directory where PNG files are written
    method : "allo" suppresses left-eigenvector figures
    """
    save_dir = Path(save_dir)
    sweep_data = sorted(sweep_data, key=lambda d: d['wind'])
    wind_values = np.array([d['wind'] for d in sweep_data])

    def _stack_and_align(key_r, key_i):
        all_r = np.stack([d[key_r] for d in sweep_data], axis=0)
        all_i = np.stack([d[key_i] for d in sweep_data], axis=0)
        for w in range(len(wind_values)):
            all_r[w], all_i[w] = align_phase(all_r[w], all_i[w])
        return all_r, all_i

    gt_rr, gt_ri = _stack_and_align('gt_right_real', 'gt_right_imag')
    ln_rr, ln_ri = _stack_and_align('learned_right_real', 'learned_right_imag')

    for comp, label, display in [
        ("real", "Re",  gt_rr),
        ("imag", "Im",  gt_ri),
        ("abs",  "|·|", np.sqrt(gt_rr ** 2 + gt_ri ** 2)),
    ]:
        save_figure(display, wind_values, canonical_states,
                    grid_width, grid_height, eig_indices,
                    f"GT right eigenvectors [{label}(ψ_k)] × wind",
                    save_dir / f"wind_sweep_gt_right_{comp}.png")

    save_figure(ln_rr, wind_values, canonical_states,
                grid_width, grid_height, eig_indices,
                "Learned right eigenvectors [Re(ψ_k)] × wind",
                save_dir / "wind_sweep_learned_right_real.png")

    if method != "allo":
        gt_lr, gt_li = _stack_and_align('gt_left_real', 'gt_left_imag')
        for comp, label, display in [
            ("real", "Re",  gt_lr),
            ("imag", "Im",  gt_li),
            ("abs",  "|·|", np.sqrt(gt_lr ** 2 + gt_li ** 2)),
        ]:
            save_figure(display, wind_values, canonical_states,
                        grid_width, grid_height, eig_indices,
                        f"GT left eigenvectors [{label}(ψ_k)] × wind",
                        save_dir / f"wind_sweep_gt_left_{comp}.png")