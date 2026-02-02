import matplotlib.pyplot as plt
import numpy as np



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