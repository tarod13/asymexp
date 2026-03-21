"""
Plotting utilities for reward-shaping Q-learning results.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import visualize_source_vs_target_hitting_times
from src.utils.envs import get_env_transition_markers


def plot_results(
    results: dict,
    output_path: Path,
) -> None:
    """
    Save a three-panel figure (x-axis: training steps):
      Left   – evaluation success rate (↑ better)
      Centre – mean episode length, all episodes (↓ better)
      Right  – mean episode length, successful episodes only (↓ better)
    Each panel shows mean ± std across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, data) in enumerate(results.items()):
        c            = colours[idx % len(colours)]
        eval_sr      = data["eval_sr"]       # [chunks, seeds]
        eval_len_all = data["eval_len_all"]  # [chunks, seeds]
        eval_len_suc = data["eval_len_suc"]  # [chunks, seeds]
        eval_steps   = data["eval_steps"]    # [chunks]

        for metric, ax in [
            (eval_sr,      axes[0]),
            (eval_len_all, axes[1]),
            (eval_len_suc, axes[2]),
        ]:
            mean = np.nanmean(metric, axis=1)
            std  = np.nanstd(metric,  axis=1)
            ax.plot(eval_steps, mean, label=label, color=c,
                    linewidth=1.5, marker="o", markersize=3)
            ax.fill_between(eval_steps, mean - std, mean + std, color=c, alpha=0.15)

    axes[0].set_title("Eval: success rate  (↑ better)", fontsize=11)
    axes[0].set_ylabel("Success rate", fontsize=10)
    axes[0].set_ylim(-0.05, 1.10)

    axes[1].set_title("Eval: avg episode length – all  (↓ better)", fontsize=11)
    axes[1].set_ylabel("Steps", fontsize=10)

    axes[2].set_title("Eval: avg episode length – successful only  (↓ better)", fontsize=11)
    axes[2].set_ylabel("Steps", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Training steps", fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("Reward shaping with Laplacian hitting times", fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


def plot_hitting_times_grid(
    hitting_times: np.ndarray,
    canonical_states: np.ndarray,
    env,
    output_dir: Path,
    ncols: int = 8,
) -> None:
    """
    Save grid-overlaid hitting-time maps for every canonical state.
    Each state appears as both target (times TO it) and source (times FROM it).
    Produces 4 PNGs: linear/log × shared/independent color scale.
    """
    door_markers = get_env_transition_markers(env)
    all_indices = list(range(len(canonical_states)))
    output_dir.mkdir(parents=True, exist_ok=True)

    for log_scale in (False, True):
        suffix = "_log" if log_scale else ""
        for shared in (True, False):
            scale_str = "shared" if shared else "independent"
            fname = output_dir / f"hitting_times{suffix}_{scale_str}_scale.png"
            visualize_source_vs_target_hitting_times(
                state_indices=all_indices,
                hitting_time_matrix=hitting_times,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                ncols=ncols,
                save_path=str(fname),
                log_scale=log_scale,
                shared_colorbar=shared,
            )
            plt.close()
            print(f"  Hitting-times plot → {fname}")
