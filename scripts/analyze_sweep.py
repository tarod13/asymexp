"""
Analyze a batch_size × learning_rate sweep.

Loads all runs sharing the same --exp_name under a results_dir, groups them by
(batch_size, learning_rate), averages over seeds, and saves plots.

Usage:
    python scripts/analyze_sweep.py \
        --results_dir ./results/sweeps \
        --exp_name batch_lr_sweep \
        --output_dir ./results/sweeps/analysis \
        [--env_type file]
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# =============================================================================
# Helpers
# =============================================================================

def load_run(run_dir: Path):
    """
    Load args and metrics_history from a single run directory.
    Returns (args_dict, metrics_list) or None if the run is incomplete.
    """
    args_file = run_dir / "args.json"
    metrics_file = run_dir / "metrics_history.json"
    if not args_file.exists() or not metrics_file.exists():
        return None
    try:
        with open(args_file) as f:
            args = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)
        if not metrics:
            return None
        return args, metrics
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Skip runs with corrupted/incomplete JSON files
        print(f"    Warning: Skipping {run_dir.name} - {type(e).__name__}: {e}")
        return None


def collect_runs(results_dir: Path, exp_name: str, env_type: str):
    """
    Scan results_dir/{env_type}/* for runs matching exp_name.
    Returns a list of (batch_size, learning_rate, seed, metrics_list).
    """
    env_dir = results_dir / env_type
    if not env_dir.exists():
        raise FileNotFoundError(f"No results found at {env_dir}")

    runs = []
    skipped = 0
    for run_dir in sorted(env_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        result = load_run(run_dir)
        if result is None:
            skipped += 1
            continue
        args, metrics = result
        if args.get("exp_name") != exp_name:
            continue
        bs = args["batch_size"]
        lr = args["learning_rate"]
        seed = args["seed"]
        runs.append((bs, lr, seed, metrics))

    print(f"Loaded {len(runs)} runs, skipped {skipped} incomplete/unrelated.")
    return runs


def metrics_to_arrays(all_metrics: list[list[dict]]):
    """
    Given a list of metrics histories (one per seed), align them on
    gradient_step and return {metric_name: (n_seeds, n_steps) array} plus
    the common steps array.
    """
    # Collect all steps across all seeds; use the intersection
    step_sets = [set(m["gradient_step"] for m in hist) for hist in all_metrics]
    common_steps = sorted(step_sets[0].intersection(*step_sets[1:]))
    if not common_steps:
        # Fall back to union (will have NaNs)
        common_steps = sorted(step_sets[0].union(*step_sets[1:]))

    # Build index for fast lookup
    indexed = []
    for hist in all_metrics:
        idx = {m["gradient_step"]: m for m in hist}
        indexed.append(idx)

    # Determine all metric keys (excluding gradient_step)
    all_keys = set()
    for hist in all_metrics:
        for m in hist:
            all_keys.update(m.keys())
    all_keys.discard("gradient_step")

    arrays = {}
    for key in all_keys:
        mat = np.full((len(all_metrics), len(common_steps)), np.nan)
        for s, idx in enumerate(indexed):
            for t, step in enumerate(common_steps):
                if step in idx and key in idx[step]:
                    val = idx[step][key]
                    if isinstance(val, (int, float)):
                        mat[s, t] = val
        arrays[key] = mat

    return np.array(common_steps), arrays


def savefig(fig, path: Path, dpi=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# Plot helpers
# =============================================================================

LR_LABEL = {
    3e-3: "3e-3", 1e-3: "1e-3",
    3e-4: "3e-4", 1e-4: "1e-4",
    3e-5: "3e-5", 1e-5: "1e-5",
    3e-6: "3e-6", 1e-6: "1e-6",
    3e-7: "3e-7", 1e-7: "1e-7",
}

def lr_label(lr):
    # Match to nearest known value
    for k, v in LR_LABEL.items():
        if abs(lr - k) / max(k, 1e-12) < 0.01:
            return v
    return f"{lr:.1e}"


def _add_curve(ax, steps, mean, std, label, color, alpha_fill=0.15):
    ax.plot(steps, mean, label=label, color=color, linewidth=1.2)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=alpha_fill)


def plot_metric_grid(
    grouped,        # dict: (bs, lr) -> (steps, mean_arr, std_arr)
    metric_key,
    batch_sizes,
    learning_rates,
    output_path,
    ylabel,
    title_prefix,
):
    """
    One subplot per batch size (rows), one line per learning rate (colors).
    """
    n_bs = len(batch_sizes)
    colors = cm.plasma(np.linspace(0.05, 0.95, len(learning_rates)))

    fig, axes = plt.subplots(
        n_bs, 1,
        figsize=(12, 2.8 * n_bs),
        sharex=False,
    )
    if n_bs == 1:
        axes = [axes]

    for ax, bs in zip(axes, batch_sizes):
        ax.set_title(f"batch_size = {bs}", fontsize=9, pad=3)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("gradient step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.5)

        for color, lr in zip(colors, learning_rates):
            key = (bs, lr)
            if key not in grouped or metric_key not in grouped[key][2]:
                continue
            steps, _, metric_dict = grouped[key]
            mean = metric_dict[metric_key]["mean"]
            std = metric_dict[metric_key]["std"]
            mask = ~np.isnan(mean)
            if mask.sum() == 0:
                continue
            _add_curve(ax, steps[mask], mean[mask], std[mask],
                       label=f"lr={lr_label(lr)}", color=color)

    # Shared legend on first subplot
    if axes:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, fontsize=7,
                           ncol=min(5, len(learning_rates)),
                           loc="upper left", framealpha=0.7)

    fig.suptitle(f"{title_prefix} — {metric_key}", fontsize=11, y=1.002)
    plt.tight_layout()
    savefig(fig, output_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results/sweeps")
    parser.add_argument("--exp_name", type=str, default="batch_lr_sweep")
    parser.add_argument("--env_type", type=str, default="file")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis" / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Collect all runs
    # -------------------------------------------------------------------------
    print(f"\nScanning {results_dir / args.env_type} for exp_name={args.exp_name!r} ...")
    raw_runs = collect_runs(results_dir, args.exp_name, args.env_type)
    if not raw_runs:
        print("No matching runs found. Exiting.")
        return

    # -------------------------------------------------------------------------
    # 2. Group by (batch_size, lr), collect metrics over seeds
    # -------------------------------------------------------------------------
    by_combo = defaultdict(list)  # (bs, lr) -> list of metrics_list
    all_bs = set()
    all_lr = set()
    for bs, lr, seed, metrics in raw_runs:
        by_combo[(bs, lr)].append(metrics)
        all_bs.add(bs)
        all_lr.add(lr)

    batch_sizes = sorted(all_bs)
    learning_rates = sorted(all_lr)

    print(f"\nBatch sizes:    {batch_sizes}")
    print(f"Learning rates: {[lr_label(lr) for lr in learning_rates]}")
    print(f"Combos with data: {len(by_combo)} / {len(batch_sizes) * len(learning_rates)}")

    # Print seed counts
    print("\nSeeds per combo:")
    for bs in batch_sizes:
        row = "  bs={:<6d}".format(bs)
        for lr in learning_rates:
            n = len(by_combo.get((bs, lr), []))
            row += f"  {n}"
        print(row)

    # -------------------------------------------------------------------------
    # 3. Align metrics and compute mean/std per combo
    # -------------------------------------------------------------------------
    # grouped[(bs, lr)] = (steps_array, n_seeds, {metric: {mean, std}})
    grouped = {}
    for (bs, lr), seed_metrics in by_combo.items():
        steps, arrays = metrics_to_arrays(seed_metrics)
        metric_stats = {}
        for key, mat in arrays.items():
            # Only keep metrics that have at least one non-NaN value
            valid = ~np.all(np.isnan(mat), axis=1)
            if valid.sum() == 0:
                continue
            mat_valid = mat[valid]
            metric_stats[key] = {
                "mean": np.nanmean(mat_valid, axis=0),
                "std": np.nanstd(mat_valid, axis=0),
                "n": valid.sum(),
            }
        grouped[(bs, lr)] = (steps, len(seed_metrics), metric_stats)

    # -------------------------------------------------------------------------
    # 4. Identify metrics to plot
    # -------------------------------------------------------------------------
    # Sample all available keys from first combo
    all_metric_keys = set()
    for steps, n, metric_dict in grouped.values():
        all_metric_keys.update(metric_dict.keys())

    cosine_sim_avg_keys = sorted(
        k for k in all_metric_keys if "cosine_sim_avg" in k
    )
    cosine_sim_indiv_keys = sorted(
        k for k in all_metric_keys
        if "cosine_sim" in k and "avg" not in k
    )
    other_keys = sorted(
        k for k in all_metric_keys
        if "cosine_sim" not in k and k != "gradient_step"
    )

    print(f"\nAverage cosine similarity keys: {cosine_sim_avg_keys}")
    print(f"Individual cosine similarity keys: {len(cosine_sim_indiv_keys)} keys")
    print(f"Other metric keys: {len(other_keys)} keys")

    title = f"{args.exp_name} ({args.env_type})"

    # -------------------------------------------------------------------------
    # 5a. Average cosine similarity evolution (left + right)
    # -------------------------------------------------------------------------
    print("\nPlotting average cosine similarities ...")
    for key in cosine_sim_avg_keys:
        plot_metric_grid(
            grouped, key, batch_sizes, learning_rates,
            output_path=output_dir / "avg_cosine" / f"{key}.png",
            ylabel="cosine similarity",
            title_prefix=title,
        )

    # -------------------------------------------------------------------------
    # 5b. Individual cosine similarity evolution
    # -------------------------------------------------------------------------
    print("Plotting individual cosine similarities ...")
    # Group by side (left / right) and eigenvector index
    left_indiv = sorted(k for k in cosine_sim_indiv_keys if k.startswith("left"))
    right_indiv = sorted(k for k in cosine_sim_indiv_keys if k.startswith("right"))

    for key in left_indiv + right_indiv:
        plot_metric_grid(
            grouped, key, batch_sizes, learning_rates,
            output_path=output_dir / "indiv_cosine" / f"{key}.png",
            ylabel="cosine similarity",
            title_prefix=title,
        )

    # Combined: all left individual on one grid, all right individual on one grid
    for side, side_keys in [("left", left_indiv), ("right", right_indiv)]:
        if not side_keys:
            continue
        n_bs = len(batch_sizes)
        n_evec = len(side_keys)
        colors_lr = cm.plasma(np.linspace(0.05, 0.95, len(learning_rates)))
        colors_ev = cm.tab20(np.linspace(0, 1, n_evec))

        # One subplot per batch size, multiple lines per eigenvector per LR
        # (too dense if n_evec * n_lr lines — use one figure per LR instead)
        for lr_color, lr in zip(colors_lr, learning_rates):
            fig, axes = plt.subplots(n_bs, 1, figsize=(12, 2.8 * n_bs), sharex=False)
            if n_bs == 1:
                axes = [axes]
            for ax, bs in zip(axes, batch_sizes):
                ax.set_title(f"bs={bs}, lr={lr_label(lr)}", fontsize=9, pad=3)
                ax.set_ylabel("cosine similarity", fontsize=8)
                ax.set_xlabel("gradient step", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, linewidth=0.4, alpha=0.5)
                key_combo = (bs, lr)
                if key_combo not in grouped:
                    continue
                steps, _, metric_dict = grouped[key_combo]
                for ev_color, key in zip(colors_ev, side_keys):
                    if key not in metric_dict:
                        continue
                    mean = metric_dict[key]["mean"]
                    std = metric_dict[key]["std"]
                    mask = ~np.isnan(mean)
                    if mask.sum() == 0:
                        continue
                    idx = key.split("_")[-1]  # e.g., "3"
                    _add_curve(ax, steps[mask], mean[mask], std[mask],
                               label=f"ev{idx}", color=ev_color)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=6,
                              ncol=min(4, n_evec), loc="lower right", framealpha=0.7)
            fig.suptitle(f"{title} — {side} eigenvectors, lr={lr_label(lr)}", fontsize=11, y=1.002)
            plt.tight_layout()
            savefig(fig, output_dir / "indiv_cosine" / f"{side}_all_ev_lr{lr_label(lr)}.png")

    # -------------------------------------------------------------------------
    # 5c. Other metrics
    # -------------------------------------------------------------------------
    print("Plotting other metrics ...")
    for key in other_keys:
        plot_metric_grid(
            grouped, key, batch_sizes, learning_rates,
            output_path=output_dir / "other_metrics" / f"{key}.png",
            ylabel=key,
            title_prefix=title,
        )

    # -------------------------------------------------------------------------
    # 6. Summary heatmap: final average cosine similarity
    # -------------------------------------------------------------------------
    print("Plotting summary heatmaps ...")
    for avg_key in cosine_sim_avg_keys:
        fig, ax = plt.subplots(figsize=(len(learning_rates) * 1.1 + 1, len(batch_sizes) * 0.8 + 1))

        heat = np.full((len(batch_sizes), len(learning_rates)), np.nan)
        for i, bs in enumerate(batch_sizes):
            for j, lr in enumerate(learning_rates):
                combo = (bs, lr)
                if combo not in grouped:
                    continue
                _, _, metric_dict = grouped[combo]
                if avg_key not in metric_dict:
                    continue
                mean = metric_dict[avg_key]["mean"]
                # Use last non-NaN value
                valid = ~np.isnan(mean)
                if valid.any():
                    heat[i, j] = mean[valid][-1]

        im = ax.imshow(heat, aspect="auto", vmin=0, vmax=1,
                       cmap="viridis", origin="upper")
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([lr_label(lr) for lr in learning_rates], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(batch_sizes)))
        ax.set_yticklabels([str(bs) for bs in batch_sizes], fontsize=8)
        ax.set_xlabel("learning rate", fontsize=10)
        ax.set_ylabel("batch size", fontsize=10)
        ax.set_title(f"{title}\nFinal {avg_key}", fontsize=11)
        plt.colorbar(im, ax=ax, label="cosine similarity")

        # Annotate cells with values
        for i in range(len(batch_sizes)):
            for j in range(len(learning_rates)):
                val = heat[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        plt.tight_layout()
        savefig(fig, output_dir / f"heatmap_{avg_key}.png")

    # -------------------------------------------------------------------------
    # 7. Save a summary JSON
    # -------------------------------------------------------------------------
    summary = {}
    for (bs, lr), (steps, n_seeds, metric_dict) in grouped.items():
        k = f"bs{bs}_lr{lr_label(lr)}"
        summary[k] = {"n_seeds": int(n_seeds), "n_steps": int(len(steps))}
        for avg_key in cosine_sim_avg_keys:
            if avg_key in metric_dict:
                mean = metric_dict[avg_key]["mean"]
                valid = ~np.isnan(mean)
                if valid.any():
                    summary[k][f"final_{avg_key}"] = float(mean[valid][-1])

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")

    print(f"\nAll plots saved under {output_dir}")


if __name__ == "__main__":
    main()
