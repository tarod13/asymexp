#!/usr/bin/env python3
"""
Visualize how ground-truth right eigenvectors evolve as the wind parameter
changes across the 31-run wind sweep (sweep_wind_allo.sh).

Each task run saves its results under
    <results_dir>/task_<id>/file/<run_name>/
        gt_right_real.npy   (num_canonical_states, num_eigvecs)
        gt_right_imag.npy
        gt_eigenvalues_real.npy
        gt_eigenvalues_imag.npy
        args.json           contains 'wind'
        viz_metadata.pkl    contains 'canonical_states', 'grid_width', 'grid_height'

The figure has one row per eigenvector (skipping k=0, the stationary mode):
  - Left panel  : spaghetti plot — eigenvector component at each state vs wind.
                  Lines coloured by the state's column (x) position.
  - Right panels: heatmaps on the grid at a few representative wind values.

Usage
-----
    python scripts/plot_wind_sweep.py
    python scripts/plot_wind_sweep.py --results_dir ./results/wind_sweep \\
                                      --output wind_eigvecs.png \\
                                      --num_eigvecs 4 \\
                                      --heatmap_winds -0.99 -0.5 0 0.5 0.99
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ──────────────────────────────────────────────────────────────

def find_run_dir(task_dir: Path) -> Path | None:
    """Return the most-recently modified run directory under task_dir/file/."""
    file_dir = task_dir / "file"
    if not file_dir.exists():
        return None
    candidates = [d for d in file_dir.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def load_run(run_dir: Path) -> dict | None:
    """Load eigenvectors, eigenvalues, wind value, and grid metadata."""
    try:
        with open(run_dir / "args.json") as f:
            args = json.load(f)

        right_real = np.load(run_dir / "gt_right_real.npy")   # (S, K)
        right_imag = np.load(run_dir / "gt_right_imag.npy")
        eig_real   = np.load(run_dir / "gt_eigenvalues_real.npy")
        eig_imag   = np.load(run_dir / "gt_eigenvalues_imag.npy")

        with open(run_dir / "viz_metadata.pkl", "rb") as f:
            meta = pickle.load(f)

        return dict(
            wind             = float(args["wind"]),
            right_real       = right_real,
            right_imag       = right_imag,
            eig_real         = eig_real,
            eig_imag         = eig_imag,
            canonical_states = np.array(meta["canonical_states"]),
            grid_width       = int(meta["grid_width"]),
            grid_height      = int(meta["grid_height"]),
        )
    except Exception as exc:
        print(f"  Warning: could not load {run_dir}: {exc}")
        return None


# ── Eigenvector utilities ─────────────────────────────────────────────────────

def align_phase(right_real: np.ndarray, right_imag: np.ndarray) -> tuple:
    """
    Remove the arbitrary U(1) phase from each column of the (complex) eigenvector
    matrix: rotate so the largest-magnitude component is real and positive.
    """
    K = right_real.shape[1]
    out_real = right_real.copy()
    out_imag = right_imag.copy()
    for k in range(K):
        cplx    = right_real[:, k] + 1j * right_imag[:, k]
        max_idx = np.argmax(np.abs(cplx))
        phase   = np.angle(cplx[max_idx])
        rotated = cplx * np.exp(-1j * phase)
        out_real[:, k] = rotated.real
        out_imag[:, k] = rotated.imag
    return out_real, out_imag


def eigvec_to_grid(values: np.ndarray,
                   canonical_states: np.ndarray,
                   width: int, height: int) -> np.ndarray:
    """Place per-state eigenvector values on a (height, width) grid (NaN = wall)."""
    grid = np.full((height, width), np.nan)
    for i, s in enumerate(canonical_states):
        x = int(s) % width
        y = int(s) // width
        grid[y, x] = values[i]
    return grid


# ── Plotting ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise eigenvector evolution across a wind sweep."
    )
    p.add_argument("--results_dir", default="./results/wind_sweep",
                   help="Directory that contains task_0/, task_1/, …")
    p.add_argument("--output", default="./wind_sweep_eigenvectors.png",
                   help="Output figure path")
    p.add_argument("--num_eigvecs", type=int, default=4,
                   help="Number of non-trivial eigenvectors to show (rows). "
                        "Eigenvector k=0 (stationary) is shown as a bonus row.")
    p.add_argument("--component", choices=["real", "imag", "abs"], default="real",
                   help="Component of the right eigenvector to display")
    p.add_argument("--heatmap_winds", type=float, nargs="+",
                   default=[-0.99, -0.5, 0.0, 0.5, 0.99],
                   help="Wind values for which to draw grid heatmaps (right panels). "
                        "Nearest available wind in the sweep is used.")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir   = Path(args.results_dir)
    num_eigvecs   = args.num_eigvecs   # non-trivial eigenvectors shown (k=1…)
    component     = args.component
    heatmap_winds = sorted(set(args.heatmap_winds))

    # ── Collect runs ──────────────────────────────────────────────────────────
    print(f"Scanning {results_dir} for task directories…")
    runs = []
    task_dirs = sorted(
        results_dir.glob("task_*"),
        key=lambda p: int(p.name.split("_")[1])
    )
    for task_dir in task_dirs:
        run_dir = find_run_dir(task_dir)
        if run_dir is None:
            print(f"  {task_dir.name}: no run found, skipping")
            continue
        data = load_run(run_dir)
        if data is None:
            continue
        runs.append(data)
        print(f"  {task_dir.name}: wind={data['wind']:+.4f}  "
              f"shape={data['right_real'].shape}")

    if not runs:
        print("No runs found. Exiting.")
        return

    runs.sort(key=lambda r: r["wind"])
    wind_values = np.array([r["wind"] for r in runs])
    n_wind      = len(wind_values)

    # Grid metadata from the first run (same for all)
    canonical_states = runs[0]["canonical_states"]
    grid_width       = runs[0]["grid_width"]
    grid_height      = runs[0]["grid_height"]
    num_states       = len(canonical_states)

    # ── Phase-align eigenvectors ──────────────────────────────────────────────
    # Shape: (n_wind, num_states, K)
    all_real = np.stack([r["right_real"] for r in runs], axis=0)
    all_imag = np.stack([r["right_imag"] for r in runs], axis=0)
    all_eig_real = np.stack([r["eig_real"] for r in runs], axis=0)
    all_eig_imag = np.stack([r["eig_imag"] for r in runs], axis=0)

    for w in range(n_wind):
        all_real[w], all_imag[w] = align_phase(all_real[w], all_imag[w])

    # Select display component
    if component == "real":
        display        = all_real
        comp_label     = "Re"
    elif component == "imag":
        display        = all_imag
        comp_label     = "Im"
    else:
        display        = np.sqrt(all_real ** 2 + all_imag ** 2)
        comp_label     = "|·|"

    # ── Select heatmap wind indices ───────────────────────────────────────────
    hm_indices = []
    for w_target in heatmap_winds:
        idx = int(np.argmin(np.abs(wind_values - w_target)))
        if idx not in hm_indices:
            hm_indices.append(idx)
    hm_indices.sort()
    n_hm = len(hm_indices)

    # ── Eigenvectors to plot: k=0 (stationary) + k=1…num_eigvecs ─────────────
    eig_indices = list(range(0, num_eigvecs + 1))  # includes stationary (k=0)
    n_rows      = len(eig_indices)

    # ── Build figure ─────────────────────────────────────────────────────────
    # Each row: [wide spaghetti | n_hm small heatmaps]
    spag_w  = 5.0   # inches — spaghetti column
    hm_w    = 1.0   # inches — each heatmap column
    row_h   = 2.0   # inches — each eigenvector row
    cb_w    = 0.15  # inches — state-colour bar on the far left

    fig_w = cb_w + 0.1 + spag_w + n_hm * hm_w + 0.7   # 0.7 for heatmap colorbar
    fig_h = n_rows * row_h + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Ground-truth right eigenvectors vs wind  [{comp_label}(ψ_k)]",
        fontsize=12, y=0.99
    )

    # Outer GridSpec: state-colour bar | main area
    outer = gridspec.GridSpec(
        1, 2,
        width_ratios=[cb_w, spag_w + n_hm * hm_w + 0.7],
        wspace=0.05,
        left=0.02, right=0.98, top=0.94, bottom=0.07
    )

    # Inner GridSpec: rows=eigenvectors, cols=spaghetti + heatmaps
    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1 + n_hm,
        subplot_spec=outer[1],
        width_ratios=[spag_w] + [hm_w] * n_hm,
        hspace=0.55,
        wspace=0.15,
    )

    # State colours: x-position → RdYlBu colourmap
    x_pos       = np.array([int(s) % grid_width for s in canonical_states])
    state_norm  = mcolors.Normalize(vmin=0, vmax=grid_width - 1)
    state_cmap  = cm.RdYlBu_r
    state_colors = state_cmap(state_norm(x_pos))

    for row, k in enumerate(eig_indices):
        eig_vals = display[:, :, k]   # (n_wind, num_states)
        label    = f"ψ₀  (stationary)" if k == 0 else f"ψ{k}"

        # ── Spaghetti ────────────────────────────────────────────────────────
        ax_sp = fig.add_subplot(inner[row, 0])
        for si in range(num_states):
            ax_sp.plot(
                wind_values, eig_vals[:, si],
                color=state_colors[si],
                linewidth=0.9, alpha=0.85,
            )
        ax_sp.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax_sp.set_xlim(wind_values[0], wind_values[-1])
        ax_sp.set_ylabel(f"{comp_label}({label})", fontsize=8)
        if row == n_rows - 1:
            ax_sp.set_xlabel("wind", fontsize=9)
        else:
            ax_sp.set_xticklabels([])
        ax_sp.tick_params(labelsize=7)
        ax_sp.grid(True, alpha=0.2)

        # ── Heatmaps ─────────────────────────────────────────────────────────
        vabs = max(np.nanmax(np.abs(eig_vals)), 1e-9)
        hm_norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)

        for col, w_idx in enumerate(hm_indices):
            ax_hm = fig.add_subplot(inner[row, 1 + col])
            grid_img = eigvec_to_grid(
                eig_vals[w_idx], canonical_states, grid_width, grid_height
            )
            im = ax_hm.imshow(
                grid_img,
                cmap="RdBu_r", norm=hm_norm,
                origin="upper", aspect="equal",
                interpolation="nearest",
            )
            ax_hm.set_xticks([])
            ax_hm.set_yticks([])

            # Wind label above first eigenvector row only
            if row == 0:
                ax_hm.set_title(f"w={wind_values[w_idx]:+.2f}", fontsize=7)

            # Colorbar on the rightmost heatmap column
            if col == n_hm - 1:
                cbar = fig.colorbar(im, ax=ax_hm, fraction=0.08, pad=0.06)
                cbar.ax.tick_params(labelsize=5)

    # ── State-colour bar (far-left narrow axes) ───────────────────────────────
    cbar_ax = fig.add_subplot(outer[0])
    cbar_ax.set_axis_off()
    sm = cm.ScalarMappable(cmap=state_cmap, norm=state_norm)
    sm.set_array([])
    cb = fig.colorbar(
        sm, ax=cbar_ax,
        fraction=1.0, pad=0.0,
        orientation="vertical",
    )
    cb.set_label("column (x)", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([0, grid_width - 1])
    cb.set_ticklabels(["0 (left)", f"{grid_width-1} (right)"])

    # ── Save ─────────────────────────────────────────────────────────────────
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
