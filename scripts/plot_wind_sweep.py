#!/usr/bin/env python3
"""
Visualize how ground-truth right eigenvectors evolve as the wind parameter
changes across the 31-run wind sweep (sweep_wind_allo.sh).

Each task run saves its results under
    <results_dir>/task_<id>/file/<run_name>/
        gt_right_real.npy   (num_canonical_states, num_eigvecs)
        gt_right_imag.npy
        args.json           contains 'wind'
        viz_metadata.pkl    contains 'canonical_states', 'grid_width', 'grid_height'

The figure is a grid of heatmaps:
    rows    = eigenvectors (k=0 stationary, then k=1…num_eigvecs)
    columns = all available wind values (one per run found)

Usage
-----
    python scripts/plot_wind_sweep.py
    python scripts/plot_wind_sweep.py --results_dir ./results/wind_sweep \\
                                      --output wind_eigvecs.png \\
                                      --num_eigvecs 4 \\
                                      --component real
"""

import argparse
import json
import pickle
from pathlib import Path

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
    try:
        with open(run_dir / "args.json") as f:
            args = json.load(f)
        right_real = np.load(run_dir / "gt_right_real.npy")   # (S, K)
        right_imag = np.load(run_dir / "gt_right_imag.npy")
        with open(run_dir / "viz_metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return dict(
            wind             = float(args["wind"]),
            right_real       = right_real,
            right_imag       = right_imag,
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
    Rotate each eigenvector column so its largest-magnitude component is
    real and positive, removing the arbitrary U(1) phase from jnp.linalg.eig.
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
    """Place per-state values on a (height, width) grid; walls stay NaN."""
    grid = np.full((height, width), np.nan)
    for i, s in enumerate(canonical_states):
        grid[int(s) // width, int(s) % width] = values[i]
    return grid


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid of eigenvector heatmaps across all wind values."
    )
    p.add_argument("--results_dir", default="./results/wind_sweep",
                   help="Directory containing task_0/, task_1/, …")
    p.add_argument("--output", default="./wind_sweep_eigenvectors.png")
    p.add_argument("--num_eigvecs", type=int, default=4,
                   help="Number of non-trivial eigenvectors to show "
                        "(rows after k=0). Total rows = num_eigvecs + 1.")
    p.add_argument("--component", choices=["real", "imag", "abs"], default="real",
                   help="Component of the right eigenvector to display.")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    num_eigvecs = args.num_eigvecs
    component   = args.component

    # ── Load runs ─────────────────────────────────────────────────────────────
    print(f"Scanning {results_dir} …")
    runs = []
    for task_dir in sorted(results_dir.glob("task_*"),
                           key=lambda p: int(p.name.split("_")[1])):
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
    wind_values      = np.array([r["wind"] for r in runs])
    n_wind           = len(wind_values)
    canonical_states = runs[0]["canonical_states"]
    grid_width       = runs[0]["grid_width"]
    grid_height      = runs[0]["grid_height"]

    # ── Phase-align and stack ─────────────────────────────────────────────────
    # Shape: (n_wind, num_states, K)
    all_real = np.stack([r["right_real"] for r in runs], axis=0)
    all_imag = np.stack([r["right_imag"] for r in runs], axis=0)
    for w in range(n_wind):
        all_real[w], all_imag[w] = align_phase(all_real[w], all_imag[w])

    if component == "real":
        display, comp_label = all_real, "Re"
    elif component == "imag":
        display, comp_label = all_imag, "Im"
    else:
        display = np.sqrt(all_real ** 2 + all_imag ** 2)
        comp_label = "|·|"

    # Eigenvector rows: k=0 (stationary) then k=1…num_eigvecs
    eig_indices = list(range(num_eigvecs + 1))
    n_rows      = len(eig_indices)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Each heatmap cell is square (grid_width × grid_height aspect).
    cell_w  = 0.55   # inches per wind column
    cell_h  = cell_w * grid_height / grid_width
    cb_w    = 0.18   # inches for the per-row colorbar
    pad_l   = 0.55   # left margin for row labels
    pad_top = 0.35   # top margin for wind labels
    pad_bot = 0.25

    fig_w = pad_l + n_wind * cell_w + cb_w + 0.15
    fig_h = pad_top + n_rows * cell_h + pad_bot

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Ground-truth right eigenvectors  [{comp_label}(ψ_k)]  ×  wind",
        fontsize=10, y=1.0
    )

    gs = gridspec.GridSpec(
        n_rows, n_wind + 1,                          # +1 for colorbar column
        width_ratios=[1.0] * n_wind + [cb_w / cell_w],
        hspace=0.08,
        wspace=0.04,
        left=pad_l / fig_w,
        right=1.0 - 0.08 / fig_w,
        top=1.0 - pad_top / fig_h,
        bottom=pad_bot / fig_h,
    )

    for row, k in enumerate(eig_indices):
        eig_data = display[:, :, k]   # (n_wind, num_states)

        # Shared symmetric color scale across all wind values for this row
        vabs    = max(float(np.nanmax(np.abs(eig_data))), 1e-9)
        hm_norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)

        row_label = "ψ₀ (stationary)" if k == 0 else f"ψ{k}"

        for col, w_idx in enumerate(range(n_wind)):
            ax = fig.add_subplot(gs[row, col])
            grid_img = eigvec_to_grid(
                eig_data[w_idx], canonical_states, grid_width, grid_height
            )
            ax.imshow(
                grid_img,
                cmap="RdBu_r", norm=hm_norm,
                origin="upper", aspect="equal",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Wind value labels on first row only
            if row == 0:
                ax.set_title(f"{wind_values[w_idx]:+.2f}",
                             fontsize=4.5, pad=2)

        # Row label on the left of each row
        fig.text(
            pad_l / fig_w - 0.005, 1.0 - (pad_top + (row + 0.5) * cell_h) / fig_h,
            row_label,
            ha="right", va="center", fontsize=8,
            transform=fig.transFigure,
        )

        # Colorbar in the dedicated last column
        ax_cb = fig.add_subplot(gs[row, n_wind])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=hm_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cb)
        cbar.ax.tick_params(labelsize=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
