#!/usr/bin/env python3
"""
Visualize how eigenvectors evolve as the wind parameter changes across the
31-run wind sweep (sweep_wind_allo.sh).

Produces seven figures in <output_dir>/:
    gt_right_real.png   – ground-truth right eigenvectors, real part
    gt_right_imag.png   –                                  imaginary part
    gt_right_abs.png    –                                  magnitude
    gt_left_real.png    – ground-truth left  eigenvectors, real part
    gt_left_imag.png    –                                  imaginary part
    gt_left_abs.png     –                                  magnitude
    allo_right_real.png – ALLO learned right eigenvectors, real part
                          (left ≈ right and allo is real-valued, so omitted)

Each task run saves its results under
    <results_dir>/task_<id>/<env_file_name>/<run_name>/
        gt_right_real.npy                        (num_canonical_states, num_eigvecs)
        gt_right_imag.npy
        gt_left_real.npy
        gt_left_imag.npy
        final_learned_right_real_normalized.npy  (or final_learned_right_real.npy)
        args.json           contains 'wind'
        viz_metadata.pkl    contains 'canonical_states', 'grid_width', 'grid_height'

The figure is a grid of heatmaps:
    rows    = eigenvectors (k=0 stationary, then k=1…num_eigvecs)
    columns = all available wind values (one per run found)
    one single colorbar shared across the whole figure

Usage
-----
    python scripts/plot_wind_sweep.py
    python scripts/plot_wind_sweep.py --results_dir ./results/wind_sweep \\
                                      --output_dir  ./figures \\
                                      --num_eigvecs 4
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

def find_run_dir(task_dir: Path, timestamp: int | None = None) -> Path | None:
    """Return a run directory under task_dir/<env_dir>/.

    If *timestamp* is given, return the run whose name ends with that Unix
    timestamp (exact match).  Otherwise return the run with the largest
    embedded timestamp (i.e. the most recently started run).
    """
    candidates = []
    for subdir in task_dir.iterdir():
        if subdir.is_dir():
            candidates.extend(
                d for d in subdir.iterdir()
                if d.is_dir() and (d / "args.json").exists()
            )
    if not candidates:
        return None

    def run_timestamp(d: Path) -> int:
        tail = d.name.rsplit("__", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    if timestamp is not None:
        matches = [d for d in candidates if run_timestamp(d) == timestamp]
        if not matches:
            return None
        return matches[0]

    return max(candidates, key=run_timestamp)


def load_run(run_dir: Path) -> dict | None:
    try:
        with open(run_dir / "args.json") as f:
            args = json.load(f)
        right_real = np.load(run_dir / "gt_right_real.npy")   # (S, K)
        right_imag = np.load(run_dir / "gt_right_imag.npy")
        left_real  = np.load(run_dir / "gt_left_real.npy")
        left_imag  = np.load(run_dir / "gt_left_imag.npy")

        # ALLO learned (right only, real only — normalized preferred)
        allo_path = run_dir / "final_learned_right_real_normalized.npy"
        if not allo_path.exists():
            allo_path = run_dir / "final_learned_right_real.npy"
        allo_right_real = np.load(allo_path) if allo_path.exists() else None

        with open(run_dir / "viz_metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return dict(
            wind            = float(args["wind"]),
            right_real      = right_real,
            right_imag      = right_imag,
            left_real       = left_real,
            left_imag       = left_imag,
            allo_right_real = allo_right_real,
            canonical_states = np.array(meta["canonical_states"]),
            grid_width      = int(meta["grid_width"]),
            grid_height     = int(meta["grid_height"]),
        )
    except Exception as exc:
        print(f"  Warning: could not load {run_dir}: {exc}")
        return None


# ── Eigenvector utilities ─────────────────────────────────────────────────────

def align_phase(real: np.ndarray, imag: np.ndarray) -> tuple:
    """
    Rotate each eigenvector column so its largest-magnitude component is
    real and positive, removing the arbitrary U(1) phase from jnp.linalg.eig.
    """
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


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(display: np.ndarray,
                wind_values: np.ndarray,
                canonical_states: np.ndarray,
                grid_width: int, grid_height: int,
                eig_indices: list,
                title: str) -> plt.Figure:
    """
    Build the heatmap grid figure.

    display: (n_wind, num_states, K)
    """
    n_wind = len(wind_values)
    n_rows = len(eig_indices)

    # Single symmetric colorscale shared across ALL rows and columns
    vabs    = max(float(np.nanmax(np.abs(display[:, :, eig_indices]))), 1e-9)
    hm_norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)

    # Figure dimensions
    cell_w  = 0.55                          # inches per wind column
    cell_h  = cell_w * grid_height / grid_width
    cb_w    = 0.25                          # inches for the single colorbar
    cb_gap  = 0.08                          # gap between heatmaps and colorbar
    pad_l   = 0.50                          # left margin for row labels
    pad_top = 0.35                          # top margin for wind labels
    pad_bot = 0.25

    fig_w = pad_l + n_wind * cell_w + cb_gap + cb_w + 0.10
    fig_h = pad_top + n_rows * cell_h + pad_bot

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=10, y=1.0)

    # GridSpec: n_rows rows × (n_wind + 1) columns; last column = colorbar
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
        eig_data  = display[:, :, k]        # (n_wind, num_states)
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
            # Remove all axis decorations
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Wind value labels on top row only
            if row == 0:
                ax.set_title(f"{wind_values[col]:+.2f}",
                             fontsize=4.5, pad=2)

        # Row label to the left of each row
        fig.text(
            pad_l / fig_w - 0.005,
            1.0 - (pad_top + (row + 0.5) * cell_h) / fig_h,
            row_label,
            ha="right", va="center", fontsize=8,
            transform=fig.transFigure,
        )

    # Single colorbar spanning all rows in the dedicated last column
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
    fig = make_figure(display, wind_values, canonical_states,
                      grid_width, grid_height, eig_indices, title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid of eigenvector heatmaps across all wind values."
    )
    p.add_argument("--results_dir", default="./results/wind_sweep",
                   help="Directory containing task_0/, task_1/, …")
    p.add_argument("--output_dir", default="./results/wind_sweep",
                   help="Directory where all output figures are saved.")
    p.add_argument("--num_eigvecs", type=int, default=4,
                   help="Number of non-trivial eigenvectors to show "
                        "(rows after k=0). Total rows = num_eigvecs + 1.")
    p.add_argument("--timestamp", type=int, default=None,
                   help="Unix timestamp suffix of the run to plot "
                        "(e.g. 1712345678). Defaults to the most recently "
                        "started run in each task directory.")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    num_eigvecs = args.num_eigvecs
    timestamp   = args.timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load runs ─────────────────────────────────────────────────────────────
    print(f"Scanning {results_dir} …"
          + (f"  (timestamp={timestamp})" if timestamp else "  (latest run)"))
    runs = []
    for task_dir in sorted(results_dir.glob("task_*"),
                           key=lambda p: int(p.name.split("_")[1])):
        run_dir = find_run_dir(task_dir, timestamp=timestamp)
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

    eig_indices = list(range(num_eigvecs + 1))

    # ── Ground-truth figures (right and left × real, imag, abs) ───────────────
    for side in ("right", "left"):
        all_real = np.stack([r[f"{side}_real"] for r in runs], axis=0)  # (n_wind, S, K)
        all_imag = np.stack([r[f"{side}_imag"] for r in runs], axis=0)
        for w in range(n_wind):
            all_real[w], all_imag[w] = align_phase(all_real[w], all_imag[w])

        for comp, comp_label, display in [
            ("real", "Re",  all_real),
            ("imag", "Im",  all_imag),
            ("abs",  "|·|", np.sqrt(all_real ** 2 + all_imag ** 2)),
        ]:
            title = (f"Ground-truth {side} eigenvectors  "
                     f"[{comp_label}(ψ_k)]  ×  wind")
            save_figure(display, wind_values, canonical_states,
                        grid_width, grid_height, eig_indices, title,
                        output_dir / f"gt_{side}_{comp}.png")

    # ── ALLO figure (right real only) ─────────────────────────────────────────
    allo_runs = [r for r in runs if r["allo_right_real"] is not None]
    if allo_runs:
        allo_wind    = np.array([r["wind"] for r in allo_runs])
        allo_display = np.stack([r["allo_right_real"] for r in allo_runs], axis=0)
        save_figure(allo_display, allo_wind, canonical_states,
                    grid_width, grid_height, eig_indices,
                    "ALLO learned right eigenvectors  [Re(ψ_k)]  ×  wind",
                    output_dir / "allo_right_real.png")
    else:
        print("No ALLO learned eigenvector files found; skipping allo_right_real.png")


if __name__ == "__main__":
    main()
