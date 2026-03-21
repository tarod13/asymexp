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

import numpy as np

from src.utils.plotting import align_phase, eigvec_to_grid, make_figure, save_figure


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
