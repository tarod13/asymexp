#!/usr/bin/env python3
"""
Combined eigenvector visualization across all environments.

Reads training results produced by scripts/eval_parallel_train.sh and generates
a set of large figures where:
  - Rows  = eigenvector indices  (0 … num_eigenvector_pairs-1)
  - Columns = environments        (one per trained run)

Two sets of 8 figures are produced (learned + ground-truth), one per component:
  right-real, right-imag, left-real, left-imag,
  right-magnitude, right-phase, left-magnitude, left-phase

Usage:
  python plot_parallel_eval.py --manifest-dir ./results/eval_manifest \
                               --output-dir   ./results/parallel_eval_plots

  # Subset of environments (useful for quick checks):
  python plot_parallel_eval.py --manifest-dir ./results/eval_manifest \
                               --output-dir   ./results/parallel_eval_plots \
                               --envs GridRoom-4-Doors GridRoom-1

The manifest directory must contain one *.txt file per environment, each
holding the absolute/relative path to the corresponding results directory
(written automatically by eval_parallel_train.sh).
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Make sure the repo root is on the path
sys.path.append(str(Path(__file__).parent))

from src.utils.plotting import visualize_eigenvector_on_grid, visualize_hitting_time_on_grid
from src.utils.metrics import compute_hitting_times_from_eigenvectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env_data(results_dir: Path) -> dict:
    """Load ground-truth and final learned eigenvectors plus grid metadata."""
    results_dir = Path(results_dir)

    with open(results_dir / "viz_metadata.pkl", "rb") as f:
        viz_meta = pickle.load(f)

    def npy(name):
        return np.load(results_dir / name)

    data = dict(
        viz_meta=viz_meta,
        # Ground truth eigenvectors
        gt_right_real=npy("gt_right_real.npy"),   # (num_states, K)
        gt_right_imag=npy("gt_right_imag.npy"),
        gt_left_real=npy("gt_left_real.npy"),
        gt_left_imag=npy("gt_left_imag.npy"),
        # Ground truth eigenvalues
        gt_eigenvalues_real=npy("gt_eigenvalues_real.npy"),
        gt_eigenvalues_imag=npy("gt_eigenvalues_imag.npy"),
    )

    # Learned (final checkpoint) eigenvectors
    for key in ("final_learned_right_real", "final_learned_right_imag",
                "final_learned_left_real",  "final_learned_left_imag"):
        path = results_dir / f"{key}.npy"
        if path.exists():
            data[key.replace("final_learned_", "learned_")] = np.load(path)
        else:
            print(f"  Warning: {path} not found; using zeros.")
            data[key.replace("final_learned_", "learned_")] = np.zeros_like(
                data["gt_right_real"])

    # Learned eigenvalues — read from the saved model (params['lambda_real/imag'])
    model_path = results_dir / "models" / "final_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        data["learned_eigenvalues_real"] = np.array(model["params"]["lambda_real"]).squeeze()
        data["learned_eigenvalues_imag"] = np.array(model["params"]["lambda_imag"]).squeeze()
    else:
        print(f"  Warning: {model_path} not found; learned hitting times will be skipped.")
        data["learned_eigenvalues_real"] = None
        data["learned_eigenvalues_imag"] = None

    return data


def get_component(data: dict, side: str, component: str, source: str) -> np.ndarray:
    """Return eigenvector matrix (num_states, K) for the requested component.

    Parameters
    ----------
    side      : 'left' or 'right'
    component : 'real' | 'imag' | 'magnitude' | 'phase'
    source    : 'gt' | 'learned'
    """
    prefix = "gt" if source == "gt" else "learned"
    real = data[f"{prefix}_{side}_real"]
    imag = data[f"{prefix}_{side}_imag"]

    if component == "real":
        return real
    elif component == "imag":
        return imag
    elif component == "magnitude":
        return np.sqrt(real ** 2 + imag ** 2)
    elif component == "phase":
        return np.arctan2(imag, real)
    else:
        raise ValueError(f"Unknown component: {component!r}")


def cmap_for(component: str) -> str:
    if component == "magnitude":
        return "viridis"
    elif component == "phase":
        return "twilight"
    else:
        return "RdBu_r"


def vrange_for(component: str, values: np.ndarray):
    """Return (vmin, vmax) appropriate for the component type."""
    if component == "phase":
        return -np.pi, np.pi
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = np.max(np.abs(finite))
    if component == "magnitude":
        return 0.0, max(vmax, 1e-6)
    # real / imag — symmetric
    vmax = max(vmax, 1e-6)
    return -vmax, vmax


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

COMPONENTS = [
    ("right", "real"),
    ("right", "imag"),
    ("left",  "real"),
    ("left",  "imag"),
    ("right", "magnitude"),
    ("right", "phase"),
    ("left",  "magnitude"),
    ("left",  "phase"),
]

SOURCES = ["learned", "gt"]


def make_component_figure(
    env_names: list,
    all_data: dict,
    side: str,
    component: str,
    source: str,
    num_eigs: int,
    subplot_size: float = 2.2,
) -> plt.Figure:
    """Create a figure with rows=eigenvectors and columns=environments.

    Parameters
    ----------
    env_names  : ordered list of environment names
    all_data   : {env_name: data_dict}
    side       : 'left' or 'right'
    component  : 'real' | 'imag' | 'magnitude' | 'phase'
    source     : 'gt' or 'learned'
    num_eigs   : number of eigenvector pairs to visualise
    subplot_size: approximate size (inches) for each subplot cell
    """
    nrows = num_eigs
    ncols = len(env_names)

    fig_w = subplot_size * ncols + 1.5   # extra space for colorbar
    fig_h = subplot_size * nrows + 0.6   # extra space for column headers

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Reserve right-hand column for a shared colour bar
    gs = gridspec.GridSpec(
        nrows, ncols + 1,
        figure=fig,
        width_ratios=[1.0] * ncols + [0.05],
        hspace=0.08,
        wspace=0.06,
        left=0.04,
        right=0.93,
        top=0.93,
        bottom=0.03,
    )

    source_label = "Ground Truth" if source == "gt" else "Learned"
    fig.suptitle(
        f"{source_label} — {side.capitalize()} eigenvectors — {component}",
        fontsize=11,
        y=0.98,
    )

    cmap = cmap_for(component)

    # Collect global value range across all environments and eigenvectors
    all_values = []
    for env_name in env_names:
        mat = get_component(all_data[env_name], side, component, source)
        all_values.append(mat[:, :num_eigs].ravel())
    all_values = np.concatenate(all_values)
    vmin, vmax = vrange_for(component, all_values)

    for col, env_name in enumerate(env_names):
        data = all_data[env_name]
        vm = data["viz_meta"]
        mat = get_component(data, side, component, source)  # (num_states, K)
        portals = vm.get("door_markers") or None

        for row in range(nrows):
            ax = fig.add_subplot(gs[row, col])

            eig_vals = mat[:, row] if row < mat.shape[1] else np.zeros(mat.shape[0])

            visualize_eigenvector_on_grid(
                eigenvector_idx=row,
                eigenvector_values=np.array(eig_vals),
                canonical_states=vm["canonical_states"],
                grid_width=vm["grid_width"],
                grid_height=vm["grid_height"],
                portals=portals,
                title=None,
                ax=ax,
                cmap=cmap,
                show_colorbar=False,
                wall_color="gray",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                # Environment name as column header
                short = env_name.replace("GridRoom-", "GR-")
                ax.set_title(short, fontsize=7, pad=2)

            if col == 0:
                ax.set_ylabel(f"Eig {row}", fontsize=7, labelpad=2)

    # Shared colour bar in the last grid column
    cax = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=7)

    return fig


def compute_hitting_times_k(data: dict, source: str, k: int) -> np.ndarray:
    """Compute hitting times using the first *k* eigenvectors (including stationary).

    Parameters
    ----------
    data   : env data dict from load_env_data
    source : 'gt' or 'learned'
    k      : total number of eigenvectors to use (minimum 2)
    """
    vm = data["viz_meta"]
    gamma = vm.get("gamma", 0.9)
    delta = vm.get("delta", 0.0)

    if source == "gt":
        left_real = data["gt_left_real"][:, :k]
        left_imag = data["gt_left_imag"][:, :k]
        right_real = data["gt_right_real"][:, :k]
        right_imag = data["gt_right_imag"][:, :k]
        eig_real = data["gt_eigenvalues_real"][:k]
        eig_imag = data["gt_eigenvalues_imag"][:k]
        eigenvalue_type = "laplacian"
    else:
        left_real = data["learned_left_real"][:, :k]
        left_imag = data["learned_left_imag"][:, :k]
        right_real = data["learned_right_real"][:, :k]
        right_imag = data["learned_right_imag"][:, :k]
        eig_real = data["learned_eigenvalues_real"][:k]
        eig_imag = data["learned_eigenvalues_imag"][:k]
        eigenvalue_type = "kernel"

    import jax.numpy as jnp
    ht = compute_hitting_times_from_eigenvectors(
        left_real=jnp.array(left_real),
        left_imag=jnp.array(left_imag),
        right_real=jnp.array(right_real),
        right_imag=jnp.array(right_imag),
        eigenvalues_real=jnp.array(eig_real),
        eigenvalues_imag=jnp.array(eig_imag),
        gamma=gamma,
        delta=delta,
        eigenvalue_type=eigenvalue_type,
    )
    return np.array(ht)


def make_hitting_time_figure(
    env_names: list,
    all_data: dict,
    source: str,
    num_eigs: int,
    subplot_size: float = 2.2,
) -> plt.Figure:
    """Grid of hitting-time maps: rows = num eigenvectors used, columns = environments.

    Each cell shows hitting times FROM a reference state (centre of canonical states)
    to all reachable states, computed using the first k eigenvectors (k = 2 … num_eigs).

    Parameters
    ----------
    source    : 'gt' or 'learned'
    num_eigs  : maximum number of eigenvectors available (rows use k=2..num_eigs)
    """
    k_values = list(range(2, num_eigs + 1))
    nrows = len(k_values)
    ncols = len(env_names)

    fig_w = subplot_size * ncols * 1.25 + 0.3
    fig_h = subplot_size * nrows + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        hspace=0.15,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.93,
        bottom=0.03,
    )

    source_label = "Ground Truth" if source == "gt" else "Learned"
    fig.suptitle(
        f"{source_label} — Hitting times (FROM centre state) vs. # eigenvectors",
        fontsize=11,
        y=0.98,
    )

    for col, env_name in enumerate(env_names):
        data = all_data[env_name]
        vm = data["viz_meta"]
        canonical_states = vm["canonical_states"]
        portals = vm.get("door_markers") or None
        ref_idx = len(canonical_states) // 2   # centre of canonical state list

        for row, k in enumerate(k_values):
            ax = fig.add_subplot(gs[row, col])
            try:
                ht = compute_hitting_times_k(data, source, k)
                ht_from_ref = ht[ref_idx, :]
            except Exception as e:
                print(f"    WARNING: hitting time failed (env={env_name}, k={k}): {e}")
                ax.set_visible(False)
                continue

            vmin = 0.0
            finite = ht_from_ref[np.isfinite(ht_from_ref)]
            vmax = float(np.max(finite)) if finite.size > 0 else 1.0

            visualize_hitting_time_on_grid(
                hitting_time_values=ht_from_ref,
                center_state_idx=ref_idx,
                canonical_states=canonical_states,
                grid_width=vm["grid_width"],
                grid_height=vm["grid_height"],
                portals=portals,
                title=None,
                ax=ax,
                cmap="viridis",
                show_colorbar=True,
                wall_color="gray",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                short = env_name.replace("GridRoom-", "GR-")
                ax.set_title(short, fontsize=7, pad=2)

            if col == 0:
                ax.set_ylabel(f"k={k}", fontsize=7, labelpad=2)

    return fig


def make_component_figure_independent_cb(
    env_names: list,
    all_data: dict,
    side: str,
    component: str,
    source: str,
    num_eigs: int,
    subplot_size: float = 2.2,
) -> plt.Figure:
    """Like make_component_figure but each subplot has its own independent colorbar."""
    nrows = num_eigs
    ncols = len(env_names)

    # Extra horizontal space per column to accommodate individual colorbars
    fig_w = subplot_size * ncols * 1.25 + 0.3
    fig_h = subplot_size * nrows + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        hspace=0.15,
        wspace=0.35,
        left=0.04,
        right=0.97,
        top=0.93,
        bottom=0.03,
    )

    source_label = "Ground Truth" if source == "gt" else "Learned"
    fig.suptitle(
        f"{source_label} — {side.capitalize()} eigenvectors — {component} (independent colorbars)",
        fontsize=11,
        y=0.98,
    )

    cmap = cmap_for(component)

    for col, env_name in enumerate(env_names):
        data = all_data[env_name]
        vm = data["viz_meta"]
        mat = get_component(data, side, component, source)  # (num_states, K)
        portals = vm.get("door_markers") or None

        for row in range(nrows):
            ax = fig.add_subplot(gs[row, col])

            eig_vals = mat[:, row] if row < mat.shape[1] else np.zeros(mat.shape[0])
            vmin, vmax = vrange_for(component, eig_vals)

            visualize_eigenvector_on_grid(
                eigenvector_idx=row,
                eigenvector_values=np.array(eig_vals),
                canonical_states=vm["canonical_states"],
                grid_width=vm["grid_width"],
                grid_height=vm["grid_height"],
                portals=portals,
                title=None,
                ax=ax,
                cmap=cmap,
                show_colorbar=True,
                wall_color="gray",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                short = env_name.replace("GridRoom-", "GR-")
                ax.set_title(short, fontsize=7, pad=2)

            if col == 0:
                ax.set_ylabel(f"Eig {row}", fontsize=7, labelpad=2)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest-dir",
        default="./results/eval_manifest",
        help="Directory containing one *.txt file per environment with the "
             "path to its results directory.",
    )
    p.add_argument(
        "--output-dir",
        default="./results/parallel_eval_plots",
        help="Directory where output figures are saved.",
    )
    p.add_argument(
        "--envs",
        nargs="*",
        default=None,
        help="Subset of environment names to include (default: all in manifest).",
    )
    p.add_argument(
        "--num-eigs",
        type=int,
        default=None,
        help="Number of eigenvectors to show per plot (default: all available).",
    )
    p.add_argument(
        "--subplot-size",
        type=float,
        default=2.2,
        help="Approximate size (inches) of each subplot cell (default: 2.2).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures (default: 150).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_dir.exists():
        print(f"ERROR: Manifest directory not found: {manifest_dir}")
        sys.exit(1)

    # Collect environments from manifest
    manifest_files = sorted(manifest_dir.glob("*.txt"))
    if not manifest_files:
        print(f"ERROR: No *.txt manifest files found in {manifest_dir}")
        sys.exit(1)

    env_names = []
    results_paths = {}
    for mf in manifest_files:
        env_name = mf.stem
        if args.envs and env_name not in args.envs:
            continue
        results_dir = Path(mf.read_text().strip())
        if not results_dir.exists():
            print(f"WARNING: Results dir not found for {env_name}: {results_dir} — skipping.")
            continue
        env_names.append(env_name)
        results_paths[env_name] = results_dir

    if not env_names:
        print("ERROR: No valid environment results found.")
        sys.exit(1)

    print(f"Found {len(env_names)} environments: {env_names}")

    # Load data for all environments
    print("Loading data…")
    all_data = {}
    for env_name in env_names:
        print(f"  Loading {env_name} from {results_paths[env_name]}")
        try:
            all_data[env_name] = load_env_data(results_paths[env_name])
        except FileNotFoundError as e:
            print(f"  ERROR loading {env_name}: {e} — skipping.")
            env_names.remove(env_name)

    if not all_data:
        print("ERROR: No data could be loaded.")
        sys.exit(1)

    # Determine number of eigenvectors (use minimum across envs so layouts match)
    available_eigs = [all_data[n]["gt_right_real"].shape[1] for n in all_data]
    num_eigs = args.num_eigs or min(available_eigs)
    print(f"Visualising {num_eigs} eigenvectors.")

    REAL_COMPONENTS = [(side, comp) for side, comp in COMPONENTS if comp == "real"]

    # Generate figures
    for source in SOURCES:
        for side, component in COMPONENTS:
            label = f"{source}_{side}_{component}"
            print(f"  Generating: {label}")
            try:
                fig = make_component_figure(
                    env_names=env_names,
                    all_data=all_data,
                    side=side,
                    component=component,
                    source=source,
                    num_eigs=num_eigs,
                    subplot_size=args.subplot_size,
                )
                out_path = output_dir / f"{label}.png"
                fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"    Saved → {out_path}")
            except Exception as e:
                print(f"    ERROR generating {label}: {e}")
                plt.close("all")

        # Additional plots with independent colorbars for real parts only
        for side, component in REAL_COMPONENTS:
            label = f"{source}_{side}_{component}_indep_cb"
            print(f"  Generating: {label}")
            try:
                fig = make_component_figure_independent_cb(
                    env_names=env_names,
                    all_data=all_data,
                    side=side,
                    component=component,
                    source=source,
                    num_eigs=num_eigs,
                    subplot_size=args.subplot_size,
                )
                out_path = output_dir / f"{label}.png"
                fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"    Saved → {out_path}")
            except Exception as e:
                print(f"    ERROR generating {label}: {e}")
                plt.close("all")

    # Hitting time figures
    for source in SOURCES:
        # Skip learned if eigenvalues were not saved
        if source == "learned":
            missing = [n for n in all_data
                       if all_data[n].get("learned_eigenvalues_real") is None]
            if missing:
                print(f"  Skipping learned hitting times (eigenvalues missing for: {missing})")
                continue

        label = f"hitting_times_{source}"
        print(f"  Generating: {label}")
        try:
            fig = make_hitting_time_figure(
                env_names=env_names,
                all_data=all_data,
                source=source,
                num_eigs=num_eigs,
                subplot_size=args.subplot_size,
            )
            out_path = output_dir / f"{label}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved → {out_path}")
        except Exception as e:
            print(f"    ERROR generating {label}: {e}")
            plt.close("all")

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
