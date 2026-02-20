"""
Main script for eigendecomposition analysis of environments with irreversible doors.

This script:
1. Generates transition data for environments with one-way doors
2. Computes eigendecomposition without symmetrization (batched)
3. Computes hitting times
4. Visualizes eigenvectors
5. Visualizes hitting times (Asymmetry analysis)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def find_project_root():
    """Find project root. Tries ASYMEXP_ROOT env var, then searches upward."""
    if "ASYMEXP_ROOT" in os.environ:
        root = Path(os.environ["ASYMEXP_ROOT"])
        if root.is_dir():
            return root

    current = Path.cwd()
    while current != current.parent:
        if ((current / ".git").is_dir() and
            (current / "experiments").is_dir() and
            (current / "src").is_dir()):
            return current
        current = current.parent

    # Fallback: try parent of parent of this script
    script_path = Path(__file__).resolve()
    potential_root = script_path.parent.parent.parent
    if (potential_root / "src").is_dir():
        return potential_root

    return Path.cwd()


# Add project root to path for imports
project_root = find_project_root()
sys.path.insert(0, str(project_root))

from src.utils.laplacian import compute_eigendecomposition_batched
from src.utils.metrics import compute_hitting_times_batched
from src.utils.plotting import (
    create_eigenvector_visualization_report,
    create_hitting_time_visualization_report,
)
from src.utils.envs import get_canonical_free_states
from src.envs.env import create_environment_from_text
from src.data_collection import generate_batched_door_environments


def visualize_doors_on_grid(
    doors: List,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    obstacles: List[Tuple[int, int]],
    output_path: str
):
    """Create a simple visualization showing door locations."""
    fig, ax = plt.subplots(figsize=(10, 10))

    for obs_x, obs_y in obstacles:
        rect = mpatches.Rectangle(
            (obs_x - 0.5, obs_y - 0.5), 1, 1,
            linewidth=0, edgecolor='none', facecolor='gray', alpha=0.7
        )
        ax.add_patch(rect)

    for i in range(grid_height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(grid_width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)

    rect_thickness = 0.15
    rect_width = 0.7
    margin = 0.02

    for idx, (s_canonical, a_forward, s_prime_canonical, a_reverse) in enumerate(doors):
        s_full = int(canonical_states[s_canonical])
        s_y = s_full // grid_width
        s_x = s_full % grid_width

        if a_forward == 0:  # Up
            rect_x, rect_y = s_x - rect_width/2, s_y - 0.5 - rect_thickness/2
            rect_w, rect_h = rect_width, rect_thickness
            triangle_pts = [
                (s_x, s_y - 0.5 - rect_thickness/2 + margin),
                (s_x - rect_thickness/2 + margin, s_y - 0.5 + rect_thickness/2 - margin),
                (s_x + rect_thickness/2 - margin, s_y - 0.5 + rect_thickness/2 - margin)
            ]
        elif a_forward == 1:  # Right
            rect_x, rect_y = s_x + 0.5 - rect_thickness/2, s_y - rect_width/2
            rect_w, rect_h = rect_thickness, rect_width
            triangle_pts = [
                (s_x + 0.5 + rect_thickness/2 - margin, s_y),
                (s_x + 0.5 - rect_thickness/2 + margin, s_y - rect_thickness/2 + margin),
                (s_x + 0.5 - rect_thickness/2 + margin, s_y + rect_thickness/2 - margin)
            ]
        elif a_forward == 2:  # Down
            rect_x, rect_y = s_x - rect_width/2, s_y + 0.5 - rect_thickness/2
            rect_w, rect_h = rect_width, rect_thickness
            triangle_pts = [
                (s_x, s_y + 0.5 + rect_thickness/2 - margin),
                (s_x - rect_thickness/2 + margin, s_y + 0.5 - rect_thickness/2 + margin),
                (s_x + rect_thickness/2 - margin, s_y + 0.5 - rect_thickness/2 + margin)
            ]
        else:  # Left
            rect_x, rect_y = s_x - 0.5 - rect_thickness/2, s_y - rect_width/2
            rect_w, rect_h = rect_thickness, rect_width
            triangle_pts = [
                (s_x - 0.5 - rect_thickness/2 + margin, s_y),
                (s_x - 0.5 + rect_thickness/2 - margin, s_y - rect_thickness/2 + margin),
                (s_x - 0.5 + rect_thickness/2 - margin, s_y + rect_thickness/2 - margin)
            ]

        ax.add_patch(mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=0, edgecolor='none', facecolor='black'))
        ax.add_patch(mpatches.Polygon(triangle_pts, facecolor='white', edgecolor='none'))

    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(grid_height - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Irreversible Doors ({len(doors)} total)\nBlack rectangles with arrows show one-way passages', fontsize=14)
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved door visualization to {output_path}")


def run_door_analysis(
    batched_transition_counts: jnp.ndarray,
    metadata: Dict,
    k: int = 20,
    output_dir: str = "experiments/02/results",
    save_results: bool = True,
    num_eigenvectors_to_visualize: int = 6,
    num_targets_to_visualize: int = 6,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    wall_color: str = 'gray',
    log_scale: bool = False
) -> Dict:
    """Run eigendecomposition analysis on door environments."""
    num_envs = batched_transition_counts.shape[0]
    num_canonical_states = batched_transition_counts.shape[1]
    grid_width = metadata.get("grid_width", 13)
    canonical_states = metadata.get("canonical_states", jnp.arange(num_canonical_states))

    print("\n" + "=" * 80)
    print("EIGENDECOMPOSITION ANALYSIS - IRREVERSIBLE DOOR ENVIRONMENTS")
    print("=" * 80)
    print(f"Number of environments: {num_envs}")
    print(f"Canonical states per environment: {num_canonical_states}")
    print(f"Doors per environment: {metadata.get('num_doors_per_env', 'unknown')}")

    # [1/5] Compute batched eigendecomposition
    print("\n[1/5] Computing batched eigendecomposition...")
    batched_eigendecomp = compute_eigendecomposition_batched(
        batched_transition_counts,
        k=k,
        smoothing=1e-5,
        normalize=True,
        sort_by_magnitude=True
    )
    print(f"  Computed {k} eigenvalues/vectors for {num_envs} environments")

    # [2/5] Analyze eigenvalue spectrum
    print("\n[2/5] Analyzing eigenvalue spectrum...")
    eigenvalues_imag = batched_eigendecomp["eigenvalues_imag"]
    num_real_per_env = jnp.sum(jnp.abs(eigenvalues_imag) < 1e-8, axis=1)

    print(f"  Real eigenvalues per env: mean={jnp.mean(num_real_per_env):.1f}")
    print(f"  Complex eigenvalues per env: mean={k - jnp.mean(num_real_per_env):.1f}")

    # [3/5] Compute hitting times
    print("\n[3/5] Computing hitting times for all environments...")
    batched_hitting_times, batched_hitting_times_imag_error = compute_hitting_times_batched(batched_eigendecomp)
    print(f"  Hitting times imaginary error: mean={jnp.mean(batched_hitting_times_imag_error):.6e}")

    # Compile results
    first_env_eigendecomp = {key: value[0] for key, value in batched_eigendecomp.items()}
    first_env_hitting_times = batched_hitting_times[0]

    results = {
        "batched_eigendecomposition": batched_eigendecomp,
        "batched_hitting_times": batched_hitting_times,
        "batched_hitting_times_imaginary_error": batched_hitting_times_imag_error,
        "metadata": metadata,
        "parameters": {"k": k, "num_envs": num_envs}
    }

    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "door_eigendecomposition_results.pkl"
        print(f"\nSaving results to {results_file}...")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

        # Visualize doors
        if "first_env_doors" in metadata and len(metadata["first_env_doors"]) > 0:
            visualize_doors_on_grid(
                doors=metadata["first_env_doors"],
                canonical_states=canonical_states,
                grid_width=grid_width,
                grid_height=metadata.get("grid_height", grid_width),
                obstacles=metadata.get("obstacles", []),
                output_path=str(output_path / "door_locations.png")
            )

    # Convert doors for visualization
    door_markers = {}
    if "first_env_doors" in metadata:
        for s_canonical, a, s_prime_canonical, a_reverse in metadata["first_env_doors"]:
            s_full = int(canonical_states[s_canonical])
            s_prime_full = int(canonical_states[s_prime_canonical])
            door_markers[(s_full, a)] = s_prime_full

    # [4/5] Visualize eigenvectors
    print("\n[4/5] Visualizing eigenvectors (using first environment)...")
    if save_results:
        grid_height = metadata.get("grid_height", grid_width)
        create_eigenvector_visualization_report(
            eigendecomposition=first_env_eigendecomp,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=door_markers,
            output_dir=str(output_path / "visualizations"),
            num_eigenvectors=num_eigenvectors_to_visualize,
            nrows=nrows,
            ncols=ncols,
            wall_color=wall_color
        )

    # [5/5] Visualize hitting times (Asymmetry)
    print("\n[5/5] Visualizing hitting times (Asymmetry analysis)...")
    if save_results:
        ht_ncols = ncols if ncols is not None else min(5, num_targets_to_visualize)

        create_hitting_time_visualization_report(
            hitting_time_matrix=first_env_hitting_times,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=door_markers,
            output_dir=str(output_path / "visualizations"),
            num_targets=num_targets_to_visualize,
            ncols=ht_ncols,
            wall_color=wall_color,
            log_scale=log_scale
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Eigendecomposition analysis of environments with irreversible doors")
    parser.add_argument("--base-env", type=str, default="GridRoom-4", help="Base environment name")
    parser.add_argument("--num-envs", type=int, default=10, help="Number of door environments")
    parser.add_argument("--num-doors", type=int, default=5, help="Number of doors per environment")
    parser.add_argument("--num-rollouts", type=int, default=100, help="Number of rollouts per environment")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of steps per rollout")
    parser.add_argument("--k", type=int, default=20, help="Number of eigenvectors to compute")
    parser.add_argument("--output-dir", type=str, default="experiments/02/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nrows", type=int, default=None, help="Rows for eigenvector grid")
    parser.add_argument("--ncols", type=int, default=None, help="Columns for eigenvector grid")
    parser.add_argument("--wall-color", type=str, default="gray", help="Color for walls")
    parser.add_argument("--num-eigenvectors", type=int, default=6, help="Eigenvectors to visualize")
    parser.add_argument("--num-targets", type=int, default=6, help="Number of target states for hitting times")
    parser.add_argument("--log-scale", action="store_true", help="Use logarithmic scale for hitting times")

    args = parser.parse_args()

    # Load base environment
    print(f"Loading base environment: {args.base_env}")
    base_env = create_environment_from_text(file_name=args.base_env, max_steps=args.num_steps)
    canonical_states = get_canonical_free_states(base_env)

    print(f"  Grid size: {base_env.width}x{base_env.height}")
    print(f"  Free states: {len(canonical_states)}")

    # Generate door environments
    print(f"\nGenerating {args.num_envs} environments with {args.num_doors} doors each...")
    batched_counts, metadata = generate_batched_door_environments(
        base_env=base_env,
        canonical_states=canonical_states,
        num_envs=args.num_envs,
        num_doors=args.num_doors,
        num_rollouts_per_env=args.num_rollouts,
        num_steps=args.num_steps,
        seed=args.seed
    )

    # Run analysis
    results = run_door_analysis(
        batched_transition_counts=batched_counts,
        metadata=metadata,
        k=args.k,
        output_dir=args.output_dir,
        save_results=True,
        num_eigenvectors_to_visualize=args.num_eigenvectors,
        num_targets_to_visualize=args.num_targets,
        nrows=args.nrows,
        ncols=args.ncols,
        wall_color=args.wall_color,
        log_scale=args.log_scale
    )

    return results


if __name__ == "__main__":
    main()
