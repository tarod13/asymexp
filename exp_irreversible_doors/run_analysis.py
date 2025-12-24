"""
Main script for eigendecomposition analysis of environments with irreversible doors.

This script:
1. Generates transition data for environments with one-way doors
2. Computes eigendecomposition without symmetrization (batched)
3. Visualizes left and right eigenvectors
4. Saves results and generates visualizations
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from exp_complex_basis.eigendecomposition import (
    compute_eigendecomposition_batched,
)
from exp_complex_basis.eigenvector_visualization import (
    create_eigenvector_visualization_report,
)
from exp_irreversible_doors.door_environment import (
    generate_batched_door_environments,
)
from src.envs.env import create_environment_from_text


def get_canonical_free_states(base_env):
    """
    Get the canonical set of free (non-obstacle) states from the base environment.

    Args:
        base_env: Base GridWorld environment

    Returns:
        canonical_states: Array of free state indices, sorted
    """
    width = base_env.width
    height = base_env.height

    # Get all state indices
    all_states = set(range(width * height))

    # Get obstacle state indices
    obstacle_states = set()
    if base_env.has_obstacles:
        for obs in base_env.obstacles:
            obs_x, obs_y = int(obs[0]), int(obs[1])
            if 0 <= obs_x < width and 0 <= obs_y < height:
                state_idx = obs_y * width + obs_x
                obstacle_states.add(state_idx)

    # Free states = all states - obstacles
    free_states = sorted(list(all_states - obstacle_states))

    return jnp.array(free_states, dtype=jnp.int32)


def visualize_doors_on_grid(
    doors: List,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    output_path: str
):
    """
    Create a simple visualization showing door locations.

    Args:
        doors: List of door tuples (s, a, s', a_reverse)
        canonical_states: Canonical state mapping
        grid_width: Grid width
        grid_height: Grid height
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid
    for i in range(grid_height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(grid_width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)

    # Draw doors with arrows
    colors = plt.cm.rainbow(np.linspace(0, 1, len(doors)))

    for idx, (s_canonical, a_forward, s_prime_canonical, a_reverse) in enumerate(doors):
        s_full = int(canonical_states[s_canonical])
        s_prime_full = int(canonical_states[s_prime_canonical])

        s_y = s_full // grid_width
        s_x = s_full % grid_width
        s_prime_y = s_prime_full // grid_width
        s_prime_x = s_prime_full % grid_width

        # Draw arrow from s to s' (allowed direction)
        ax.annotate('',
                   xy=(s_prime_x, s_prime_y),
                   xytext=(s_x, s_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors[idx]))

        # Draw X at reverse location (blocked direction)
        ax.plot(s_x, s_y, 'x', color=colors[idx], markersize=15, markeredgewidth=3)

    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(grid_height - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Irreversible Doors ({len(doors)} total)\nArrow=Allowed, X=Blocked', fontsize=14)
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
    output_dir: str = "exp_irreversible_doors/results",
    save_results: bool = True,
    num_eigenvectors_to_visualize: int = 6,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    wall_color: str = 'gray'
) -> Dict:
    """
    Run eigendecomposition analysis on door environments.

    Args:
        batched_transition_counts: Shape [num_envs, num_states, num_actions, num_states]
        metadata: Metadata dictionary
        k: Number of eigenvectors to compute
        output_dir: Directory to save results
        save_results: Whether to save results
        num_eigenvectors_to_visualize: Number of eigenvectors to visualize
        nrows: Visualization grid rows
        ncols: Visualization grid columns
        wall_color: Wall color for visualizations

    Returns:
        Dictionary containing analysis results
    """
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

    # Compute batched eigendecomposition
    print("\n[1/3] Computing batched eigendecomposition...")
    batched_eigendecomp = compute_eigendecomposition_batched(
        batched_transition_counts,
        k=k,
        smoothing=1e-5,
        normalize=True,
        sort_by_magnitude=True
    )
    print(f"  Computed {k} eigenvalues/vectors for {num_envs} environments")

    # Analyze eigenvalue spectrum
    print("\n[2/3] Analyzing eigenvalue spectrum...")
    eigenvalues_real = batched_eigendecomp["eigenvalues_real"]
    eigenvalues_imag = batched_eigendecomp["eigenvalues_imag"]
    eigenvalue_mags = jnp.abs(batched_eigendecomp["eigenvalues"])

    imag_threshold = 1e-8
    is_real = jnp.abs(eigenvalues_imag) < imag_threshold
    num_real_per_env = jnp.sum(is_real, axis=1)

    print(f"  Real eigenvalues per env: mean={jnp.mean(num_real_per_env):.1f}, std={jnp.std(num_real_per_env):.1f}")
    print(f"  Complex eigenvalues per env: mean={k - jnp.mean(num_real_per_env):.1f}")
    print(f"  Max imaginary component: {jnp.max(jnp.abs(eigenvalues_imag)):.6f}")
    print(f"  Mean imaginary component: {jnp.mean(jnp.abs(eigenvalues_imag)):.6f}")

    print(f"\n  Top 5 eigenvalue magnitudes (averaged across envs):")
    avg_mags = jnp.mean(eigenvalue_mags, axis=0)
    for i in range(min(5, k)):
        print(f"    λ_{i}: {avg_mags[i]:.6f}")

    # Visualize eigenvectors
    print("\n[3/3] Visualizing eigenvectors (using first environment)...")
    first_env_eigendecomp = {
        key: value[0] for key, value in batched_eigendecomp.items()
    }

    # Compile results
    results = {
        "batched_eigendecomposition": batched_eigendecomp,
        "metadata": metadata,
        "parameters": {
            "k": k,
            "num_envs": num_envs,
        }
    }

    # Save results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "door_eigendecomposition_results.pkl"
        print(f"\nSaving results to {results_file}...")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

        # Save summary
        summary_file = output_path / "analysis_summary.txt"
        with open(summary_file, "w") as f:
            f.write("IRREVERSIBLE DOOR EIGENDECOMPOSITION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Number of environments: {num_envs}\n")
            f.write(f"Canonical states per env: {num_canonical_states}\n")
            f.write(f"Doors per env: {metadata.get('num_doors_per_env', 'unknown')}\n")
            f.write(f"Number of eigenvalues computed: {k}\n")
            f.write(f"Real eigenvalues per env (mean): {jnp.mean(num_real_per_env):.1f}\n")
            f.write(f"Complex eigenvalues per env (mean): {k - jnp.mean(num_real_per_env):.1f}\n\n")
            f.write("Eigenvector visualizations:\n")
            f.write(f"  - Left eigenvectors (real and imaginary components)\n")
            f.write(f"  - Right eigenvectors (real and imaginary components)\n")
            f.write(f"\nVisualizations saved to: {output_path}/visualizations/\n")

        print(f"Summary saved to {summary_file}")

        # Visualize doors
        if "first_env_doors" in metadata and len(metadata["first_env_doors"]) > 0:
            visualize_doors_on_grid(
                doors=metadata["first_env_doors"],
                canonical_states=canonical_states,
                grid_width=grid_width,
                grid_height=metadata.get("grid_height", grid_width),
                output_path=str(output_path / "door_locations.png")
            )

        # Generate eigenvector visualizations
        print("\nGenerating eigenvector visualizations...")
        grid_height = metadata.get("grid_height", grid_width)

        # Convert doors to portal-like format for visualization (just to show door locations)
        # Doors are (s, a, s', a_reverse), convert to {(s_full, a): s'_full}
        door_markers = {}
        if "first_env_doors" in metadata:
            for s_canonical, a, s_prime_canonical, a_reverse in metadata["first_env_doors"]:
                s_full = int(canonical_states[s_canonical])
                s_prime_full = int(canonical_states[s_prime_canonical])
                door_markers[(s_full, a)] = s_prime_full

        create_eigenvector_visualization_report(
            eigendecomposition=first_env_eigendecomp,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=door_markers,  # Use door locations as markers
            output_dir=str(output_path / "visualizations"),
            num_eigenvectors=num_eigenvectors_to_visualize,
            nrows=nrows,
            ncols=ncols,
            wall_color=wall_color
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Eigendecomposition analysis of environments with irreversible doors"
    )
    parser.add_argument(
        "--base-env",
        type=str,
        default="GridRoom-4",
        help="Base environment name (default: GridRoom-4)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=10,
        help="Number of door environments (default: 10)"
    )
    parser.add_argument(
        "--num-doors",
        type=int,
        default=5,
        help="Number of doors per environment (default: 5)"
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=100,
        help="Number of rollouts per environment (default: 100)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of steps per rollout (default: 100)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of eigenvectors to compute (default: 20)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp_irreversible_doors/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows for visualization grid layout (default: auto)"
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=None,
        help="Number of columns for visualization grid layout (default: auto)"
    )
    parser.add_argument(
        "--wall-color",
        type=str,
        default="gray",
        help="Color for wall/obstacle cells in visualizations (default: gray)"
    )
    parser.add_argument(
        "--num-eigenvectors",
        type=int,
        default=6,
        help="Number of eigenvectors to visualize (default: 6)"
    )

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

    print(f"  Generated transition counts: {batched_counts.shape}")

    # Run analysis
    results = run_door_analysis(
        batched_transition_counts=batched_counts,
        metadata=metadata,
        k=args.k,
        output_dir=args.output_dir,
        save_results=True,
        num_eigenvectors_to_visualize=args.num_eigenvectors,
        nrows=args.nrows,
        ncols=args.ncols,
        wall_color=args.wall_color
    )

    return results


if __name__ == "__main__":
    main()
