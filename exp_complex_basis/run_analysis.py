"""
Main script for running eigendecomposition analysis on non-symmetrized dynamics matrices.

This script:
1. Loads or generates transition data for multiple environments
2. Computes eigendecomposition without symmetrization (batched)
3. Analyzes distances in eigenspace (real and imaginary components)
4. Compares with actual environment distances
5. Aggregates results across environments
6. Saves results and generates visualizations
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
    analyze_eigenvalue_spectrum,
    get_nonsymmetrized_transition_matrix,
)
from exp_complex_basis.eigenvector_visualization import (
    create_eigenvector_visualization_report,
)
from src.data_collection import collect_transition_counts_batched_portals
from src.envs.env import create_environment_from_text
from src.envs.portal_gridworld import create_random_portal_env


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


def generate_portal_environments_data(
    base_env_name: str = "GridRoom-4",
    num_envs: int = 10,
    num_portals: int = 10,
    num_rollouts_per_env: int = 100,
    num_steps: int = 100,
    seed: int = 42
):
    """
    Generate transition data from multiple portal environments.

    Args:
        base_env_name: Name of base environment text file
        num_envs: Number of different portal environments
        num_portals: Number of portals per environment
        num_rollouts_per_env: Number of rollouts per environment
        num_steps: Number of steps per rollout
        seed: Random seed

    Returns:
        Tuple of (transition_counts, metadata)
    """
    print(f"Generating data from {num_envs} portal environments...")
    print(f"  Base environment: {base_env_name}")
    print(f"  Portals per env: {num_portals}")
    print(f"  Rollouts per env: {num_rollouts_per_env}")
    print(f"  Steps per rollout: {num_steps}")

    # Load base environment
    base_env = create_environment_from_text(file_name=base_env_name, max_steps=num_steps)
    num_states = base_env.observation_space_size()
    num_actions = base_env.action_space_size()

    print(f"  Total states: {num_states}")
    print(f"  Actions: {num_actions}")

    # Get canonical free states
    canonical_states = get_canonical_free_states(base_env)
    num_canonical_states = len(canonical_states)
    print(f"  Free states: {num_canonical_states}")

    # Generate portal configurations
    rng = np.random.RandomState(seed)
    portal_configs_list = []
    portal_masks_list = []

    for env_idx in range(num_envs):
        env_seed = rng.randint(0, 2**31)
        portal_env = create_random_portal_env(
            base_env=base_env,
            num_portals=num_portals,
            seed=env_seed
        )

        # Get portal config (source_state, action, dest_state in canonical space)
        # portal_env.portals is a dict: {(state, action): dest_state}
        portals = []
        for (source_state, action), dest_state in portal_env.portals.items():
            # Convert to canonical indices
            source_canonical = jnp.where(canonical_states == source_state, size=1, fill_value=-1)[0][0]
            dest_canonical = jnp.where(canonical_states == dest_state, size=1, fill_value=-1)[0][0]
            portals.append([int(source_canonical), int(action), int(dest_canonical)])

        # Pad to max_portals
        while len(portals) < num_portals:
            portals.append([0, 0, 0])

        portal_configs_list.append(portals[:num_portals])
        portal_masks_list.append([True] * len(portal_env.portals) + [False] * (num_portals - len(portal_env.portals)))

    portal_configs = jnp.array(portal_configs_list, dtype=jnp.int32)
    portal_masks = jnp.array(portal_masks_list, dtype=bool)

    # Collect transition counts in parallel
    print(f"\nCollecting transitions...")
    transition_counts, metrics = collect_transition_counts_batched_portals(
        base_env=base_env,
        portal_configs=portal_configs,
        portal_masks=portal_masks,
        num_rollouts_per_env=num_rollouts_per_env,
        num_steps=num_steps,
        canonical_states=canonical_states,
        seed=seed
    )

    print(f"  Collected transition counts: {transition_counts.shape}")
    print(f"  Shape: [num_envs={num_envs}, num_states={num_canonical_states}, num_actions={num_actions}, num_states={num_canonical_states}]")

    # Get obstacle information from base environment
    obstacles = []
    if base_env.has_obstacles:
        obstacles = [(int(obs[0]), int(obs[1])) for obs in base_env.obstacles]

    metadata = {
        "num_envs": num_envs,
        "num_states": num_states,
        "num_canonical_states": num_canonical_states,
        "num_actions": num_actions,
        "canonical_states": canonical_states,
        "base_env_name": base_env_name,
        "num_portals": num_portals,
        "grid_width": base_env.width,
        "grid_height": base_env.height,
        "obstacles": obstacles,
        "first_env_portals": portal_configs_list[0] if num_envs > 0 else [],
    }

    return transition_counts, metadata


def load_or_generate_data(
    data_path: Optional[str] = None,
    generate_new: bool = False,
    **generation_kwargs
):
    """
    Load transition data from file or generate new data.

    Args:
        data_path: Path to saved transition data
        generate_new: Whether to generate new data
        **generation_kwargs: Arguments for generate_portal_environments_data

    Returns:
        Tuple of (transition_counts, metadata)
    """
    if not generate_new and data_path is not None and Path(data_path).exists():
        print(f"Loading data from {data_path}...")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data["transition_counts"], data.get("metadata", {})
    else:
        return generate_portal_environments_data(**generation_kwargs)


def run_eigendecomposition_analysis(
    batched_transition_counts: jnp.ndarray,
    metadata: Dict,
    k: int = 20,
    k_values_to_analyze: Optional[List[int]] = None,
    output_dir: str = "exp_complex_basis/results",
    save_results: bool = True
) -> Dict:
    """
    Run complete eigendecomposition analysis on batched transition data.

    Args:
        batched_transition_counts: Shape [num_envs, num_states, num_actions, num_states]
        metadata: Metadata dictionary
        k: Number of eigenvectors to compute
        k_values_to_analyze: List of k values for distance analysis
        output_dir: Directory to save results
        save_results: Whether to save results to disk

    Returns:
        Dictionary containing all analysis results
    """
    num_envs = batched_transition_counts.shape[0]
    num_canonical_states = batched_transition_counts.shape[1]
    grid_width = metadata.get("grid_width", 13)
    canonical_states = metadata.get("canonical_states", jnp.arange(num_canonical_states))

    print("\n" + "=" * 80)
    print("EIGENDECOMPOSITION ANALYSIS OF NON-SYMMETRIZED DYNAMICS MATRICES")
    print("=" * 80)
    print(f"Number of environments: {num_envs}")
    print(f"Canonical states per environment: {num_canonical_states}")

    # Step 1: Compute batched eigendecomposition
    print("\n[1/4] Computing batched eigendecomposition...")
    batched_eigendecomp = compute_eigendecomposition_batched(
        batched_transition_counts,
        k=k,
        smoothing=1e-5,
        normalize=True,
        sort_by_magnitude=True
    )
    print(f"  Computed {k} eigenvalues/vectors for {num_envs} environments")
    print(f"  Batched eigenvalues shape: {batched_eigendecomp['eigenvalues'].shape}")

    # Step 2: Analyze eigenvalue spectrum (aggregate across envs)
    print("\n[2/4] Analyzing eigenvalue spectrum...")

    # Aggregate eigenvalue statistics
    eigenvalues_real = batched_eigendecomp["eigenvalues_real"]
    eigenvalues_imag = batched_eigendecomp["eigenvalues_imag"]
    eigenvalue_mags = jnp.abs(batched_eigendecomp["eigenvalues"])

    # Count real vs complex (across all envs)
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

    # Step 3: Visualize eigenvectors for first environment as example
    print("\n[3/3] Visualizing eigenvectors (using first environment as example)...")

    # Use first environment for visualization
    first_env_eigendecomp = {
        key: value[0] for key, value in batched_eigendecomp.items()
    }

    # Compile all results
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

        results_file = output_path / "eigendecomposition_results_batched.pkl"
        print(f"\nSaving results to {results_file}...")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        print("Results saved!")

        # Save a summary text file
        summary_file = output_path / "analysis_summary_batched.txt"
        with open(summary_file, "w") as f:
            f.write("BATCHED EIGENDECOMPOSITION ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Number of environments: {num_envs}\n")
            f.write(f"Canonical states per env: {num_canonical_states}\n")
            f.write(f"Number of eigenvalues computed: {k}\n")
            f.write(f"Real eigenvalues per env (mean): {jnp.mean(num_real_per_env):.1f}\n")
            f.write(f"Complex eigenvalues per env (mean): {k - jnp.mean(num_real_per_env):.1f}\n\n")
            f.write("Eigenvector visualizations:\n")
            f.write(f"  - Left eigenvectors (real and imaginary components)\n")
            f.write(f"  - Right eigenvectors (real and imaginary components)\n")
            f.write(f"\nVisualizations saved to: {output_path}/visualizations/\n")

        print(f"Summary saved to {summary_file}")

        # Generate eigenvector visualizations
        print("\nGenerating eigenvector visualizations...")
        obstacles = metadata.get("obstacles", [])
        grid_height = metadata.get("grid_height", grid_width)

        # Reconstruct portals from first environment (for visualization)
        portals = None
        if "first_env_portals" in metadata and len(metadata["first_env_portals"]) > 0:
            portals = {}
            for portal in metadata["first_env_portals"]:
                if len(portal) == 3:
                    # portal is [source_canonical, action, dest_canonical]
                    # Convert back to full state space for visualization
                    source_canonical, action, dest_canonical = portal
                    if source_canonical >= 0 and dest_canonical >= 0:
                        source_full = int(canonical_states[source_canonical])
                        dest_full = int(canonical_states[dest_canonical])
                        portals[(source_full, action)] = dest_full

        create_eigenvector_visualization_report(
            eigendecomposition=first_env_eigendecomp,
            canonical_states=canonical_states,
            grid_width=grid_width,
            grid_height=grid_height,
            portals=portals,
            output_dir=str(output_path / "visualizations"),
            num_eigenvectors=6
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Eigendecomposition analysis of non-symmetrized dynamics matrices"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to saved transition data (pickle file)"
    )
    parser.add_argument(
        "--generate-new",
        action="store_true",
        help="Generate new transition data instead of loading"
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
        help="Number of portal environments (default: 10)"
    )
    parser.add_argument(
        "--num-portals",
        type=int,
        default=10,
        help="Number of portals per environment (default: 10)"
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
        default="exp_complex_basis/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Load or generate data
    transition_counts, metadata = load_or_generate_data(
        data_path=args.data_path,
        generate_new=args.generate_new,
        base_env_name=args.base_env,
        num_envs=args.num_envs,
        num_portals=args.num_portals,
        num_rollouts_per_env=args.num_rollouts,
        num_steps=args.num_steps,
        seed=args.seed
    )

    # Run analysis
    results = run_eigendecomposition_analysis(
        batched_transition_counts=transition_counts,
        metadata=metadata,
        k=args.k,
        output_dir=args.output_dir,
        save_results=True
    )

    return results


if __name__ == "__main__":
    main()
