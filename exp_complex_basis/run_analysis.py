"""
Main script for running eigendecomposition analysis on non-symmetrized dynamics matrices.

This script:
1. Loads or generates transition data
2. Computes eigendecomposition without symmetrization
3. Analyzes distances in eigenspace (real and imaginary components)
4. Compares with actual environment distances
5. Saves results and generates visualizations
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from exp_complex_basis.eigendecomposition import (
    compute_eigendecomposition_from_counts,
    analyze_eigenvalue_spectrum,
    get_nonsymmetrized_transition_matrix,
)
from exp_complex_basis.distance_analysis import (
    analyze_distance_relationships,
)
from src.data_collection import collect_transition_counts_batched_portals
from src.envs.env import make_env


def generate_transition_data(
    num_envs: int = 10,
    num_episodes: int = 100,
    max_steps: int = 100,
    env_kwargs: Optional[Dict] = None,
    seed: int = 42
):
    """
    Generate transition data from portal environments.

    Args:
        num_envs: Number of parallel environments
        num_episodes: Number of episodes per environment
        max_steps: Maximum steps per episode
        env_kwargs: Additional environment kwargs
        seed: Random seed

    Returns:
        Transition counts array [num_states, num_actions, num_states]
    """
    if env_kwargs is None:
        env_kwargs = {}

    print(f"Generating transition data from {num_envs} environments...")

    # Create environments
    envs = [make_env(**env_kwargs) for _ in range(num_envs)]

    # Collect transition counts
    key = jax.random.PRNGKey(seed)
    transition_counts = collect_transition_counts_batched_portals(
        envs=envs,
        num_episodes=num_episodes,
        max_steps=max_steps,
        key=key
    )

    print(f"Collected transitions. Shape: {transition_counts.shape}")
    return transition_counts


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
        **generation_kwargs: Arguments for generate_transition_data

    Returns:
        Tuple of (transition_counts, metadata)
    """
    if not generate_new and data_path is not None and Path(data_path).exists():
        print(f"Loading data from {data_path}...")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data["transition_counts"], data.get("metadata", {})
    else:
        transition_counts = generate_transition_data(**generation_kwargs)
        metadata = {
            "num_states": transition_counts.shape[0],
            "num_actions": transition_counts.shape[1],
            "generation_kwargs": generation_kwargs,
        }
        return transition_counts, metadata


def run_eigendecomposition_analysis(
    transition_counts: jnp.ndarray,
    grid_width: int = 13,
    k: int = 20,
    k_values_to_analyze: Optional[list] = None,
    output_dir: str = "exp_complex_basis/results",
    save_results: bool = True
) -> Dict:
    """
    Run complete eigendecomposition analysis.

    Args:
        transition_counts: Transition count matrix
        grid_width: Width of the grid environment
        k: Number of eigenvectors to compute
        k_values_to_analyze: List of k values for distance analysis
        output_dir: Directory to save results
        save_results: Whether to save results to disk

    Returns:
        Dictionary containing all analysis results
    """
    print("\n" + "=" * 80)
    print("EIGENDECOMPOSITION ANALYSIS OF NON-SYMMETRIZED DYNAMICS MATRIX")
    print("=" * 80)

    # Step 1: Build non-symmetrized transition matrix
    print("\n[1/5] Building non-symmetrized transition matrix...")
    transition_matrix = get_nonsymmetrized_transition_matrix(
        transition_counts,
        smoothing=1e-5,
        normalize=True
    )
    print(f"  Transition matrix shape: {transition_matrix.shape}")
    print(f"  Matrix is symmetric: {np.allclose(transition_matrix, transition_matrix.T)}")

    # Step 2: Compute eigendecomposition
    print("\n[2/5] Computing eigendecomposition...")
    eigendecomp = compute_eigendecomposition_from_counts(
        transition_counts,
        k=k,
        smoothing=1e-5,
        normalize=True,
        sort_by_magnitude=True
    )
    print(f"  Computed {len(eigendecomp['eigenvalues'])} eigenvalues/vectors")

    # Step 3: Analyze eigenvalue spectrum
    print("\n[3/5] Analyzing eigenvalue spectrum...")
    spectrum_analysis = analyze_eigenvalue_spectrum(eigendecomp)
    print(f"  Real eigenvalues: {spectrum_analysis['num_real']}")
    print(f"  Complex eigenvalues: {spectrum_analysis['num_complex']}")
    print(f"  Max imaginary component: {spectrum_analysis['max_imaginary_component']:.6f}")
    print(f"  Mean imaginary component: {spectrum_analysis['mean_imaginary_component']:.6f}")
    print(f"\n  Top 10 eigenvalue magnitudes:")
    for i, mag in enumerate(spectrum_analysis['eigenvalue_magnitudes'][:10]):
        real = eigendecomp['eigenvalues_real'][i]
        imag = eigendecomp['eigenvalues_imag'][i]
        print(f"    λ_{i}: {mag:.6f} (real: {real:.6f}, imag: {imag:.6f})")

    # Step 4: Prepare states for distance analysis
    print("\n[4/5] Computing distances...")
    num_states = transition_matrix.shape[0]
    states = jnp.arange(num_states)

    # Analyze distance relationships
    if k_values_to_analyze is None:
        k_values_to_analyze = [5, 10, 20, None]

    distance_analysis = analyze_distance_relationships(
        eigendecomposition=eigendecomp,
        states=states,
        grid_width=grid_width,
        transition_matrix=transition_matrix,
        k_values=k_values_to_analyze,
        eigenspace_metric="euclidean"
    )

    # Step 5: Print summary of distance comparisons
    print("\n[5/5] Distance comparison results:")
    print("\nCorrelation between eigenspace and environment distances:")
    print("-" * 80)

    for k_label, k_results in distance_analysis["eigenspace_comparisons"].items():
        print(f"\n{k_label}:")
        for env_type, comparisons in k_results["comparisons"].items():
            print(f"  {env_type.upper()} distances:")
            for component, metrics in comparisons.items():
                corr = metrics["correlation"]
                spearman = metrics["spearman_correlation"]
                print(f"    {component:10s}: Pearson={corr:+.4f}, Spearman={spearman:+.4f}")

    # Compile all results
    results = {
        "transition_matrix": transition_matrix,
        "eigendecomposition": eigendecomp,
        "spectrum_analysis": spectrum_analysis,
        "distance_analysis": distance_analysis,
        "parameters": {
            "k": k,
            "grid_width": grid_width,
            "k_values_analyzed": k_values_to_analyze,
        }
    }

    # Save results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "eigendecomposition_results.pkl"
        print(f"\nSaving results to {results_file}...")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        print("Results saved!")

        # Save a summary text file
        summary_file = output_path / "analysis_summary.txt"
        with open(summary_file, "w") as f:
            f.write("EIGENDECOMPOSITION ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Transition matrix shape: {transition_matrix.shape}\n")
            f.write(f"Number of eigenvalues computed: {k}\n")
            f.write(f"Real eigenvalues: {spectrum_analysis['num_real']}\n")
            f.write(f"Complex eigenvalues: {spectrum_analysis['num_complex']}\n")
            f.write(f"Max imaginary component: {spectrum_analysis['max_imaginary_component']:.6f}\n\n")

            f.write("Distance Correlations:\n")
            f.write("-" * 80 + "\n")
            for k_label, k_results in distance_analysis["eigenspace_comparisons"].items():
                f.write(f"\n{k_label}:\n")
                for env_type, comparisons in k_results["comparisons"].items():
                    f.write(f"  {env_type}:\n")
                    for component, metrics in comparisons.items():
                        f.write(f"    {component}: r={metrics['correlation']:+.4f}\n")

        print(f"Summary saved to {summary_file}")

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
        "--num-envs",
        type=int,
        default=10,
        help="Number of environments for data generation"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes per environment"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of eigenvectors to compute"
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=13,
        help="Width of the grid environment"
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
        help="Random seed"
    )

    args = parser.parse_args()

    # Load or generate data
    transition_counts, metadata = load_or_generate_data(
        data_path=args.data_path,
        generate_new=args.generate_new,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed
    )

    # Run analysis
    results = run_eigendecomposition_analysis(
        transition_counts=transition_counts,
        grid_width=args.grid_width,
        k=args.k,
        output_dir=args.output_dir,
        save_results=True
    )

    return results


if __name__ == "__main__":
    main()
