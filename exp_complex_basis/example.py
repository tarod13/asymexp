"""
Example script demonstrating the eigendecomposition analysis.

This is a minimal example showing how to use the exp_complex_basis package.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from exp_complex_basis import (
    compute_eigendecomposition,
    compute_eigenspace_distances,
    compute_environment_distances,
    compare_distances,
)


def create_simple_transition_matrix(n_states: int = 10, seed: int = 42):
    """Create a simple random transition matrix for demonstration."""
    key = jax.random.PRNGKey(seed)

    # Create random transition counts
    transition_counts = jax.random.uniform(key, shape=(n_states, n_states))

    # Make it stochastic (rows sum to 1)
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    return transition_matrix


def main():
    print("=" * 80)
    print("SIMPLE EIGENDECOMPOSITION ANALYSIS EXAMPLE")
    print("=" * 80)

    # Create a simple transition matrix
    print("\n[1] Creating transition matrix...")
    n_states = 20
    transition_matrix = create_simple_transition_matrix(n_states)
    print(f"    Created {n_states}x{n_states} transition matrix")
    print(f"    Symmetric: {np.allclose(transition_matrix, transition_matrix.T)}")

    # Compute eigendecomposition
    print("\n[2] Computing eigendecomposition...")
    k = 10
    eigendecomp = compute_eigendecomposition(transition_matrix, k=k, sort_by_magnitude=True)
    print(f"    Computed {k} eigenvalues/vectors")

    # Analyze spectrum
    print("\n[3] Eigenvalue spectrum:")
    for i in range(min(5, k)):
        real = eigendecomp['eigenvalues_real'][i]
        imag = eigendecomp['eigenvalues_imag'][i]
        mag = abs(eigendecomp['eigenvalues'][i])
        print(f"    λ_{i}: magnitude={mag:.4f}, real={real:.4f}, imag={imag:.4f}")

    # Compute eigenspace distances
    print("\n[4] Computing eigenspace distances...")
    eigen_dists = compute_eigenspace_distances(
        eigendecomp,
        metric="euclidean",
        use_real=True,
        use_imag=True,
        k=k
    )
    print(f"    Real component distances: {eigen_dists['distances_real'].shape}")
    print(f"    Imaginary component distances: {eigen_dists['distances_imag'].shape}")
    print(f"    Combined distances: {eigen_dists['distances_combined'].shape}")

    # Compute environment distances (simple grid)
    print("\n[5] Computing environment distances...")
    states = jnp.arange(n_states)
    grid_width = int(np.ceil(np.sqrt(n_states)))
    env_dists = compute_environment_distances(
        states,
        grid_width=grid_width,
        transition_matrix=transition_matrix,
        include_shortest_path=True
    )
    print(f"    Euclidean distances: {env_dists['euclidean'].shape}")
    print(f"    Manhattan distances: {env_dists['manhattan'].shape}")
    print(f"    Shortest path distances: {env_dists['shortest_path'].shape}")

    # Compare distances
    print("\n[6] Comparing distances:")
    print("\n    Real components vs Environment:")
    for env_type, env_dist in env_dists.items():
        comparison = compare_distances(
            eigen_dists['distances_real'],
            env_dist
        )
        print(f"      vs {env_type:15s}: Pearson r = {comparison['correlation']:+.4f}")

    print("\n    Imaginary components vs Environment:")
    for env_type, env_dist in env_dists.items():
        comparison = compare_distances(
            eigen_dists['distances_imag'],
            env_dist
        )
        print(f"      vs {env_type:15s}: Pearson r = {comparison['correlation']:+.4f}")

    print("\n    Combined (real + imag) vs Environment:")
    for env_type, env_dist in env_dists.items():
        comparison = compare_distances(
            eigen_dists['distances_combined'],
            env_dist
        )
        print(f"      vs {env_type:15s}: Pearson r = {comparison['correlation']:+.4f}")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
