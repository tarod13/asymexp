"""
Distance analysis between states in eigenspace and environment space.

This module provides functions to:
1. Compute pairwise distances between states using eigenspace representations
2. Compute actual distances in the environment (shortest path, Euclidean, etc.)
3. Compare and analyze the relationship between eigenspace and environment distances
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
import numpy as np
from functools import partial


def compute_pairwise_distances(vectors: jnp.ndarray, metric: str = "euclidean") -> jnp.ndarray:
    """
    Compute pairwise distances between vectors.

    Args:
        vectors: Shape [num_states, num_features]
        metric: Distance metric ("euclidean", "cosine", "manhattan")

    Returns:
        Distance matrix of shape [num_states, num_states]
    """
    if metric == "euclidean":
        # Efficient computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T b
        norms_sq = jnp.sum(vectors ** 2, axis=1, keepdims=True)
        dot_product = jnp.dot(vectors, vectors.T)
        distances_sq = norms_sq + norms_sq.T - 2 * dot_product
        distances = jnp.sqrt(jnp.maximum(distances_sq, 0))  # Handle numerical errors
    elif metric == "manhattan":
        # L1 distance
        distances = jnp.sum(jnp.abs(vectors[:, None, :] - vectors[None, :, :]), axis=2)
    elif metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / jnp.maximum(norms, 1e-10)
        similarities = jnp.dot(normalized, normalized.T)
        distances = 1 - similarities
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distances


def compute_eigenspace_distances(
    eigendecomposition: Dict[str, jnp.ndarray],
    metric: str = "euclidean",
    use_real: bool = True,
    use_imag: bool = True,
    k: Optional[int] = None
) -> Dict[str, jnp.ndarray]:
    """
    Compute pairwise distances between states using eigenspace representations.

    Args:
        eigendecomposition: Output from compute_eigendecomposition
        metric: Distance metric to use
        use_real: Whether to include real components
        use_imag: Whether to include imaginary components
        k: Number of eigenvectors to use (None = use all)

    Returns:
        Dictionary containing:
            - distances_real: Distances using only real components
            - distances_imag: Distances using only imaginary components
            - distances_combined: Distances using both real and imaginary components
    """
    eigenvectors_real = eigendecomposition["eigenvectors_real"]
    eigenvectors_imag = eigendecomposition["eigenvectors_imag"]

    # Limit to top k eigenvectors if specified
    if k is not None:
        eigenvectors_real = eigenvectors_real[:, :k]
        eigenvectors_imag = eigenvectors_imag[:, :k]

    # Compute distances for real components
    distances_real = compute_pairwise_distances(eigenvectors_real, metric=metric)

    # Compute distances for imaginary components
    distances_imag = compute_pairwise_distances(eigenvectors_imag, metric=metric)

    # Compute combined distances
    if use_real and use_imag:
        # Concatenate real and imaginary parts
        combined_vectors = jnp.concatenate([eigenvectors_real, eigenvectors_imag], axis=1)
    elif use_real:
        combined_vectors = eigenvectors_real
    elif use_imag:
        combined_vectors = eigenvectors_imag
    else:
        raise ValueError("At least one of use_real or use_imag must be True")

    distances_combined = compute_pairwise_distances(combined_vectors, metric=metric)

    return {
        "distances_real": distances_real,
        "distances_imag": distances_imag,
        "distances_combined": distances_combined,
    }


def compute_grid_euclidean_distances(
    states: jnp.ndarray,
    grid_width: int
) -> jnp.ndarray:
    """
    Compute Euclidean distances between grid states.

    Args:
        states: Array of state indices (flattened grid positions)
        grid_width: Width of the grid

    Returns:
        Distance matrix of shape [num_states, num_states]
    """
    # Convert flat indices to (x, y) coordinates
    y_coords = states // grid_width
    x_coords = states % grid_width

    # Compute pairwise Euclidean distances
    dx = x_coords[:, None] - x_coords[None, :]
    dy = y_coords[:, None] - y_coords[None, :]
    distances = jnp.sqrt(dx ** 2 + dy ** 2)

    return distances


def compute_grid_manhattan_distances(
    states: jnp.ndarray,
    grid_width: int
) -> jnp.ndarray:
    """
    Compute Manhattan (L1) distances between grid states.

    Args:
        states: Array of state indices (flattened grid positions)
        grid_width: Width of the grid

    Returns:
        Distance matrix of shape [num_states, num_states]
    """
    # Convert flat indices to (x, y) coordinates
    y_coords = states // grid_width
    x_coords = states % grid_width

    # Compute pairwise Manhattan distances
    dx = jnp.abs(x_coords[:, None] - x_coords[None, :])
    dy = jnp.abs(y_coords[:, None] - y_coords[None, :])
    distances = dx + dy

    return distances


def compute_shortest_path_distances(
    transition_matrix: jnp.ndarray,
    max_iterations: int = 1000
) -> jnp.ndarray:
    """
    Compute shortest path distances using Floyd-Warshall algorithm.

    Args:
        transition_matrix: Shape [num_states, num_states], adjacency matrix
        max_iterations: Maximum number of iterations (for JAX compatibility)

    Returns:
        Distance matrix where entry [i,j] is shortest path from i to j
    """
    num_states = transition_matrix.shape[0]

    # Initialize distance matrix
    # Set distance to 1 where there's a transition, infinity otherwise
    distances = jnp.where(
        transition_matrix > 0,
        jnp.ones_like(transition_matrix),
        jnp.full_like(transition_matrix, jnp.inf)
    )
    # Distance from state to itself is 0
    distances = distances.at[jnp.arange(num_states), jnp.arange(num_states)].set(0)

    # Floyd-Warshall algorithm
    for k in range(num_states):
        # For each intermediate node k, check if path through k is shorter
        distances = jnp.minimum(
            distances,
            distances[:, k:k+1] + distances[k:k+1, :]
        )

    return distances


def compute_environment_distances(
    states: jnp.ndarray,
    grid_width: int,
    transition_matrix: Optional[jnp.ndarray] = None,
    include_shortest_path: bool = False
) -> Dict[str, jnp.ndarray]:
    """
    Compute various environment-based distances between states.

    Args:
        states: Array of state indices
        grid_width: Width of the grid
        transition_matrix: Optional transition matrix for shortest path computation
        include_shortest_path: Whether to compute shortest path distances (can be slow)

    Returns:
        Dictionary containing:
            - euclidean: Euclidean distances in grid space
            - manhattan: Manhattan distances in grid space
            - shortest_path: Shortest path distances (if include_shortest_path=True)
    """
    distances = {
        "euclidean": compute_grid_euclidean_distances(states, grid_width),
        "manhattan": compute_grid_manhattan_distances(states, grid_width),
    }

    if include_shortest_path and transition_matrix is not None:
        distances["shortest_path"] = compute_shortest_path_distances(transition_matrix)

    return distances


def compare_distances(
    eigenspace_distances: jnp.ndarray,
    environment_distances: jnp.ndarray,
    exclude_diagonal: bool = True
) -> Dict[str, float]:
    """
    Compare eigenspace distances with environment distances.

    Computes correlation and other metrics to understand the relationship
    between distances in eigenspace and actual environment.

    Args:
        eigenspace_distances: Shape [num_states, num_states]
        environment_distances: Shape [num_states, num_states]
        exclude_diagonal: Whether to exclude diagonal (self-distances) from analysis

    Returns:
        Dictionary with comparison metrics:
            - correlation: Pearson correlation coefficient
            - spearman_correlation: Spearman rank correlation
            - mean_squared_error: MSE between distances
            - mean_absolute_error: MAE between distances
    """
    # Flatten distance matrices
    if exclude_diagonal:
        # Get upper triangle (excluding diagonal)
        num_states = eigenspace_distances.shape[0]
        mask = jnp.triu(jnp.ones((num_states, num_states), dtype=bool), k=1)
        eigen_flat = eigenspace_distances[mask]
        env_flat = environment_distances[mask]
    else:
        eigen_flat = eigenspace_distances.flatten()
        env_flat = environment_distances.flatten()

    # Remove any infinite values (from shortest path)
    finite_mask = jnp.isfinite(eigen_flat) & jnp.isfinite(env_flat)
    eigen_flat = eigen_flat[finite_mask]
    env_flat = env_flat[finite_mask]

    # Pearson correlation
    correlation = jnp.corrcoef(eigen_flat, env_flat)[0, 1]

    # Spearman correlation (rank-based)
    # For JAX, we'll compute it using sorted ranks
    eigen_ranks = jnp.argsort(jnp.argsort(eigen_flat))
    env_ranks = jnp.argsort(jnp.argsort(env_flat))
    spearman = jnp.corrcoef(eigen_ranks.astype(float), env_ranks.astype(float))[0, 1]

    # Mean squared error
    mse = jnp.mean((eigen_flat - env_flat) ** 2)

    # Mean absolute error
    mae = jnp.mean(jnp.abs(eigen_flat - env_flat))

    return {
        "correlation": float(correlation),
        "spearman_correlation": float(spearman),
        "mean_squared_error": float(mse),
        "mean_absolute_error": float(mae),
        "num_pairs": int(jnp.sum(finite_mask)),
    }


def analyze_distance_relationships(
    eigendecomposition: Dict[str, jnp.ndarray],
    states: jnp.ndarray,
    grid_width: int,
    transition_matrix: Optional[jnp.ndarray] = None,
    k_values: Optional[List[int]] = None,
    eigenspace_metric: str = "euclidean"
) -> Dict[str, any]:
    """
    Comprehensive analysis of relationships between eigenspace and environment distances.

    Args:
        eigendecomposition: Output from compute_eigendecomposition
        states: Array of state indices
        grid_width: Width of the grid
        transition_matrix: Optional for shortest path distances
        k_values: List of k values (number of eigenvectors) to analyze
        eigenspace_metric: Distance metric for eigenspace

    Returns:
        Comprehensive analysis results
    """
    if k_values is None:
        k_values = [5, 10, 20, None]  # None means use all

    # Compute environment distances
    env_distances = compute_environment_distances(
        states,
        grid_width,
        transition_matrix=transition_matrix,
        include_shortest_path=(transition_matrix is not None)
    )

    results = {
        "environment_distances": env_distances,
        "eigenspace_comparisons": {}
    }

    # Analyze for different numbers of eigenvectors
    for k in k_values:
        k_label = f"k={k}" if k is not None else "k=all"

        eigen_dists = compute_eigenspace_distances(
            eigendecomposition,
            metric=eigenspace_metric,
            use_real=True,
            use_imag=True,
            k=k
        )

        # Compare with each environment distance type
        comparisons = {}
        for env_type, env_dist in env_distances.items():
            comparisons[env_type] = {
                "real": compare_distances(eigen_dists["distances_real"], env_dist),
                "imag": compare_distances(eigen_dists["distances_imag"], env_dist),
                "combined": compare_distances(eigen_dists["distances_combined"], env_dist),
            }

        results["eigenspace_comparisons"][k_label] = {
            "distances": eigen_dists,
            "comparisons": comparisons
        }

    return results


def analyze_distance_relationships_batched(
    batched_eigendecomposition: Dict[str, jnp.ndarray],
    states: jnp.ndarray,
    grid_width: int,
    batched_transition_matrices: Optional[jnp.ndarray] = None,
    k_values: Optional[List[int]] = None,
    eigenspace_metric: str = "euclidean"
) -> Dict[str, any]:
    """
    Analyze distance relationships for multiple environments in parallel.

    Args:
        batched_eigendecomposition: Batched eigendecomposition results
            - eigenvalues: [num_envs, k]
            - eigenvectors_real: [num_envs, num_states, k]
            - etc.
        states: Array of state indices (same for all environments)
        grid_width: Width of the grid
        batched_transition_matrices: Optional [num_envs, num_states, num_states]
        k_values: List of k values to analyze
        eigenspace_metric: Distance metric

    Returns:
        Dictionary with per-environment and aggregated results
    """
    num_envs = batched_eigendecomposition["eigenvalues"].shape[0]

    # Compute environment distances (same for all envs)
    env_distances = compute_environment_distances(
        states,
        grid_width,
        transition_matrix=None,  # Don't compute shortest path in batch
        include_shortest_path=False
    )

    # Process each environment
    per_env_results = []
    for env_idx in range(num_envs):
        # Extract single environment eigendecomposition
        single_eigendecomp = {
            key: value[env_idx] for key, value in batched_eigendecomposition.items()
        }

        # Get transition matrix for this env if available
        trans_mat = batched_transition_matrices[env_idx] if batched_transition_matrices is not None else None

        # Analyze this environment
        result = analyze_distance_relationships(
            eigendecomposition=single_eigendecomp,
            states=states,
            grid_width=grid_width,
            transition_matrix=trans_mat,
            k_values=k_values,
            eigenspace_metric=eigenspace_metric
        )
        per_env_results.append(result)

    # Aggregate results across environments
    aggregated_results = aggregate_multi_environment_results(per_env_results, k_values)

    return {
        "num_envs": num_envs,
        "per_env_results": per_env_results,
        "aggregated_results": aggregated_results,
        "environment_distances": env_distances,
    }


def aggregate_multi_environment_results(
    per_env_results: List[Dict],
    k_values: Optional[List[int]] = None
) -> Dict[str, any]:
    """
    Aggregate analysis results across multiple environments.

    Args:
        per_env_results: List of analysis results, one per environment
        k_values: List of k values that were analyzed

    Returns:
        Dictionary with aggregated statistics (mean, std, min, max)
    """
    if k_values is None:
        k_values = [5, 10, 20, None]

    num_envs = len(per_env_results)
    aggregated = {}

    for k in k_values:
        k_label = f"k={k}" if k is not None else "k=all"
        aggregated[k_label] = {}

        # Extract comparison metrics for this k across all environments
        for env_result in per_env_results:
            comparisons = env_result["eigenspace_comparisons"][k_label]["comparisons"]

            for env_type in comparisons.keys():
                if env_type not in aggregated[k_label]:
                    aggregated[k_label][env_type] = {
                        "real": {"correlations": [], "spearman": [], "mse": [], "mae": []},
                        "imag": {"correlations": [], "spearman": [], "mse": [], "mae": []},
                        "combined": {"correlations": [], "spearman": [], "mse": [], "mae": []},
                    }

                for component in ["real", "imag", "combined"]:
                    metrics = comparisons[env_type][component]
                    aggregated[k_label][env_type][component]["correlations"].append(metrics["correlation"])
                    aggregated[k_label][env_type][component]["spearman"].append(metrics["spearman_correlation"])
                    aggregated[k_label][env_type][component]["mse"].append(metrics["mean_squared_error"])
                    aggregated[k_label][env_type][component]["mae"].append(metrics["mean_absolute_error"])

        # Compute statistics
        for env_type in aggregated[k_label].keys():
            for component in ["real", "imag", "combined"]:
                corrs = np.array(aggregated[k_label][env_type][component]["correlations"])
                spear = np.array(aggregated[k_label][env_type][component]["spearman"])
                mse = np.array(aggregated[k_label][env_type][component]["mse"])
                mae = np.array(aggregated[k_label][env_type][component]["mae"])

                aggregated[k_label][env_type][component] = {
                    "correlation": {
                        "mean": float(np.mean(corrs)),
                        "std": float(np.std(corrs)),
                        "min": float(np.min(corrs)),
                        "max": float(np.max(corrs)),
                        "median": float(np.median(corrs)),
                    },
                    "spearman_correlation": {
                        "mean": float(np.mean(spear)),
                        "std": float(np.std(spear)),
                        "min": float(np.min(spear)),
                        "max": float(np.max(spear)),
                        "median": float(np.median(spear)),
                    },
                    "mean_squared_error": {
                        "mean": float(np.mean(mse)),
                        "std": float(np.std(mse)),
                        "min": float(np.min(mse)),
                        "max": float(np.max(mse)),
                        "median": float(np.median(mse)),
                    },
                    "mean_absolute_error": {
                        "mean": float(np.mean(mae)),
                        "std": float(np.std(mae)),
                        "min": float(np.min(mae)),
                        "max": float(np.max(mae)),
                        "median": float(np.median(mae)),
                    },
                }

    return aggregated
