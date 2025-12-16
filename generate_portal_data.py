"""
Generate dataset of environments with random portals, collect rollouts, and compute SVD.

This script creates M different portal-based environments, collects rollouts from each,
computes the SVD decomposition of their transition matrices, and saves everything to disk.
"""

import os
import pickle
from typing import Dict, List, Tuple
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.env import create_environment_from_text
from src.envs.portal_gridworld import create_random_portal_env
from src.data_collection import collect_data_purejaxrl_style, collect_transition_counts, collect_transition_counts_batched_portals
from src.transition_estimation import calculate_normalized_transitions
from src.svd import transition_svd_jax


def collect_rollouts(env, num_rollouts: int, horizon: int, seed: int = 42):
    """
    Collect rollouts from an environment using uniform random policy.

    Args:
        env: Environment to collect from
        num_rollouts: Number of rollout episodes
        horizon: Length of each rollout
        seed: Random seed

    Returns:
        buffer_state: Flashbax buffer containing transitions
        transitions: List of (state, action, next_state) tuples
    """
    # Collect data using existing infrastructure
    # num_rollouts is treated as num_envs (parallel environments)
    # horizon is the number of steps per environment
    buffer_state, metrics = collect_data_purejaxrl_style(
        env=env,
        num_envs=num_rollouts,
        num_steps=horizon,
        seed=seed,
        precision=32
    )

    # Extract transitions as list of (s, a, s') tuples
    state_indices = np.array(buffer_state['experience'].state_idx.reshape(-1))
    actions = np.array(buffer_state['experience'].action.reshape(-1))
    next_state_indices = np.array(buffer_state['experience'].next_state_idx.reshape(-1))

    transitions = list(zip(state_indices, actions, next_state_indices))

    return buffer_state, transitions


def get_canonical_free_states(base_env):
    """
    Get the canonical set of free (non-obstacle) states from the base environment.

    This creates a fixed mapping that will be used across all environments,
    ensuring state indices correspond to the same actual positions.

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
    free_states = sorted(all_states - obstacle_states)

    return np.array(free_states, dtype=np.int32)


def compute_svd_from_counts(env, num_envs, num_steps, num_states, canonical_states, k=10, seed=42):
    """
    Compute SVD directly from transition counts (memory efficient).

    This collects transition counts without storing trajectories, making it
    much more memory efficient and allowing massive parallelization.

    Args:
        env: Environment to collect from
        num_envs: Number of parallel environments (e.g., 10000)
        num_steps: Number of steps per environment (e.g., 100)
        num_states: Total number of states in full state space
        canonical_states: Array of free state indices to use
        k: Number of singular vectors to keep
        seed: Random seed

    Returns:
        U: Left singular vectors (num_canonical_states x k)
        s: Singular values (k,)
        Vh: Right singular vectors transposed (k x num_canonical_states)
    """
    # Collect transition counts directly (no trajectory storage)
    transition_counts_full, metrics = collect_transition_counts(
        env=env,
        num_envs=num_envs,
        num_steps=num_steps,
        num_states=num_states,
        seed=seed
    )

    # Sum over actions to get state-to-state transition counts
    transition_counts_full = transition_counts_full.sum(axis=1)  # [num_states, num_states]

    # Extract submatrix for canonical states only
    num_canonical = len(canonical_states)
    transition_counts = transition_counts_full[jnp.ix_(canonical_states, canonical_states)]

    # Symmetrize
    transition_counts = (transition_counts + transition_counts.T) / 2.0

    # Add smoothing factor
    smooth_factor = 1e-5
    transition_probs = transition_counts + smooth_factor

    # Normalize (with iterations for better symmetry)
    for _ in range(2):
        row_sums = transition_probs.sum(axis=1, keepdims=True)
        transition_probs = transition_probs / row_sums

        # Re-symmetrize after normalization
        transition_probs = jnp.tril(transition_probs)
        transition_probs = transition_probs + transition_probs.T - jnp.diag(jnp.diag(transition_probs))

    # Compute SVD
    U, s, Vh = transition_svd_jax(transition_probs, k=k)

    return U, s, Vh


@partial(jax.jit, static_argnums=1)
def compute_svd_batched(batched_transition_counts, k=10):
    """
    Compute SVD for multiple portal environments in parallel using vmap.

    Args:
        batched_transition_counts: Array of shape [num_portal_envs, num_canonical_states, num_actions, num_canonical_states]
                                   Already in canonical (free states only) space
        k: Number of singular vectors to keep (must be static for JIT)

    Returns:
        U_batch: Left singular vectors [num_portal_envs, num_canonical_states, k]
        s_batch: Singular values [num_portal_envs, k]
        Vh_batch: Right singular vectors transposed [num_portal_envs, k, num_canonical_states]
    """
    def process_one_env(env_transition_counts):
        """Process transition counts for a single environment and compute SVD."""
        # Sum over actions to get state-to-state transition counts
        # Input is already in canonical space, no need to extract
        transition_counts = env_transition_counts.sum(axis=1)  # [num_canonical_states, num_canonical_states]

        # Symmetrize
        transition_counts = (transition_counts + transition_counts.T) / 2.0

        # Add smoothing factor
        smooth_factor = 1e-5
        transition_probs = transition_counts + smooth_factor

        # Normalize (with iterations for better symmetry)
        for _ in range(2):
            row_sums = transition_probs.sum(axis=1, keepdims=True)
            transition_probs = transition_probs / row_sums

            # Re-symmetrize after normalization
            transition_probs = jnp.tril(transition_probs)
            transition_probs = transition_probs + transition_probs.T - jnp.diag(jnp.diag(transition_probs))

        # Compute SVD
        U, s, Vh = transition_svd_jax(transition_probs, k=k)

        return U, s, Vh

    # Vmap over all portal environments
    U_batch, s_batch, Vh_batch = jax.vmap(process_one_env)(batched_transition_counts)

    return U_batch, s_batch, Vh_batch


def compute_transition_matrix_svd(buffer_state, num_states: int, canonical_states, k: int = 10):
    """
    Compute SVD decomposition using canonical state mapping.

    Args:
        buffer_state: Flashbax buffer containing transitions
        num_states: Total number of states in the environment (including obstacles)
        canonical_states: Fixed array of free state indices to use
        k: Number of singular vectors to keep

    Returns:
        U: Left singular vectors (num_canonical_states x k)
        s: Singular values (k,)
        Vh: Right singular vectors transposed (k x num_canonical_states)
    """
    # Import here to avoid circular dependency
    from src.transition_estimation import calculate_transition_counts_jax

    # Get raw transition counts for full state space
    transition_counts_full = calculate_transition_counts_jax(buffer_state, num_states)

    # Extract submatrix for canonical states only
    num_canonical = len(canonical_states)
    transition_counts = transition_counts_full[np.ix_(canonical_states, canonical_states)]

    # Symmetrize
    transition_counts = (transition_counts + transition_counts.T) / 2.0

    # Add smoothing factor
    smooth_factor = 1e-5
    transition_probs = transition_counts + smooth_factor

    # Normalize (with iterations for better symmetry)
    for _ in range(2):
        row_sums = transition_probs.sum(axis=1, keepdims=True)
        transition_probs = transition_probs / row_sums

        # Re-symmetrize after normalization
        transition_probs = jnp.tril(transition_probs)
        transition_probs = transition_probs + transition_probs.T - jnp.diag(jnp.diag(transition_probs))

    # Compute SVD
    U, s, Vh = transition_svd_jax(transition_probs, k=k)

    return U, s, Vh


def generate_dataset(
    num_train_envs: int = 100,
    num_eval_envs: int = 20,
    num_rollouts: int = 1,
    horizon: int = 10000,
    num_portals: int = 10,
    num_eigenvectors: int = 10,
    base_env_name: str = "GridRoom-4",
    output_dir: str = "data/portal_envs",
    seed: int = 42,
):
    """
    Generate a complete dataset of portal environments with rollouts and SVD.

    Args:
        num_train_envs: Number of training environments to create
        num_eval_envs: Number of evaluation environments to create
        num_rollouts: Number of rollouts per environment (N)
        horizon: Horizon length for each rollout (H)
        num_portals: Number of random portals per environment
        num_eigenvectors: Number of eigenvectors to keep (k)
        base_env_name: Name of base environment (e.g., "GridRoom-4")
        output_dir: Directory to save data
        seed: Random seed

    Returns:
        dataset: Dictionary containing all generated data
    """
    num_envs = num_train_envs + num_eval_envs
    print(f"Generating dataset with {num_envs} environments...")
    print(f"  Training environments: {num_train_envs}")
    print(f"  Evaluation environments: {num_eval_envs}")
    print(f"  Rollouts per env: {num_rollouts}")
    print(f"  Horizon: {horizon}")
    print(f"  Portals per env: {num_portals}")
    print(f"  Eigenvectors to keep: {num_eigenvectors}")
    print(f"  Base environment: {base_env_name}")
    print(f"  Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load base environment
    print(f"\nLoading base environment: {base_env_name}")
    base_env = create_environment_from_text(file_name=base_env_name, max_steps=horizon)
    num_states = base_env.observation_space_size()
    num_actions = base_env.action_space_size()

    print(f"  Number of states: {num_states}")
    print(f"  Number of actions: {num_actions}")

    # Get canonical free states (same mapping for all environments)
    canonical_states = get_canonical_free_states(base_env)
    num_canonical_states = len(canonical_states)
    print(f"  Canonical free states: {num_canonical_states}")

    # Storage for dataset
    dataset = {
        "metadata": {
            "num_train_envs": num_train_envs,
            "num_eval_envs": num_eval_envs,
            "num_envs": num_envs,
            "num_rollouts": num_rollouts,
            "horizon": horizon,
            "num_portals": num_portals,
            "num_eigenvectors": num_eigenvectors,
            "num_states": num_states,
            "num_actions": num_actions,
            "num_canonical_states": num_canonical_states,
            "canonical_states": np.array(canonical_states),
            "base_env_name": base_env_name,
            "seed": seed,
        },
        "train_environments": [],
        "train_rollouts": [],
        "train_svd_decompositions": [],
        "eval_environments": [],
        "eval_rollouts": [],
        "eval_svd_decompositions": [],
    }

    rng = np.random.RandomState(seed)

    # Generate data for each environment
    for env_idx in range(num_envs):
        is_train = env_idx < num_train_envs
        split_name = "train" if is_train else "eval"

        if (env_idx + 1) % 10 == 0:
            print(f"\nProcessing {split_name} environment {env_idx + 1}/{num_envs}...")

        # Create environment with random portals
        env_seed = rng.randint(0, 2**31)
        portal_env = create_random_portal_env(
            base_env=base_env,
            num_portals=num_portals,
            seed=env_seed
        )

        # Collect rollouts
        rollout_seed = rng.randint(0, 2**31)
        buffer_state, transitions = collect_rollouts(
            env=portal_env,
            num_rollouts=num_rollouts,
            horizon=horizon,
            seed=rollout_seed
        )

        # Compute SVD using canonical states
        U, s, Vh = compute_transition_matrix_svd(
            buffer_state=buffer_state,
            num_states=num_states,
            canonical_states=canonical_states,
            k=num_eigenvectors
        )

        # Store data in appropriate split
        env_data = {
            "env_idx": env_idx,
            "portals": portal_env.portals,
            "seed": env_seed,
        }

        rollout_data = {
            "env_idx": env_idx,
            "transitions": transitions,
            "num_transitions": len(transitions),
            "seed": rollout_seed,
        }

        svd_data = {
            "env_idx": env_idx,
            "U": np.array(U),  # Left singular vectors (num_canonical_states x k)
            "s": np.array(s),  # Singular values (k,)
            "Vh": np.array(Vh),  # Right singular vectors (k x num_canonical_states)
        }

        if is_train:
            dataset["train_environments"].append(env_data)
            dataset["train_rollouts"].append(rollout_data)
            dataset["train_svd_decompositions"].append(svd_data)
        else:
            dataset["eval_environments"].append(env_data)
            dataset["eval_rollouts"].append(rollout_data)
            dataset["eval_svd_decompositions"].append(svd_data)

    print(f"\n✓ Generated data for {num_train_envs} train + {num_eval_envs} eval = {num_envs} total environments")

    # Save dataset
    output_path = os.path.join(output_dir, "dataset.pkl")
    print(f"\nSaving dataset to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"✓ Dataset saved successfully!")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Training environments: {num_train_envs}")
    print(f"Evaluation environments: {num_eval_envs}")
    print(f"Total environments: {num_envs}")
    total_train_transitions = sum(r['num_transitions'] for r in dataset['train_rollouts'])
    total_eval_transitions = sum(r['num_transitions'] for r in dataset['eval_rollouts'])
    print(f"Total train transitions: {total_train_transitions}")
    print(f"Total eval transitions: {total_eval_transitions}")
    print(f"Average transitions per env: {(total_train_transitions + total_eval_transitions) / num_envs:.1f}")

    # Statistics about canonical states
    print(f"Full state space size: {num_states}")
    print(f"Canonical free states: {num_canonical_states}")
    print(f"State space reduction: {(1 - num_canonical_states/num_states)*100:.1f}%")
    print(f"SVD decompositions shape: U ({num_canonical_states} x {num_eigenvectors}), s ({num_eigenvectors})")
    print("=" * 60)

    return dataset


def load_dataset(data_dir: str = "data/portal_envs"):
    """
    Load a previously generated dataset.

    Args:
        data_dir: Directory containing dataset.pkl

    Returns:
        dataset: Dictionary containing all data
    """
    dataset_path = os.path.join(data_dir, "dataset.pkl")
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    metadata = dataset["metadata"]
    print(f"\nLoaded dataset:")
    print(f"  Training environments: {metadata['num_train_envs']}")
    print(f"  Evaluation environments: {metadata['num_eval_envs']}")
    print(f"  Total environments: {metadata['num_envs']}")
    print(f"  Rollouts per env: {metadata['num_rollouts']}")
    print(f"  Horizon: {metadata['horizon']}")
    print(f"  Portals per env: {metadata['num_portals']}")
    print(f"  Eigenvectors: {metadata['num_eigenvectors']}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate portal environment dataset")
    parser.add_argument("--num_train_envs", type=int, default=100, help="Number of training environments")
    parser.add_argument("--num_eval_envs", type=int, default=20, help="Number of evaluation environments")
    parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts per environment (N)")
    parser.add_argument("--horizon", type=int, default=10000, help="Horizon length (H)")
    parser.add_argument("--num_portals", type=int, default=10, help="Number of portals per environment")
    parser.add_argument("--num_eigenvectors", type=int, default=10, help="Number of eigenvectors to keep (k)")
    parser.add_argument("--base_env", type=str, default="GridRoom-4", help="Base environment name")
    parser.add_argument("--output_dir", type=str, default="data/portal_envs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    dataset = generate_dataset(
        num_train_envs=args.num_train_envs,
        num_eval_envs=args.num_eval_envs,
        num_rollouts=args.num_rollouts,
        horizon=args.horizon,
        num_portals=args.num_portals,
        num_eigenvectors=args.num_eigenvectors,
        base_env_name=args.base_env,
        output_dir=args.output_dir,
        seed=args.seed,
    )
