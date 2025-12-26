"""
Environment generator for irreversible doors using DoorGridWorldEnv.

This module creates grid environments with one-way passages (irreversible doors)
by using the DoorGridWorldEnv class which implements doors in the environment dynamics.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict

from src.envs.door_gridworld import (
    create_door_gridworld_from_base,
    create_random_doors,
    find_reversible_transitions,
    get_reverse_action,
)
from src.data_collection import collect_transition_counts


def get_canonical_free_states(base_env):
    """
    Get the canonical set of free (non-obstacle) states from the base environment.

    Args:
        base_env: GridWorld environment

    Returns:
        canonical_states: Array of free state indices, sorted
    """
    width = base_env.width
    height = base_env.height
    all_states = set(range(width * height))
    obstacle_states = set()

    if base_env.has_obstacles:
        for obs in base_env.obstacles:
            obs_x, obs_y = int(obs[0]), int(obs[1])
            if 0 <= obs_x < width and 0 <= obs_y < height:
                state_idx = obs_y * width + obs_x
                obstacle_states.add(state_idx)

    free_states = sorted(list(all_states - obstacle_states))
    return jnp.array(free_states, dtype=jnp.int32)


def create_door_environment_data(
    base_env,
    canonical_states: jnp.ndarray,
    num_doors: int,
    num_rollouts: int,
    num_steps: int,
    seed: int = 42
) -> Tuple[jnp.ndarray, Dict]:
    """
    Create a single environment with irreversible doors and collect transition data.

    Uses DoorGridWorldEnv to implement doors in the environment dynamics,
    so the agent actually experiences the one-way passages during data collection.

    Args:
        base_env: Base grid environment
        canonical_states: Free state indices
        num_doors: Number of doors to create
        num_rollouts: Number of rollouts for data collection
        num_steps: Steps per rollout
        seed: Random seed

    Returns:
        Tuple of (transition_counts, metadata)
        - transition_counts: Shape [num_canonical_states, num_actions, num_canonical_states]
        - metadata: Dictionary with door configuration and environment info
    """
    num_canonical_states = len(canonical_states)
    num_states_full = base_env.width * base_env.height

    # Create random doors
    door_config = create_random_doors(
        base_env,
        canonical_states,
        num_doors=num_doors,
        seed=seed
    )

    # Create environment with doors in the dynamics
    door_env = create_door_gridworld_from_base(
        base_env,
        door_config['doors'],
        canonical_states
    )

    # Collect transition data from door environment
    # The agent will never experience blocked transitions
    transition_counts_full, _ = collect_transition_counts(
        env=door_env,
        num_envs=num_rollouts,
        num_steps=num_steps,
        num_states=num_states_full,
        seed=seed
    )

    # Extract canonical state subspace
    transition_counts_canonical = np.zeros(
        (num_canonical_states, base_env.action_space, num_canonical_states)
    )

    for i_canonical, i_full in enumerate(canonical_states):
        for j_canonical, j_full in enumerate(canonical_states):
            transition_counts_canonical[i_canonical, :, j_canonical] = transition_counts_full[
                int(i_full), :, int(j_full)
            ]

    metadata = {
        "doors": door_config["doors"],
        "num_doors": door_config["num_doors"],
        "total_reversible": door_config.get("total_reversible", 0),
        "canonical_states": canonical_states,
        "grid_width": base_env.width,
        "grid_height": base_env.height,
    }

    return jnp.array(transition_counts_canonical), metadata


def generate_batched_door_environments(
    base_env,
    canonical_states: jnp.ndarray,
    num_envs: int,
    num_doors: int,
    num_rollouts_per_env: int,
    num_steps: int,
    seed: int = 42
) -> Tuple[jnp.ndarray, Dict]:
    """
    Generate multiple door environments with batched processing.

    Each environment has different randomly-placed doors, implemented
    using DoorGridWorldEnv so doors are in the actual dynamics.

    Args:
        base_env: Base grid environment
        canonical_states: Free state indices
        num_envs: Number of environments to generate
        num_doors: Number of doors per environment
        num_rollouts_per_env: Rollouts per environment
        num_steps: Steps per rollout
        seed: Base random seed

    Returns:
        Tuple of (batched_transition_counts, metadata)
        - batched_transition_counts: Shape [num_envs, num_canonical_states, num_actions, num_canonical_states]
        - metadata: Dictionary with all door configurations and environment info
    """
    num_canonical_states = len(canonical_states)
    num_actions = base_env.action_space

    all_transition_counts = []
    all_doors = []

    rng = np.random.RandomState(seed)

    print(f"Generating {num_envs} door environments...")
    for env_idx in range(num_envs):
        env_seed = rng.randint(0, 2**31)

        # Create door environment and collect data
        transition_counts, env_metadata = create_door_environment_data(
            base_env=base_env,
            canonical_states=canonical_states,
            num_doors=num_doors,
            num_rollouts=num_rollouts_per_env,
            num_steps=num_steps,
            seed=env_seed
        )

        all_transition_counts.append(transition_counts)
        all_doors.append(env_metadata["doors"])

        if (env_idx + 1) % 10 == 0:
            print(f"  Generated {env_idx + 1}/{num_envs} environments")

    batched_counts = jnp.stack(all_transition_counts, axis=0)

    # Get obstacle information
    obstacles = []
    if base_env.has_obstacles:
        obstacles = [(int(obs[0]), int(obs[1])) for obs in base_env.obstacles]

    metadata = {
        "num_envs": num_envs,
        "num_canonical_states": num_canonical_states,
        "num_actions": num_actions,
        "canonical_states": canonical_states,
        "grid_width": base_env.width,
        "grid_height": base_env.height,
        "obstacles": obstacles,
        "all_doors": all_doors,
        "first_env_doors": all_doors[0] if num_envs > 0 else [],
        "num_doors_per_env": num_doors,
    }

    print(f"Completed! Generated {num_envs} environments with {num_doors} doors each")

    return batched_counts, metadata
