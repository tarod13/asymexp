"""
Environment generator for irreversible doors.

This module creates grid environments with one-way passages (irreversible doors).
Unlike portals that teleport to arbitrary locations, doors maintain normal grid
connectivity but make certain transitions one-way only.

For a door at (s, a) -> s':
- You CAN go from s to s' via action a: p(s' | s, a) = 1
- You CANNOT go back from s' to s via the reverse action: p(s | s', a_reverse) = 0
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Dict, Optional


def get_reverse_action(action: int) -> int:
    """
    Get the reverse action for grid navigation.

    Assumes action mapping from GridWorldEnv: 0=up, 1=right, 2=down, 3=left

    Args:
        action: Action index (0-3)

    Returns:
        Reverse action index
    """
    reverse_map = {
        0: 2,  # up <-> down
        1: 3,  # right <-> left
        2: 0,  # down <-> up
        3: 1,  # left <-> right
    }
    return reverse_map.get(action, action)


def get_next_position(position_xy: np.ndarray, action: int, base_env) -> Tuple[int, int]:
    """
    Compute the next position from a given position and action.

    Args:
        position_xy: Current position [x, y]
        action: Action to take
        base_env: GridWorld environment

    Returns:
        Next position (x, y)
    """
    # Get action effect
    action_effects = np.array([
        [0, -1],  # 0: up
        [1, 0],   # 1: right
        [0, 1],   # 2: down
        [-1, 0],  # 3: left
    ])

    action_effect = action_effects[action]

    # Calculate tentative new position
    new_position = position_xy + action_effect

    # Apply boundary constraints
    new_position[0] = np.clip(new_position[0], 0, base_env.width - 1)
    new_position[1] = np.clip(new_position[1], 0, base_env.height - 1)

    # Check if new position is an obstacle
    if base_env.has_obstacles:
        for obs in base_env.obstacles:
            if np.all(new_position == obs):
                # Hit obstacle, stay in place
                return tuple(position_xy)

    return tuple(new_position)


def find_reversible_transitions(
    base_env,
    canonical_states: jnp.ndarray
) -> List[Tuple[int, int, int, int]]:
    """
    Find all state-action pairs that have reversible transitions.

    A transition (s, a) -> s' is reversible if there exists an action a'
    such that (s', a') -> s.

    Args:
        base_env: Base grid environment
        canonical_states: Array of free (non-obstacle) state indices

    Returns:
        List of tuples (s_canonical, a, s'_canonical, a') where:
        - s_canonical: source state in canonical space
        - a: forward action
        - s'_canonical: successor state in canonical space
        - a': reverse action (from s' back to s)
    """
    reversible_pairs = []
    num_actions = base_env.action_space_size()
    width = base_env.width

    # Create mapping from full state to canonical index
    state_to_canonical = {int(state): idx for idx, state in enumerate(canonical_states)}

    for s_canonical_idx, s_full in enumerate(canonical_states):
        s_full = int(s_full)
        s_y = s_full // width
        s_x = s_full % width

        for a in range(num_actions):
            # Get successor state for (s, a)
            next_x, next_y = get_next_position(np.array([s_x, s_y]), a, base_env)
            s_prime_full = next_y * width + next_x

            # Check if successor is in canonical states (not an obstacle)
            if s_prime_full not in state_to_canonical:
                continue

            s_prime_canonical = state_to_canonical[s_prime_full]

            # Skip self-loops (e.g., hitting walls)
            if s_prime_full == s_full:
                continue

            # Check if reverse action exists
            a_reverse = get_reverse_action(a)

            # Get successor for reverse action
            s_prime_y = s_prime_full // width
            s_prime_x = s_prime_full % width
            reverse_x, reverse_y = get_next_position(
                np.array([s_prime_x, s_prime_y]),
                a_reverse,
                base_env
            )
            s_reverse_full = reverse_y * width + reverse_x

            # Check if reverse action leads back to s
            if s_reverse_full == s_full:
                reversible_pairs.append((s_canonical_idx, a, s_prime_canonical, a_reverse))

    return reversible_pairs


def create_irreversible_doors(
    base_env,
    canonical_states: jnp.ndarray,
    num_doors: int,
    seed: int = 42
) -> Dict:
    """
    Create a set of irreversible doors by selecting random reversible transitions
    and blocking the reverse direction.

    Args:
        base_env: Base grid environment
        canonical_states: Array of free state indices
        num_doors: Number of irreversible doors to create
        seed: Random seed for door selection

    Returns:
        Dictionary containing:
        - doors: List of door definitions [(s_canonical, a, s'_canonical, a_reverse), ...]
        - num_doors: Actual number of doors created
    """
    # Find all reversible transitions
    reversible_pairs = find_reversible_transitions(base_env, canonical_states)

    if len(reversible_pairs) == 0:
        print("Warning: No reversible transitions found!")
        return {"doors": [], "num_doors": 0}

    # Randomly select doors
    rng = np.random.RandomState(seed)
    num_doors_actual = min(num_doors, len(reversible_pairs))

    selected_indices = rng.choice(len(reversible_pairs), size=num_doors_actual, replace=False)
    doors = [reversible_pairs[i] for i in selected_indices]

    return {
        "doors": doors,
        "num_doors": num_doors_actual,
        "total_reversible": len(reversible_pairs)
    }


def apply_doors_to_transition_matrix(
    transition_counts: jnp.ndarray,
    doors: List[Tuple[int, int, int, int]]
) -> jnp.ndarray:
    """
    Apply irreversible doors to a transition count matrix by blocking reverse transitions.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states]
        doors: List of door tuples (s, a, s', a_reverse)

    Returns:
        Modified transition counts with doors applied
    """
    # Work with numpy for easier indexing
    modified_counts = np.array(transition_counts)

    for s_canonical, a_forward, s_prime_canonical, a_reverse in doors:
        # Block the reverse transition: p(s | s', a_reverse) = 0
        # This means setting counts for (s', a_reverse, s) to 0
        modified_counts[s_prime_canonical, a_reverse, s_canonical] = 0

    return jnp.array(modified_counts)


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

    Args:
        base_env: Base grid environment
        canonical_states: Free state indices
        num_doors: Number of doors to create
        num_rollouts: Number of rollouts for data collection
        num_steps: Steps per rollout
        seed: Random seed

    Returns:
        Tuple of (transition_counts, metadata)
    """
    # Create doors
    door_config = create_irreversible_doors(base_env, canonical_states, num_doors, seed)

    # Collect base transition counts using standard function
    from src.data_collection import collect_transition_counts

    num_canonical_states = len(canonical_states)

    # Collect transition data
    transition_counts_full, _ = collect_transition_counts(
        env=base_env,
        num_envs=num_rollouts,
        num_steps=num_steps,
        num_states=base_env.width * base_env.height,
        seed=seed
    )

    # Project to canonical state space
    # transition_counts_full is [num_states, num_actions, num_states]
    # We need to extract only canonical states
    transition_counts_canonical = np.zeros(
        (num_canonical_states, base_env.action_space_size(), num_canonical_states)
    )

    for i_canonical, i_full in enumerate(canonical_states):
        for j_canonical, j_full in enumerate(canonical_states):
            transition_counts_canonical[i_canonical, :, j_canonical] = transition_counts_full[
                int(i_full), :, int(j_full)
            ]

    # Apply doors to modify transition dynamics
    modified_counts = apply_doors_to_transition_matrix(
        jnp.array(transition_counts_canonical),
        door_config["doors"]
    )

    metadata = {
        "doors": door_config["doors"],
        "num_doors": door_config["num_doors"],
        "total_reversible": door_config.get("total_reversible", 0),
        "canonical_states": canonical_states,
        "grid_width": base_env.width,
        "grid_height": base_env.height,
    }

    return modified_counts, metadata


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
    """
    num_canonical_states = len(canonical_states)
    num_actions = base_env.action_space_size()

    all_transition_counts = []
    all_doors = []

    rng = np.random.RandomState(seed)

    for env_idx in range(num_envs):
        env_seed = rng.randint(0, 2**31)

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

    return batched_counts, metadata
