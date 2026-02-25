"""
Utility functions for creating GridWorldEnv instances with irreversible doors.

Doors are one-way passages: the forward transition (s → s') via action a is
allowed, but the reverse (s' → s) via the opposite action is blocked.
The GridWorldEnv class handles the actual door logic; these helpers exist to
build the blocked_transitions set from higher-level door descriptions.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List, Set

from src.envs.gridworld import GridWorldEnv


def create_door_gridworld_from_base(
    base_env: GridWorldEnv,
    doors: List[Tuple[int, int, int, int]],
    canonical_states: jnp.ndarray = None,
) -> GridWorldEnv:
    """
    Create a GridWorldEnv from a base environment and a list of door definitions.

    Each door makes a transition one-way by blocking the reverse direction.

    Args:
        base_env: Base GridWorld environment
        doors: List of (s_canonical, a_forward, s'_canonical, a_reverse) tuples.
               The reverse transition (s', a_reverse) will be blocked.
        canonical_states: Maps canonical indices to full state indices.
                          If None, assumes 1:1 mapping.

    Returns:
        GridWorldEnv with the combined blocked_transitions and portals.
    """
    if canonical_states is None:
        canonical_states = jnp.arange(base_env.width * base_env.height)

    # Preserve existing doors
    blocked_transitions = set(base_env.blocked_transitions) if base_env.has_doors else set()

    # Add new doors
    for s_canonical, a_forward, s_prime_canonical, a_reverse in doors:
        s_prime_full = int(canonical_states[s_prime_canonical])
        blocked_transitions.add((s_prime_full, a_reverse))

    # Preserve existing portals
    portals = dict(base_env.portals) if base_env.has_portals else {}

    return GridWorldEnv(
        width=base_env.width,
        height=base_env.height,
        obstacles=base_env.obstacles if base_env.has_obstacles else None,
        start_pos=base_env.start_pos if base_env.fixed_start else None,
        goal_pos=base_env.goal_pos if base_env.has_goal else None,
        max_steps=base_env.max_steps,
        precision=32 if base_env.dtype == jnp.int32 else 64,
        blocked_transitions=blocked_transitions,
        portals=portals if portals else None,
    )


def get_reverse_action(action: int) -> int:
    """
    Get the reverse action for grid navigation.

    Assumes action mapping: 0=up, 1=right, 2=down, 3=left
    """
    return {0: 2, 1: 3, 2: 0, 3: 1}[action]


def find_reversible_transitions(
    base_env: GridWorldEnv,
    canonical_states: jnp.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Find all (s, a, s', a') tuples where (s, a) -> s' is reversible.

    A transition is reversible if applying the reverse action a' from s'
    leads back to s.

    Returns:
        List of (s_canonical, a, s'_canonical, a') tuples.
    """
    reversible_pairs = []
    width = base_env.width

    action_effects = np.array([
        [0, -1],  # 0: up
        [1, 0],   # 1: right
        [0, 1],   # 2: down
        [-1, 0],  # 3: left
    ])

    state_to_canonical = {int(state): idx for idx, state in enumerate(canonical_states)}

    for s_canonical_idx, s_full in enumerate(canonical_states):
        s_full = int(s_full)
        s_y, s_x = divmod(s_full, width)

        for a in range(base_env.action_space):
            dx, dy = action_effects[a]
            nx, ny = np.clip(s_x + dx, 0, base_env.width - 1), np.clip(s_y + dy, 0, base_env.height - 1)

            if base_env.has_obstacles:
                if any(np.all(np.array([nx, ny]) == obs) for obs in base_env.obstacles):
                    continue

            s_prime_full = int(ny * width + nx)
            if s_prime_full == s_full:
                continue
            if s_prime_full not in state_to_canonical:
                continue

            s_prime_canonical = state_to_canonical[s_prime_full]
            a_reverse = get_reverse_action(a)
            rdx, rdy = action_effects[a_reverse]
            rx, ry = np.clip(nx + rdx, 0, base_env.width - 1), np.clip(ny + rdy, 0, base_env.height - 1)

            if base_env.has_obstacles:
                if any(np.all(np.array([rx, ry]) == obs) for obs in base_env.obstacles):
                    continue

            if int(ry * width + rx) == s_full:
                reversible_pairs.append((s_canonical_idx, a, s_prime_canonical, a_reverse))

    return reversible_pairs


def create_random_doors(
    base_env: GridWorldEnv,
    canonical_states: jnp.ndarray,
    num_doors: int,
    seed: int = 42,
) -> Dict:
    """
    Create irreversible doors by randomly selecting reversible transitions.

    Returns:
        Dict with keys: doors, num_doors, total_reversible.
    """
    reversible_pairs = find_reversible_transitions(base_env, canonical_states)

    if len(reversible_pairs) == 0:
        print("Warning: No reversible transitions found!")
        return {"doors": [], "num_doors": 0, "total_reversible": 0}

    rng = np.random.RandomState(seed)
    num_doors_actual = min(num_doors, len(reversible_pairs))
    selected = rng.choice(len(reversible_pairs), size=num_doors_actual, replace=False)

    return {
        "doors": [reversible_pairs[i] for i in selected],
        "num_doors": num_doors_actual,
        "total_reversible": len(reversible_pairs),
    }

