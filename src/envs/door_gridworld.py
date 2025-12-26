"""
GridWorld environment with irreversible doors.

Irreversible doors are one-way passages: you can go from state s to s' via action a,
but the reverse transition from s' back to s via the reverse action is blocked.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List, Set
from src.envs.gridworld import GridWorldEnv, GridWorldState


class DoorGridWorldEnv(GridWorldEnv):
    """
    GridWorld environment with irreversible doors that block specific transitions.

    A door blocks a specific (state, action) pair. When the agent tries to take
    that action from that state, they stay in place instead of moving.
    """

    def __init__(self, blocked_transitions: Set[Tuple[int, int]] = None, **kwargs):
        """
        Initialize the DoorGridWorld environment.

        Args:
            blocked_transitions: Set of (state_idx, action) pairs that are blocked
            **kwargs: Arguments passed to GridWorldEnv
        """
        super().__init__(**kwargs)

        # Store blocked transitions as a set: (state_idx, action)
        self.blocked_transitions = blocked_transitions if blocked_transitions is not None else set()

        # Convert to JAX arrays for efficient lookup
        if len(self.blocked_transitions) > 0:
            blocked_list = list(self.blocked_transitions)
            self.blocked_states = jnp.array([k[0] for k in blocked_list], dtype=self.dtype)
            self.blocked_actions = jnp.array([k[1] for k in blocked_list], dtype=self.dtype)
            self.has_doors = True
        else:
            self.blocked_states = jnp.array([], dtype=self.dtype)
            self.blocked_actions = jnp.array([], dtype=self.dtype)
            self.has_doors = False

    def step(self, key, state, action, params=None):
        """Take a step in the environment, checking for blocked doors first."""
        # Get current position and status
        position = state.position
        terminal = state.terminal
        steps = state.steps

        # Get current state index
        current_state_idx = self.get_state_representation(state)

        # Only take action if not in terminal state
        def _step_impl(args):
            position, steps, current_state_idx = args

            # Check if this (state, action) pair is blocked by a door
            def check_blocked(i):
                """Check if blocked transition i matches current state and action."""
                matches_state = self.blocked_states[i] == current_state_idx
                matches_action = self.blocked_actions[i] == action
                return matches_state & matches_action

            # Check if current transition is blocked
            is_blocked = jax.lax.cond(
                self.has_doors,
                lambda _: jnp.any(jax.vmap(check_blocked)(jnp.arange(len(self.blocked_states)))),
                lambda _: jnp.array(False),
                operand=None
            )

            def stay_in_place(_):
                """Door is blocking this transition - stay in place."""
                return position, steps + 1

            def apply_normal_physics(_):
                """Apply normal gridworld physics."""
                # Get action effect
                action_effect = self.action_effects[action]

                # Calculate tentative new position
                new_position_raw = position + action_effect

                # Apply boundary constraints
                new_position = jnp.clip(
                    new_position_raw,
                    jnp.array([0, 0]),
                    jnp.array([self.width - 1, self.height - 1])
                )

                # Check if new position is an obstacle
                is_obstacle = jax.lax.cond(
                    self.has_obstacles,
                    lambda _: jnp.any(jnp.all(new_position == self.obstacles, axis=1)),
                    lambda _: jnp.array(False),
                    operand=None
                )

                # If obstacle, stay in place
                final_position = jnp.where(
                    is_obstacle,
                    position,
                    new_position
                )

                return final_position, steps + 1

            # Either stay in place (blocked) or apply normal physics
            final_position, new_steps = jax.lax.cond(
                is_blocked,
                stay_in_place,
                apply_normal_physics,
                operand=None
            )

            return final_position, new_steps

        # Either take action or keep same state if terminal
        position, steps = jax.lax.cond(
            terminal,
            lambda args: (args[0], args[1]),  # If terminal, stay in place
            _step_impl,  # If not terminal, apply step function
            (position, steps, current_state_idx)
        )

        # Check if at goal or max steps reached
        at_goal = jax.lax.cond(
            self.has_goal,
            lambda _: jnp.all(position == self.goal_pos),
            lambda _: jnp.array(False),
            operand=None
        )
        max_steps_reached = steps >= self.max_steps

        # Determine if terminal state
        new_terminal = at_goal | max_steps_reached

        # Create new state
        new_state = GridWorldState(
            position=position,
            terminal=new_terminal,
            steps=steps
        )

        # Compute reward (1.0 for reaching goal, 0 otherwise)
        reward = jnp.where(at_goal, 1.0, 0.0)

        return new_state, reward


def create_door_gridworld_from_base(
    base_env: GridWorldEnv,
    doors: List[Tuple[int, int, int, int]],
    canonical_states: jnp.ndarray = None
) -> DoorGridWorldEnv:
    """
    Create a DoorGridWorld from a base environment and door configuration.

    Args:
        base_env: Base GridWorld environment
        doors: List of door tuples (s_canonical, a_forward, s'_canonical, a_reverse)
               These specify which reverse transitions to block
        canonical_states: Array mapping canonical indices to full state indices
                         (if None, assumes 1:1 mapping)

    Returns:
        DoorGridWorldEnv with blocked reverse transitions
    """
    # If no canonical states mapping, assume identity
    if canonical_states is None:
        num_states = base_env.width * base_env.height
        canonical_states = jnp.arange(num_states)

    # Build set of blocked transitions from doors
    # Each door blocks the REVERSE transition
    blocked_transitions = set()
    for s_canonical, a_forward, s_prime_canonical, a_reverse in doors:
        # Block (s', a_reverse) so that you can't go from s' back to s
        s_prime_full = int(canonical_states[s_prime_canonical])
        blocked_transitions.add((s_prime_full, a_reverse))

    # Create new environment with blocked transitions
    door_env = DoorGridWorldEnv(
        width=base_env.width,
        height=base_env.height,
        obstacles=base_env.obstacles if base_env.has_obstacles else None,
        start_pos=base_env.start_pos if base_env.fixed_start else None,
        goal_pos=base_env.goal_pos if base_env.has_goal else None,
        max_steps=base_env.max_steps,
        precision=32 if base_env.dtype == jnp.int32 else 64,
        blocked_transitions=blocked_transitions,
    )

    return door_env


def get_reverse_action(action: int) -> int:
    """
    Get the reverse action for grid navigation.

    Assumes action mapping: 0=up, 1=right, 2=down, 3=left

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


def find_reversible_transitions(
    base_env: GridWorldEnv,
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
    num_actions = base_env.action_space
    width = base_env.width

    # Create mapping from full state to canonical index
    state_to_canonical = {int(state): idx for idx, state in enumerate(canonical_states)}

    # Action effects
    action_effects = np.array([
        [0, -1],  # 0: up
        [1, 0],   # 1: right
        [0, 1],   # 2: down
        [-1, 0],  # 3: left
    ])

    for s_canonical_idx, s_full in enumerate(canonical_states):
        s_full = int(s_full)
        s_y = s_full // width
        s_x = s_full % width

        for a in range(num_actions):
            # Get successor state for (s, a)
            action_effect = action_effects[a]
            next_pos = np.array([s_x, s_y]) + action_effect

            # Apply boundary constraints
            next_pos[0] = np.clip(next_pos[0], 0, base_env.width - 1)
            next_pos[1] = np.clip(next_pos[1], 0, base_env.height - 1)

            # Check if new position is an obstacle
            if base_env.has_obstacles:
                is_obstacle = False
                for obs in base_env.obstacles:
                    if np.all(next_pos == obs):
                        is_obstacle = True
                        break
                if is_obstacle:
                    continue  # Skip obstacle positions

            next_x, next_y = next_pos
            s_prime_full = next_y * width + next_x

            # Skip self-loops (e.g., hitting walls/boundaries)
            if s_prime_full == s_full:
                continue

            # Check if successor is in canonical states
            if s_prime_full not in state_to_canonical:
                continue

            s_prime_canonical = state_to_canonical[s_prime_full]

            # Check if reverse action exists
            a_reverse = get_reverse_action(a)

            # Get successor for reverse action from s'
            s_prime_y = s_prime_full // width
            s_prime_x = s_prime_full % width
            reverse_effect = action_effects[a_reverse]
            reverse_pos = np.array([s_prime_x, s_prime_y]) + reverse_effect

            # Apply boundary constraints
            reverse_pos[0] = np.clip(reverse_pos[0], 0, base_env.width - 1)
            reverse_pos[1] = np.clip(reverse_pos[1], 0, base_env.height - 1)

            # Check if reverse position is an obstacle
            if base_env.has_obstacles:
                is_obstacle = False
                for obs in base_env.obstacles:
                    if np.all(reverse_pos == obs):
                        is_obstacle = True
                        break
                if is_obstacle:
                    continue

            reverse_x, reverse_y = reverse_pos
            s_reverse_full = reverse_y * width + reverse_x

            # Check if reverse action leads back to s
            if s_reverse_full == s_full:
                reversible_pairs.append((s_canonical_idx, a, s_prime_canonical, a_reverse))

    return reversible_pairs


def create_random_doors(
    base_env: GridWorldEnv,
    canonical_states: jnp.ndarray,
    num_doors: int,
    seed: int = 42
) -> Dict:
    """
    Create irreversible doors by randomly selecting reversible transitions.

    Args:
        base_env: Base grid environment
        canonical_states: Array of free state indices
        num_doors: Number of doors to create
        seed: Random seed

    Returns:
        Dictionary containing:
        - doors: List of door definitions
        - num_doors: Actual number of doors created
        - total_reversible: Total number of reversible transitions
    """
    # Find all reversible transitions
    reversible_pairs = find_reversible_transitions(base_env, canonical_states)

    if len(reversible_pairs) == 0:
        print("Warning: No reversible transitions found!")
        return {"doors": [], "num_doors": 0, "total_reversible": 0}

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
