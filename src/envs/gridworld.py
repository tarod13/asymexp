import jax
import jax.numpy as jnp
from typing import NamedTuple, Any


# Define the environment state for GridWorld
class GridWorldState(NamedTuple):
    position: jnp.ndarray  # (x, y) coordinates
    terminal: jnp.ndarray  # Boolean flag for terminal state
    steps: jnp.ndarray  # Step counter


# GridWorld Environment
class GridWorldEnv:
    def __init__(
        self,
        width,
        height,
        obstacles=None,
        start_pos=None,
        goal_pos=None,
        max_steps=100,
        precision=32,
        asymmetric_transitions=None,
        portals=None,
        **kwargs,
    ):
        """
        Initialize the GridWorld environment.

        Args:
            width: Grid width
            height: Grid height
            obstacles: List of (x, y) coordinates for obstacles
            start_pos: Starting position (x, y) or None for random start
            goal_pos: Goal position (x, y) or None for no goal
            max_steps: Maximum steps per episode
            precision: 32 or 64
            asymmetric_transitions: Dict mapping (state_idx, action) -> float,
                where the float is the probability [0, 1] that the reverse
                transition succeeds.  0 = fully blocked (hard door), 1 = fully
                reversible (no effect).  Agent stays in place when the sampled
                uniform exceeds the probability.
            portals: Dict mapping (state_idx, action) -> dest_state_idx.
                Portals override normal physics and teleport the agent.
                Portals are checked before doors.
        """
        self.width = width
        self.height = height
        if precision == 32:
            self.dtype = jnp.int32
        elif precision == 64:
            self.dtype = jnp.int64
        else:
            raise ValueError("Precision must be 32 or 64")

        # Store the explicit start position if provided
        self.fixed_start = start_pos is not None
        self.start_pos = jnp.array(start_pos if start_pos is not None else (-1, -1), dtype=self.dtype)

        # Setup goal
        self.has_goal = goal_pos is not None
        self.goal_pos = jnp.array(goal_pos if goal_pos is not None else (-1, -1), dtype=self.dtype)

        # Setup obstacles
        if obstacles is not None and len(obstacles) > 0:
            self.has_obstacles = True
            self.obstacles = jnp.array(obstacles, dtype=self.dtype)
        else:
            self.has_obstacles = False
            self.obstacles = jnp.array([[-1, -1]], dtype=self.dtype)

        self.action_space = 4  # up, right, down, left
        self.max_steps = max_steps

        # Define action effects (up, right, down, left)
        self.action_effects = jnp.array([
            [0, -1],  # up
            [1, 0],   # right
            [0, 1],   # down
            [-1, 0],  # left
        ], dtype=self.dtype)

        # --- Doors (asymmetric_transitions) ---
        self.asymmetric_transitions = asymmetric_transitions if asymmetric_transitions is not None else {}
        if len(self.asymmetric_transitions) > 0:
            asym_list = list(self.asymmetric_transitions.items())
            self.asym_states  = jnp.array([k[0] for k, _ in asym_list], dtype=self.dtype)
            self.asym_actions = jnp.array([k[1] for k, _ in asym_list], dtype=self.dtype)
            self.asym_probs   = jnp.array([v      for _, v in asym_list], dtype=jnp.float32)
            self.has_doors = True
        else:
            self.asym_states  = jnp.array([], dtype=self.dtype)
            self.asym_actions = jnp.array([], dtype=self.dtype)
            self.asym_probs   = jnp.array([], dtype=jnp.float32)
            self.has_doors = False

        # --- Portals ---
        self.portals = portals if portals is not None else {}
        if len(self.portals) > 0:
            portal_keys = list(self.portals.keys())
            self.portal_states       = jnp.array([k[0] for k in portal_keys], dtype=self.dtype)
            self.portal_actions      = jnp.array([k[1] for k in portal_keys], dtype=self.dtype)
            self.portal_destinations = jnp.array(list(self.portals.values()),  dtype=self.dtype)
            self.has_portals = True
        else:
            self.portal_states       = jnp.array([], dtype=self.dtype)
            self.portal_actions      = jnp.array([], dtype=self.dtype)
            self.portal_destinations = jnp.array([], dtype=self.dtype)
            self.has_portals = False

        # Precompute valid start positions (positions that are not obstacles or goal)
        self.valid_positions = self._compute_valid_positions()

    def _compute_valid_positions(self):
        """Compute all valid positions for starting (not obstacles, not goal)."""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                pos = jnp.array([x, y], dtype=self.dtype)

                # Check if position is an obstacle
                is_obstacle = self.has_obstacles and jnp.any(jnp.all(pos == self.obstacles, axis=1))

                # Check if position is the goal
                is_goal = self.has_goal and jnp.all(pos == self.goal_pos)

                # If it's neither an obstacle nor the goal, it's valid
                if not is_obstacle and not is_goal:
                    valid_positions.append((x, y))

        return jnp.array(valid_positions, dtype=self.dtype)

    def _is_valid_position(self, position):
        """Check if a position is valid (not an obstacle)."""
        if self.has_obstacles:
            is_obstacle = jnp.any(jnp.all(position == self.obstacles, axis=1))
            return jnp.logical_not(is_obstacle)
        return jnp.array(True)

    def _get_random_start_position(self, key):
        """Get a random valid start position."""
        # Choose a random index from valid positions
        if len(self.valid_positions) > 0:
            idx = jax.random.randint(key, (), 0, len(self.valid_positions))
            return self.valid_positions[idx]
        else:
            # Fallback to (0,0) if no valid positions
            return jnp.array([0, 0], dtype=self.dtype)

    def _state_idx_to_position(self, state_idx):
        """Convert a flat state index to (x, y) position."""
        x = state_idx % self.width
        y = state_idx // self.width
        return jnp.array([x, y], dtype=self.dtype)

    def reset(self, key, params=None):
        """Reset the environment and return initial state."""
        # Decide on the start position
        if self.fixed_start:
            position = self.start_pos
        else:
            # Get random start position
            position = self._get_random_start_position(key)

        return GridWorldState(
            position=position,
            terminal=jnp.array(False),
            steps=jnp.array(0, dtype=self.dtype)
        )

    def step(self, key, state, action, params=None):
        """Take a step in the environment.

        Priority order for non-terminal states:
          1. Portal  — teleport to destination if (state, action) is a portal
          2. Door    — stay in place if (state, action) is blocked
          3. Physics — normal movement with boundary/obstacle checks
        """
        position = state.position
        terminal = state.terminal
        steps    = state.steps
        current_state_idx = self.get_state_representation(state)

        key, door_key = jax.random.split(key)

        def _step_impl(args):
            pos, st, csidx = args

            # ----------------------------------------------------------
            # 1. Portal check
            # ----------------------------------------------------------
            def check_portal(i):
                return (self.portal_states[i] == csidx) & (self.portal_actions[i] == action)

            portal_match = jax.lax.cond(
                self.has_portals,
                lambda _: jnp.any(jax.vmap(check_portal)(jnp.arange(len(self.portal_states)))),
                lambda _: jnp.array(False),
                operand=None,
            )

            def apply_portal(_):
                matches    = jax.vmap(check_portal)(jnp.arange(len(self.portal_states)))
                portal_idx = jnp.argmax(matches)
                dest_pos   = self._state_idx_to_position(self.portal_destinations[portal_idx])
                return dest_pos, st + 1

            # ----------------------------------------------------------
            # 2. Door check (only reached when no portal matched)
            # ----------------------------------------------------------
            def check_door(i):
                return (self.asym_states[i] == csidx) & (self.asym_actions[i] == action)

            def after_portal_check(_):
                # Find whether this (state, action) has a door and its reversal prob.
                def find_door(_):
                    matches   = jax.vmap(check_door)(jnp.arange(len(self.asym_states)))
                    has_match = jnp.any(matches)
                    door_idx  = jnp.argmax(matches)
                    prob      = jnp.where(has_match, self.asym_probs[door_idx], jnp.array(1.0))
                    return has_match, prob

                has_door, door_prob = jax.lax.cond(
                    self.has_doors,
                    find_door,
                    lambda _: (jnp.array(False), jnp.array(1.0)),
                    operand=None,
                )

                # Sample: transition is blocked when u >= door_prob
                u          = jax.random.uniform(door_key)
                is_blocked = has_door & (u >= door_prob)

                # -------------------------------------------------------
                # 3. Normal physics (only reached when not blocked)
                # -------------------------------------------------------
                def normal_physics(_):
                    new_pos_raw = pos + self.action_effects[action]
                    new_pos = jnp.clip(
                        new_pos_raw,
                        jnp.array([0, 0]),
                        jnp.array([self.width - 1, self.height - 1]),
                    )
                    is_obstacle = jax.lax.cond(
                        self.has_obstacles,
                        lambda _: jnp.any(jnp.all(new_pos == self.obstacles, axis=1)),
                        lambda _: jnp.array(False),
                        operand=None,
                    )
                    final_pos = jnp.where(is_obstacle, pos, new_pos)
                    return final_pos, st + 1

                return jax.lax.cond(
                    is_blocked,
                    lambda _: (pos, st + 1),   # door blocked → stay
                    normal_physics,
                    operand=None,
                )

            return jax.lax.cond(portal_match, apply_portal, after_portal_check, operand=None)

        # Either advance (non-terminal) or keep position (terminal)
        position, steps = jax.lax.cond(
            terminal,
            lambda args: (args[0], args[1]),
            _step_impl,
            (position, steps, current_state_idx),
        )

        at_goal = jax.lax.cond(
            self.has_goal,
            lambda _: jnp.all(position == self.goal_pos),
            lambda _: jnp.array(False),
            operand=None,
        )
        max_steps_reached = steps >= self.max_steps
        new_terminal = jnp.logical_or(terminal, jnp.logical_or(at_goal, max_steps_reached))

        next_state = GridWorldState(
            position=position,
            terminal=new_terminal,
            steps=steps,
        )

        reward = jnp.array(0.0)
        done   = new_terminal
        return next_state, reward, done, {}

    def get_state_representation(self, state):
        """Convert state to a flattened index."""
        return state.position[1] * self.width + state.position[0]

    def action_space_size(self):
        """Return the size of the action space."""
        return self.action_space

    def observation_space_size(self):
        """Return the size of the observation space."""
        return self.width * self.height

    def get_params(self):
        """Return environment parameters."""
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": self.obstacles,
            "start_pos": self.start_pos if self.fixed_start else None,
            "goal_pos": self.goal_pos if self.has_goal else None,
            "max_steps": self.max_steps,
            "fixed_start": self.fixed_start,
            "asymmetric_transitions": self.asymmetric_transitions,
            "portals": self.portals,
        }
