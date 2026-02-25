import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List
from src.envs.gridworld import GridWorldEnv, GridWorldState


class PortalGridWorldEnv(GridWorldEnv):
    """
    GridWorld environment with portals that override normal transitions.

    A portal is defined as ((s, a), s') where taking action a in state s
    teleports the agent to state s' instead of following normal physics.
    """

    def __init__(self, portals: Dict[Tuple[int, int], int] = None, **kwargs):
        """
        Initialize the PortalGridWorld environment.

        Args:
            portals: Dictionary mapping (state_idx, action) -> next_state_idx
            **kwargs: Arguments passed to GridWorldEnv
        """
        super().__init__(**kwargs)

        # Store portals as a dictionary: (state_idx, action) -> next_state_idx
        self.portals = portals if portals is not None else {}

        # Convert portals to JAX arrays for efficient lookup
        if len(self.portals) > 0:
            portal_keys = list(self.portals.keys())
            portal_values = list(self.portals.values())

            # Separate state indices and actions
            self.portal_states = jnp.array([k[0] for k in portal_keys], dtype=self.dtype)
            self.portal_actions = jnp.array([k[1] for k in portal_keys], dtype=self.dtype)
            self.portal_destinations = jnp.array(portal_values, dtype=self.dtype)
            self.has_portals = True
        else:
            self.portal_states = jnp.array([], dtype=self.dtype)
            self.portal_actions = jnp.array([], dtype=self.dtype)
            self.portal_destinations = jnp.array([], dtype=self.dtype)
            self.has_portals = False

    def _state_idx_to_position(self, state_idx):
        """Convert state index to (x, y) position."""
        x = state_idx % self.width
        y = state_idx // self.width
        return jnp.array([x, y], dtype=self.dtype)

    def step(self, key, state, action, params=None):
        """Take a step in the environment, checking for portals first."""
        # Get current position and status
        position = state.position
        terminal = state.terminal
        steps = state.steps

        # Get current state index
        current_state_idx = self.get_state_representation(state)

        # Only take action if not in terminal state
        def _step_impl(args):
            position, steps, current_state_idx = args

            # Check if there's a portal for (current_state, action)
            def check_portal(i):
                """Check if portal i matches current state and action."""
                matches_state = self.portal_states[i] == current_state_idx
                matches_action = self.portal_actions[i] == action
                return matches_state & matches_action

            # Look for matching portal
            portal_match = jax.lax.cond(
                self.has_portals,
                lambda _: jnp.any(jax.vmap(check_portal)(jnp.arange(len(self.portal_states)))),
                lambda _: jnp.array(False),
                operand=None
            )

            def apply_portal(_):
                """Apply portal teleportation."""
                # Find which portal matched
                matches = jax.vmap(check_portal)(jnp.arange(len(self.portal_states)))
                portal_idx = jnp.argmax(matches)  # Get first matching portal
                destination_idx = self.portal_destinations[portal_idx]

                # Convert destination index to position
                destination_pos = self._state_idx_to_position(destination_idx)
                return destination_pos, steps + 1

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

            # Either apply portal or normal physics
            final_position, new_steps = jax.lax.cond(
                portal_match,
                apply_portal,
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
        new_terminal = jnp.logical_or(terminal, jnp.logical_or(at_goal, max_steps_reached))

        # Create new state
        next_state = GridWorldState(
            position=position,
            terminal=new_terminal,
            steps=steps
        )

        # Reward is always 0.0
        reward = jnp.array(0.0)

        # Done flag for buffer
        done = new_terminal

        return next_state, reward, done, {}

    def get_params(self):
        """Return environment parameters including portals."""
        base_params = super().get_params()
        base_params["portals"] = self.portals
        return base_params


