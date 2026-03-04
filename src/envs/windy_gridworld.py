import jax
import jax.numpy as jnp

from src.envs.gridworld import GridWorldEnv, GridWorldState


class WindyGridWorldEnv(GridWorldEnv):
    """GridWorld parametrized by a wind value in the open interval (-1, 1).

    On every step the wind displacement is added on top of the agent's chosen
    action displacement.  The combined vector is then clipped to the grid
    boundaries and checked against obstacles as usual.

    Stochastic wind rule:
        - With probability ``abs(wind)`` a wind displacement is applied:
            * ``wind > 0``: delta = (+1, 0)  (rightward gust)
            * ``wind < 0``: delta = (-1, 0)  (leftward gust)
        - With probability ``1 - abs(wind)`` no extra displacement is added.

    Examples:
        - Agent chooses DOWN,  wind blows LEFT  → displacement (-1, +1) (diagonal).
        - Agent chooses RIGHT, wind blows LEFT  → displacement  (0,  0) (stay).
        - Agent chooses UP,    no wind gust     → displacement  (0, -1) (normal).

    Doors and portals are still triggered by the agent's original action;
    the additive wind affects only normal-physics movement.

    Args:
        wind: float strictly in (-1, 1).
        **kwargs: forwarded to :class:`GridWorldEnv`.
    """

    # Wind displacement vectors (x, y) for left / right gusts
    _WIND_LEFT  = jnp.array([-1, 0], dtype=jnp.int32)
    _WIND_RIGHT = jnp.array([ 1, 0], dtype=jnp.int32)
    _WIND_NONE  = jnp.array([ 0, 0], dtype=jnp.int32)

    def __init__(self, wind: float = 0.0, **kwargs):
        if not (-1 < wind < 1):
            raise ValueError(
                f"wind must be in the open interval (-1, 1), got {wind}"
            )
        self.wind = wind
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, key, state, action, params=None):
        """Take a step with additive wind displacement.

        Priority order (same as base class):
          1. Portal  — teleport if (state, action) is a portal.
          2. Door    — stay in place if (state, action) is blocked.
          3. Physics — combined (action + wind) displacement with
                       boundary clipping and obstacle checks.
        """
        position          = state.position
        terminal          = state.terminal
        steps             = state.steps
        current_state_idx = self.get_state_representation(state)

        key, wind_key, door_key = jax.random.split(key, 3)

        # ------------------------------------------------------------------
        # Compute stochastic wind delta
        # ------------------------------------------------------------------
        wind_dir = self._WIND_LEFT if self.wind < 0 else self._WIND_RIGHT
        # Cast wind_dir to match the grid dtype
        wind_dir = wind_dir.astype(self.dtype)
        wind_none = self._WIND_NONE.astype(self.dtype)

        u = jax.random.uniform(wind_key)
        wind_delta = jax.lax.cond(
            u < abs(self.wind),
            lambda _: wind_dir,
            lambda _: wind_none,
            operand=None,
        )

        # ------------------------------------------------------------------
        # Inner step (mirrors GridWorldEnv._step_impl with wind_delta added)
        # ------------------------------------------------------------------
        def _step_impl(args):
            pos, st, csidx = args

            # 3. Normal physics — combined displacement
            def normal_physics(_):
                new_pos_raw = pos + self.action_effects[action] + wind_delta
                new_pos = jnp.clip(
                    new_pos_raw,
                    jnp.array([0, 0]),
                    jnp.array([self.width - 1, self.height - 1]),
                )
                if self.has_obstacles:
                    is_obstacle = jnp.any(jnp.all(new_pos == self.obstacles, axis=1))
                    final_pos   = jnp.where(is_obstacle, pos, new_pos)
                else:
                    final_pos = new_pos
                return final_pos, st + 1

            # 2. Door check (keyed on the agent's original action)
            if self.has_doors:
                def check_door(i):
                    return (self.asym_states[i] == csidx) & (self.asym_actions[i] == action)

                matches   = jax.vmap(check_door)(jnp.arange(len(self.asym_states)))
                has_door  = jnp.any(matches)
                door_idx  = jnp.argmax(matches)
                door_prob = jnp.where(has_door, self.asym_probs[door_idx], jnp.array(1.0))
                u_door    = jax.random.uniform(door_key)
                is_blocked = has_door & (u_door >= door_prob)
                after_door = jax.lax.cond(
                    is_blocked,
                    lambda _: (pos, st + 1),
                    normal_physics,
                    operand=None,
                )
            else:
                after_door = normal_physics(None)

            # 1. Portal check (keyed on the agent's original action)
            if self.has_portals:
                def check_portal(i):
                    return (self.portal_states[i] == csidx) & (self.portal_actions[i] == action)

                matches      = jax.vmap(check_portal)(jnp.arange(len(self.portal_states)))
                portal_match = jnp.any(matches)
                portal_idx   = jnp.argmax(matches)
                portal_dest  = self._state_idx_to_position(
                    self.portal_destinations[portal_idx]
                )
                return jax.lax.cond(
                    portal_match,
                    lambda _: (portal_dest, st + 1),
                    lambda _: after_door,
                    operand=None,
                )
            else:
                return after_door

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
        new_terminal = jnp.logical_or(
            terminal, jnp.logical_or(at_goal, max_steps_reached)
        )

        next_state = GridWorldState(
            position=position,
            terminal=new_terminal,
            steps=steps,
        )

        reward = jnp.array(0.0)
        done   = new_terminal
        return next_state, reward, done, {}

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def get_params(self):
        params = super().get_params()
        params["wind"] = self.wind
        return params
