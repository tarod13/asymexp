import jax
import jax.numpy as jnp

from src.envs.gridworld import GridWorldEnv, GridWorldState


# Actions used internally for wind deflection (must match action_effects ordering)
_ACTION_RIGHT = 1
_ACTION_LEFT = 3


class WindyGridWorldEnv(GridWorldEnv):
    """GridWorld parametrized by a wind value in the open interval (-1, 1).

    On every step, before the agent's chosen action is executed, wind is
    applied stochastically:

    - ``wind > 0`` (rightward wind): with probability ``abs(wind)`` the
      agent is pushed RIGHT instead of executing the chosen action.
    - ``wind < 0`` (leftward wind): with probability ``abs(wind)`` the
      agent is pushed LEFT instead of executing the chosen action.
    - ``wind == 0``: no deflection (identical to the base GridWorldEnv).

    The deflected action then goes through all normal physics (boundaries,
    obstacles, doors, portals).

    Args:
        wind: float strictly in (-1, 1).
        **kwargs: forwarded to :class:`GridWorldEnv`.
    """

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
        """Take a step, possibly deflecting the action due to wind."""
        key, wind_key = jax.random.split(key)

        wind_magnitude = abs(self.wind)
        wind_action = _ACTION_LEFT if self.wind < 0 else _ACTION_RIGHT

        u = jax.random.uniform(wind_key)
        effective_action = jax.lax.cond(
            u < wind_magnitude,
            lambda _: wind_action,
            lambda _: action,
            operand=None,
        )

        return super().step(key, state, effective_action, params)

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def get_params(self):
        params = super().get_params()
        params["wind"] = self.wind
        return params
