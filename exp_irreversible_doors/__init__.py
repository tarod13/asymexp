"""
Experiment package for eigendecomposition analysis with irreversible doors.

This package creates grid environments with one-way passages (irreversible doors)
to study asymmetric transition dynamics. Unlike portals that teleport to arbitrary
locations, doors maintain normal grid connectivity but make certain transitions
one-way only.

Key difference from portals:
- Portals: (s, a) -> arbitrary s' (teleportation)
- Doors: (s, a) -> adjacent s', but p(s | s', a_reverse) = 0 (one-way passage)

Doors are implemented in the environment dynamics via DoorGridWorldEnv.
"""

from .door_environment import (
    get_canonical_free_states,
    create_door_environment_data,
    generate_batched_door_environments,
)

# Re-export from src.envs.door_gridworld for convenience
from src.envs.door_gridworld import (
    create_random_doors,
    get_reverse_action,
    find_reversible_transitions,
    create_door_gridworld_from_base,
    DoorGridWorldEnv,
)

__all__ = [
    # From door_environment
    "get_canonical_free_states",
    "create_door_environment_data",
    "generate_batched_door_environments",
    # From src.envs.door_gridworld
    "create_random_doors",
    "get_reverse_action",
    "find_reversible_transitions",
    "create_door_gridworld_from_base",
    "DoorGridWorldEnv",
]
