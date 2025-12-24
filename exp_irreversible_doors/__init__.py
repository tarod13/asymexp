"""
Experiment package for eigendecomposition analysis with irreversible doors.

This package creates grid environments with one-way passages (irreversible doors)
to study asymmetric transition dynamics. Unlike portals that teleport to arbitrary
locations, doors maintain normal grid connectivity but make certain transitions
one-way only.

Key difference from portals:
- Portals: (s, a) -> arbitrary s' (teleportation)
- Doors: (s, a) -> adjacent s', but p(s | s', a_reverse) = 0 (one-way passage)
"""

from .door_environment import (
    create_irreversible_doors,
    apply_doors_to_transition_matrix,
    generate_batched_door_environments,
    get_reverse_action,
    find_reversible_transitions,
)

__all__ = [
    "create_irreversible_doors",
    "apply_doors_to_transition_matrix",
    "generate_batched_door_environments",
    "get_reverse_action",
    "find_reversible_transitions",
]
