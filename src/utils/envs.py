

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  

import jax.numpy as jnp

from src.config.ded_clf import Args
from src.envs.env import create_environment_from_text, EXAMPLE_ENVIRONMENTS


def create_gridworld_env(args: Args):
    """
    Create a gridworld environment from text file or example.

    Args:
        args: Arguments containing env_type, env_file, or env_file_name

    Returns:
        env: GridWorld environment instance
    """
    if args.env_type == 'file':
        # Load from file path
        if args.env_file is None and args.env_file_name is None:
            raise ValueError("Must specify either env_file or env_file_name when env_type='file'")

        env = create_environment_from_text(
            file_path=args.env_file,
            file_name=args.env_file_name,
            max_steps=args.max_episode_length,
            precision=32,
        )
    else:
        # Load from example environments
        if args.env_type not in EXAMPLE_ENVIRONMENTS:
            raise ValueError(f"Unknown env_type: {args.env_type}. "
                           f"Must be one of {list(EXAMPLE_ENVIRONMENTS.keys())} or 'file'")

        text_content = EXAMPLE_ENVIRONMENTS[args.env_type]
        env = create_environment_from_text(
            text_content=text_content,
            max_steps=args.max_episode_length,
            precision=32,
        )

    print(f"Loaded environment: {args.env_type}")
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Number of obstacles: {len(env.obstacles) if env.has_obstacles else 0}")
    if env.has_doors:
        print(f"  Doors: {len(env.blocked_transitions)} blocked transitions")
    if env.has_portals:
        print(f"  Portals: {len(env.portals)} portal transitions")
    if not env.has_doors and not env.has_portals:
        print(f"  No doors or portals")

    return env


def get_env_transition_markers(env) -> dict:
    """
    Return a dict {(source_state_full, action): dest_state_full} for all
    non-standard transitions in the environment (doors and/or portals).

    Doors: reconstructs (source, forward_action) -> dest from the stored
           (dest, reverse_action) blocked_transitions.
    Portals: reads the portals dict directly.

    Both are collected into the same dict (portals take precedence over doors
    for the same (source, action) key, matching the step() priority).
    """
    markers = {}

    if env.has_doors:
        action_reverse = {0: 2, 1: 3, 2: 0, 3: 1}
        action_delta   = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        for state_full, action in env.blocked_transitions:
            state_full = int(state_full)
            action = int(action)
            forward_action = action_reverse[action]
            dx, dy = action_delta[action]
            dest_y, dest_x = divmod(state_full, env.width)
            source_x = dest_x + dx
            source_y = dest_y + dy
            if 0 <= source_x < env.width and 0 <= source_y < env.height:
                source_full = source_y * env.width + source_x
                if (source_full, forward_action) not in markers:
                    markers[(source_full, forward_action)] = state_full

    if env.has_portals:
        for (source_full, action), dest_full in env.portals.items():
            markers[(int(source_full), int(action))] = int(dest_full)

    return markers


def get_canonical_free_states(env):
    """
    Get the canonical set of free (non-obstacle) states from the environment.

    Args:
        env: GridWorld environment

    Returns:
        canonical_states: Array of free state indices, sorted
    """
    width = env.width
    height = env.height

    # Get all state indices
    all_states = set(range(width * height))

    # Get obstacle state indices
    obstacle_states = set()
    if env.has_obstacles:
        for obs in env.obstacles:
            obs_x, obs_y = int(obs[0]), int(obs[1])
            if 0 <= obs_x < width and 0 <= obs_y < height:
                state_idx = obs_y * width + obs_x
                obstacle_states.add(state_idx)

    # Free states = all states - obstacles
    free_states = sorted(all_states - obstacle_states)

    return jnp.array(free_states, dtype=jnp.int32)