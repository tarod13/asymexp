import os
import re

from src.envs.gridworld import GridWorldEnv
from src.envs.windy_gridworld import WindyGridWorldEnv


def get_env_path(file_name):
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, 'txt', f'{file_name}.txt')


def create_environment_from_text(text_content=None, file_name=None, file_path=None, windy=False, **env_kwargs):
    """
    Create a GridWorld environment from a text representation.

    Args:
        text_content: String containing the environment layout (if provided directly)
        file_name: Name of a file in src/envs/txt/ (without .txt extension)
        file_path: Path to a text file containing the environment layout

    Text format (two modes):

    1. Legacy format (space/character separated):
        - ' ' or '.' for empty space
        - 'X' or '#' for obstacle
        - 'S' for starting position
        - 'G' for goal position

    2. New format (comma-separated, supports multiple elements per tile):
        - Tiles separated by commas, rows separated by newlines
        - '.' for empty space
        - 'X' or '#' for obstacle
        - 'S' for starting position
        - 'G' for goal position
        - 'D{actions}' or 'D{actions}:{prob}' for doors (asymmetric passages)
          - First character 'D' marks door, each following character is an action
          - Optional ':{prob}' sets reversal probability in [0, 1] (default 0 = fully blocked)
          - e.g., 'DD' = hard door Down, 'DD:0.3' = door Down with 30% reversal chance
          - 'DDL:0.5' = doors Down and Left, both with 50% reversal chance
          - Actions: U (up), D (down), L (left), R (right)
        - 'P{action}:{dest}' for portals (non-adjacent teleports from this tile)
          - Source is always the tile's own position; one token per direction
          - e.g., 'PU:12' = going Up from this tile teleports to state 12
          - 'PU:12PD:5' = two portals from this tile (Up->12, Down->5)
        - Multiple elements can be combined in a tile: 'X', 'DD', 'DDLR', etc.

    Returns:
        env: GridWorldEnv instance (with doors and/or portals if specified in the file)
    """
    if file_name is not None:
        file_path = get_env_path(file_name)
        try:
            with open(file_path, 'r') as f:
                text_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Environment file not found: {file_path}")

    elif file_path is not None:
        with open(file_path, 'r') as f:
            text_content = f.read()

    if text_content is None:
        raise ValueError("Either text_content, file_name, or file_path must be provided")

    lines = text_content.strip().split('\n')
    is_comma_format = any(',' in line for line in lines)

    if is_comma_format:
        return _parse_comma_format(lines, windy=windy, **env_kwargs)
    else:
        return _parse_legacy_format(lines, windy=windy, **env_kwargs)


def _parse_legacy_format(lines, windy=False, **env_kwargs):
    """Parse the legacy character-by-character format."""
    height = len(lines)
    width = max(len(line) for line in lines)

    obstacles = []
    start_pos = None
    goal_pos = None

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char in 'X#':
                obstacles.append((x, y))
            elif char == 'S':
                start_pos = (x, y)
            elif char == 'G':
                goal_pos = (x, y)

    env_class = WindyGridWorldEnv if windy else GridWorldEnv
    return env_class(
        width=width,
        height=height,
        obstacles=obstacles,
        start_pos=start_pos,
        goal_pos=goal_pos,
        **env_kwargs,
    )


def _parse_comma_format(lines, windy=False, **env_kwargs):
    """Parse the new comma-separated format with support for doors and multiple elements per tile."""
    grid = []
    for line in lines:
        if line.strip():
            row = [tile.strip() for tile in line.split(',')]
            grid.append(row)

    height = len(grid)
    width = max(len(row) for row in grid) if grid else 0

    for row in grid:
        while len(row) < width:
            row.append('.')

    obstacles = []
    start_pos = None
    goal_pos = None
    asymmetric_transitions = {}  # (state_idx, action) -> reversal_prob
    portals = {}  # (source_state_idx, action) -> dest_state_idx

    action_map = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
    action_effects_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    reverse_action_map = {0: 2, 1: 3, 2: 0, 3: 1}

    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            if 'X' in tile or '#' in tile:
                obstacles.append((x, y))
            if 'S' in tile:
                start_pos = (x, y)
            if 'G' in tile:
                goal_pos = (x, y)

            # Door format: D{actions} or D{actions}:{prob}
            door_pattern = r'D([UDLR]+)(?::([0-9]*\.?[0-9]+))?'
            for match in re.finditer(door_pattern, tile):
                actions_str = match.group(1)
                reversal_prob = float(match.group(2)) if match.group(2) is not None else 0.0

                for action_char in actions_str:
                    action = action_map[action_char]
                    dx, dy = action_effects_map[action]
                    dest_x, dest_y = x + dx, y + dy

                    if 0 <= dest_x < width and 0 <= dest_y < height:
                        dest_state = dest_y * width + dest_x
                        reverse_action = reverse_action_map[action]
                        asymmetric_transitions[(dest_state, reverse_action)] = reversal_prob

            # Portal format: P{action}:{dest}
            source_state = y * width + x
            portal_pattern = r'P([UDLR]):(\d+)'
            for match in re.finditer(portal_pattern, tile):
                action_char = match.group(1)
                dest_state = int(match.group(2))
                action = action_map[action_char]
                portals[(source_state, action)] = dest_state

    env_class = WindyGridWorldEnv if windy else GridWorldEnv
    return env_class(
        width=width,
        height=height,
        obstacles=obstacles,
        start_pos=start_pos,
        goal_pos=goal_pos,
        asymmetric_transitions=asymmetric_transitions if asymmetric_transitions else None,
        portals=portals if portals else None,
        **env_kwargs,
    )


def save_environment_to_text(env, file_path):
    """Save a GridWorld environment to a text file."""
    grid = [[' ' for _ in range(env.width)] for _ in range(env.height)]

    if env.has_obstacles:
        for obs in env.obstacles:
            if 0 <= obs[0] < env.width and 0 <= obs[1] < env.height:
                grid[obs[1]][obs[0]] = 'X'

    if hasattr(env, 'start_pos'):
        x, y = env.start_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'S'

    if env.has_goal:
        x, y = env.goal_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'G'

    text_content = '\n'.join(''.join(row) for row in grid)

    with open(file_path, 'w') as f:
        f.write(text_content)

    return text_content


def print_environment(env):
    """Print a visual representation of the environment to the console."""
    grid = [[' ' for _ in range(env.width)] for _ in range(env.height)]

    if env.has_obstacles:
        for obs in env.obstacles:
            if 0 <= obs[0] < env.width and 0 <= obs[1] < env.height:
                grid[obs[1]][obs[0]] = 'X'

    if hasattr(env, 'start_pos'):
        x, y = env.start_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'S'

    if env.has_goal:
        x, y = env.goal_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'G'

    print('┌' + '─' * env.width + '┐')
    for row in grid:
        print('│' + ''.join(row) + '│')
    print('└' + '─' * env.width + '┘')
