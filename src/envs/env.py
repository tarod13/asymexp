import os
import re

from src.envs.gridworld import GridWorldEnv



# Example environment layouts as strings
EXAMPLE_ENVIRONMENTS = {
    "empty": (
"""S






        G"""
),

    "obstacles": (
"""S
XXX
    X
    X
    X

    XXXX
        G"""
),

    "maze": (
"""S XXXXXXX
X X     X
X X XXX X
X   X   X
XXX XXX X
X       X
X XXXXX X
X      GX
XXXXXXXXX"""
),

    "spiral": (
"""XXXXXXXXX
X      XX
X XXXX XX
X XG X XX
X XX X XX
X    X XX
XXXXXX XX
S      XX
XXXXXXXXX"""
),

    "room4":(
"""XXXXXXXXXXXXX
X     X     X
X           X
X     X     X
X     X     X
X     X     X
XXX XXXXXX XX
X      X    X
X      X    X
X           X
X      X    X
X      X    X
XXXXXXXXXXXXX"""
),

    # Example using new comma-separated format with doors
    "room4_doors": (
"""X,X,X,X,X,X,X,X,X,X,X,X,X
X,.,.,.,.,.,X,.,.,.,.,.,X
X,.,.,.,.,.,.,.,.,.,.,.,X
X,.,.,.,.,.,X,.,.,.,.,.,X
X,.,.,.,.,.,X,.,.,.,.,.,X
X,.,.,.,.,.,X,.,.,.,.,.,X
X,X,X,DD,X,X,X,X,X,X,X,X
X,.,.,.,.,.,.,X,.,.,.,.,X
X,.,.,.,.,.,.,X,.,.,.,.,X
X,.,.,.,.,.,.,.,.,.,.,.,X
X,.,.,.,.,.,.,X,.,.,.,.,X
X,.,.,.,.,.,.,X,.,.,.,.,X
X,X,X,X,X,X,X,X,X,X,X,X,X"""
),

    # Example with multiple doors creating hard-to-reach states
    "asymmetric_maze": (
"""X,X,X,X,X,X,X
X,.,.,DD,.,.,X
X,.,X,X,X,.,X
X,.,.,.,.,.,X
X,X,X,DU,X,X,X
X,.,.,.,.,.,X
X,X,X,X,X,X,X"""
)}

def get_env_path(file_name):
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, 'txt', f'{file_name}.txt')


def create_environment_from_text(text_content=None, file_name=None, file_path=None, **env_kwargs):
    """
    Create a GridWorld environment from a text representation.

    Args:
        text_content: String containing the environment layout (if provided directly)
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
          - 'PU:12PD:5' = two portals from this tile (Up→12, Down→5)
        - Multiple elements can be combined in a tile: 'X', 'DD', 'DDLR', etc.

    Returns:
        env: GridWorldEnv instance (with doors and/or portals if specified in the file)
    """
    # Load text with file name if provided
    if file_name is not None:
        file_path = get_env_path(file_name)
        try:
            with open(file_path, 'r') as f:
                text_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Environment file not found: {file_path}")

    # Load text from file if path is provided
    elif file_path is not None:
        with open(file_path, 'r') as f:
            text_content = f.read()

    if text_content is None:
        raise ValueError("Either text_content or file_path must be provided")

    # Detect format: if line contains comma, use new format
    lines = text_content.strip().split('\n')
    is_comma_format = any(',' in line for line in lines)

    if is_comma_format:
        return _parse_comma_format(lines, **env_kwargs)
    else:
        return _parse_legacy_format(lines, **env_kwargs)


def _parse_legacy_format(lines, **env_kwargs):
    """Parse the legacy character-by-character format."""
    # Get grid dimensions
    height = len(lines)
    width = max(len(line) for line in lines)

    # Initialize grid elements
    obstacles = []
    start_pos = None
    goal_pos = None

    # Process each character in the grid
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char in 'X#':  # Obstacle
                obstacles.append((x, y))
            elif char == 'S':  # Start
                start_pos = (x, y)
            elif char == 'G':  # Goal
                goal_pos = (x, y)

    # Create and return the environment
    return GridWorldEnv(
        width=width,
        height=height,
        obstacles=obstacles,
        start_pos=start_pos,
        goal_pos=goal_pos,
        **env_kwargs,
    )


def _parse_comma_format(lines, **env_kwargs):
    """Parse the new comma-separated format with support for doors and multiple elements per tile."""

    # Parse grid
    grid = []
    for line in lines:
        if line.strip():  # Skip empty lines
            row = [tile.strip() for tile in line.split(',')]
            grid.append(row)

    height = len(grid)
    width = max(len(row) for row in grid) if grid else 0

    # Pad rows to have same width
    for row in grid:
        while len(row) < width:
            row.append('.')

    # Initialize grid elements
    obstacles = []
    start_pos = None
    goal_pos = None
    asymmetric_transitions = {}  # For doors: (state_idx, action) -> reversal_prob
    portals = {}  # For portals: (source_state_idx, action): dest_state_idx

    # Action mapping
    action_map = {'U': 0, 'R': 1, 'D': 2, 'L': 3}

    # Process each tile
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            # A tile can contain multiple elements
            if 'X' in tile or '#' in tile:
                obstacles.append((x, y))
            if 'S' in tile:
                start_pos = (x, y)
            if 'G' in tile:
                goal_pos = (x, y)

            # Parse door specifications:
            # Format: D{actions} or D{actions}:{prob}
            # Each action character creates a separate door from this tile.
            # The optional :{prob} sets the reversal probability (default 0 = fully blocked).
            # Example: DD = hard door Down, DD:0.3 = door Down with 30% reversal chance
            action_effects_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
            reverse_action_map = {0: 2, 1: 3, 2: 0, 3: 1}

            door_pattern = r'D([UDLR]+)(?::([0-9]*\.?[0-9]+))?'
            for match in re.finditer(door_pattern, tile):
                actions_str  = match.group(1)
                reversal_prob = float(match.group(2)) if match.group(2) is not None else 0.0

                for action_char in actions_str:
                    action = action_map[action_char]
                    dx, dy = action_effects_map[action]
                    dest_x, dest_y = x + dx, y + dy

                    if 0 <= dest_x < width and 0 <= dest_y < height:
                        dest_state     = dest_y * width + dest_x
                        reverse_action = reverse_action_map[action]
                        asymmetric_transitions[(dest_state, reverse_action)] = reversal_prob

            # Portal format: P{action}:{dest}
            # Source is this tile; one token per direction.
            # Example: PU:12 = going Up from this tile teleports to state 12
            #          PU:12PD:5 = two portals from this tile
            source_state  = y * width + x
            portal_pattern = r'P([UDLR]):(\d+)'
            for match in re.finditer(portal_pattern, tile):
                action_char = match.group(1)
                dest_state  = int(match.group(2))
                action      = action_map[action_char]
                portals[(source_state, action)] = dest_state

    # Always create a single GridWorldEnv; doors and portals are optional
    return GridWorldEnv(
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
    """
    Save a GridWorld environment to a text file.

    Args:
        env: GridWorld environment instance
        file_path: Path to save the text representation
    """
    # Create a grid of empty spaces
    grid = [[' ' for _ in range(env.width)] for _ in range(env.height)]

    # Mark obstacles
    if env.has_obstacles:
        for obs in env.obstacles:
            if 0 <= obs[0] < env.width and 0 <= obs[1] < env.height:
                grid[obs[1]][obs[0]] = 'X'

    # Mark start position
    if hasattr(env, 'start_pos'):
        x, y = env.start_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'S'

    # Mark goal if exists
    if env.has_goal:
        x, y = env.goal_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'G'

    # Join the grid into a string
    text_content = '\n'.join(''.join(row) for row in grid)

    # Write to file
    with open(file_path, 'w') as f:
        f.write(text_content)

    return text_content


def print_environment(env):
    """
    Print a visual representation of the environment to the console.

    Args:
        env: GridWorld environment instance
    """
    # Create a grid of empty spaces
    grid = [[' ' for _ in range(env.width)] for _ in range(env.height)]

    # Mark obstacles
    if env.has_obstacles:
        for obs in env.obstacles:
            if 0 <= obs[0] < env.width and 0 <= obs[1] < env.height:
                grid[obs[1]][obs[0]] = 'X'

    # Mark start position
    if hasattr(env, 'start_pos'):
        x, y = env.start_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'S'

    # Mark goal if exists
    if env.has_goal:
        x, y = env.goal_pos
        if 0 <= x < env.width and 0 <= y < env.height:
            grid[y][x] = 'G'

    # Print the grid with borders
    print('┌' + '─' * env.width + '┐')
    for row in grid:
        print('│' + ''.join(row) + '│')
    print('└' + '─' * env.width + '┘')


def get_example_environment(name, **env_kwargs):
        """
        Get one of the predefined example environments.

        Args:
            name: Name of the example environment

        Returns:
            env: GridWorld environment instance
        """
        if name not in EXAMPLE_ENVIRONMENTS:
            raise ValueError(f"Unknown environment name: {name}. Available environments: {list(EXAMPLE_ENVIRONMENTS.keys())}")

        return create_environment_from_text(EXAMPLE_ENVIRONMENTS[name], **env_kwargs)
