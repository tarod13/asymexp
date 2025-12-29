import os
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
        - 'D{action}' for door (one-way passage from this tile)
          e.g., 'DD' = door going Down from this tile
          Actions: U (up), D (down), L (left), R (right)
        - 'D{source}>{dest}{action}' for non-adjacent doors/teleports (advanced)
        - Multiple elements can be combined: 'X', 'DD', 'DD,DR' (multiple doors)

    Returns:
        env: GridWorld environment instance (may be DoorGridWorldEnv if doors are specified)
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
    from src.envs.door_gridworld import DoorGridWorldEnv

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
    blocked_transitions = set()  # For doors: (state_idx, action)

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
            # Simple format: D{action} - door from this tile in the given direction
            # Example: DD means door going Down from this tile
            import re

            # Simple door format: D{action} (e.g., DD, DR, DU, DL)
            simple_door_pattern = r'D([UDLR])'
            for match in re.finditer(simple_door_pattern, tile):
                action_char = match.group(1)
                action = action_map[action_char]

                # Calculate source and destination states
                source_state = y * width + x

                # Calculate destination based on action
                action_effects = {
                    0: (0, -1),  # Up
                    1: (1, 0),   # Right
                    2: (0, 1),   # Down
                    3: (-1, 0),  # Left
                }
                dx, dy = action_effects[action]
                dest_x, dest_y = x + dx, y + dy

                # Only add door if destination is within bounds
                if 0 <= dest_x < width and 0 <= dest_y < height:
                    dest_state = dest_y * width + dest_x

                    # Block the reverse transition
                    reverse_action_map = {0: 2, 1: 3, 2: 0, 3: 1}  # U<->D, L<->R
                    reverse_action = reverse_action_map[action]
                    blocked_transitions.add((dest_state, reverse_action))

            # Complex door format (for non-adjacent): D{source}>{dest}{action}
            # Example: D5>12U means door from state 5 to state 12 via Up action
            # (Kept for backward compatibility and teleportation-like doors)
            complex_door_pattern = r'D(\d+)>(\d+)([UDLR])'
            for match in re.finditer(complex_door_pattern, tile):
                source_state = int(match.group(1))
                dest_state = int(match.group(2))
                action_char = match.group(3)
                action = action_map[action_char]

                # Block the reverse transition
                reverse_action_map = {0: 2, 1: 3, 2: 0, 3: 1}  # U<->D, L<->R
                reverse_action = reverse_action_map[action]
                blocked_transitions.add((dest_state, reverse_action))

    # Create environment
    if blocked_transitions:
        return DoorGridWorldEnv(
            width=width,
            height=height,
            obstacles=obstacles,
            start_pos=start_pos,
            goal_pos=goal_pos,
            blocked_transitions=blocked_transitions,
            **env_kwargs,
        )
    else:
        return GridWorldEnv(
            width=width,
            height=height,
            obstacles=obstacles,
            start_pos=start_pos,
            goal_pos=goal_pos,
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
