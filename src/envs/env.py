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

    Text format:
        - ' ' or '.' for empty space
        - 'X' or '#' for obstacle
        - 'S' for starting position
        - 'G' for goal position

    Returns:
        env: GridWorld environment instance
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

    # Parse the text content
    lines = text_content.split('\n')

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
