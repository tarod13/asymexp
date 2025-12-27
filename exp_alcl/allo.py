"""
Train a model to learn eigenvectors of the symmetrized transition matrix.

This script uses (x,y) coordinates as inputs to the network and learns
the eigenvectors through an augmented Lagrangian optimization approach.
"""

import os
import sys
import random
import time
import json
import pickle
from typing import Dict
from pathlib import Path
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tyro
from tqdm import tqdm
from flax.training.train_state import TrainState
import flax.linen as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.envs.gridworld import GridWorldEnv
from src.envs.env import create_environment_from_text, EXAMPLE_ENVIRONMENTS
from src.envs.door_gridworld import (
    create_door_gridworld_from_base,
    create_random_doors,
)
from src.data_collection import collect_transition_counts_and_episodes
from exp_complex_basis.eigenvector_visualization import (
    visualize_multiple_eigenvectors,
    visualize_eigenvector_on_grid,
)
from exp_alcl.episodic_replay_buffer import EpisodicReplayBuffer


# Simple MLP network for (x,y) coordinates
class CoordinateEncoder(nn.Module):
    num_features: int
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    @nn.compact
    def __call__(self, xy_coords):
        """
        Args:
            xy_coords: (batch_size, 2) array of (x,y) coordinates

        Returns:
            features: (batch_size, num_features) learned eigenvector features
        """
        x = xy_coords

        # Input layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Hidden layers
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        # Output layer
        features = nn.Dense(self.num_features)(x)

        return features, {}


def get_symmetrized_transition_matrix(
    transition_counts: jnp.ndarray,
    smoothing: float = 1e-5,
    make_stochastic: bool = True,
    make_doubly_stochastic: bool = False,
) -> jnp.ndarray:
    """
    Build symmetrized transition matrix from counts.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]
        smoothing: Small value to add for numerical stability
        make_stochastic: Whether to row-normalize to get proper transition probabilities
        make_doubly_stochastic: Whether to iteratively normalize rows and columns for doubly stochastic matrix
    Returns:
        Symmetrized transition matrix of shape [num_states, num_states]
    """
    # Sum over actions if needed
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)  # [num_states, num_states]
    else:
        transition_matrix = transition_counts

    if make_stochastic:
        row_sums = jnp.sum(transition_matrix.clip(1), axis=1, keepdims=True)
        transition_matrix = transition_matrix.clip(1) / jnp.maximum(row_sums, 1e-10)

    # Symmetrize
    transition_matrix = (transition_matrix + transition_matrix.T) / 2.0

    # Normalize if requested (with iterations for better symmetry)
    if make_doubly_stochastic:
        # Add smoothing factor
        transition_matrix = transition_matrix + smoothing

        for _ in range(2):
            row_sums = jnp.sum(transition_matrix, axis=1, keepdims=True)
            transition_matrix = transition_matrix / jnp.maximum(row_sums, 1e-10)

            # Re-symmetrize after normalization
            transition_matrix = jnp.tril(transition_matrix)
            transition_matrix = transition_matrix + transition_matrix.T - jnp.diag(jnp.diag(transition_matrix))

    return transition_matrix


def compute_symmetrized_eigendecomposition(
    transition_matrix: jnp.ndarray,
    k: int = None
) -> Dict[str, jnp.ndarray]:
    """
    Compute eigendecomposition of a symmetric matrix.

    For symmetric matrices, eigenvalues are real and eigenvectors are orthogonal.

    Args:
        transition_matrix: Shape [num_states, num_states], symmetric
        k: Number of top eigenvalues/vectors to keep (None = keep all)

    Returns:
        Dictionary containing (compatible with visualization code):
            - eigenvalues: Shape [k] (complex, but with zero imaginary parts)
            - eigenvalues_real: Shape [k]
            - eigenvalues_imag: Shape [k] (all zeros)
            - right_eigenvectors_real: Shape [num_states, k]
            - right_eigenvectors_imag: Shape [num_states, k] (all zeros)
            - left_eigenvectors_real: Shape [num_states, k] (same as right for symmetric)
            - left_eigenvectors_imag: Shape [num_states, k] (all zeros)
    """
    # Compute eigendecomposition (for symmetric matrices, use eigh for better stability)
    eigenvalues, eigenvectors = jnp.linalg.eigh(transition_matrix)

    # Sort by descending eigenvalue magnitude
    sorted_indices = jnp.argsort(-jnp.abs(eigenvalues))
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Keep only top k if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    num_states = eigenvectors.shape[0]
    num_eigs = eigenvalues.shape[0]

    # Create format compatible with visualization code
    return {
        "eigenvalues": eigenvalues.astype(jnp.complex64),
        "eigenvalues_real": eigenvalues,
        "eigenvalues_imag": jnp.zeros_like(eigenvalues),
        "right_eigenvectors_real": eigenvectors,
        "right_eigenvectors_imag": jnp.zeros((num_states, num_eigs)),
        "left_eigenvectors_real": eigenvectors,  # Same as right for symmetric matrices
        "left_eigenvectors_imag": jnp.zeros((num_states, num_eigs)),
    }


# Training arguments
@dataclass
class Args:
    # Environment
    env_type: str = "room4"  # 'room4', 'maze', 'spiral', 'obstacles', 'empty', or 'file'
    env_file: str = None  # Path to environment text file (if env_type='file')
    env_file_name: str = None  # Name of environment file in src/envs/txt/ (e.g., 'GridRoom-4')
    max_episode_length: int = 100

    # Data collection
    num_envs: int = 1000
    num_steps: int = 100

    # Irreversible doors
    use_doors: bool = False
    num_doors: int = 5  # Number of irreversible doors to create
    door_seed: int = 42  # Seed for door placement

    # Model
    num_eigenvectors: int = 10
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_gradient_steps: int = 10000
    gamma: float = 0.99

    # Episodic replay buffer
    geometric_gamma: float = 0.99  # Decay for truncated geometric distribution (higher = prefer shorter time gaps)
    max_time_offset: int = None  # Maximum time offset for sampling (None = episode length)

    # Augmented Lagrangian parameters
    duals_initial_val: float = -2.0
    barrier_initial_val: float = 0.5
    max_barrier_coefs: float = 0.5
    step_size_duals: float = 1.0
    step_size_duals_I: float = 0.0
    integral_decay: float = 0.99
    init_dual_diag: bool = False

    # Regularization
    graph_epsilon: float = 0.01
    graph_variance_scale: float = 0.1
    perturbation_type: str = 'none'  # 'exponential', 'squared', 'squared-null-grad', 'none'
    turn_off_above_threshold: bool = False
    cum_error_threshold: float = 0.1

    # Logging and saving
    log_freq: int = 100
    plot_freq: int = 1000
    save_freq: int = 1000
    save_model: bool = True
    results_dir: str = "./results"

    # Misc
    seed: int = 42
    exp_name: str = None
    exp_number: int = 0


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

    return env


def plot_learning_curves(metrics_history: Dict, save_path: str):
    """Plot and save learning curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Extract data
    steps = [m['gradient_step'] for m in metrics_history]

    # Plot 1: Total loss
    axes[0, 0].plot(steps, [m['allo'] for m in metrics_history])
    axes[0, 0].set_xlabel('Gradient Step')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('ALLO Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Graph loss
    axes[0, 1].plot(steps, [m['graph_loss'] for m in metrics_history])
    axes[0, 1].set_xlabel('Gradient Step')
    axes[0, 1].set_ylabel('Graph Loss')
    axes[0, 1].set_title('Graph Drawing Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Total error
    axes[0, 2].plot(steps, [m['total_error'] for m in metrics_history])
    axes[0, 2].set_xlabel('Gradient Step')
    axes[0, 2].set_ylabel('Total Error')
    axes[0, 2].set_title('Orthogonality Error')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Dual loss
    axes[1, 0].plot(steps, [m['dual_loss'] for m in metrics_history], label='Positive')
    axes[1, 0].plot(steps, [m['dual_loss_neg'] for m in metrics_history], label='Negative')
    axes[1, 0].set_xlabel('Gradient Step')
    axes[1, 0].set_ylabel('Dual Loss')
    axes[1, 0].set_title('Dual Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Barrier loss
    axes[1, 1].plot(steps, [m['barrier_loss'] for m in metrics_history])
    axes[1, 1].set_xlabel('Gradient Step')
    axes[1, 1].set_ylabel('Barrier Loss')
    axes[1, 1].set_title('Barrier Loss')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Gradient norm
    axes[1, 2].plot(steps, [m['grad_norm'] for m in metrics_history])
    axes[1, 2].set_xlabel('Gradient Step')
    axes[1, 2].set_ylabel('Gradient Norm')
    axes[1, 2].set_title('Gradient Norm')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {save_path}")


def plot_dual_variable_evolution(metrics_history, ground_truth_eigenvalues, gamma, save_path, num_eigenvectors=11):
    """
    Plot the evolution of dual variables (eigenvalue estimates) vs ground truth eigenvalues.

    The duals are eigenvalues of the Laplacian L = I - SR_gamma, where SR_gamma is the
    successor representation with discount gamma. They are converted to approximate
    eigenvalues of the transition matrix using:
    approx_eigenvalue = (gamma + 0.5*dual) / (gamma * (1 + 0.5*dual))

    The 0.5 factor arises from the sampling scheme with the episodic replay buffer.

    Args:
        metrics_history: List of metric dictionaries
        ground_truth_eigenvalues: Array of ground truth eigenvalues of the transition matrix
        gamma: Discount factor used in the successor representation
        save_path: Path to save the plot
        num_eigenvectors: Number of eigenvectors to plot
    """
    steps = [m['gradient_step'] for m in metrics_history]
    num_plot = min(num_eigenvectors, len(ground_truth_eigenvalues))

    # Create figure with two rows: eigenvalue approximation and errors
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Approximate eigenvalues vs ground truth
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, num_plot))

    for i in range(num_plot):
        dual_key = f'dual_{i}'
        if dual_key in metrics_history[0]:
            # Get dual values and convert to approximate eigenvalues
            dual_values = np.array([m[dual_key] for m in metrics_history])
            # Apply 0.5 factor before conversion (due to sampling scheme)
            dual_values_scaled = 0.5 * dual_values
            approx_eigenvalues = (gamma + dual_values_scaled) / (gamma * (1 + dual_values_scaled))

            ax1.plot(steps, approx_eigenvalues, label=f'Approx λ_{i}', color=colors[i], linewidth=1.5)

            # Plot ground truth as horizontal dashed line
            gt_value = float(ground_truth_eigenvalues[i].real)
            ax1.axhline(y=gt_value, color=colors[i], linestyle='--', alpha=0.5, linewidth=1.5)

    ax1.set_xlabel('Gradient Step', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Approximate Eigenvalues from Duals (solid) vs Ground Truth (dashed)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute errors
    ax2 = axes[1]

    for i in range(num_plot):
        dual_key = f'dual_{i}'
        if dual_key in metrics_history[0]:
            # Get dual values and convert to approximate eigenvalues
            dual_values = np.array([m[dual_key] for m in metrics_history])
            # Apply 0.5 factor before conversion (due to sampling scheme)
            dual_values_scaled = 0.5 * dual_values
            approx_eigenvalues = (gamma + dual_values_scaled) / (gamma * (1 + dual_values_scaled))

            gt_value = float(ground_truth_eigenvalues[i].real)
            errors = np.abs(approx_eigenvalues - gt_value)
            ax2.plot(steps, errors, label=f'|Approx λ_{i} - λ_{i}|', color=colors[i], linewidth=1.5)

    ax2.set_xlabel('Gradient Step', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Absolute Errors in Eigenvalue Approximations', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dual variable evolution plot saved to {save_path}")


def state_idx_to_xy(state_idx: int, width: int) -> tuple:
    """Convert state index to (x,y) coordinates."""
    y = state_idx // width
    x = state_idx % width
    return x, y


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


def collect_data_and_compute_eigenvectors(env, args: Args):
    """
    Collect transition data and compute ground truth eigenvectors.

    Args:
        env: Base environment (possibly with doors already applied)

    Returns:
        transition_matrix: Symmetrized transition matrix
        eigendecomp: Dictionary with eigenvalues and eigenvectors
        state_coords: Array of (x,y) coordinates for each state
        canonical_states: Array of free state indices
        doors: Door configuration (if use_doors=True)
        data_env: The environment used for data collection (with doors if applicable)
    """
    print("Collecting transition data...")
    num_states = env.width * env.height

    # Get canonical (free) states from base environment
    canonical_states = get_canonical_free_states(env)
    num_canonical = len(canonical_states)
    print(f"Number of free states: {num_canonical} (out of {num_states} total)")

    # Create door environment if requested
    door_config = None
    data_env = env  # Default to base environment

    if args.use_doors:
        print(f"\nCreating {args.num_doors} irreversible doors...")
        door_config = create_random_doors(
            env,
            canonical_states,
            num_doors=args.num_doors,
            seed=args.door_seed
        )
        print(f"  Created {door_config['num_doors']} doors (out of {door_config['total_reversible']} reversible transitions)")

        # Create environment with doors in the dynamics
        data_env = create_door_gridworld_from_base(env, door_config['doors'], canonical_states)
        print("  Created DoorGridWorld environment with irreversible transitions")

    # Collect transition counts and episodes in a single efficient pass
    print("Collecting transition data and episodes...")
    transition_counts_full, raw_episodes, metrics = collect_transition_counts_and_episodes(
        env=data_env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_states=num_states,
        seed=args.seed,
    )

    print(f"Collected {metrics['total_transitions']} transitions.")

    # Map full state indices to canonical indices
    full_to_canonical = {int(full_idx): canon_idx for canon_idx, full_idx in enumerate(canonical_states)}

    # Initialize replay buffer
    # Find max valid episode length for buffer sizing
    max_valid_length = int(raw_episodes['lengths'].max()) + 1
    replay_buffer = EpisodicReplayBuffer(
        max_episodes=args.num_envs,
        max_episode_length=max_valid_length,
        observation_type='canonical_state',
        seed=args.seed
    )

    # Convert raw episodes (OGBench format) to canonical state space and add to buffer
    # Each row may contain multiple trajectories separated by terminal flags
    for ep_idx in range(args.num_envs):
        episode_length = int(raw_episodes['lengths'][ep_idx])

        # Get ALL observations and terminals up to write_idx
        episode_obs_full = raw_episodes['observations'][ep_idx, :episode_length + 1]
        episode_terminals = raw_episodes['terminals'][ep_idx, :episode_length + 1]

        # Convert ALL observations to canonical state indices (don't filter by valids here)
        episode_obs_canonical = []
        episode_terminals_canonical = []
        for i, state_idx in enumerate(episode_obs_full):
            state_idx = int(state_idx)
            # Only convert states that exist in canonical mapping
            if state_idx in full_to_canonical:
                episode_obs_canonical.append(full_to_canonical[state_idx])
                episode_terminals_canonical.append(int(episode_terminals[i]))

        # Add ALL data to buffer if it has at least 2 states
        # The buffer will use terminals to determine trajectory boundaries during sampling
        if len(episode_obs_canonical) >= 2:
            # Reshape to (n, 1) as expected by replay buffer's transform function
            obs_array = np.array(episode_obs_canonical, dtype=np.int32).reshape(-1, 1)
            terminals_array = np.array(episode_terminals_canonical, dtype=np.int32)
            episode_dict = {
                'obs': obs_array,
                'terminals': terminals_array,
            }
            replay_buffer.add_episode(episode_dict)

    print(f"Added {len(replay_buffer)} trajectory sequences to replay buffer (may contain multiple episodes)")

    # Extract canonical state subspace
    transition_counts = transition_counts_full[jnp.ix_(canonical_states, jnp.arange(env.action_space), canonical_states)]

    # Build symmetrized transition matrix
    print("\nBuilding symmetrized transition matrix...")
    transition_matrix = get_symmetrized_transition_matrix(transition_counts)

    # Compute eigendecomposition
    print(f"Computing eigendecomposition (top {args.num_eigenvectors} eigenvectors)...")
    eigendecomp = compute_symmetrized_eigendecomposition(
        transition_matrix,
        k=args.num_eigenvectors,
    )

    print(f"Top {min(5, args.num_eigenvectors)} eigenvalues:")
    for i in range(min(5, args.num_eigenvectors)):
        print(f"  λ_{i}: {eigendecomp['eigenvalues'][i]:.6f}")

    # Create state coordinate mapping (only for canonical/free states)
    state_coords = []
    for state_idx in canonical_states:
        state_idx = int(state_idx)
        x, y = state_idx_to_xy(state_idx, env.width)
        state_coords.append([x, y])
    state_coords = jnp.array(state_coords, dtype=jnp.float32)

    # Center and scale coordinates to approximately [-1, 1] range
    # Center around grid center
    center = jnp.array([env.width / 2.0, env.height / 2.0], dtype=jnp.float32)
    state_coords = state_coords - center

    # Scale by half the maximum dimension (to get range approximately [-1, 1])
    scale = max(env.width, env.height) / 2.0
    state_coords = state_coords / scale

    print(f"\nCoordinate normalization:")
    print(f"  Center: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"  Scale: {scale:.2f}")
    print(f"  Coordinate range: x=[{state_coords[:, 0].min():.3f}, {state_coords[:, 0].max():.3f}], "
          f"y=[{state_coords[:, 1].min():.3f}, {state_coords[:, 1].max():.3f}]")

    return transition_matrix, eigendecomp, state_coords, canonical_states, door_config, data_env, replay_buffer


def learn_eigenvectors(args):
    """Main training loop to learn eigenvectors."""

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]

    # Set up run
    run_name = f"{args.env_type}__{args.exp_name}__{args.exp_number}__{args.seed}__{int(time.time())}"

    # Create results directories
    results_dir = Path(args.results_dir) / args.env_type / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    models_dir = results_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save args
    with open(results_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Results will be saved to: {results_dir}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, encoder_key = jax.random.split(rng_key, 2)

    # Create environment and collect data
    env = create_gridworld_env(args)
    transition_matrix, eigendecomp, state_coords, canonical_states, door_config, data_env, replay_buffer = collect_data_and_compute_eigenvectors(env, args)

    gt_eigenvalues = eigendecomp['eigenvalues_real']
    gt_eigenvectors = eigendecomp['right_eigenvectors_real']

    print(f"\nState coordinates shape: {state_coords.shape}")
    print(f"Ground truth eigenvectors shape: {gt_eigenvectors.shape}")

    # Save door configuration if doors were used
    if door_config is not None:
        door_save_path = results_dir / "door_config.pkl"
        with open(door_save_path, 'wb') as f:
            pickle.dump({
                'doors': door_config['doors'],
                'num_doors': door_config['num_doors'],
                'total_reversible': door_config['total_reversible'],
                'canonical_states': np.array(canonical_states),
            }, f)
        print(f"Door configuration saved to {door_save_path}")

    # Initialize the encoder
    encoder = CoordinateEncoder(
        num_features=args.num_eigenvectors,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )

    # Initialize dual variables
    if args.init_dual_diag:
        initial_dual_mask = jnp.eye(args.num_eigenvectors)
    else:
        initial_dual_mask = jnp.tril(jnp.ones((args.num_eigenvectors, args.num_eigenvectors)))

    # Create dummy input for initialization
    dummy_input = state_coords[:1]  # (1, 2)

    initial_params = {
        'encoder': encoder.init(encoder_key, dummy_input),
        'duals': args.duals_initial_val * initial_dual_mask,
        'barrier_coefs': args.barrier_initial_val * jnp.ones((1, 1)),
        'error_integral': jnp.zeros((args.num_eigenvectors, args.num_eigenvectors)),
    }

    # Create optimizer
    encoder_tx = optax.adam(learning_rate=args.learning_rate)
    sgd_tx = optax.sgd(learning_rate=args.learning_rate)

    # Create masks for different parameter groups
    encoder_mask = {
        'encoder': True,
        'duals': False,
        'barrier_coefs': False,
        'error_integral': False,
    }
    other_mask = {
        'encoder': False,
        'duals': True,
        'barrier_coefs': True,
        'error_integral': False,
    }

    tx = optax.chain(
        optax.masked(encoder_tx, encoder_mask),
        optax.masked(sgd_tx, other_mask)
    )

    encoder_state = TrainState.create(
        apply_fn=encoder.apply,
        params=initial_params,
        tx=tx,
    )
    encoder.apply = jax.jit(encoder.apply)

    # Define the update function
    @jax.jit
    def update_encoder(
        encoder_state: TrainState,
        state_coords_batch: jnp.ndarray,
        next_state_coords_batch: jnp.ndarray,
        state_coords_batch_2: jnp.ndarray,
    ):
        def check_previous_entries_below_threshold(matrix, threshold):
            # Create a matrix that contains 1 where the absolute value is below threshold
            below_threshold = (jnp.abs(matrix) < threshold).astype(jnp.float32)

            # Compute a scan that checks if all values up to each position are below threshold
            row_all_below = jnp.prod(below_threshold, axis=1, keepdims=True)

            # Finally, for each row i, check if all previous rows satisfy the condition
            cumulative_results = jnp.cumprod(row_all_below, axis=0)

            result_with_zero = jnp.ones((matrix.shape[0] + 1, 1), dtype=cumulative_results.dtype)
            result_with_zero = result_with_zero.at[1:, :].set(cumulative_results)
            final_results = result_with_zero[:-1, :]

            return final_results

        def encoder_loss(params):
            # Compute representations
            encoder_params = params['encoder']
            phi = encoder.apply(encoder_params, state_coords_batch)[0]
            phi_2 = encoder.apply(encoder_params, state_coords_batch_2)[0]
            next_phi = encoder.apply(encoder_params, next_state_coords_batch)[0]

            # Get sizes
            d = args.num_eigenvectors
            n = phi.shape[0]

            # Get duals
            dual_variables = params['duals']
            barrier_coefficients = params['barrier_coefs']
            diagonal_duals = jnp.diag(dual_variables)
            eigenvalue_sum = -0.5 * diagonal_duals.sum()

            # Compute inner product matrices
            inner_product_matrix_1 = jnp.einsum(
                'ij,ik->jk', phi, jax.lax.stop_gradient(phi)) / n
            inner_product_matrix_2 = jnp.einsum(
                'ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n

            error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(d))
            error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(d))

            # Compute error matrices below threshold
            cum_error_matrix_1_below_threshold = jax.lax.cond(
                args.turn_off_above_threshold,
                lambda x: check_previous_entries_below_threshold(x, args.cum_error_threshold),
                lambda x: jnp.ones([x.shape[0], 1]),
                error_matrix_1,
            )
            cum_error_matrix_1_below_threshold = jax.lax.stop_gradient(cum_error_matrix_1_below_threshold)

            # Compute dual loss
            error_integral = params['error_integral']
            dual_loss_pos = (
                jax.lax.stop_gradient(dual_variables)
                * cum_error_matrix_1_below_threshold * error_matrix_1
            ).sum()

            dual_loss_P = jax.lax.stop_gradient(args.step_size_duals * error_matrix_1)
            dual_loss_I = args.step_size_duals_I * jax.lax.stop_gradient(error_integral)
            dual_loss_neg = -(dual_variables * cum_error_matrix_1_below_threshold * (dual_loss_P + dual_loss_I)).sum()

            # Compute barrier loss
            quadratic_error_matrix = 2 * cum_error_matrix_1_below_threshold * error_matrix_1 * jax.lax.stop_gradient(error_matrix_2)
            quadratic_error = quadratic_error_matrix.sum()
            barrier_loss_pos = jax.lax.stop_gradient(barrier_coefficients[0, 0]) * quadratic_error
            barrier_loss_neg = -barrier_coefficients[0, 0] * jax.lax.stop_gradient(jnp.absolute(quadratic_error))

            # Compute representation variances
            phi_centered = (phi - jnp.mean(phi, axis=0, keepdims=True))
            phi_variances = (phi_centered ** 2).mean(0, keepdims=True)

            delta = jnp.exp(-1 / args.graph_variance_scale)
            if args.perturbation_type == 'squared-null-grad':
                graph_perturbation = args.graph_epsilon * (
                    ((phi_centered - 1) ** 2).mean(0, keepdims=True)
                ).clip(0, 1)
            elif args.perturbation_type == 'squared':
                graph_perturbation = args.graph_epsilon * (
                    ((jnp.absolute(phi_centered - 1) + delta) ** 2 - delta ** 2).mean(0, keepdims=True)
                ).clip(0, 1)
            elif args.perturbation_type == 'exponential':
                graph_perturbation = args.graph_epsilon * (
                    jnp.exp(-phi_variances / args.graph_variance_scale)
                    - delta
                ).clip(0, 1)
            else:
                graph_perturbation = jnp.zeros_like(phi_variances)

            graph_perturbation = graph_perturbation.at[0, 0].set(0.0)

            # Compute graph drawing losses
            diff = (phi - next_phi) * cum_error_matrix_1_below_threshold.reshape(1, -1)
            graph_losses = 0.5 * ((diff) ** 2).mean(0, keepdims=True)
            graph_loss = (graph_losses + graph_perturbation).sum()

            # Compute auxiliary metrics
            norm_phi = (phi ** 2).mean(0, keepdims=True)
            norm_errors_1 = jnp.diag(error_matrix_1)
            distance_to_constraint_manifold = jnp.tril(error_matrix_1 ** 2).sum()
            total_norm_error = jnp.absolute(norm_errors_1).sum()
            total_error = jnp.absolute(error_matrix_1).sum()
            total_two_component_error = jnp.absolute(error_matrix_1[:, :min(2, d)]).sum()

            # Total loss
            positive_loss = graph_loss + dual_loss_pos + barrier_loss_pos
            negative_loss = dual_loss_neg + barrier_loss_neg

            allo = positive_loss + negative_loss

            # Auxiliary metrics
            aux = {
                'graph_loss': graph_loss,
                'dual_loss': dual_loss_pos,
                'dual_loss_neg': dual_loss_neg,
                'barrier_loss': barrier_loss_pos,
                'approx_eigenvalue_sum': eigenvalue_sum,
                'barrier_coef': barrier_coefficients[0, 0],
                'total_norm_error': total_norm_error,
                'total_error': total_error,
                'total_two_component_error': total_two_component_error,
                'distance_to_constraint_manifold': distance_to_constraint_manifold,
                'distance_to_origin': norm_phi.sum(),
            }

            # Add dual variables to aux
            for i in range(min(11, args.num_eigenvectors)):
                aux[f'dual_{i}'] = dual_variables[i, i]
                aux[f'graph_perturbation_{i}'] = graph_perturbation[0, i]

                for j in range(0, min(2, i)):
                    aux[f'dual_{i}_{j}'] = dual_variables[i, j]

            return allo, (error_matrix_1, aux)

        # Compute loss and gradients
        (allo, (error_matrix, aux)), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        # Apply optimizer updates
        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        # Perform custom integral update with the error matrix
        new_params['error_integral'] = args.integral_decay * new_params['error_integral'] + error_matrix

        # Clip the barrier coefficients
        new_params['barrier_coefs'] = jnp.clip(new_params['barrier_coefs'], 0, args.max_barrier_coefs)

        # Create new state
        new_encoder_state = encoder_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=encoder_state.step + 1
        )

        # Get grad norm
        grads_flat, _ = jax.tree_util.tree_flatten(grads)
        grads_vector = jnp.concatenate([jnp.ravel(g) for g in grads_flat])
        grad_norm = jnp.linalg.norm(grads_vector)
        aux['grad_norm'] = grad_norm

        return new_encoder_state, allo, aux

    # Start the training process
    print("\nStarting training...")
    start_time = time.time()
    num_states = state_coords.shape[0]

    # Track metrics history
    metrics_history = []

    # Convert doors to portal markers for visualization
    door_markers = {}
    if door_config is not None and 'doors' in door_config:
        for s_canonical, a_forward, s_prime_canonical, a_reverse in door_config['doors']:
            s_full = int(canonical_states[s_canonical])
            s_prime_full = int(canonical_states[s_prime_canonical])
            door_markers[(s_full, a_forward)] = s_prime_full

    # Plot ground truth eigenvectors
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(args.num_eigenvectors)),
        eigendecomposition=eigendecomp,
        canonical_states=canonical_states,
        grid_width=env.width,
        grid_height=env.height,
        portals=door_markers if door_markers else None,
        eigenvector_type='right',
        component='real',
        ncols=min(4, args.num_eigenvectors),
        wall_color='gray',
        save_path=str(plots_dir / "ground_truth_eigenvectors.png"),
        shared_colorbar=True
    )
    plt.close()

    for gradient_step in tqdm(range(args.num_gradient_steps)):
        # Sample batches from episodic replay buffer using truncated geometric distribution
        batch1 = replay_buffer.sample(args.batch_size, discount=args.geometric_gamma)
        batch2 = replay_buffer.sample(args.batch_size, discount=args.geometric_gamma)

        # Extract state indices (canonical state indices)
        state_indices = jnp.array(batch1.obs)
        next_state_indices = jnp.array(batch1.next_obs)
        state_indices_2 = jnp.array(batch2.obs)

        # Get coordinates
        coords_batch = state_coords[state_indices]
        next_coords_batch = state_coords[next_state_indices]
        coords_batch_2 = state_coords[state_indices_2]

        # Update
        encoder_state, allo, metrics = update_encoder(
            encoder_state,
            coords_batch,
            next_coords_batch,
            coords_batch_2,
        )

        # Logging
        is_log_step = (
            ((gradient_step % args.log_freq) == 0)
            or (gradient_step == args.num_gradient_steps - 1)
        )
        if is_log_step:
            # Store metrics
            metrics_dict = {
                "gradient_step": gradient_step,
                "allo": float(allo.item()),
                "sps": int(gradient_step / (time.time() - start_time)),
            }
            for k, v in metrics.items():
                metrics_dict[k] = float(v.item())
            metrics_history.append(metrics_dict)

            if gradient_step % (args.log_freq * 10) == 0:
                print(f"Step {gradient_step}: loss={allo.item():.4f}, "
                      f"total_error={metrics['total_error'].item():.4f}")

        # Plot learned eigenvectors periodically
        is_plot_step = (
            ((gradient_step % args.plot_freq) == 0 and gradient_step > 0)
            or (gradient_step == args.num_gradient_steps - 1)
        )
        if is_plot_step:
            # Compute learned eigenvectors on all states
            learned_features = encoder.apply(encoder_state.params['encoder'], state_coords)[0]

            # Create a temporary eigendecomposition dict for visualization
            learned_eigendecomp = {
                'eigenvalues': jnp.zeros(args.num_eigenvectors, dtype=jnp.complex64),
                'eigenvalues_real': jnp.zeros(args.num_eigenvectors),
                'eigenvalues_imag': jnp.zeros(args.num_eigenvectors),
                'right_eigenvectors_real': learned_features,
                'right_eigenvectors_imag': jnp.zeros_like(learned_features),
                'left_eigenvectors_real': learned_features,
                'left_eigenvectors_imag': jnp.zeros_like(learned_features),
            }

            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvectors)),
                eigendecomposition=learned_eigendecomp,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='right',
                component='real',
                ncols=min(4, args.num_eigenvectors),
                wall_color='gray',
                save_path=str(plots_dir / f"learned_eigenvectors_step_{gradient_step}.png"),
                shared_colorbar=True
            )
            plt.close()

            # Plot learning curves
            plot_learning_curves(metrics_history, str(plots_dir / "learning_curves.png"))

            # Plot dual variable evolution vs ground truth eigenvalues
            plot_dual_variable_evolution(
                metrics_history,
                gt_eigenvalues,
                args.geometric_gamma,
                str(plots_dir / "dual_variable_evolution.png"),
                num_eigenvectors=args.num_eigenvectors
            )

    print("\nTraining complete!")

    # Save final metrics
    with open(results_dir / "metrics_history.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Metrics saved to {results_dir / 'metrics_history.json'}")

    # Save final model
    if args.save_model:
        save_path = models_dir / "final_model.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump({
                'params': encoder_state.params,
                'args': vars(args),
                'gt_eigenvalues': np.array(gt_eigenvalues),
                'gt_eigenvectors': np.array(gt_eigenvectors),
            }, f)
        print(f"Model saved to {save_path}")

    # Save final learned eigenvectors
    final_learned_features = encoder.apply(encoder_state.params['encoder'], state_coords)[0]
    np.save(results_dir / "final_learned_eigenvectors.npy", np.array(final_learned_features))

    # Create final comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Plot first ground truth eigenvector (skip constant eigenvector 0)
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=np.array(gt_eigenvectors[:, 1]),
        canonical_states=canonical_states,
        grid_width=env.width,
        grid_height=env.height,
        portals=door_markers if door_markers else None,
        title='Ground Truth Eigenvector 1',
        ax=axes[0],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    # Plot first learned feature
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=np.array(final_learned_features[:, 1]),
        canonical_states=canonical_states,
        grid_width=env.width,
        grid_height=env.height,
        portals=door_markers if door_markers else None,
        title='Learned Feature 1',
        ax=axes[1],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    plt.tight_layout()
    plt.savefig(plots_dir / "final_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll results saved to: {results_dir}")

    return encoder_state, results_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    learn_eigenvectors(args)
