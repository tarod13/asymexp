"""
Train a model to learn complex eigenvectors of non-symmetric Laplacians.
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
from exp_complex_basis.eigendecomposition import compute_eigendecomposition
from exp_alcl.episodic_replay_buffer import EpisodicReplayBuffer


# Simple MLP network for (x,y) coordinates
class CoordinateEncoder(nn.Module):
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    @nn.compact
    def __call__(self, xy_coords):
        """
        Args:
            xy_coords: (batch_size, 2) array of (x,y) coordinates

        Returns:
            features_dict: Dictionary containing:
                - left_real: (batch_size, num_features) left eigenvector real components
                - left_imag: (batch_size, num_features) left eigenvector imaginary components
                - right_real: (batch_size, num_features) right eigenvector real components
                - right_imag: (batch_size, num_features) right eigenvector imaginary components
        """
        x = xy_coords

        # Shared backbone
        # Input layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Hidden layers
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        # Separate heads for each component with independent transformations
        # Each head has at least one independent layer before the final prediction

        # Left real head
        left_real_hidden = nn.Dense(self.hidden_dim)(x)
        left_real_hidden = nn.relu(left_real_hidden)
        left_real = nn.Dense(1)(left_real_hidden)

        # Left imaginary head
        left_imag_hidden = nn.Dense(self.hidden_dim)(x)
        left_imag_hidden = nn.relu(left_imag_hidden)
        left_imag = nn.Dense(1)(left_imag_hidden)

        # Right real head
        right_real_hidden = nn.Dense(self.hidden_dim)(x)
        right_real_hidden = nn.relu(right_real_hidden)
        right_real = nn.Dense(1)(right_real_hidden)

        # Right imaginary head
        right_imag_hidden = nn.Dense(self.hidden_dim)(x)
        right_imag_hidden = nn.relu(right_imag_hidden)
        right_imag = nn.Dense(1)(right_imag_hidden)

        features_dict = {
            'left_real': left_real,
            'left_imag': left_imag,
            'right_real': right_real,
            'right_imag': right_imag,
        }

        return features_dict, {}


def get_transition_matrix(
    transition_counts: jnp.ndarray,
    make_stochastic: bool = True,
) -> jnp.ndarray:
    """
    Build transition matrix from counts (non-symmetrized).

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]
        make_stochastic: Whether to row-normalize to get proper transition probabilities
    Returns:
        Transition matrix of shape [num_states, num_states]
    """
    # Sum over actions if needed
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)  # [num_states, num_states]
    else:
        transition_matrix = transition_counts

    if make_stochastic:
        row_sums = jnp.sum(transition_matrix.clip(1e-8), axis=1, keepdims=True)
        transition_matrix = transition_matrix.clip(1e-8) / row_sums

    return transition_matrix


def compute_successor_representation(
    transition_matrix: jnp.ndarray,
    gamma: float,
    max_horizon: int = None,
) -> jnp.ndarray:
    """
    Compute the successor representation SR_γ = (I - γP)^(-1) or finite-horizon version.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor
        max_horizon: If provided, compute finite-horizon SR_γ^(T) = Σ_{k=0}^{T} γ^k P^k
                     Otherwise, compute infinite-horizon SR_γ = (I - γP)^(-1)

    Returns:
        Successor representation of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    if max_horizon is None:
        # Infinite-horizon: SR_γ = (I - γP)^(-1)
        sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix)
    else:
        # Finite-horizon: SR_γ^(T) = Σ_{k=0}^{T} γ^k P^k
        # Using closed form: SR_γ^(T) = (I - γP)^{-1}(I - γ^{T+1}P^{T+1})
        # (Derived by multiplying both sides on the left by (I - γP))

        # Compute P^{T+1} using built-in matrix power (uses efficient algorithm)
        P_T_plus_1 = jnp.linalg.matrix_power(transition_matrix, max_horizon + 1)

        # Apply finite geometric series formula (left multiplication is mathematically correct)
        gamma_power = gamma ** (max_horizon + 1)
        sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix) @ (identity - gamma_power * P_T_plus_1)

    return sr_matrix


def compute_sampling_distribution(
    transition_counts: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the empirical sampling distribution from transition counts.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]

    Returns:
        Diagonal matrix D where D_{ii} is the empirical sampling probability of state i
    """
    # Sum over actions and next states to get visit counts per state
    if len(transition_counts.shape) == 3:
        # Sum over actions and next states
        state_visit_counts = jnp.sum(transition_counts, axis=(1, 2))
    else:
        # Sum over next states
        state_visit_counts = jnp.sum(transition_counts, axis=1)

    # Normalize to get probabilities
    total_visits = jnp.sum(state_visit_counts)
    sampling_probs = state_visit_counts / total_visits

    # Create diagonal matrix
    D = jnp.diag(sampling_probs)

    return D


def compute_laplacian(
    transition_matrix: jnp.ndarray,
    gamma: float,
    delta: float = 0.0,
    max_horizon: int = None,
) -> jnp.ndarray:
    """
    Compute the non-symmetric Laplacian L = (1+δ)I - (1-γ)P·SR_γ.

    This definition matches what the episodic replay buffer approximates with
    geometric sampling (k >= 1). The transition matrix P applied to SR_γ gives
    the expected discounted future occupancy starting from the next state.

    The δ parameter shifts eigenvalues away from zero, improving numerical stability.
    With δ > 0, the smallest eigenvalue is δ instead of 0.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor used in successor representation
        delta: Eigenvalue shift parameter (default 0.0). Recommended: 0.1 for stability.
        max_horizon: If provided, use finite-horizon SR_γ^(T) instead of infinite-horizon

    Returns:
        Non-symmetric Laplacian of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    # Compute successor representation (infinite or finite-horizon)
    sr_matrix = compute_successor_representation(transition_matrix, gamma, max_horizon)

    # Compute Laplacian: L = (1+δ)I - (1-γ)P·SR_γ
    laplacian = (1 + delta) * identity - (1 - gamma) * (transition_matrix @ sr_matrix)

    return laplacian


# Training arguments
@dataclass
class Args:
    # Environment
    env_type: str = "room4"  # 'room4', 'maze', 'spiral', 'obstacles', 'empty', or 'file'
    env_file: str | None = None  # Path to environment text file (if env_type='file')
    env_file_name: str | None = None  # Name of environment file in src/envs/txt/ (e.g., 'GridRoom-4')
    max_episode_length: int = 1000

    # Data collection
    num_envs: int = 1000
    num_steps: int = 1000

    # Irreversible doors
    use_doors: bool = False
    num_doors: int = 5  # Number of irreversible doors to create
    door_seed: int = 42  # Seed for door placement

    # Model
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_gradient_steps: int = 20000
    gamma: float = 0.2  # Discount factor for successor representation
    delta: float = 0.1  # Eigenvalue shift parameter: L = (1+δ)I - M (improves numerical stability)
    lambda_x: float = 10.0  # Exponential decay parameter for CLF
    lambda_xy: float = 10.0  # Exponential decay parameter for CLF for xy phase

    # Episodic replay buffer
    max_time_offset: int | None = None  # Maximum time offset for sampling (None = episode length)
    
    # Logging and saving
    log_freq: int = 100
    plot_freq: int = 1000
    save_freq: int = 1000
    checkpoint_freq: int = 5000  # How often to save checkpoints (in gradient steps)
    save_model: bool = True
    plot_during_training: bool = False  # If True, creates plots during training (slow). If False, only exports data.
    results_dir: str = "./results"

    # Resuming training
    resume_from: str | None = None  # Path to results directory to resume from (e.g., './results/room4/room4__allo__0__42__1234567890')

    # Misc
    seed: int = 42
    exp_name: str | None = None
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

    # Check if environment has doors from file
    from src.envs.door_gridworld import DoorGridWorldEnv
    if isinstance(env, DoorGridWorldEnv):
        print(f"  Environment has doors from file: {len(env.blocked_transitions)} blocked transitions")
    else:
        print(f"  Environment type: {type(env).__name__} (no file-defined doors)")

    return env


def save_checkpoint(
    encoder_state: TrainState,
    metrics_history: list,
    gradient_step: int,
    save_path: Path,
    args: Args,
    rng_state: np.ndarray = None
):
    """
    Save a training checkpoint.

    Args:
        encoder_state: Current training state
        metrics_history: List of metrics dictionaries
        gradient_step: Current gradient step
        save_path: Path to save the checkpoint
        args: Training arguments
        rng_state: Current random state (optional)
    """
    checkpoint = {
        'gradient_step': gradient_step,
        'params': encoder_state.params,
        'opt_state': encoder_state.opt_state,
        'metrics_history': metrics_history,
        'args': vars(args),
    }

    if rng_state is not None:
        checkpoint['rng_state'] = rng_state

    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Resuming from gradient step: {checkpoint['gradient_step']}")

    return checkpoint


def compute_complex_cosine_similarities(
    learned_real: jnp.ndarray,
    learned_imag: jnp.ndarray,
    gt_real: jnp.ndarray,
    gt_imag: jnp.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute absolute value of real part of complex cosine similarity between learned and ground truth.

    Uses standard complex inner product with conjugate:
    For complex vectors u = u_real + i·u_imag, v = v_real + i·v_imag:
        <u, v> = conj(u)^T v = (u_real^T v_real + u_imag^T v_imag) + i(u_real^T v_imag - u_imag^T v_real)
        ||u|| = sqrt(u_real^T u_real + u_imag^T u_imag)
        cos(θ) = <u, v> / (||u|| ||v||)
        Result: |cos(θ)|

    Args:
        learned_real: Learned eigenvector real parts [num_states, num_eigenvectors]
        learned_imag: Learned eigenvector imaginary parts [num_states, num_eigenvectors]
        gt_real: Ground truth eigenvector real parts [num_states, num_eigenvectors]
        gt_imag: Ground truth eigenvector imaginary parts [num_states, num_eigenvectors]
        prefix: Prefix for metric names (e.g., "left_" or "right_")

    Returns:
        Dictionary containing:
            - {prefix}cosine_sim_{i}: Absolute value of real part of cosine similarity for component i
            - {prefix}cosine_sim_avg: Average across all components
    """
    num_components = learned_real.shape[1]

    similarities = {}
    cosine_sims = []

    for i in range(num_components):
        # Extract complex vectors
        u_real = learned_real[:, i]
        u_imag = learned_imag[:, i]
        v_real = gt_real[:, i]
        v_imag = gt_imag[:, i]

        # Standard complex inner product (with conjugate): <u, v> = conj(u)^T v
        # Real part: u_real^T v_real + u_imag^T v_imag
        # Imag part: u_real^T v_imag - u_imag^T v_real
        inner_real = jnp.dot(u_real, v_real) + jnp.dot(u_imag, v_imag)
        inner_imag = jnp.dot(u_real, v_imag) - jnp.dot(u_imag, v_real)

        # Magnitudes: ||u|| = sqrt(u_real^T u_real + u_imag^T u_imag)
        u_norm = jnp.sqrt(jnp.dot(u_real, u_real) + jnp.dot(u_imag, u_imag))
        v_norm = jnp.sqrt(jnp.dot(v_real, v_real) + jnp.dot(v_imag, v_imag))

        # Complex cosine similarity
        cos_real = inner_real / (u_norm * v_norm + 1e-10)
        cos_imag = inner_imag / (u_norm * v_norm + 1e-10)

        # Take absolute value of real part
        abs_cos_real = (cos_real**2 + cos_imag**2)**0.5
        similarities[f'{prefix}cosine_sim_{i}'] = float(abs_cos_real)
        cosine_sims.append(float(abs_cos_real))

    # Average across all components
    similarities[f'{prefix}cosine_sim_avg'] = float(np.mean(cosine_sims))

    return similarities


def normalize_eigenvectors_for_comparison(
    left_real: jnp.ndarray,
    left_imag: jnp.ndarray,
    right_real: jnp.ndarray,
    right_imag: jnp.ndarray,
    sampling_probs: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Apply normalization transformations to eigenvectors for proper comparison.

    The learned left eigenvectors correspond to eigenvectors of the adjoint with respect
    to the inner product determined by the replay buffer state distribution D.

    Normalization procedure:
    1. For RIGHT eigenvectors:
       - Find component with largest magnitude
       - Divide by that component (fixes arbitrary phase)
       - Normalize to unit norm

    2. For LEFT eigenvectors:
       - Multiply by the largest component from corresponding right eigenvector
       - Multiply by the norm of the corresponding right eigenvector (before normalization)
       - Scale each entry by the stationary state distribution (to convert from adjoint)

    This ensures proper comparison between learned and ground truth eigenvectors while
    accounting for:
    - Arbitrary complex scaling freedom
    - Adjoint vs. true left eigenvector relationship

    Args:
        left_real: Left eigenvector real parts [num_states, num_eigenvectors]
        left_imag: Left eigenvector imaginary parts [num_states, num_eigenvectors]
        right_real: Right eigenvector real parts [num_states, num_eigenvectors]
        right_imag: Right eigenvector imaginary parts [num_states, num_eigenvectors]
        sampling_probs: State distribution probabilities [num_states]

    Returns:
        Dictionary containing:
            - 'left_real': Normalized left eigenvector real parts
            - 'left_imag': Normalized left eigenvector imaginary parts
            - 'right_real': Normalized right eigenvector real parts
            - 'right_imag': Normalized right eigenvector imaginary parts
    """
    num_components = right_real.shape[1]
    num_states = right_real.shape[0]

    # Normalize eigenvectors
    normalized_right_real = jnp.zeros_like(right_real)
    normalized_right_imag = jnp.zeros_like(right_imag)
    normalized_left_real = jnp.zeros_like(left_real)
    normalized_left_imag = jnp.zeros_like(left_imag)

    for i in range(num_components):
        # Step 1: Process right eigenvectors
        right_r = right_real[:, i]
        right_i = right_imag[:, i]

        # Find component with largest magnitude
        magnitudes = jnp.sqrt(right_r**2 + right_i**2)
        max_idx = jnp.argmax(magnitudes)
        max_component_real = right_r[max_idx]
        max_component_imag = right_i[max_idx]

        # Complex division: divide by max component
        # (a + bi) / (c + di) = [(ac + bd) + i(bc - ad)] / (c^2 + d^2)
        denom = max_component_real**2 + max_component_imag**2 + 1e-10
        scaled_right_real = (right_r * max_component_real + right_i * max_component_imag) / denom
        scaled_right_imag = (right_i * max_component_real - right_r * max_component_imag) / denom

        # Normalize to unit norm
        right_norm = jnp.sqrt(jnp.sum(scaled_right_real**2 + scaled_right_imag**2))
        normalized_right_real = normalized_right_real.at[:, i].set(scaled_right_real / (right_norm + 1e-10))
        normalized_right_imag = normalized_right_imag.at[:, i].set(scaled_right_imag / (right_norm + 1e-10))

        # Step 2: Process left eigenvectors (complementary normalization)
        left_r = left_real[:, i]
        left_i = left_imag[:, i]

        # Compute the original norm of the right eigenvector (before normalization)
        original_right_norm = jnp.sqrt(jnp.sum(right_r**2 + right_i**2))

        # Multiply by max component (conjugate for complex multiplication)
        # (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        scaled_left_real = (left_r * max_component_real - left_i * max_component_imag)
        scaled_left_imag = (left_r * max_component_imag + left_i * max_component_real)

        # Multiply by the original norm of the right eigenvector
        scaled_left_real = scaled_left_real * original_right_norm
        scaled_left_imag = scaled_left_imag * original_right_norm

        # Scale each entry by the stationary state distribution
        # This converts from adjoint eigenvectors to true left eigenvectors
        scaled_left_real = scaled_left_real * sampling_probs
        scaled_left_imag = scaled_left_imag * sampling_probs

        normalized_left_real = normalized_left_real.at[:, i].set(scaled_left_real)
        normalized_left_imag = normalized_left_imag.at[:, i].set(scaled_left_imag)

    return {
        'left_real': normalized_left_real,
        'left_imag': normalized_left_imag,
        'right_real': normalized_right_real,
        'right_imag': normalized_right_imag,
    }


def compute_complex_cosine_similarities_with_normalization(
    learned_left_real: jnp.ndarray,
    learned_left_imag: jnp.ndarray,
    learned_right_real: jnp.ndarray,
    learned_right_imag: jnp.ndarray,
    gt_left_real: jnp.ndarray,
    gt_left_imag: jnp.ndarray,
    gt_right_real: jnp.ndarray,
    gt_right_imag: jnp.ndarray,
    sampling_probs: jnp.ndarray,
) -> Dict[str, float]:
    """
    Compute cosine similarities with proper normalization for adjoint eigenvectors.

    The learned left eigenvectors (psi) correspond to complex conjugates of eigenvectors
    of the adjoint with respect to the inner product determined by the replay buffer
    state distribution D.

    Normalization procedure:
    1. For RIGHT eigenvectors:
       - Find component with largest magnitude
       - Divide by that component (fixes arbitrary phase)
       - Normalize to unit norm

    2. For LEFT eigenvectors:
       - Multiply by the largest component from corresponding right eigenvector
       - Multiply by the norm of the corresponding right eigenvector (before normalization)
       - Scale each entry by the stationary state distribution (to convert from adjoint)

    Important: Ground truth eigenvectors are already normalized and should NOT be
    normalized again. We only normalize the learned eigenvectors.

    Also: Since learned psi are complex conjugates of adjoint eigenvectors, we use
    the complex conjugate of learned psi when computing cosine similarities.

    Args:
        learned_right_real: Learned right eigenvector real parts [num_states, num_eigenvectors]
        learned_right_imag: Learned right eigenvector imaginary parts [num_states, num_eigenvectors]
        gt_left_real: Ground truth left eigenvector real parts [num_states, num_eigenvectors]
        gt_left_imag: Ground truth left eigenvector imaginary parts [num_states, num_eigenvectors]
        gt_right_real: Ground truth right eigenvector real parts [num_states, num_eigenvectors]
        gt_right_imag: Ground truth right eigenvector imaginary parts [num_states, num_eigenvectors]
        sampling_probs: State distribution probabilities [num_states]

    Returns:
        Dictionary containing cosine similarities for left and right eigenvectors
    """
    # Normalize learned eigenvectors only (ground truth is already normalized)
    learned_normalized = normalize_eigenvectors_for_comparison(
        left_real=learned_left_real,
        left_imag=learned_left_imag,
        right_real=learned_right_real,
        right_imag=learned_right_imag,
        sampling_probs=sampling_probs
    )

    # For left eigenvectors
    left_sims = compute_complex_cosine_similarities(
        learned_normalized['left_real'], learned_normalized['left_imag'],  # Conjugate
        gt_left_real, gt_left_imag,  # Ground truth (already normalized)
        prefix="left_"
    )

    # For right eigenvectors
    right_sims = compute_complex_cosine_similarities(
        learned_normalized['right_real'], learned_normalized['right_imag'],
        gt_right_real, gt_right_imag,  # Ground truth (already normalized)
        prefix="right_"
    )

    # Combine results
    result = {}
    result.update(left_sims)
    result.update(right_sims)

    return result


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


def create_replay_buffer_only(env, canonical_states, args: Args):
    """
    Create replay buffer by collecting episodes (without recomputing eigenvectors).
    Used when resuming training to avoid redundant computation.

    Args:
        env: Environment to collect data from
        canonical_states: Array of free state indices
        args: Training arguments

    Returns:
        replay_buffer: EpisodicReplayBuffer filled with collected episodes
    """
    print("Collecting episodes for replay buffer...")
    num_states = env.width * env.height

    # Collect transition counts and episodes
    transition_counts_full, raw_episodes, metrics = collect_transition_counts_and_episodes(
        env=env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_states=num_states,
        seed=args.seed,
    )
    print(f"Collected {metrics['total_transitions']} transitions.")

    # Initialize replay buffer
    max_valid_length = int(raw_episodes['lengths'].max()) + 1
    replay_buffer = EpisodicReplayBuffer(
        max_episodes=args.num_envs,
        max_episode_length=max_valid_length,
        observation_type='canonical_state',
        seed=args.seed
    )

    # Convert and add episodes to buffer using vectorized method
    print("\nPopulating replay buffer with episodes...")
    replay_buffer.populate_with_tabular_episodes(
        canonical_states=canonical_states,
        episodes=raw_episodes,
    )
    print(f"  Added {len(replay_buffer)} trajectory sequences to replay buffer")
    return replay_buffer


def collect_data_and_compute_eigenvectors(env, args: Args):
    """
    Collect transition data and compute ground truth complex eigenvectors.

    Computes the non-symmetric Laplacian: L = I - (1-γ)P·SR_γ
    This Laplacian admits complex eigenvalues and distinct left/right eigenvectors.
    The definition matches what the graph loss approximates via geometric sampling.

    Args:
        env: Base environment (possibly with doors already applied)

    Returns:
        laplacian_matrix: Non-symmetric Laplacian L = I - (1-γ)P·SR_γ
        eigendecomp: Dictionary with complex eigenvalues and left/right eigenvectors
        state_coords: Array of (x,y) coordinates for each state
        canonical_states: Array of free state indices
        sampling_probs: 1D array of empirical sampling probabilities for each canonical state
        door_config: Door configuration (if use_doors=True)
        data_env: The environment used for data collection (with doors if applicable)
        replay_buffer: Episodic replay buffer filled with collected episodes
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

    # Initialize replay buffer
    # Find max valid episode length for buffer sizing
    max_valid_length = int(raw_episodes['lengths'].max()) + 1
    replay_buffer = EpisodicReplayBuffer(
        max_episodes=args.num_envs,
        max_episode_length=max_valid_length,
        observation_type='canonical_state',
        seed=args.seed
    )

    # Convert and add episodes to buffer using vectorized method
    print("\nPopulating replay buffer with episodes...")
    replay_buffer.populate_with_tabular_episodes(
        canonical_states=canonical_states,
        episodes=raw_episodes,
    )
    print(f"  Added {len(replay_buffer)} trajectory sequences to replay buffer")

    # Extract canonical state subspace
    transition_counts = transition_counts_full[jnp.ix_(canonical_states, jnp.arange(env.action_space), canonical_states)]

    # Build transition matrix (non-symmetrized)
    print("\nBuilding transition matrix...")
    transition_matrix = get_transition_matrix(transition_counts)

    # Compute sampling distribution D
    print("Computing empirical sampling distribution...")
    sampling_distribution = compute_sampling_distribution(transition_counts)
    sampling_probs = jnp.diag(sampling_distribution)
    print(f"  Sampling prob range: [{sampling_probs.min():.6f}, {sampling_probs.max():.6f}]")
    print(f"  Sampling prob std: {sampling_probs.std():.6f}")

    # Compute non-symmetric Laplacian for complex eigenvectors
    print(f"\nComputing non-symmetric Laplacian with gamma={args.gamma}, delta={args.delta}...")
    print(f"  L = (1+δ)I - (1-γ)P·SR_γ (matches geometric sampling with k >= 1)")

    laplacian_matrix = compute_laplacian(transition_matrix, args.gamma, delta=args.delta)
    eigendecomp = compute_eigendecomposition(
        laplacian_matrix,
        k=1,
        ascending=True  # For Laplacians, we want smallest eigenvalues
    )

    print(f"\nSmallest {min(5, 1)} eigenvalues (complex):")
    print("  Eigenvalue (real + imag)")
    for i in range(min(5, 1)):
        ev_real = eigendecomp['eigenvalues_real'][i]
        ev_imag = eigendecomp['eigenvalues_imag'][i]
        print(f"  λ_{i}: {ev_real:.6f} + {ev_imag:.6f}i")

    # Print ranges of eigenvector values to debug visualization issues
    print(f"\nEigenvector value ranges (first {min(5, 1)} eigenvectors):")
    for i in range(min(5, 1)):
        # Right eigenvectors
        right_real = eigendecomp['right_eigenvectors_real'][:, i]
        right_imag = eigendecomp['right_eigenvectors_imag'][:, i]
        print(f"  Right eigenvector {i} (real): min={np.min(right_real):.6f}, max={np.max(right_real):.6f}, mean={np.mean(right_real):.6f}")
        print(f"  Right eigenvector {i} (imag): min={np.min(right_imag):.6f}, max={np.max(right_imag):.6f}, mean={np.mean(right_imag):.6f}")

        # Left eigenvectors
        left_real = eigendecomp['left_eigenvectors_real'][:, i]
        left_imag = eigendecomp['left_eigenvectors_imag'][:, i]
        print(f"  Left eigenvector {i} (real):  min={np.min(left_real):.6f}, max={np.max(left_real):.6f}, mean={np.mean(left_real):.6f}")
        print(f"  Left eigenvector {i} (imag):  min={np.min(left_imag):.6f}, max={np.max(left_imag):.6f}, mean={np.mean(left_imag):.6f}")
        print()

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

    return laplacian_matrix, eigendecomp, state_coords, canonical_states, sampling_probs, door_config, data_env, replay_buffer, transition_matrix


def learn_eigenvectors(args):
    """Main training loop to learn eigenvectors."""

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]

    # Check if resuming from a previous run
    checkpoint_data = None
    start_step = 0
    metrics_history = []

    if args.resume_from is not None:
        # Load checkpoint from previous run
        resume_dir = Path(args.resume_from)
        if not resume_dir.exists():
            raise ValueError(f"Resume directory does not exist: {resume_dir}")

        checkpoint_path = resume_dir / "models" / "checkpoint.pkl"
        checkpoint_data = load_checkpoint(checkpoint_path)

        # Use the same results directory
        results_dir = resume_dir
        start_step = checkpoint_data['gradient_step'] + 1  # Start from next step
        metrics_history = checkpoint_data['metrics_history']

        # Load the original args but keep the new training parameters
        original_args = checkpoint_data['args']
        # Update only the new training parameters (allow changing learning rate, num_steps, etc.)
        for key in ['learning_rate', 'num_gradient_steps', 'log_freq', 'plot_freq', 'checkpoint_freq']:
            if hasattr(args, key):
                original_args[key] = getattr(args, key)

        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING FROM: {results_dir}")
        print(f"  Starting from step: {start_step}")
        print(f"  Loaded metrics history: {len(metrics_history)} entries")
        print(f"{'='*60}\n")
    else:
        # Set up new run
        run_name = f"{args.env_type}__{args.exp_name}__{args.exp_number}__{args.seed}__{int(time.time())}"

        # Create results directories
        results_dir = Path(args.results_dir) / args.env_type / run_name
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Results will be saved to: {results_dir}")

    # Create subdirectories
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    models_dir = results_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save args (overwrite if resuming to track parameter changes)
    with open(results_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, encoder_key = jax.random.split(rng_key, 2)

    # Restore RNG state if resuming
    if checkpoint_data is not None and 'rng_state' in checkpoint_data:
        np.random.set_state(checkpoint_data['rng_state'])

    # Load or create environment and data
    if checkpoint_data is not None:
        # Load saved data from previous run
        print("Loading saved environment data...")
        env = create_gridworld_env(args)

        # Load saved eigenvectors and state coordinates (complex)
        gt_eigenvalues_real = jnp.array(np.load(results_dir / "gt_eigenvalues_real.npy"))
        gt_eigenvalues_imag = jnp.array(np.load(results_dir / "gt_eigenvalues_imag.npy"))
        gt_left_real = jnp.array(np.load(results_dir / "gt_left_real.npy"))
        gt_left_imag = jnp.array(np.load(results_dir / "gt_left_imag.npy"))
        gt_right_real = jnp.array(np.load(results_dir / "gt_right_real.npy"))
        gt_right_imag = jnp.array(np.load(results_dir / "gt_right_imag.npy"))

        state_coords = jnp.array(np.load(results_dir / "state_coords.npy"))

        # Try to load sampling distribution if it exists
        sampling_dist_path = results_dir / "sampling_distribution.npy"
        if sampling_dist_path.exists():
            sampling_probs = jnp.array(np.load(sampling_dist_path))
        else:
            sampling_probs = None

        # Load visualization metadata
        with open(results_dir / "viz_metadata.pkl", 'rb') as f:
            viz_metadata = pickle.load(f)
        canonical_states = viz_metadata['canonical_states']

        # Reconstruct door_config if it exists
        door_config = None
        door_config_path = results_dir / "door_config.pkl"
        if door_config_path.exists():
            with open(door_config_path, 'rb') as f:
                door_config = pickle.load(f)

        # Recreate replay buffer (much faster than recomputing everything)
        # Note: We don't save/load the replay buffer; we recreate it
        # This is acceptable since sampling is random anyway
        replay_buffer = create_replay_buffer_only(env, canonical_states, args)

        # Recreate data_env if doors were used
        if door_config is not None and 'doors' in door_config:
            from src.envs.door_gridworld import create_door_gridworld_from_base
            data_env = create_door_gridworld_from_base(env, door_config['doors'], canonical_states)
        else:
            data_env = env

        print("Loaded saved data successfully")
    else:
        # Create environment and collect data (new run)
        env = create_gridworld_env(args)
        laplacian_matrix, eigendecomp, state_coords, canonical_states, \
            sampling_probs, door_config, data_env, replay_buffer, transition_matrix = \
                collect_data_and_compute_eigenvectors(env, args)

        # Extract ground truth eigenvalues and eigenvectors (complex)
        gt_eigenvalues = eigendecomp['eigenvalues']  # Complex eigenvalues
        gt_eigenvalues_real = eigendecomp['eigenvalues_real']
        gt_eigenvalues_imag = eigendecomp['eigenvalues_imag']
        gt_left_real = eigendecomp['left_eigenvectors_real']
        gt_left_imag = eigendecomp['left_eigenvectors_imag']
        gt_right_real = eigendecomp['right_eigenvectors_real']
        gt_right_imag = eigendecomp['right_eigenvectors_imag']

    print(f"\nState coordinates shape: {state_coords.shape}")
    print(f"Ground truth right eigenvectors shape: {gt_right_real.shape}")
    print(f"Ground truth left eigenvectors shape: {gt_left_real.shape}")

    # Save data for new runs (skip if resuming)
    if checkpoint_data is None:
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

        # Save ground truth eigendecomposition and state coords
        # Non-symmetric Laplacian: L = I - (1-γ)P·SR_γ (complex eigenvalues and eigenvectors)
        np.save(results_dir / "gt_eigenvalues_real.npy", np.array(gt_eigenvalues_real))
        np.save(results_dir / "gt_eigenvalues_imag.npy", np.array(gt_eigenvalues_imag))
        np.save(results_dir / "gt_left_real.npy", np.array(gt_left_real))
        np.save(results_dir / "gt_left_imag.npy", np.array(gt_left_imag))
        np.save(results_dir / "gt_right_real.npy", np.array(gt_right_real))
        np.save(results_dir / "gt_right_imag.npy", np.array(gt_right_imag))
        # State coordinates
        np.save(results_dir / "state_coords.npy", np.array(state_coords))
        # Sampling distribution
        np.save(results_dir / "sampling_distribution.npy", np.array(sampling_probs))

        # Visualize and save sampling distribution
        if args.plot_during_training:
            # Convert doors to portal markers for visualization
            door_markers = {}
            if door_config is not None and 'doors' in door_config:
                for s_canonical, a_forward, s_prime_canonical, a_reverse in door_config['doors']:
                    s_full = int(canonical_states[s_canonical])
                    s_prime_full = int(canonical_states[s_prime_canonical])
                    door_markers[(s_full, a_forward)] = s_prime_full

    # Initialize the encoder
    encoder = CoordinateEncoder(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )

    # Create optimizer
    encoder_tx = optax.adam(learning_rate=args.learning_rate)
    sgd_tx = optax.sgd(learning_rate=args.learning_rate)

    # Create masks for different parameter groups
    encoder_mask = {
        'encoder': True,
    }
    other_mask = {
        'encoder': False,
    }

    tx = optax.chain(
        optax.masked(encoder_tx, encoder_mask),
        optax.masked(sgd_tx, other_mask)
    )

    # Initialize or restore encoder state
    if checkpoint_data is not None:
        # Restore from checkpoint
        # Convert loaded arrays to JAX device arrays for optimal performance
        import jax.tree_util as tree_util

        def to_device_array(x):
            """Ensure array is a JAX device array (optimized for GPU/TPU)."""
            if isinstance(x, (jnp.ndarray, np.ndarray)):
                # Force conversion to JAX array and materialize on device
                return jnp.array(x)
            return x

        # Optimize loaded parameters and optimizer state
        params_optimized = tree_util.tree_map(to_device_array, checkpoint_data['params'])
        opt_state_optimized = tree_util.tree_map(to_device_array, checkpoint_data['opt_state'])

        encoder_state = TrainState.create(
            apply_fn=encoder.apply,
            params=params_optimized,
            tx=tx,
        )
        # Restore optimizer state
        encoder_state = encoder_state.replace(opt_state=opt_state_optimized)
        print("Restored encoder state from checkpoint (optimized for device)")
    else:
        # Create dummy input for initialization
        dummy_input = state_coords[:1]  # (1, 2)

        initial_params = {
            'encoder': encoder.init(encoder_key, dummy_input),
        }

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

        def encoder_loss(params):
            # Compute representations for complex eigenvectors
            encoder_params = params['encoder']
            features_1 = encoder.apply(encoder_params, state_coords_batch)[0]
            features_2 = encoder.apply(encoder_params, state_coords_batch_2)[0]
            next_features = encoder.apply(encoder_params, next_state_coords_batch)[0]

            # Extract right eigenvectors (real and imaginary parts)
            # features_1 contains: left_real, left_imag, right_real, right_imag
            x_r = features_1['right_real']  # Right eigenvectors, real part
            x_i = features_1['right_imag']  # Right eigenvectors, imaginary part
            y_r = features_2['left_real']
            y_i = features_2['left_imag']

            x_r_2 = features_2['right_real']
            x_i_2 = features_2['right_imag']

            next_x_r = next_features['right_real']
            next_x_i = next_features['right_imag']
            next_y_r = next_features['left_real']
            next_y_i = next_features['left_imag']

            # Get sizes
            n = x_r.shape[0]

            # Compute current unnormalized eigenvalue estimates
            lambda_x_r = (x_r * next_x_r).mean() + (x_i * next_x_i).mean()
            lambda_x_i = (x_r * next_x_i).mean() - (x_i * next_x_r).mean()
            lambda_y_r = (y_r * next_y_r).mean() + (y_i * next_y_i).mean()
            lambda_y_i = (y_r * next_y_i).mean() - (y_i * next_y_r).mean()

            # Compute squared norms
            norm_x_r_sq = (x_r ** 2).mean()
            norm_x_i_sq = (x_i ** 2).mean()
            norm_x_sq = norm_x_r_sq + norm_x_i_sq

            norm_y_r_sq = (next_y_r ** 2).mean()
            norm_y_i_sq = (next_y_i ** 2).mean()
            norm_y_sq = norm_y_r_sq + norm_y_i_sq

            # Compute graph losses
            f_x_real = - next_x_r  + lambda_x_r * x_r - lambda_x_i * x_i
            f_x_imag = - next_x_i  + lambda_x_i * x_r + lambda_x_r * x_i
            f_y_real = - y_r  + lambda_y_r * next_y_r - lambda_y_i * next_y_i
            f_y_imag = - y_i  + lambda_y_i * next_y_r + lambda_y_r * next_y_i

            graph_loss_x_real = (x_r * jax.lax.stop_gradient(f_x_real)).mean()
            graph_loss_x_imag = (x_i * jax.lax.stop_gradient(f_x_imag)).mean()
            graph_loss_x = graph_loss_x_real + graph_loss_x_imag

            graph_loss_y_real = (next_y_r * jax.lax.stop_gradient(f_y_real)).mean()
            graph_loss_y_imag = (next_y_i * jax.lax.stop_gradient(f_y_imag)).mean()
            graph_loss_y = graph_loss_y_real + graph_loss_y_imag

            graph_loss = graph_loss_x + graph_loss_y

            # Compute Lyapunov functions and their gradients
            # 1. For x norm
            norm_x_error = norm_x_sq - 1
            V_x_norm = norm_x_error ** 2 / 2
            nabla_x_r_V_x_norm = 2 * norm_x_error * x_r
            nabla_x_i_V_x_norm = 2 * norm_x_error * x_i
            nabla_y_r_V_x_norm = jnp.zeros_like(y_r)
            nabla_y_i_V_x_norm = jnp.zeros_like(y_i)

            # 2. For y norm
            norm_y_error = norm_y_sq - 1
            V_y_norm = norm_y_error ** 2 / 2
            nabla_x_r_V_y_norm = jnp.zeros_like(x_r)
            nabla_x_i_V_y_norm = jnp.zeros_like(x_i)
            nabla_y_r_V_y_norm = 2 * norm_y_error * next_y_r
            nabla_y_i_V_y_norm = 2 * norm_y_error * next_y_i

            # 3. For xy phase
            phase_xy = (y_r * x_i).mean() - (y_i * x_r).mean()
            V_xy_phase = phase_xy ** 2 / 2
            nabla_x_r_V_xy_phase = - phase_xy * y_i
            nabla_x_i_V_xy_phase = phase_xy * y_r
            nabla_y_r_V_xy_phase = phase_xy * next_x_i
            nabla_y_i_V_xy_phase = - phase_xy * next_x_r

            # 4. Global
            V = V_x_norm + V_y_norm + V_xy_phase
            nabla_x_r_V = nabla_x_r_V_x_norm + nabla_x_r_V_y_norm + nabla_x_r_V_xy_phase
            nabla_x_i_V = nabla_x_i_V_x_norm + nabla_x_i_V_y_norm + nabla_x_i_V_xy_phase
            nabla_y_r_V = nabla_y_r_V_x_norm + nabla_y_r_V_y_norm + nabla_y_r_V_xy_phase
            nabla_y_i_V = nabla_y_i_V_x_norm + nabla_y_i_V_y_norm + nabla_y_i_V_xy_phase

            norm_nabla_x_r_V_sq = (nabla_x_r_V ** 2).mean()
            norm_nabla_x_i_V_sq = (nabla_x_i_V ** 2).mean()
            norm_nabla_y_r_V_sq = (nabla_y_r_V ** 2).mean()
            norm_nabla_y_i_V_sq = (nabla_y_i_V ** 2).mean()
            norm_nabla_V_sq = (
                norm_nabla_x_r_V_sq + norm_nabla_x_i_V_sq 
                + norm_nabla_y_r_V_sq + norm_nabla_y_i_V_sq
            )

            # Compute Control Lyapunov Function (CLF) loss
            f_x_r_dot_nabla_V = (f_x_real * nabla_x_r_V).mean()
            f_x_i_dot_nabla_V = (f_x_imag * nabla_x_i_V).mean()
            f_y_r_dot_nabla_V = (f_y_real * nabla_y_r_V).mean()
            f_y_i_dot_nabla_V = (f_y_imag * nabla_y_i_V).mean()
            f_dot_nabla_V = f_x_r_dot_nabla_V + f_x_i_dot_nabla_V + f_y_r_dot_nabla_V + f_y_i_dot_nabla_V

            clf_num = f_dot_nabla_V + args.lambda_x * V
            barrier = jnp.maximum(0, clf_num) / (norm_nabla_V_sq + 1e-8)
            u_x_r = barrier * nabla_x_r_V
            u_x_i = barrier * nabla_x_i_V
            u_y_r = barrier * nabla_y_r_V
            u_y_i = barrier * nabla_y_i_V
            clf_loss_x_r = (x_r * jax.lax.stop_gradient(u_x_r)).mean()
            clf_loss_x_i = (x_i * jax.lax.stop_gradient(u_x_i)).mean()
            clf_loss_y_r = (next_y_r * jax.lax.stop_gradient(u_y_r)).mean()
            clf_loss_y_i = (next_y_i * jax.lax.stop_gradient(u_y_i)).mean()
            clf_loss = clf_loss_x_r + clf_loss_x_i + clf_loss_y_r + clf_loss_y_i

            # Total loss
            total_loss = graph_loss + clf_loss

            # Auxiliary metrics
            aux = {
                'total_loss': total_loss,
                'graph_loss': graph_loss,
                'clf_loss': clf_loss,
                'graph_loss_x_real': graph_loss_x_real,
                'graph_loss_x_imag': graph_loss_x_imag,
                'clf_loss_x_real': clf_loss_x_r,
                'clf_loss_x_imag': clf_loss_x_i,
                'graph_loss_y_real': graph_loss_y_real,
                'graph_loss_y_imag': graph_loss_y_imag,
                'clf_loss_y_real': clf_loss_y_r,
                'clf_loss_y_imag': clf_loss_y_i,
                'lambda_x_real': lambda_x_r,
                'lambda_x_imag': lambda_x_i,
                'lambda_y_real': lambda_y_r,
                'lambda_y_imag': lambda_y_i,
                'V_x_norm': V_x_norm,
                'V_y_norm': V_y_norm,
                'V_xy_phase': V_xy_phase,
                'norm_x_sq': norm_x_sq,
                'norm_y_sq': norm_y_sq,
                'barrier': barrier,
            }

            return total_loss, aux

        # Compute loss and gradients
        (total_loss, aux), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        # Apply optimizer updates
        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

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

        return new_encoder_state, total_loss, aux

    # Start the training process
    if checkpoint_data is None:
        print("\nStarting training...")
    else:
        print(f"\nResuming training from step {start_step}...")

    start_time = time.time()
    num_states = state_coords.shape[0]

    # Convert doors to portal markers for visualization
    door_markers = {}

    # First, extract doors from random door config if available
    if door_config is not None and 'doors' in door_config:
        for s_canonical, a_forward, s_prime_canonical, a_reverse in door_config['doors']:
            s_full = int(canonical_states[s_canonical])
            s_prime_full = int(canonical_states[s_prime_canonical])
            door_markers[(s_full, a_forward)] = s_prime_full

    # Also extract doors directly from environment (for file-defined doors)
    from src.envs.door_gridworld import DoorGridWorldEnv
    if isinstance(data_env, DoorGridWorldEnv) and data_env.has_doors:
        # Extract doors from blocked_transitions
        # blocked_transitions are (state, action) pairs
        # We need to figure out the forward transition for visualization
        full_to_canonical = {int(full_idx): canon_idx for canon_idx, full_idx in enumerate(canonical_states)}

        print(f"  DEBUG: Blocked transitions from environment:")
        action_names = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
        for state, act in data_env.blocked_transitions:
            y, x = state // data_env.width, state % data_env.width
            print(f"    State ({x}, {y}) cannot perform action {action_names[act]}")

        for state_full, action in data_env.blocked_transitions:
            # This blocks the transition from state_full via action
            # To visualize, we need the REVERSE door (the one that's allowed)
            # The blocked transition is the reverse, so we need to find the forward transition
            reverse_action_map = {0: 2, 1: 3, 2: 0, 3: 1}  # U<->D, L<->R
            forward_action = reverse_action_map[action]

            # Calculate the source state (which is the destination of the blocked transition)
            # The blocked transition is (dest, reverse_action), so we need (source, forward_action)
            action_effects = {
                0: (0, -1),  # Up
                1: (1, 0),   # Right
                2: (0, 1),   # Down
                3: (-1, 0),  # Left
            }
            dx, dy = action_effects[action]
            dest_y = state_full // data_env.width
            dest_x = state_full % data_env.width
            source_x = dest_x + dx
            source_y = dest_y + dy

            # Check if source is valid
            if 0 <= source_x < data_env.width and 0 <= source_y < data_env.height:
                source_full = source_y * data_env.width + source_x
                # Add to door_markers if not already there
                if (source_full, forward_action) not in door_markers:
                    door_markers[(source_full, forward_action)] = state_full

        if len(door_markers) > (len(door_config['doors']) if door_config else 0):
            num_file_doors = len(door_markers) - (len(door_config['doors']) if door_config else 0)
            print(f"  Added {num_file_doors} file-defined doors to visualization")
            print(f"  DEBUG: Door markers from file:")
            action_names = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
            for (src_state, action), dest_state in door_markers.items():
                src_y, src_x = src_state // data_env.width, src_state % data_env.width
                dest_y, dest_x = dest_state // data_env.width, dest_state % data_env.width
                print(f"    Door from ({src_x}, {src_y}) via {action_names[action]} to ({dest_x}, {dest_y})")

    # Save visualization metadata for new runs (skip if resuming)
    if checkpoint_data is None:
        viz_metadata = {
            'canonical_states': np.array(canonical_states),
            'grid_width': env.width,
            'grid_height': env.height,
            'door_markers': door_markers,
            'num_eigenvectors': 1,
            'gamma': args.gamma,
        }
        with open(results_dir / "viz_metadata.pkl", 'wb') as f:
            pickle.dump(viz_metadata, f)

        # Optionally plot ground truth eigenvectors immediately
        if args.plot_during_training:
            # Plot non-symmetric Laplacian eigenvectors (complex)
            eigendecomp_viz = {
                'eigenvalues': (gt_eigenvalues_real + 1j * gt_eigenvalues_imag).astype(jnp.complex64),
                'eigenvalues_real': gt_eigenvalues_real,
                'eigenvalues_imag': gt_eigenvalues_imag,
                'right_eigenvectors_real': gt_right_real,
                'right_eigenvectors_imag': gt_right_imag,
                'left_eigenvectors_real': gt_left_real,
                'left_eigenvectors_imag': gt_left_imag,
            }

            # Visualize right eigenvectors (real part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(1)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='right',
                component='real',
                ncols=min(4, 1),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_right_eigenvectors_real.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize right eigenvectors (imaginary part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(1)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='right',
                component='imag',
                ncols=min(4, 1),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_right_eigenvectors_imag.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize left eigenvectors (real part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(1)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='left',
                component='real',
                ncols=min(4, 1),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_left_eigenvectors_real.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize left eigenvectors (imaginary part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(1)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='left',
                component='imag',
                ncols=min(4, 1),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_left_eigenvectors_imag.png"),
                shared_colorbar=True
            )
            plt.close()

    # Warmup: trigger JIT compilation with loaded checkpoint state
    if checkpoint_data is not None:
        print("Running warmup step to trigger JIT compilation with loaded state...")
        # Sample a small batch for warmup
        warmup_batch = replay_buffer.sample(min(args.batch_size, 32), discount=args.gamma)
        warmup_indices = jnp.array(warmup_batch.obs)
        warmup_next_indices = jnp.array(warmup_batch.next_obs)
        warmup_coords = state_coords[warmup_indices]
        warmup_next_coords = state_coords[warmup_next_indices]

        # Run one update to compile with the loaded state (discard result)
        warmup_state, warmup_loss, _ = update_encoder(
            encoder_state,
            warmup_coords,
            warmup_next_coords,
            warmup_coords,  # Use same batch for second set
        )

        # Block until compilation is complete
        jax.block_until_ready(warmup_state.params)
        print("Warmup complete - JIT compilation finished")
        # Note: We discard warmup_state and continue with the original encoder_state

    # Timing diagnostics for performance debugging
    timing_samples = {'sample': [], 'update': [], 'log': [], 'total': []}

    for gradient_step in tqdm(range(start_step, args.num_gradient_steps)):
        step_start = time.time()

        # Sample batches from episodic replay buffer using truncated geometric distribution
        sample_start = time.time()
        batch1 = replay_buffer.sample(args.batch_size, discount=args.gamma)
        batch2 = replay_buffer.sample(args.batch_size, discount=args.gamma)

        # Extract state indices (canonical state indices)
        state_indices = jnp.array(batch1.obs)
        next_state_indices = jnp.array(batch1.next_obs)
        state_indices_2 = jnp.array(batch2.obs)

        # Get coordinates
        coords_batch = state_coords[state_indices]
        next_coords_batch = state_coords[next_state_indices]
        coords_batch_2 = state_coords[state_indices_2]
        sample_time = time.time() - sample_start

        # Update
        update_start = time.time()
        encoder_state, total_loss, metrics = update_encoder(
            encoder_state,
            coords_batch,
            next_coords_batch,
            coords_batch_2,
        )
        # Block until GPU computation completes for accurate timing
        jax.block_until_ready(encoder_state.params)
        update_time = time.time() - update_start

        # Logging
        is_log_step = (
            ((gradient_step % args.log_freq) == 0)
            or (gradient_step == args.num_gradient_steps - 1)
        )
        log_start = time.time()
        if is_log_step:
            # Compute learned eigenvectors on all states for cosine similarity
            features_dict = encoder.apply(encoder_state.params['encoder'], state_coords)[0]

            # Compute complex cosine similarities with proper normalization
            # This accounts for:
            # 1. Arbitrary complex scaling of eigenvectors
            # 2. Adjoint vs. true left eigenvector relationship (via sampling distribution)
            cosine_sims = compute_complex_cosine_similarities_with_normalization(
                learned_left_real=features_dict['left_real'],
                learned_left_imag=features_dict['left_imag'],
                learned_right_real=features_dict['right_real'],
                learned_right_imag=features_dict['right_imag'],
                gt_left_real=gt_left_real,
                gt_left_imag=gt_left_imag,
                gt_right_real=gt_right_real,
                gt_right_imag=gt_right_imag,
                sampling_probs=sampling_probs
            )

            # Store metrics
            elapsed_time = time.time() - start_time
            steps_completed = gradient_step - start_step
            metrics_dict = {
                "gradient_step": gradient_step,
                "total_loss": float(total_loss.item()),
                "sps": int(steps_completed / max(elapsed_time, 1e-6)),  # Steps since start/resume
            }
            for k, v in metrics.items():
                metrics_dict[k] = float(v.item())

            # Add cosine similarities to metrics (both left and right)
            for k, v in cosine_sims.items():
                metrics_dict[k] = v

            metrics_history.append(metrics_dict)

            if gradient_step % (args.log_freq * 10) == 0:
                right_sim = cosine_sims['right_cosine_sim_avg']
                left_sim = cosine_sims['left_cosine_sim_avg']
                print(f"Step {gradient_step}: total_loss={total_loss.item():.4f}, "
                      f"left_sim={left_sim:.4f}, right_sim={right_sim:.4f}")
        log_time = time.time() - log_start

        # Collect timing samples (last 1000 steps for statistics)
        step_total_time = time.time() - step_start
        timing_samples['sample'].append(sample_time)
        timing_samples['update'].append(update_time)
        timing_samples['log'].append(log_time)
        timing_samples['total'].append(step_total_time)

        # Keep only recent samples to avoid memory growth
        for key in timing_samples:
            if len(timing_samples[key]) > 1000:
                timing_samples[key] = timing_samples[key][-1000:]

        # Report timing statistics periodically
        if gradient_step > 0 and gradient_step % (args.log_freq * 10) == 0:
            avg_update = np.mean(timing_samples['update'][-100:]) * 1000  # ms
            avg_sample = np.mean(timing_samples['sample'][-100:]) * 1000  # ms
            avg_total = np.mean(timing_samples['total'][-100:]) * 1000  # ms
            print(f"  Timing (avg last 100 steps): sample={avg_sample:.2f}ms, "
                  f"update={avg_update:.2f}ms, total={avg_total:.2f}ms")

        # Save metrics history and optionally plot periodically
        is_plot_step = (
            ((gradient_step % args.plot_freq) == 0 and gradient_step > 0)
            or (gradient_step == args.num_gradient_steps - 1)
        )
        if is_plot_step:
            # Save metrics history periodically (for live plotting)
            # Note: Saving without indent is much faster for large histories
            with open(results_dir / "metrics_history.json", 'w') as f:
                json.dump(metrics_history, f)

            # Compute and save latest learned eigenvectors (overwrite each time)
            features_dict = encoder.apply(encoder_state.params['encoder'], state_coords)[0]

            # Save raw eigenvectors
            np.save(results_dir / "latest_learned_left_real.npy", np.array(features_dict['left_real']))
            np.save(results_dir / "latest_learned_left_imag.npy", np.array(features_dict['left_imag']))
            np.save(results_dir / "latest_learned_right_real.npy", np.array(features_dict['right_real']))
            np.save(results_dir / "latest_learned_right_imag.npy", np.array(features_dict['right_imag']))

            # Compute and save normalized eigenvectors
            normalized_features = normalize_eigenvectors_for_comparison(
                left_real=features_dict['left_real'],
                left_imag=features_dict['left_imag'],
                right_real=features_dict['right_real'],
                right_imag=features_dict['right_imag'],
                sampling_probs=sampling_probs
            )
            np.save(results_dir / "latest_learned_left_real_normalized.npy", np.array(normalized_features['left_real']))
            np.save(results_dir / "latest_learned_left_imag_normalized.npy", np.array(normalized_features['left_imag']))
            np.save(results_dir / "latest_learned_right_real_normalized.npy", np.array(normalized_features['right_real']))
            np.save(results_dir / "latest_learned_right_imag_normalized.npy", np.array(normalized_features['right_imag']))

            # Optionally create plots during training (slower)
            if args.plot_during_training:
                # Create a temporary eigendecomposition dict for visualization
                learned_eigendecomp = {
                    'eigenvalues': jnp.zeros(1, dtype=jnp.complex64),
                    'eigenvalues_real': jnp.zeros(1),
                    'eigenvalues_imag': jnp.zeros(1),
                    'left_eigenvectors_real': features_dict['left_real'],
                    'left_eigenvectors_imag': features_dict['left_imag'],
                    'right_eigenvectors_real': features_dict['right_real'],
                    'right_eigenvectors_imag': features_dict['right_imag'],
                }

            else:
                # Just log progress
                if gradient_step % (args.plot_freq * 5) == 0:
                    print(f"Saved metrics at step {gradient_step}")

        # Save checkpoint periodically
        is_checkpoint_step = (
            ((gradient_step % args.checkpoint_freq) == 0 and gradient_step > 0)
            or (gradient_step == args.num_gradient_steps - 1)
        )
        if is_checkpoint_step:
            checkpoint_path = models_dir / "checkpoint.pkl"
            # Note: As metrics_history grows, checkpoint saving may take longer
            # Consider saving only recent metrics if this becomes a bottleneck
            save_checkpoint(
                encoder_state=encoder_state,
                metrics_history=metrics_history,
                gradient_step=gradient_step,
                save_path=checkpoint_path,
                args=args,
                rng_state=np.random.get_state()
            )
            print(f"  Checkpoint size: {len(metrics_history)} metric entries")

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
                'gt_eigenvalues_real': np.array(gt_eigenvalues_real),
                'gt_eigenvalues_imag': np.array(gt_eigenvalues_imag),
                'gt_left_real': np.array(gt_left_real),
                'gt_left_imag': np.array(gt_left_imag),
                'gt_right_real': np.array(gt_right_real),
                'gt_right_imag': np.array(gt_right_imag),
            }, f)
        print(f"Model saved to {save_path}")

    # Save final learned eigenvectors
    final_features_dict = encoder.apply(encoder_state.params['encoder'], state_coords)[0]

    # Save raw eigenvectors
    np.save(results_dir / "final_learned_left_real.npy", np.array(final_features_dict['left_real']))
    np.save(results_dir / "final_learned_left_imag.npy", np.array(final_features_dict['left_imag']))
    np.save(results_dir / "final_learned_right_real.npy", np.array(final_features_dict['right_real']))
    np.save(results_dir / "final_learned_right_imag.npy", np.array(final_features_dict['right_imag']))

    # Compute and save normalized eigenvectors
    final_normalized_features = normalize_eigenvectors_for_comparison(
        left_real=final_features_dict['left_real'],
        left_imag=final_features_dict['left_imag'],
        right_real=final_features_dict['right_real'],
        right_imag=final_features_dict['right_imag'],
        sampling_probs=sampling_probs
    )
    np.save(results_dir / "final_learned_left_real_normalized.npy", np.array(final_normalized_features['left_real']))
    np.save(results_dir / "final_learned_left_imag_normalized.npy", np.array(final_normalized_features['left_imag']))
    np.save(results_dir / "final_learned_right_real_normalized.npy", np.array(final_normalized_features['right_real']))
    np.save(results_dir / "final_learned_right_imag_normalized.npy", np.array(final_normalized_features['right_imag']))

    # Optionally create final comparison plots
    if args.plot_during_training:
        # Determine which eigenvector to plot (skip constant eigenvector 0 if multiple exist)
        evec_idx = 0 if 1 == 1 else 1

        # Create 3-column comparison: Ground Truth | Raw | Normalized
        # Right eigenvectors (real part)
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        visualize_eigenvector_on_grid(
            eigenvector_idx=evec_idx,
            eigenvector_values=np.array(gt_right_real[:, evec_idx]),
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            portals=door_markers if door_markers else None,
            title=f'Ground Truth Right {evec_idx} (Real)',
            ax=axes[0],
            cmap='RdBu_r',
            show_colorbar=True,
            wall_color='gray'
        )

        visualize_eigenvector_on_grid(
            eigenvector_idx=evec_idx,
            eigenvector_values=np.array(final_features_dict['right_real'][:, evec_idx]),
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            portals=door_markers if door_markers else None,
            title=f'Raw Learned Right {evec_idx} (Real)',
            ax=axes[1],
            cmap='RdBu_r',
            show_colorbar=True,
            wall_color='gray'
        )

        visualize_eigenvector_on_grid(
            eigenvector_idx=evec_idx,
            eigenvector_values=np.array(final_normalized_features['right_real'][:, evec_idx]),
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            portals=door_markers if door_markers else None,
            title=f'Normalized Learned Right {evec_idx} (Real)',
            ax=axes[2],
            cmap='RdBu_r',
            show_colorbar=True,
            wall_color='gray'
        )

        plt.tight_layout()
        plt.savefig(plots_dir / "final_comparison_right.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nData exported. Use generate_plots_complex.py to create visualizations.")
    print(f"Example: python generate_plots_complex.py {results_dir}")

    print(f"\nAll results saved to: {results_dir}")

    return encoder_state, results_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    learn_eigenvectors(args)
