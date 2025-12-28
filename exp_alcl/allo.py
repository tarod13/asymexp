"""
Train a model to learn eigenvectors of the symmetrized Laplacian.

This script uses (x,y) coordinates as inputs to the network and learns
the eigenvectors of L = I - (SR_γ + SR_γ^T)/2 through an augmented
Lagrangian optimization approach, where SR_γ is the successor representation
matrix (Neumann series).
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
        row_sums = jnp.sum(transition_matrix.clip(1), axis=1, keepdims=True)
        transition_matrix = transition_matrix.clip(1) / row_sums

    return transition_matrix


def compute_successor_representation(
    transition_matrix: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """
    Compute the successor representation SR_γ = (I - γP)^(-1).

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor

    Returns:
        Successor representation of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    # SR_γ = (I - γP)^(-1)
    sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix)

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


def compute_symmetrized_laplacian(
    transition_matrix: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """
    Compute the symmetrized Laplacian L = I - (1-γ)(SR_γ + SR_γ^T)/2.

    This is a simplified version that assumes uniform sampling distribution.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor used in successor representation

    Returns:
        Symmetrized Laplacian of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    # Compute successor representation SR_γ = (I - γP)^(-1)
    sr_matrix = compute_successor_representation(transition_matrix, gamma)

    # Symmetrize: (SR_γ + SR_γ^T)/2
    sr_symmetrized = (sr_matrix + sr_matrix.T) / 2.0

    # Compute Laplacian: L = I - (1-γ)(SR_γ + SR_γ^T)/2
    laplacian = identity - (1 - gamma) * sr_symmetrized

    return laplacian


def compute_weighted_symmetrized_laplacian(
    transition_matrix: jnp.ndarray,
    sampling_distribution: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """
    Compute the weighted symmetrized Laplacian L = D - (1-γ)(DSR_γ + SR_γ^TD^T)/2.

    This version accounts for non-uniform sampling distribution D.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        sampling_distribution: Shape [num_states, num_states], diagonal matrix D with sampling probabilities
        gamma: Discount factor used in successor representation

    Returns:
        Weighted symmetrized Laplacian of shape [num_states, num_states]
    """
    # Compute successor representation SR_γ = (I - γP)^(-1)
    sr_matrix = compute_successor_representation(transition_matrix, gamma)

    # Compute DSR_γ and SR_γ^TD^T
    D_SR = sampling_distribution @ sr_matrix
    SR_D_T = sr_matrix.T @ sampling_distribution.T

    # Symmetrize: (DSR_γ + SR_γ^TD^T)/2
    weighted_sr_symmetrized = (D_SR + SR_D_T) / 2.0

    # Compute Laplacian: L = D - (1-γ)(DSR_γ + SR_γ^TD^T)/2
    laplacian = sampling_distribution - (1 - gamma) * weighted_sr_symmetrized

    return laplacian


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

    # Sort by ascending eigenvalue magnitude
    sorted_indices = jnp.argsort(jnp.abs(eigenvalues))
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
    num_eigenvectors: int = 10
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_gradient_steps: int = 20000
    gamma: float = 0.2  # Discount factor for successor representation

    # Episodic replay buffer
    max_time_offset: int | None = None  # Maximum time offset for sampling (None = episode length)

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


def plot_dual_variable_evolution(metrics_history, ground_truth_eigenvalues, gamma, save_path,
                                  num_eigenvectors=11, ground_truth_eigenvalues_simple=None):
    """
    Plot the evolution of dual variables (Laplacian eigenvalue estimates) vs ground truth eigenvalues.

    The duals are eigenvalues of the weighted Laplacian L = D - (1-γ)(DSR_γ + SR_γ^TD^T)/2,
    where SR_γ is the successor representation with discount gamma and D is the sampling distribution.

    The 0.5 factor arises from the sampling scheme with the episodic replay buffer,
    which samples (s_t, s_{t+k}) pairs where k is geometrically distributed. This
    effectively implements the symmetrized version with the 0.5 scaling.

    Args:
        metrics_history: List of metric dictionaries
        ground_truth_eigenvalues: Array of ground truth eigenvalues of the weighted Laplacian
        gamma: Discount factor used in the successor representation (not used in this version)
        save_path: Path to save the plot
        num_eigenvectors: Number of eigenvectors to plot
        ground_truth_eigenvalues_simple: Optional array of simple Laplacian eigenvalues for comparison
    """
    steps = [m['gradient_step'] for m in metrics_history]
    num_plot = min(num_eigenvectors, len(ground_truth_eigenvalues))

    # Create figure with two rows: eigenvalue approximation and errors
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Dual variables (Laplacian eigenvalues) vs ground truth
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, num_plot))

    for i in range(num_plot):
        dual_key = f'dual_{i}'
        if dual_key in metrics_history[0]:
            # Get dual values (these are eigenvalues of the Laplacian)
            dual_values = np.array([m[dual_key] for m in metrics_history])
            # Apply 0.5 factor due to sampling scheme
            dual_values_scaled = 0.5 * dual_values

            ax1.plot(steps, dual_values_scaled, label=f'Learned λ_{i}', color=colors[i], linewidth=1.5)

            # Plot weighted Laplacian ground truth as solid horizontal line
            gt_value = float(ground_truth_eigenvalues[i].real)
            ax1.axhline(y=gt_value, color=colors[i], linestyle='-', alpha=0.3, linewidth=2.5,
                       label=f'GT Weighted λ_{i}' if i == 0 else '')

            # Plot simple Laplacian ground truth as dashed line if provided
            if ground_truth_eigenvalues_simple is not None:
                gt_simple_value = float(ground_truth_eigenvalues_simple[i].real)
                ax1.axhline(y=gt_simple_value, color=colors[i], linestyle='--', alpha=0.3, linewidth=1.5,
                           label=f'GT Simple λ_{i}' if i == 0 else '')

    ax1.set_xlabel('Gradient Step', fontsize=12)
    ax1.set_ylabel('Laplacian Eigenvalue', fontsize=12)
    ax1.set_title('Learned Eigenvalues vs Ground Truth (solid=weighted, dashed=simple)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute errors for both versions
    ax2 = axes[1]

    for i in range(num_plot):
        dual_key = f'dual_{i}'
        if dual_key in metrics_history[0]:
            # Get dual values (eigenvalues of the Laplacian)
            dual_values = np.array([m[dual_key] for m in metrics_history])
            # Apply 0.5 factor due to sampling scheme
            dual_values_scaled = 0.5 * dual_values

            # Error vs weighted Laplacian
            gt_value = float(ground_truth_eigenvalues[i].real)
            errors = np.abs(dual_values_scaled - gt_value)
            ax2.plot(steps, errors, label=f'vs Weighted λ_{i}', color=colors[i], linewidth=1.5, linestyle='-')

            # Error vs simple Laplacian if provided
            if ground_truth_eigenvalues_simple is not None:
                gt_simple_value = float(ground_truth_eigenvalues_simple[i].real)
                errors_simple = np.abs(dual_values_scaled - gt_simple_value)
                ax2.plot(steps, errors_simple, label=f'vs Simple λ_{i}', color=colors[i],
                        linewidth=1.5, linestyle='--', alpha=0.5)

    ax2.set_xlabel('Gradient Step', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Absolute Errors (solid=weighted, dashed=simple)', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dual variable evolution plot saved to {save_path}")


def plot_sampling_distribution(
    sampling_probs: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    save_path: str,
    portals: dict = None,
):
    """
    Visualize the empirical sampling distribution D on the grid.

    Args:
        sampling_probs: 1D array of sampling probabilities for each canonical state
        canonical_states: Array of canonical state indices
        grid_width: Width of the grid
        grid_height: Height of the grid
        save_path: Path to save the visualization
        portals: Optional dictionary of portal/door markers
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create a full grid initialized with NaN for obstacles
    grid_values = np.full((grid_height, grid_width), np.nan)

    # Fill in the sampling probabilities for canonical states
    for canon_idx, full_state_idx in enumerate(canonical_states):
        y = int(full_state_idx) // grid_width
        x = int(full_state_idx) % grid_width
        grid_values[y, x] = float(sampling_probs[canon_idx])

    # Create the heatmap
    im = ax.imshow(grid_values, cmap='viridis', interpolation='nearest', origin='upper')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Sampling Probability', fontsize=12)

    # Add grid lines
    ax.set_xticks(np.arange(grid_width) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_height) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

    # Add portal markers if provided
    if portals:
        for (state, action), next_state in portals.items():
            y = state // grid_width
            x = state % grid_width
            next_y = next_state // grid_width
            next_x = next_state % grid_width

            # Draw arrow from state to next_state
            dx = next_x - x
            dy = next_y - y
            ax.arrow(x, y, dx * 0.3, dy * 0.3,
                    head_width=0.2, head_length=0.15,
                    fc='red', ec='red', linewidth=2, alpha=0.7)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Empirical Sampling Distribution D', fontsize=14, fontweight='bold')

    # Add statistics text
    stats_text = (f'Min: {np.nanmin(grid_values):.6f}\n'
                  f'Max: {np.nanmax(grid_values):.6f}\n'
                  f'Mean: {np.nanmean(grid_values):.6f}\n'
                  f'Std: {np.nanstd(grid_values):.6f}')
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sampling distribution visualization saved to {save_path}")


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


def compute_cosine_similarities(learned_features: jnp.ndarray, gt_eigenvectors: jnp.ndarray) -> Dict[str, float]:
    """
    Compute absolute cosine similarity per component between learned and ground truth eigenvectors.

    For each eigenvector component i, computes the absolute cosine similarity:
        |cos(θ_i)| = |<learned_i, gt_i>| / (||learned_i|| * ||gt_i||)

    Args:
        learned_features: Learned eigenvector features [num_states, num_eigenvectors]
        gt_eigenvectors: Ground truth eigenvectors [num_states, num_eigenvectors]

    Returns:
        Dictionary containing:
            - cosine_sim_i: Absolute cosine similarity for each component i
            - cosine_sim_avg: Average absolute cosine similarity across all components
    """
    num_components = learned_features.shape[1]

    similarities = {}
    cosine_sims = []

    for i in range(num_components):
        learned_vec = learned_features[:, i]
        gt_vec = gt_eigenvectors[:, i]

        # Compute cosine similarity
        dot_product = jnp.dot(learned_vec, gt_vec)
        learned_norm = jnp.linalg.norm(learned_vec)
        gt_norm = jnp.linalg.norm(gt_vec)

        # Absolute cosine similarity (handles sign ambiguity)
        cosine_sim = jnp.abs(dot_product / (learned_norm * gt_norm + 1e-10))

        similarities[f'cosine_sim_{i}'] = float(cosine_sim)
        cosine_sims.append(float(cosine_sim))

    # Average across all components
    similarities['cosine_sim_avg'] = float(np.mean(cosine_sims))

    return similarities


def plot_cosine_similarity_evolution(metrics_history: Dict, save_path: str, num_eigenvectors: int = None):
    """
    Plot the evolution of cosine similarities between learned and ground truth eigenvectors.

    Args:
        metrics_history: List of metric dictionaries containing cosine similarity values
        save_path: Path to save the plot
        num_eigenvectors: Number of eigenvectors to plot (None = plot all available)
    """
    steps = [m['gradient_step'] for m in metrics_history]

    # Determine number of eigenvectors to plot
    if num_eigenvectors is None:
        # Count how many cosine_sim_i keys exist
        num_eigenvectors = sum(1 for key in metrics_history[0].keys() if key.startswith('cosine_sim_') and key != 'cosine_sim_avg')

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Use a colormap for different eigenvectors
    colors = plt.cm.viridis(np.linspace(0, 1, num_eigenvectors + 1))

    # Plot cosine similarity for each component
    for i in range(num_eigenvectors):
        key = f'cosine_sim_{i}'
        if key in metrics_history[0]:
            values = [m[key] for m in metrics_history]
            ax.plot(steps, values, label=f'Component {i}', color=colors[i], linewidth=1.5, alpha=0.7)

    # Plot average cosine similarity with thicker line
    if 'cosine_sim_avg' in metrics_history[0]:
        avg_values = [m['cosine_sim_avg'] for m in metrics_history]
        ax.plot(steps, avg_values, label='Average', color='black', linewidth=2.5, linestyle='--')

    ax.set_xlabel('Gradient Step', fontsize=12)
    ax.set_ylabel('Absolute Cosine Similarity', fontsize=12)
    ax.set_title('Evolution of Absolute Cosine Similarity between Learned and Ground Truth Eigenvectors', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])  # Cosine similarity is in [0, 1]

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cosine similarity evolution plot saved to {save_path}")


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

    # Map full state indices to canonical indices
    full_to_canonical = {int(full_idx): canon_idx for canon_idx, full_idx in enumerate(canonical_states)}

    # Initialize replay buffer
    max_valid_length = int(raw_episodes['lengths'].max()) + 1
    replay_buffer = EpisodicReplayBuffer(
        max_episodes=args.num_envs,
        max_episode_length=max_valid_length,
        observation_type='canonical_state',
        seed=args.seed
    )

    # Convert and add episodes to buffer
    for ep_idx in range(args.num_envs):
        episode_length = int(raw_episodes['lengths'][ep_idx])
        episode_obs_full = raw_episodes['observations'][ep_idx, :episode_length + 1]
        episode_terminals = raw_episodes['terminals'][ep_idx, :episode_length + 1]

        episode_obs_canonical = []
        episode_terminals_canonical = []
        for i, state_idx in enumerate(episode_obs_full):
            state_idx = int(state_idx)
            if state_idx in full_to_canonical:
                episode_obs_canonical.append(full_to_canonical[state_idx])
                episode_terminals_canonical.append(int(episode_terminals[i]))

        if len(episode_obs_canonical) >= 2:
            obs_array = np.array(episode_obs_canonical, dtype=np.int32).reshape(-1, 1)
            terminals_array = np.array(episode_terminals_canonical, dtype=np.int32)
            episode_dict = {
                'obs': obs_array,
                'terminals': terminals_array,
            }
            replay_buffer.add_episode(episode_dict)

    print(f"Added {len(replay_buffer)} trajectory sequences to replay buffer")
    return replay_buffer


def collect_data_and_compute_eigenvectors(env, args: Args):
    """
    Collect transition data and compute ground truth eigenvectors.

    Computes eigenvalues for both:
    - Simple Laplacian: L = I - (1-γ)(SR_γ + SR_γ^T)/2
    - Weighted Laplacian: L = D - (1-γ)(DSR_γ + SR_γ^TD^T)/2

    Args:
        env: Base environment (possibly with doors already applied)

    Returns:
        laplacian_matrix: Weighted Laplacian L = D - (1-γ)(DSR_γ + SR_γ^TD^T)/2
        eigendecomp: Dictionary with eigenvalues and eigenvectors of the weighted Laplacian
        eigendecomp_simple: Dictionary with eigenvalues and eigenvectors of the simple Laplacian
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

    # Build transition matrix (non-symmetrized)
    print("\nBuilding transition matrix...")
    transition_matrix = get_transition_matrix(transition_counts)

    # Compute sampling distribution D
    print("Computing empirical sampling distribution...")
    sampling_distribution = compute_sampling_distribution(transition_counts)
    sampling_probs = jnp.diag(sampling_distribution)
    print(f"  Sampling prob range: [{sampling_probs.min():.6f}, {sampling_probs.max():.6f}]")
    print(f"  Sampling prob std: {sampling_probs.std():.6f}")

    # Compute BOTH Laplacian versions for comparison
    print(f"\nComputing Laplacians with gamma={args.gamma}...")

    # Version 1: Simple Laplacian L = I - (1-γ)(SR_γ + SR_γ^T)/2
    print("  Computing simple Laplacian L = I - (1-γ)(SR_γ + SR_γ^T)/2...")
    laplacian_simple = compute_symmetrized_laplacian(transition_matrix, args.gamma)
    eigendecomp_simple = compute_symmetrized_eigendecomposition(
        laplacian_simple,
        k=args.num_eigenvectors,
    )

    # Version 2: Weighted Laplacian L = D - (1-γ)(DSR_γ + SR_γ^TD^T)/2
    print("  Computing weighted Laplacian L = D - (1-γ)(DSR_γ + SR_γ^TD^T)/2...")
    laplacian_weighted = compute_weighted_symmetrized_laplacian(
        transition_matrix, sampling_distribution, args.gamma
    )
    eigendecomp_weighted = compute_symmetrized_eigendecomposition(
        laplacian_weighted,
        k=args.num_eigenvectors,
    )

    # Use the weighted version as the main baseline (what the algorithm actually learns)
    laplacian_matrix = laplacian_weighted
    eigendecomp = eigendecomp_weighted

    print(f"\nTop {min(5, args.num_eigenvectors)} eigenvalues comparison:")
    print("  Simple Laplacian | Weighted Laplacian")
    for i in range(min(5, args.num_eigenvectors)):
        simple_ev = eigendecomp_simple['eigenvalues'][i]
        weighted_ev = eigendecomp_weighted['eigenvalues'][i]
        print(f"  λ_{i}: {simple_ev:.6f} | {weighted_ev:.6f}")

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

    return laplacian_matrix, eigendecomp, eigendecomp_simple, state_coords, canonical_states, sampling_probs, door_config, data_env, replay_buffer


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

        # Load saved eigenvectors and state coordinates
        gt_eigenvalues = jnp.array(np.load(results_dir / "gt_eigenvalues.npy"))
        gt_eigenvectors = jnp.array(np.load(results_dir / "gt_eigenvectors.npy"))
        # Try to load simple eigenvalues if they exist
        simple_eigenvalues_path = results_dir / "gt_eigenvalues_simple.npy"
        if simple_eigenvalues_path.exists():
            gt_eigenvalues_simple = jnp.array(np.load(simple_eigenvalues_path))
        else:
            gt_eigenvalues_simple = None
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
        laplacian_matrix, eigendecomp, eigendecomp_simple, state_coords, canonical_states, sampling_probs, door_config, data_env, replay_buffer = collect_data_and_compute_eigenvectors(env, args)

        gt_eigenvalues = eigendecomp['eigenvalues_real']
        gt_eigenvectors = eigendecomp['right_eigenvectors_real']
        gt_eigenvalues_simple = eigendecomp_simple['eigenvalues_real']
        gt_eigenvectors_simple = eigendecomp_simple['right_eigenvectors_real']

    print(f"\nState coordinates shape: {state_coords.shape}")
    print(f"Ground truth eigenvectors shape: {gt_eigenvectors.shape}")

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
        # Weighted Laplacian (main baseline)
        np.save(results_dir / "gt_eigenvalues.npy", np.array(gt_eigenvalues))
        np.save(results_dir / "gt_eigenvectors.npy", np.array(gt_eigenvectors))
        # Simple Laplacian (for comparison)
        np.save(results_dir / "gt_eigenvalues_simple.npy", np.array(gt_eigenvalues_simple))
        np.save(results_dir / "gt_eigenvectors_simple.npy", np.array(gt_eigenvectors_simple))
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

            plot_sampling_distribution(
                sampling_probs=sampling_probs,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                save_path=str(plots_dir / "sampling_distribution.png"),
                portals=door_markers if door_markers else None
            )

    # Initialize the encoder
    encoder = CoordinateEncoder(
        num_features=args.num_eigenvectors,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )

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
        # Initialize fresh
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

            # Compute dual loss
            error_integral = params['error_integral']
            dual_loss_pos = (
                jax.lax.stop_gradient(dual_variables)
                * error_matrix_1
            ).sum()

            dual_loss_P = jax.lax.stop_gradient(args.step_size_duals * error_matrix_1)
            dual_loss_I = args.step_size_duals_I * jax.lax.stop_gradient(error_integral)
            dual_loss_neg = -(dual_variables * (dual_loss_P + dual_loss_I)).sum()

            # Compute barrier loss
            quadratic_error_matrix = 2 * error_matrix_1 * jax.lax.stop_gradient(error_matrix_2)
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
            diff = phi - next_phi
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
    if checkpoint_data is None:
        print("\nStarting training...")
    else:
        print(f"\nResuming training from step {start_step}...")

    start_time = time.time()
    num_states = state_coords.shape[0]

    # Convert doors to portal markers for visualization
    door_markers = {}
    if door_config is not None and 'doors' in door_config:
        for s_canonical, a_forward, s_prime_canonical, a_reverse in door_config['doors']:
            s_full = int(canonical_states[s_canonical])
            s_prime_full = int(canonical_states[s_prime_canonical])
            door_markers[(s_full, a_forward)] = s_prime_full

    # Save visualization metadata for new runs (skip if resuming)
    if checkpoint_data is None:
        viz_metadata = {
            'canonical_states': np.array(canonical_states),
            'grid_width': env.width,
            'grid_height': env.height,
            'door_markers': door_markers,
            'num_eigenvectors': args.num_eigenvectors,
            'gamma': args.gamma,
        }
        with open(results_dir / "viz_metadata.pkl", 'wb') as f:
            pickle.dump(viz_metadata, f)

        # Optionally plot ground truth eigenvectors immediately
        if args.plot_during_training:
            # Need to create eigendecomp for visualization
            eigendecomp_viz = {
                'eigenvalues': gt_eigenvalues.astype(jnp.complex64),
                'eigenvalues_real': gt_eigenvalues,
                'eigenvalues_imag': jnp.zeros_like(gt_eigenvalues),
                'right_eigenvectors_real': gt_eigenvectors,
                'right_eigenvectors_imag': jnp.zeros_like(gt_eigenvectors),
                'left_eigenvectors_real': gt_eigenvectors,
                'left_eigenvectors_imag': jnp.zeros_like(gt_eigenvectors),
            }
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvectors)),
                eigendecomposition=eigendecomp_viz,
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
        encoder_state, allo, metrics = update_encoder(
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
            learned_features = encoder.apply(encoder_state.params['encoder'], state_coords)[0]

            # Compute cosine similarities with ground truth
            cosine_sims = compute_cosine_similarities(learned_features, gt_eigenvectors)

            # Store metrics
            elapsed_time = time.time() - start_time
            steps_completed = gradient_step - start_step
            metrics_dict = {
                "gradient_step": gradient_step,
                "allo": float(allo.item()),
                "sps": int(steps_completed / max(elapsed_time, 1e-6)),  # Steps since start/resume
            }
            for k, v in metrics.items():
                metrics_dict[k] = float(v.item())

            # Add cosine similarities to metrics
            metrics_dict.update(cosine_sims)

            metrics_history.append(metrics_dict)

            if gradient_step % (args.log_freq * 10) == 0:
                print(f"Step {gradient_step}: loss={allo.item():.4f}, "
                      f"total_error={metrics['total_error'].item():.4f}, "
                      f"cosine_sim_avg={cosine_sims['cosine_sim_avg']:.4f}")
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
            learned_features = encoder.apply(encoder_state.params['encoder'], state_coords)[0]
            np.save(results_dir / "latest_learned_eigenvectors.npy", np.array(learned_features))

            # Optionally create plots during training (slower)
            if args.plot_during_training:
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
                    save_path=str(plots_dir / "learned_eigenvectors_latest.png"),
                    shared_colorbar=True
                )
                plt.close()

                # Plot learning curves
                plot_learning_curves(metrics_history, str(plots_dir / "learning_curves.png"))

                # Plot dual variable evolution vs ground truth eigenvalues
                plot_dual_variable_evolution(
                    metrics_history,
                    gt_eigenvalues,
                    args.gamma,
                    str(plots_dir / "dual_variable_evolution.png"),
                    num_eigenvectors=args.num_eigenvectors,
                    ground_truth_eigenvalues_simple=gt_eigenvalues_simple if 'gt_eigenvalues_simple' in locals() else None
                )

                # Plot cosine similarity evolution
                plot_cosine_similarity_evolution(
                    metrics_history,
                    str(plots_dir / "cosine_similarity_evolution.png"),
                    num_eigenvectors=args.num_eigenvectors
                )
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
                'gt_eigenvalues': np.array(gt_eigenvalues),
                'gt_eigenvectors': np.array(gt_eigenvectors),
            }, f)
        print(f"Model saved to {save_path}")

    # Save final learned eigenvectors
    final_learned_features = encoder.apply(encoder_state.params['encoder'], state_coords)[0]
    np.save(results_dir / "final_learned_eigenvectors.npy", np.array(final_learned_features))

    # Optionally create final comparison plot
    if args.plot_during_training:
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

    print(f"\nData exported. Use generate_plots.py to create visualizations.")

    print(f"\nAll results saved to: {results_dir}")

    return encoder_state, results_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    learn_eigenvectors(args)
