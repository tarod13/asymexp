"""
Train a model to learn complex eigenvectors of non-symmetric Laplacians.

This script uses (x,y) coordinates as inputs to the network and learns
the left and right eigenvectors (with real and imaginary components) of
non-symmetric Laplacians through an augmented Lagrangian optimization approach.
The network outputs 4 sets of features: left_real, left_imag, right_real, right_imag.

The loss function consists of:
1. Graph drawing loss: E[(ψ_real(s)(φ_real(s) - φ_real(s')) + ψ_img(s)(φ_img(s) - φ_img(s')))²]
2. Biorthogonality constraints: φ^T ψ = I (for both real and imaginary parts)
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
    num_features: int
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

        # Output layer - 4 × num_features for complex left and right eigenvectors
        all_features = nn.Dense(4 * self.num_features)(x)

        # Split into 4 parts: left_real, left_imag, right_real, right_imag
        left_real = all_features[:, :self.num_features]
        left_imag = all_features[:, self.num_features:2*self.num_features]
        right_real = all_features[:, 2*self.num_features:3*self.num_features]
        right_imag = all_features[:, 3*self.num_features:]

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


def compute_inverse_weighted_laplacian(
    transition_matrix: jnp.ndarray,
    sampling_distribution: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """
    Compute the inverse-weighted Laplacian L = I - (1-γ)(SR_γ + D^{-1}SR_γ^TD)/2.

    This version accounts for the cancellation of D on the left and introduction of D^{-1} on the transpose.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        sampling_distribution: Shape [num_states, num_states], diagonal matrix D with sampling probabilities
        gamma: Discount factor used in successor representation

    Returns:
        Inverse-weighted Laplacian of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    # Compute successor representation SR_γ = (I - γP)^(-1)
    sr_matrix = compute_successor_representation(transition_matrix, gamma)

    # Compute D^{-1} (inverse of diagonal matrix)
    D_inv = jnp.linalg.inv(sampling_distribution)

    # Compute SR_γ and D^{-1}SR_γ^TD
    Dinv_SRT_D = D_inv @ sr_matrix.T @ sampling_distribution

    # Symmetrize: (SR_γ + D^{-1}SR_γ^TD)/2
    inverse_weighted_sr_symmetrized = (sr_matrix + Dinv_SRT_D) / 2.0

    # Compute Laplacian: L = I - (1-γ)(SR_γ + D^{-1}SR_γ^TD)/2
    laplacian = identity - (1 - gamma) * inverse_weighted_sr_symmetrized

    return laplacian


def compute_nonsymmetric_laplacian(
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

    # Sort by ascending eigenvalue real part
    sorted_indices = jnp.argsort(jnp.real(eigenvalues))
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
    max_episode_length: int = 2

    # Data collection
    num_envs: int = 500000
    num_steps: int = 2

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
    delta: float = 0.1  # Eigenvalue shift parameter: L = (1+δ)I - M (improves numerical stability)

    # Episodic replay buffer
    max_time_offset: int | None = None  # Maximum time offset for sampling (None = episode length)

    # Augmented Lagrangian parameters
    duals_initial_val: float = 0.0
    barrier_initial_val: float = 5.0
    max_barrier_coefs: float = 5.0
    step_size_duals: float = 0.0
    step_size_duals_I: float = 0.0
    integral_decay: float = 0.99
    init_dual_diag: bool = False
    
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


def plot_learning_curves(metrics_history: Dict, save_path: str):
    """Plot and save comprehensive learning curves with all diagnostics."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Training Metrics (Comprehensive Diagnostics)', fontsize=16)

    # Extract data
    steps = [m['gradient_step'] for m in metrics_history]

    # Row 1: Main losses
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

    # Plot 3: Gradient norm
    axes[0, 2].plot(steps, [m['grad_norm'] for m in metrics_history])
    axes[0, 2].set_xlabel('Gradient Step')
    axes[0, 2].set_ylabel('Gradient Norm')
    axes[0, 2].set_title('Gradient Norm')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Dual losses and errors
    # Plot 4: Dual loss
    axes[1, 0].plot(steps, [m['dual_loss'] for m in metrics_history], label='Positive')
    axes[1, 0].plot(steps, [m['dual_loss_neg'] for m in metrics_history], label='Negative')
    axes[1, 0].set_xlabel('Gradient Step')
    axes[1, 0].set_ylabel('Dual Loss')
    axes[1, 0].set_title('Dual Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Barrier loss
    axes[1, 1].plot(steps, [m['barrier_loss'] for m in metrics_history], label='Positive')
    axes[1, 1].plot(steps, [m['barrier_loss_neg'] for m in metrics_history], label='Negative')
    axes[1, 1].set_xlabel('Gradient Step')
    axes[1, 1].set_ylabel('Barrier Loss')
    axes[1, 1].set_title('Barrier Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Total errors
    if 'total_error' in metrics_history[0]:
        axes[1, 2].plot(steps, [m['total_error'] for m in metrics_history], label='Total')
    if 'total_norm_error' in metrics_history[0]:
        axes[1, 2].plot(steps, [m['total_norm_error'] for m in metrics_history], label='Norm')
    if 'total_two_component_error' in metrics_history[0]:
        axes[1, 2].plot(steps, [m['total_two_component_error'] for m in metrics_history], label='First 2')
    axes[1, 2].set_xlabel('Gradient Step')
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].set_title('Orthogonality Errors')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Diagonal errors for first few eigenvectors
    # Plot 7: Left real diagonal errors
    axes[2, 0].set_title('Left Real Diagonal Errors')
    for i in range(min(5, 11)):
        key = f'error_left_real_{i}'
        if key in metrics_history[0]:
            axes[2, 0].plot(steps, [m[key] for m in metrics_history], label=f'ev{i}', alpha=0.7)
    axes[2, 0].set_xlabel('Gradient Step')
    axes[2, 0].set_ylabel('Error')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Left imag diagonal errors
    axes[2, 1].set_title('Left Imag Diagonal Errors')
    for i in range(min(5, 11)):
        key = f'error_left_imag_{i}'
        if key in metrics_history[0]:
            axes[2, 1].plot(steps, [m[key] for m in metrics_history], label=f'ev{i}', alpha=0.7)
    axes[2, 1].set_xlabel('Gradient Step')
    axes[2, 1].set_ylabel('Error')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 9: Right real diagonal errors
    axes[2, 2].set_title('Right Real Diagonal Errors')
    for i in range(min(5, 11)):
        key = f'error_right_real_{i}'
        if key in metrics_history[0]:
            axes[2, 2].plot(steps, [m[key] for m in metrics_history], label=f'ev{i}', alpha=0.7)
    axes[2, 2].set_xlabel('Gradient Step')
    axes[2, 2].set_ylabel('Error')
    axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(True, alpha=0.3)

    # Row 4: Barrier coefficients and distance metrics
    # Plot 10: Barrier coefficients
    axes[3, 0].plot(steps, [m['barrier_coef_left_real'] for m in metrics_history], label='Left Real')
    axes[3, 0].plot(steps, [m['barrier_coef_left_imag'] for m in metrics_history], label='Left Imag')
    axes[3, 0].plot(steps, [m['barrier_coef_right_real'] for m in metrics_history], label='Right Real')
    axes[3, 0].plot(steps, [m['barrier_coef_right_imag'] for m in metrics_history], label='Right Imag')
    axes[3, 0].set_xlabel('Gradient Step')
    axes[3, 0].set_ylabel('Coefficient')
    axes[3, 0].set_title('Barrier Coefficients')
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].grid(True, alpha=0.3)

    # Plot 11: Distance to constraint manifold
    if 'distance_to_constraint_manifold' in metrics_history[0]:
        axes[3, 1].plot(steps, [m['distance_to_constraint_manifold'] for m in metrics_history])
        axes[3, 1].set_xlabel('Gradient Step')
        axes[3, 1].set_ylabel('Distance')
        axes[3, 1].set_title('Distance to Constraint Manifold')
        axes[3, 1].grid(True, alpha=0.3)
    else:
        axes[3, 1].text(0.5, 0.5, 'Metric not available', ha='center', va='center', transform=axes[3, 1].transAxes)
        axes[3, 1].set_title('Distance to Constraint Manifold')

    # Plot 12: Distance to origin
    if 'distance_to_origin' in metrics_history[0]:
        axes[3, 2].plot(steps, [m['distance_to_origin'] for m in metrics_history])
        axes[3, 2].set_xlabel('Gradient Step')
        axes[3, 2].set_ylabel('Distance')
        axes[3, 2].set_title('Distance to Origin')
        axes[3, 2].grid(True, alpha=0.3)
    else:
        axes[3, 2].text(0.5, 0.5, 'Metric not available', ha='center', va='center', transform=axes[3, 2].transAxes)
        axes[3, 2].set_title('Distance to Origin')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive learning curves saved to {save_path}")


def plot_dual_variable_evolution(metrics_history, ground_truth_eigenvalues, gamma, save_path,
                                  num_eigenvectors=11, ground_truth_eigenvalues_simple=None,
                                  ground_truth_eigenvalues_weighted=None):
    """
    Plot the evolution of dual variables (Laplacian eigenvalue estimates) vs ground truth eigenvalues.

    The duals are eigenvalues of the non-symmetric Laplacian L = I - (1-γ)P·SR_γ,
    where P is the transition matrix, SR_γ is the successor representation, and γ is the discount factor.

    The eigenvalue estimates are computed as -0.5 * (dual_left + dual_right) where dual_left and
    dual_right are the left and right biorthogonality constraint multipliers.

    Args:
        metrics_history: List of metric dictionaries
        ground_truth_eigenvalues: Array of ground truth eigenvalues of the non-symmetric Laplacian
        gamma: Discount factor used in the successor representation
        save_path: Path to save the plot
        num_eigenvectors: Number of eigenvectors to plot
        ground_truth_eigenvalues_simple: Optional array for comparison (unused)
        ground_truth_eigenvalues_weighted: Optional array for comparison (unused)
    """
    steps = [m['gradient_step'] for m in metrics_history]
    num_plot = min(num_eigenvectors, len(ground_truth_eigenvalues))

    # Create figure with three rows: real/imag components, magnitude, and errors
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    colors = plt.cm.tab10(np.linspace(0, 1, num_plot))

    # Plot 1: Real and imaginary components of dual variables
    ax1 = axes[0]

    for i in range(num_plot):
        dual_real_key = f'dual_real_{i}'
        dual_imag_key = f'dual_imag_{i}'

        if dual_real_key in metrics_history[0] and dual_imag_key in metrics_history[0]:
            # Get dual values (these are eigenvalues of the Laplacian)
            # Already scaled as -0.5 * (dual_left + dual_right) in aux
            dual_values_real = np.array([m[dual_real_key] for m in metrics_history])
            dual_values_imag = np.array([m[dual_imag_key] for m in metrics_history])

            # Plot real and imaginary components
            ax1.plot(steps, dual_values_real, label=f'λ_{i} (real)',
                    color=colors[i], linewidth=1.5, linestyle='-')
            ax1.plot(steps, dual_values_imag, label=f'λ_{i} (imag)',
                    color=colors[i], linewidth=1.5, linestyle='--', alpha=0.7)

            # Plot ground truth (real and imaginary)
            gt_value_real = float(ground_truth_eigenvalues[i].real)
            gt_value_imag = float(ground_truth_eigenvalues[i].imag)
            ax1.axhline(y=gt_value_real, color=colors[i], linestyle='-', alpha=0.2, linewidth=2)
            ax1.axhline(y=gt_value_imag, color=colors[i], linestyle='--', alpha=0.2, linewidth=2)

    ax1.set_xlabel('Gradient Step', fontsize=12)
    ax1.set_ylabel('Dual Variable Value', fontsize=12)
    ax1.set_title('Dual Variables: Real (solid) and Imaginary (dashed) Components', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Magnitude of complex dual variables
    ax2 = axes[1]

    for i in range(num_plot):
        dual_real_key = f'dual_real_{i}'
        dual_imag_key = f'dual_imag_{i}'

        if dual_real_key in metrics_history[0] and dual_imag_key in metrics_history[0]:
            # Get dual values (already scaled as -0.5 * (dual_left + dual_right))
            dual_values_real = np.array([m[dual_real_key] for m in metrics_history])
            dual_values_imag = np.array([m[dual_imag_key] for m in metrics_history])

            # Calculate magnitude
            dual_magnitude = np.sqrt(dual_values_real**2 + dual_values_imag**2)

            ax2.plot(steps, dual_magnitude, label=f'|λ_{i}|', color=colors[i], linewidth=1.5)

            # Plot ground truth magnitude
            gt_magnitude = np.abs(ground_truth_eigenvalues[i])
            ax2.axhline(y=gt_magnitude, color=colors[i], linestyle='-', alpha=0.3, linewidth=2.5)

    ax2.set_xlabel('Gradient Step', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Magnitude of Complex Dual Variables', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Complex magnitude errors
    ax3 = axes[2]

    for i in range(num_plot):
        dual_real_key = f'dual_real_{i}'
        dual_imag_key = f'dual_imag_{i}'

        if dual_real_key in metrics_history[0] and dual_imag_key in metrics_history[0]:
            # Get dual values
            dual_values_real = np.array([m[dual_real_key] for m in metrics_history])
            dual_values_imag = np.array([m[dual_imag_key] for m in metrics_history])

            # Apply 0.5 factor due to sampling scheme
            dual_values_real_scaled = 0.5 * dual_values_real
            dual_values_imag_scaled = 0.5 * dual_values_imag

            # Calculate complex error magnitude: |dual - ground_truth|
            gt_value_real = float(ground_truth_eigenvalues[i].real)
            gt_value_imag = float(ground_truth_eigenvalues[i].imag)

            error_real = dual_values_real_scaled - gt_value_real
            error_imag = dual_values_imag_scaled - gt_value_imag
            error_magnitude = np.sqrt(error_real**2 + error_imag**2)

            ax3.plot(steps, error_magnitude, label=f'|error λ_{i}|',
                    color=colors[i], linewidth=1.5)

    ax3.set_xlabel('Gradient Step', fontsize=12)
    ax3.set_ylabel('Error Magnitude', fontsize=12)
    ax3.set_title('Complex Error Magnitude in Complex Plane', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

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
        Result: |Re(cos(θ))|

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
        abs_cos_real = jnp.abs(cos_real)
        similarities[f'{prefix}cosine_sim_{i}'] = float(abs_cos_real)
        similarities[f'{prefix}cosine_sim_imag_{i}'] = float(cos_imag)  # Also save imaginary for inspection
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
        learned_left_real: Learned left eigenvector real parts [num_states, num_eigenvectors]
        learned_left_imag: Learned left eigenvector imaginary parts [num_states, num_eigenvectors]
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
    # Normalize learned eigenvectors
    learned_normalized = normalize_eigenvectors_for_comparison(
        left_real=learned_left_real,
        left_imag=learned_left_imag,
        right_real=learned_right_real,
        right_imag=learned_right_imag,
        sampling_probs=sampling_probs
    )

    # Normalize ground truth eigenvectors
    gt_normalized = normalize_eigenvectors_for_comparison(
        left_real=gt_left_real,
        left_imag=gt_left_imag,
        right_real=gt_right_real,
        right_imag=gt_right_imag,
        sampling_probs=sampling_probs
    )

    # Compute cosine similarities using the normalized eigenvectors
    left_sims = compute_complex_cosine_similarities(
        learned_normalized['left_real'], learned_normalized['left_imag'],
        gt_normalized['left_real'], gt_normalized['left_imag'],
        prefix="left_"
    )

    right_sims = compute_complex_cosine_similarities(
        learned_normalized['right_real'], learned_normalized['right_imag'],
        gt_normalized['right_real'], gt_normalized['right_imag'],
        prefix="right_"
    )

    # Combine results
    result = {}
    result.update(left_sims)
    result.update(right_sims)

    return result


def plot_cosine_similarity_evolution(metrics_history: Dict, save_path: str, num_eigenvectors: int = None):
    """
    Plot the evolution of cosine similarities between learned and ground truth eigenvectors.

    For complex eigenvectors, plots both left and right eigenvector similarities.

    Args:
        metrics_history: List of metric dictionaries containing cosine similarity values
        save_path: Path to save the plot
        num_eigenvectors: Number of eigenvectors to plot (None = plot all available)
    """
    steps = [m['gradient_step'] for m in metrics_history]

    # Check for complex eigenvector metrics (left and right)
    has_left = 'left_cosine_sim_avg' in metrics_history[0]
    has_right = 'right_cosine_sim_avg' in metrics_history[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot left and right eigenvector cosine similarities
    if has_left:
        left_values = [m['left_cosine_sim_avg'] for m in metrics_history]
        ax.plot(steps, left_values, label='Left Eigenvectors (φ)',
                color='blue', linewidth=2.5, linestyle='-', marker='o', markersize=3, markevery=max(1, len(steps)//20))

    if has_right:
        right_values = [m['right_cosine_sim_avg'] for m in metrics_history]
        ax.plot(steps, right_values, label='Right Eigenvectors (ψ)',
                color='red', linewidth=2.5, linestyle='-', marker='s', markersize=3, markevery=max(1, len(steps)//20))

    # Plot average of both if both exist
    if has_left and has_right:
        avg_values = [(left_values[i] + right_values[i]) / 2 for i in range(len(steps))]
        ax.plot(steps, avg_values, label='Average (Left + Right)',
                color='green', linewidth=2.0, linestyle='--', alpha=0.7)

    ax.set_xlabel('Gradient Step', fontsize=12)
    ax.set_ylabel('|Re(Complex Cosine Similarity)|', fontsize=12)
    ax.set_title('Evolution of Complex Cosine Similarity\n(Absolute Value of Real Part of <conj(u),v> / (||u|| ||v||))', fontsize=14)

    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])  # Absolute value ranges from 0 to 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cosine similarity evolution plot saved to {save_path}")


def plot_all_duals_evolution(metrics_history: Dict, save_path: str, num_eigenvectors: int = 6):
    """
    Plot comprehensive dual variable evolution showing left, right, and combined duals.

    Args:
        metrics_history: List of metric dictionaries
        save_path: Path to save the plot
        num_eigenvectors: Number of eigenvectors to plot
    """
    steps = [m['gradient_step'] for m in metrics_history]

    # Create a large figure with subplots for each eigenvector
    fig, axes = plt.subplots(num_eigenvectors, 2, figsize=(16, 4*num_eigenvectors))
    fig.suptitle('Dual Variables Evolution (All Components)', fontsize=16)

    if num_eigenvectors == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_eigenvectors):
        # Left column: Real parts
        ax_real = axes[i, 0]

        # Plot left, right, and combined real duals
        if f'dual_left_real_{i}' in metrics_history[0]:
            dual_left_real = [m[f'dual_left_real_{i}'] for m in metrics_history]
            ax_real.plot(steps, dual_left_real, label='Left', alpha=0.7, linewidth=1.5)

        if f'dual_right_real_{i}' in metrics_history[0]:
            dual_right_real = [m[f'dual_right_real_{i}'] for m in metrics_history]
            ax_real.plot(steps, dual_right_real, label='Right', alpha=0.7, linewidth=1.5)

        if f'dual_real_{i}' in metrics_history[0]:
            dual_combined_real = [m[f'dual_real_{i}'] for m in metrics_history]
            ax_real.plot(steps, dual_combined_real, label='Combined (Eigenvalue)',
                        linewidth=2.5, linestyle='--', color='black')

        ax_real.set_xlabel('Gradient Step')
        ax_real.set_ylabel('Dual Value (Real)')
        ax_real.set_title(f'Eigenvector {i} - Real Part Duals')
        ax_real.legend()
        ax_real.grid(True, alpha=0.3)

        # Right column: Imaginary parts
        ax_imag = axes[i, 1]

        if f'dual_left_imag_{i}' in metrics_history[0]:
            dual_left_imag = [m[f'dual_left_imag_{i}'] for m in metrics_history]
            ax_imag.plot(steps, dual_left_imag, label='Left', alpha=0.7, linewidth=1.5)

        if f'dual_right_imag_{i}' in metrics_history[0]:
            dual_right_imag = [m[f'dual_right_imag_{i}'] for m in metrics_history]
            ax_imag.plot(steps, dual_right_imag, label='Right', alpha=0.7, linewidth=1.5)

        if f'dual_imag_{i}' in metrics_history[0]:
            dual_combined_imag = [m[f'dual_imag_{i}'] for m in metrics_history]
            ax_imag.plot(steps, dual_combined_imag, label='Combined (Eigenvalue)',
                        linewidth=2.5, linestyle='--', color='black')

        ax_imag.set_xlabel('Gradient Step')
        ax_imag.set_ylabel('Dual Value (Imag)')
        ax_imag.set_title(f'Eigenvector {i} - Imaginary Part Duals')
        ax_imag.legend()
        ax_imag.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"All duals evolution plot saved to {save_path}")


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

    # Compute non-symmetric Laplacian for complex eigenvectors
    print(f"\nComputing non-symmetric Laplacian with gamma={args.gamma}, delta={args.delta}...")
    print(f"  L = (1+δ)I - (1-γ)P·SR_γ (matches geometric sampling with k >= 1)")

    laplacian_matrix = compute_nonsymmetric_laplacian(transition_matrix, args.gamma, delta=args.delta)
    eigendecomp = compute_eigendecomposition(
        laplacian_matrix,
        k=args.num_eigenvectors,
        ascending=True  # For Laplacians, we want smallest eigenvalues
    )

    print(f"\nSmallest {min(5, args.num_eigenvectors)} eigenvalues (complex):")
    print("  Eigenvalue (real + imag)")
    for i in range(min(5, args.num_eigenvectors)):
        ev_real = eigendecomp['eigenvalues_real'][i]
        ev_imag = eigendecomp['eigenvalues_imag'][i]
        print(f"  λ_{i}: {ev_real:.6f} + {ev_imag:.6f}i")

    # Print ranges of eigenvector values to debug visualization issues
    print(f"\nEigenvector value ranges (first {min(5, args.num_eigenvectors)} eigenvectors):")
    for i in range(min(5, args.num_eigenvectors)):
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
        laplacian_matrix, eigendecomp, state_coords, canonical_states, sampling_probs, door_config, data_env, replay_buffer, transition_matrix = collect_data_and_compute_eigenvectors(env, args)

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
        'duals_left_real': False,
        'duals_left_imag': False,
        'barrier_coefs_left_real': False,
        'barrier_coefs_left_imag': False,
        'error_integral_left_real': False,
        'error_integral_left_imag': False,
        'duals_right_real': False,
        'duals_right_imag': False,
        'barrier_coefs_right_real': False,
        'barrier_coefs_right_imag': False,
        'error_integral_right_real': False,
        'error_integral_right_imag': False,
    }
    other_mask = {
        'encoder': False,
        'duals_left_real': True,
        'duals_left_imag': True,
        'barrier_coefs_left_real': True,
        'barrier_coefs_left_imag': True,
        'error_integral_left_real': False,
        'error_integral_left_imag': False,
        'duals_right_real': True,
        'duals_right_imag': True,
        'barrier_coefs_right_real': True,
        'barrier_coefs_right_imag': True,
        'error_integral_right_real': False,
        'error_integral_right_imag': False,
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
            initial_dual_mask = jnp.ones((args.num_eigenvectors, args.num_eigenvectors))

        # Create dummy input for initialization
        dummy_input = state_coords[:1]  # (1, 2)

        initial_params = {
            'encoder': encoder.init(encoder_key, dummy_input),
            'duals_left_real': jnp.tril(args.duals_initial_val * initial_dual_mask),
            'duals_left_imag': jnp.tril(args.duals_initial_val * initial_dual_mask),
            'duals_right_real': jnp.tril(args.duals_initial_val * initial_dual_mask),
            'duals_right_imag': jnp.tril(args.duals_initial_val * initial_dual_mask),
            'barrier_coefs_left_real': jnp.tril(args.barrier_initial_val * jnp.ones((1, 1))),
            'barrier_coefs_left_imag': jnp.tril(args.barrier_initial_val * jnp.ones((1, 1))),
            'barrier_coefs_right_real': jnp.tril(args.barrier_initial_val * jnp.ones((1, 1))),
            'barrier_coefs_right_imag': jnp.tril(args.barrier_initial_val * jnp.ones((1, 1))),
            'error_integral_left_real': jnp.zeros((args.num_eigenvectors, args.num_eigenvectors)),
            'error_integral_left_imag': jnp.zeros((args.num_eigenvectors, args.num_eigenvectors)),
            'error_integral_right_real': jnp.zeros((args.num_eigenvectors, args.num_eigenvectors)),
            'error_integral_right_imag': jnp.zeros((args.num_eigenvectors, args.num_eigenvectors)),
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

            # Extract left and right eigenvectors (real and imaginary parts)
            # features_1 contains: left_real, left_imag, right_real, right_imag
            psi_real = features_1['left_real']  # Left eigenvectors, real part
            psi_imag = features_1['left_imag']  # Left eigenvectors, imaginary part
            phi_real = features_1['right_real']  # Right eigenvectors, real part
            phi_imag = features_1['right_imag']  # Right eigenvectors, imaginary part

            psi_real_2 = features_2['left_real']
            psi_imag_2 = features_2['left_imag']
            phi_real_2 = features_2['right_real']
            phi_imag_2 = features_2['right_imag']

            next_phi_real = next_features['right_real']
            next_phi_imag = next_features['right_imag']

            # Get sizes
            d = args.num_eigenvectors
            n = phi_real.shape[0]

            # Get duals
            dual_variables_left_real = params['duals_left_real']
            dual_variables_left_imag = params['duals_left_imag']
            dual_variables_right_real = params['duals_right_real']
            dual_variables_right_imag = params['duals_right_imag']
            barrier_coefficients_left_real = params['barrier_coefs_left_real']
            barrier_coefficients_left_imag = params['barrier_coefs_left_imag']
            barrier_coefficients_right_real = params['barrier_coefs_right_real']
            barrier_coefficients_right_imag = params['barrier_coefs_right_imag']
            diagonal_duals_left_real = jnp.diag(dual_variables_left_real)
            diagonal_duals_left_imag = jnp.diag(dual_variables_left_imag)
            diagonal_duals_right_real = jnp.diag(dual_variables_right_real)
            diagonal_duals_right_imag = jnp.diag(dual_variables_right_imag)
            # Eigenvalue sum is the sum of left and right duals (averaged)
            eigenvalue_sum_real = -0.5 * (diagonal_duals_left_real + diagonal_duals_right_real).sum()
            eigenvalue_sum_imag = -0.5 * (diagonal_duals_left_imag + diagonal_duals_right_imag).sum()

            # Compute biorthogonality inner products: ψ^T φ
            # Real part: Re(ψ^T φ) = ψ_real^T φ_real - ψ_imag^T φ_imag
            inner_product_left_real_1 = (jnp.einsum('ij,ik->jk', psi_real, jax.lax.stop_gradient(phi_real)) -
                                   jnp.einsum('ij,ik->jk', psi_imag, jax.lax.stop_gradient(phi_imag))) / n
            inner_product_left_real_2 = (jnp.einsum('ij,ik->jk', psi_real_2, jax.lax.stop_gradient(phi_real_2)) -
                                   jnp.einsum('ij,ik->jk', psi_imag_2, jax.lax.stop_gradient(phi_imag_2))) / n
            # Imaginary part: Im(ψ^T φ) = ψ_real^T φ_imag + ψ_imag^T φ_real
            inner_product_left_imag_1 = (jnp.einsum('ij,ik->jk', psi_real, jax.lax.stop_gradient(phi_imag)) +
                                   jnp.einsum('ij,ik->jk', psi_imag, jax.lax.stop_gradient(phi_real))) / n
            inner_product_left_imag_2 = (jnp.einsum('ij,ik->jk', psi_real_2, jax.lax.stop_gradient(phi_imag_2)) +
                                   jnp.einsum('ij,ik->jk', psi_imag_2, jax.lax.stop_gradient(phi_real_2))) / n
            
            # Same for right eigenvectors
            inner_product_right_real_1 = (jnp.einsum('ij,ik->jk', phi_real, jax.lax.stop_gradient(psi_real)) -
                                    jnp.einsum('ij,ik->jk', phi_imag, jax.lax.stop_gradient(psi_imag))) / n
            inner_product_right_real_2 = (jnp.einsum('ij,ik->jk', phi_real_2, jax.lax.stop_gradient(psi_real_2)) -
                                    jnp.einsum('ij,ik->jk', phi_imag_2, jax.lax.stop_gradient(psi_imag_2))) / n
            inner_product_right_imag_1 = (jnp.einsum('ij,ik->jk', phi_real, jax.lax.stop_gradient(psi_imag)) +
                                    jnp.einsum('ij,ik->jk', phi_imag, jax.lax.stop_gradient(psi_real))) / n
            inner_product_right_imag_2 = (jnp.einsum('ij,ik->jk', phi_real_2, jax.lax.stop_gradient(psi_imag_2)) +
                                    jnp.einsum('ij,ik->jk', phi_imag_2, jax.lax.stop_gradient(psi_real_2))) / n

            # Biorthogonality error matrices
            # Real part should equal identity
            error_matrix_left_real_1 = jnp.tril(inner_product_left_real_1 - jnp.eye(d))
            error_matrix_left_real_2 = jnp.tril(inner_product_left_real_2 - jnp.eye(d))
            error_matrix_right_real_1 = jnp.tril(inner_product_right_real_1 - jnp.eye(d))
            error_matrix_right_real_2 = jnp.tril(inner_product_right_real_2 - jnp.eye(d))

            # Imaginary part should equal zero
            error_matrix_left_imag_1 = jnp.tril(inner_product_left_imag_1)
            error_matrix_left_imag_2 = jnp.tril(inner_product_left_imag_2)
            error_matrix_right_imag_1 = jnp.tril(inner_product_right_imag_1)
            error_matrix_right_imag_2 = jnp.tril(inner_product_right_imag_2)

            # Compute dual loss (for real part constraint)
            dual_loss_pos_real = (
                jax.lax.stop_gradient(dual_variables_left_real)
                * error_matrix_left_real_1
                + jax.lax.stop_gradient(dual_variables_right_real)
                * error_matrix_right_real_1
            ).sum()

            error_integral_left_real = params['error_integral_left_real']
            error_integral_right_real = params['error_integral_right_real']

            dual_loss_P_left_real = jax.lax.stop_gradient(args.step_size_duals * error_matrix_left_real_1)
            dual_loss_I_left_real = args.step_size_duals_I * jax.lax.stop_gradient(error_integral_left_real)
            dual_loss_neg_left_real = -(dual_variables_left_real * (dual_loss_P_left_real + dual_loss_I_left_real)).sum()

            dual_loss_P_right_real = jax.lax.stop_gradient(args.step_size_duals * error_matrix_right_real_1)
            dual_loss_I_right_real = args.step_size_duals_I * jax.lax.stop_gradient(error_integral_right_real)
            dual_loss_neg_right_real = -(dual_variables_right_real * (dual_loss_P_right_real + dual_loss_I_right_real)).sum()

            # Compute dual loss (for imaginary part constraint)
            dual_loss_pos_imag = (
                jax.lax.stop_gradient(dual_variables_left_imag)
                * error_matrix_left_imag_1
                + jax.lax.stop_gradient(dual_variables_right_imag)
                * error_matrix_right_imag_1
            ).sum()

            error_integral_left_imag = params['error_integral_left_imag']
            error_integral_right_imag = params['error_integral_right_imag']

            dual_loss_P_left_imag = jax.lax.stop_gradient(args.step_size_duals * error_matrix_left_imag_1)
            dual_loss_I_left_imag = args.step_size_duals_I * jax.lax.stop_gradient(error_integral_left_imag)
            dual_loss_neg_left_imag = -(dual_variables_left_imag * (dual_loss_P_left_imag + dual_loss_I_left_imag)).sum()

            dual_loss_P_right_imag = jax.lax.stop_gradient(args.step_size_duals * error_matrix_right_imag_1)
            dual_loss_I_right_imag = args.step_size_duals_I * jax.lax.stop_gradient(error_integral_right_imag)
            dual_loss_neg_right_imag = -(dual_variables_right_imag * (dual_loss_P_right_imag + dual_loss_I_right_imag)).sum()

            dual_loss_pos = dual_loss_pos_real + dual_loss_pos_imag
            dual_loss_neg_left = dual_loss_neg_left_real + dual_loss_neg_left_imag
            dual_loss_neg_right = dual_loss_neg_right_real + dual_loss_neg_right_imag
            dual_loss_neg = dual_loss_neg_left + dual_loss_neg_right

            # Compute barrier loss (for real part constraint)
            quadratic_error_matrix_left_real = 2 * error_matrix_left_real_1 * jax.lax.stop_gradient(error_matrix_left_real_2)
            quadratic_error_left_real = quadratic_error_matrix_left_real.sum()
            barrier_loss_pos_left_real = jax.lax.stop_gradient(barrier_coefficients_left_real[0, 0]) * quadratic_error_left_real
            barrier_loss_neg_left_real = -barrier_coefficients_left_real[0, 0] * jax.lax.stop_gradient(jnp.absolute(quadratic_error_left_real))

            quadratic_error_matrix_right_real = 2 * error_matrix_right_real_1 * jax.lax.stop_gradient(error_matrix_right_real_2)
            quadratic_error_right_real = quadratic_error_matrix_right_real.sum()
            barrier_loss_pos_right_real = jax.lax.stop_gradient(barrier_coefficients_right_real[0, 0]) * quadratic_error_right_real
            barrier_loss_neg_right_real = -barrier_coefficients_right_real[0, 0] * jax.lax.stop_gradient(jnp.absolute(quadratic_error_right_real))

            # Compute barrier loss (for imaginary part constraint)
            quadratic_error_matrix_left_imag = 2 * error_matrix_left_imag_1 * jax.lax.stop_gradient(error_matrix_left_imag_2)
            quadratic_error_left_imag = quadratic_error_matrix_left_imag.sum()
            barrier_loss_pos_left_imag = jax.lax.stop_gradient(barrier_coefficients_left_imag[0, 0]) * quadratic_error_left_imag
            barrier_loss_neg_left_imag = -barrier_coefficients_left_imag[0, 0] * jax.lax.stop_gradient(jnp.absolute(quadratic_error_left_imag))

            quadratic_error_matrix_right_imag = 2 * error_matrix_right_imag_1 * jax.lax.stop_gradient(error_matrix_right_imag_2)
            quadratic_error_right_imag = quadratic_error_matrix_right_imag.sum()
            barrier_loss_pos_right_imag = jax.lax.stop_gradient(barrier_coefficients_right_imag[0, 0]) * quadratic_error_right_imag
            barrier_loss_neg_right_imag = -barrier_coefficients_right_imag[0, 0] * jax.lax.stop_gradient(jnp.absolute(quadratic_error_right_imag))

            barrier_loss_pos_left = barrier_loss_pos_left_real + barrier_loss_pos_left_imag
            barrier_loss_neg_left = barrier_loss_neg_left_real + barrier_loss_neg_left_imag

            barrier_loss_pos_right = barrier_loss_pos_right_real + barrier_loss_pos_right_imag
            barrier_loss_neg_right = barrier_loss_neg_right_real + barrier_loss_neg_right_imag

            barrier_loss_pos = barrier_loss_pos_left + barrier_loss_pos_right
            barrier_loss_neg = barrier_loss_neg_left + barrier_loss_neg_right

            # Compute graph drawing loss for complex eigenvectors
            # E[(ψ_real(s)*((1+δ)φ_real(s) - φ_real(s')) - ψ_imag(s)*((1+δ)φ_imag(s) - φ_imag(s')))^2]
            # The δ parameter shifts eigenvalues: L = (1+δ)I - M
            graph_products_real = (psi_real * ((1+args.delta)*phi_real - next_phi_real)).mean(0, keepdims=True)
            graph_products_imag = (psi_imag * ((1+args.delta)*phi_imag - next_phi_imag)).mean(0, keepdims=True)
            graph_loss = ((graph_products_real - graph_products_imag)**2).sum()

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
                'barrier_loss_neg': barrier_loss_neg,
                'approx_eigenvalue_sum_real': eigenvalue_sum_real,
                'approx_eigenvalue_sum_imag': eigenvalue_sum_imag,
                'barrier_coef_left_real': barrier_coefficients_left_real[0, 0],
                'barrier_coef_left_imag': barrier_coefficients_left_imag[0, 0],
                'barrier_coef_right_real': barrier_coefficients_right_real[0, 0],
                'barrier_coef_right_imag': barrier_coefficients_right_imag[0, 0],
            }

            # Add dual variables and errors to aux
            for i in range(min(11, args.num_eigenvectors)):
                aux[f'dual_left_real_{i}'] = dual_variables_left_real[i, i]
                aux[f'dual_left_imag_{i}'] = dual_variables_left_imag[i, i]
                aux[f'dual_right_real_{i}'] = dual_variables_right_real[i, i]
                aux[f'dual_right_imag_{i}'] = dual_variables_right_imag[i, i]
                # Combined eigenvalue estimate (sum of left and right duals)
                aux[f'dual_real_{i}'] = -0.5 * (dual_variables_left_real[i, i] + dual_variables_right_real[i, i])
                aux[f'dual_imag_{i}'] = -0.5 * (dual_variables_left_imag[i, i] + dual_variables_right_imag[i, i])
                # Add diagonal errors for each eigenvector
                aux[f'error_left_real_{i}'] = error_matrix_left_real_1[i, i]
                aux[f'error_left_imag_{i}'] = error_matrix_left_imag_1[i, i]
                aux[f'error_right_real_{i}'] = error_matrix_right_real_1[i, i]
                aux[f'error_right_imag_{i}'] = error_matrix_right_imag_1[i, i]

                for j in range(0, min(2, i)):
                    aux[f'dual_left_real_{i}_{j}'] = dual_variables_left_real[i, j]
                    aux[f'dual_left_imag_{i}_{j}'] = dual_variables_left_imag[i, j]
                    aux[f'dual_right_real_{i}_{j}'] = dual_variables_right_real[i, j]
                    aux[f'dual_right_imag_{i}_{j}'] = dual_variables_right_imag[i, j]
                    # Add off-diagonal errors
                    aux[f'error_left_real_{i}_{j}'] = error_matrix_left_real_1[i, j]
                    aux[f'error_left_imag_{i}_{j}'] = error_matrix_left_imag_1[i, j]
                    aux[f'error_right_real_{i}_{j}'] = error_matrix_right_real_1[i, j]
                    aux[f'error_right_imag_{i}_{j}'] = error_matrix_right_imag_1[i, j]

            # Compute total errors for monitoring
            total_error_left_real = jnp.abs(error_matrix_left_real_1).sum()
            total_error_left_imag = jnp.abs(error_matrix_left_imag_1).sum()
            total_error_right_real = jnp.abs(error_matrix_right_real_1).sum()
            total_error_right_imag = jnp.abs(error_matrix_right_imag_1).sum()

            aux['total_error'] = total_error_left_real + total_error_left_imag + total_error_right_real + total_error_right_imag
            aux['total_norm_error'] = jnp.linalg.norm(error_matrix_left_real_1, 'fro') + jnp.linalg.norm(error_matrix_left_imag_1, 'fro') + \
                                       jnp.linalg.norm(error_matrix_right_real_1, 'fro') + jnp.linalg.norm(error_matrix_right_imag_1, 'fro')

            # Error for first two eigenvectors (useful for debugging)
            k_debug = min(2, args.num_eigenvectors)
            aux['total_two_component_error'] = jnp.abs(error_matrix_left_real_1[:k_debug, :k_debug]).sum() + \
                                                 jnp.abs(error_matrix_left_imag_1[:k_debug, :k_debug]).sum() + \
                                                 jnp.abs(error_matrix_right_real_1[:k_debug, :k_debug]).sum() + \
                                                 jnp.abs(error_matrix_right_imag_1[:k_debug, :k_debug]).sum()

            # Distance to constraint manifold (same as total_norm_error)
            aux['distance_to_constraint_manifold'] = aux['total_norm_error']

            # Distance to origin (norm of eigenvector parameters)
            aux['distance_to_origin'] = jnp.linalg.norm(psi_real) + jnp.linalg.norm(psi_imag) + \
                                         jnp.linalg.norm(phi_real) + jnp.linalg.norm(phi_imag)

            return allo, (error_matrix_left_real_1, error_matrix_left_imag_1, error_matrix_right_real_1, error_matrix_right_imag_1, aux)

        # Compute loss and gradients
        (allo, (error_matrix_left_real, error_matrix_left_imag, error_matrix_right_real, error_matrix_right_imag, aux)), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        # Apply optimizer updates
        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        # Perform custom integral update with the error matrix
        new_params['error_integral_left_real'] = args.integral_decay * new_params['error_integral_left_real'] + error_matrix_left_real
        new_params['error_integral_left_imag'] = args.integral_decay * new_params['error_integral_left_imag'] + error_matrix_left_imag
        new_params['error_integral_right_real'] = args.integral_decay * new_params['error_integral_right_real'] + error_matrix_right_real
        new_params['error_integral_right_imag'] = args.integral_decay * new_params['error_integral_right_imag'] + error_matrix_right_imag

        # Clip the barrier coefficients
        new_params['barrier_coefs_left_real'] = jnp.clip(new_params['barrier_coefs_left_real'], 0, args.max_barrier_coefs)
        new_params['barrier_coefs_left_imag'] = jnp.clip(new_params['barrier_coefs_left_imag'], 0, args.max_barrier_coefs)
        new_params['barrier_coefs_right_real'] = jnp.clip(new_params['barrier_coefs_right_real'], 0, args.max_barrier_coefs)
        new_params['barrier_coefs_right_imag'] = jnp.clip(new_params['barrier_coefs_right_imag'], 0, args.max_barrier_coefs)

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
            'num_eigenvectors': args.num_eigenvectors,
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
                save_path=str(plots_dir / "ground_truth_right_eigenvectors_real.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize right eigenvectors (imaginary part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvectors)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='right',
                component='imag',
                ncols=min(4, args.num_eigenvectors),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_right_eigenvectors_imag.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize left eigenvectors (real part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvectors)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='left',
                component='real',
                ncols=min(4, args.num_eigenvectors),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_left_eigenvectors_real.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize left eigenvectors (imaginary part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvectors)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='left',
                component='imag',
                ncols=min(4, args.num_eigenvectors),
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
                "allo": float(allo.item()),
                "sps": int(steps_completed / max(elapsed_time, 1e-6)),  # Steps since start/resume
            }
            for k, v in metrics.items():
                metrics_dict[k] = float(v.item())

            # Add cosine similarities to metrics (both left and right)
            for k, v in cosine_sims.items():
                metrics_dict[k] = v

            metrics_history.append(metrics_dict)

            if gradient_step % (args.log_freq * 10) == 0:
                left_sim = cosine_sims['left_cosine_sim_avg']
                right_sim = cosine_sims['right_cosine_sim_avg']
                print(f"Step {gradient_step}: loss={allo.item():.4f}, "
                      f"total_error={metrics['total_error'].item():.4f}, "
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
                    'eigenvalues': jnp.zeros(args.num_eigenvectors, dtype=jnp.complex64),
                    'eigenvalues_real': jnp.zeros(args.num_eigenvectors),
                    'eigenvalues_imag': jnp.zeros(args.num_eigenvectors),
                    'right_eigenvectors_real': features_dict['right_real'],
                    'right_eigenvectors_imag': features_dict['right_imag'],
                    'left_eigenvectors_real': features_dict['left_real'],
                    'left_eigenvectors_imag': features_dict['left_imag'],
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
                    ground_truth_eigenvalues_simple=gt_eigenvalues_simple if 'gt_eigenvalues_simple' in locals() else None,
                    ground_truth_eigenvalues_weighted=gt_eigenvalues_weighted if 'gt_eigenvalues_weighted' in locals() else None
                )

                # Plot cosine similarity evolution
                plot_cosine_similarity_evolution(
                    metrics_history,
                    str(plots_dir / "cosine_similarity_evolution.png"),
                    num_eigenvectors=args.num_eigenvectors
                )

                # Plot all duals evolution (left, right, combined)
                plot_all_duals_evolution(
                    metrics_history,
                    str(plots_dir / "all_duals_evolution.png"),
                    num_eigenvectors=min(6, args.num_eigenvectors)
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
        evec_idx = 0 if args.num_eigenvectors == 1 else 1

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

        # Left eigenvectors (real part)
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        visualize_eigenvector_on_grid(
            eigenvector_idx=evec_idx,
            eigenvector_values=np.array(gt_left_real[:, evec_idx]),
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            portals=door_markers if door_markers else None,
            title=f'Ground Truth Left {evec_idx} (Real)',
            ax=axes[0],
            cmap='RdBu_r',
            show_colorbar=True,
            wall_color='gray'
        )

        visualize_eigenvector_on_grid(
            eigenvector_idx=evec_idx,
            eigenvector_values=np.array(final_features_dict['left_real'][:, evec_idx]),
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            portals=door_markers if door_markers else None,
            title=f'Raw Learned Left {evec_idx} (Real)',
            ax=axes[1],
            cmap='RdBu_r',
            show_colorbar=True,
            wall_color='gray'
        )

        visualize_eigenvector_on_grid(
            eigenvector_idx=evec_idx,
            eigenvector_values=np.array(final_normalized_features['left_real'][:, evec_idx]),
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            portals=door_markers if door_markers else None,
            title=f'Normalized Learned Left {evec_idx} (Real)',
            ax=axes[2],
            cmap='RdBu_r',
            show_colorbar=True,
            wall_color='gray'
        )

        plt.tight_layout()
        plt.savefig(plots_dir / "final_comparison_left.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nData exported. Use generate_plots_complex.py to create visualizations.")
    print(f"Example: python generate_plots_complex.py {results_dir}")

    print(f"\nAll results saved to: {results_dir}")

    return encoder_state, results_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    learn_eigenvectors(args)
