"""
Train a model to learn complex eigenvectors of non-symmetric Laplacians with DIRECT optimization.

This is a simplified version that directly optimizes eigenvector matrices instead of using a neural network.
The model is simply 4 matrices of size (n_states, k):
- left_real: Left eigenvector real components
- left_imag: Left eigenvector imaginary components
- right_real: Right eigenvector real components
- right_imag: Right eigenvector imaginary components

Uses SGD optimizer (instead of Adam) to stay closer to theory.
"""

import os
import sys
import random
import time
import json
import pickle
import glob
from typing import Dict, Any
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
import matplotlib.pyplot as plt

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


def get_transition_matrix(
    transition_counts: jnp.ndarray,
    make_stochastic: bool = True,
) -> jnp.ndarray:
    """Build transition matrix from counts (non-symmetrized)."""
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)
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
    """Compute the successor representation SR_γ = (I - γP)^(-1) or finite-horizon version."""
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    if max_horizon is None:
        sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix)
    else:
        P_T_plus_1 = jnp.linalg.matrix_power(transition_matrix, max_horizon + 1)
        gamma_power = gamma ** (max_horizon + 1)
        sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix) @ (identity - gamma_power * P_T_plus_1)

    return sr_matrix


def compute_sampling_distribution(
    transition_counts: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the empirical sampling distribution from transition counts."""
    if len(transition_counts.shape) == 3:
        state_visit_counts = jnp.sum(transition_counts, axis=(1, 2))
    else:
        state_visit_counts = jnp.sum(transition_counts, axis=1)

    total_visits = jnp.sum(state_visit_counts)
    sampling_probs = state_visit_counts / total_visits

    D = jnp.diag(sampling_probs)
    return D


def compute_nonsymmetric_laplacian(
    transition_matrix: jnp.ndarray,
    gamma: float,
    delta: float = 0.0,
    max_horizon: int = None,
) -> jnp.ndarray:
    """Compute the non-symmetric Laplacian L = (1+δ)I - (1-γ)P·SR_γ."""
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    sr_matrix = compute_successor_representation(transition_matrix, gamma, max_horizon)
    laplacian = (1 + delta) * identity - (1 - gamma) * (transition_matrix @ sr_matrix)

    return laplacian


def plot_sampling_distribution(
    sampling_probs: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    save_path: str,
    portals: Dict = None
):
    """Plot the sampling distribution on the grid."""
    num_states = len(canonical_states)
    grid_values = np.full((grid_height, grid_width), np.nan)

    for canonical_idx, full_idx in enumerate(canonical_states):
        full_idx = int(full_idx)
        y = full_idx // grid_width
        x = full_idx % grid_width
        grid_values[y, x] = sampling_probs[canonical_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_values, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Sampling Probability')
    ax.set_title('Sampling Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_checkpoint(path_str: str) -> Dict[str, Any]:
    """
    Load a checkpoint from a file or directory.
    If a directory is provided, tries to find the latest checkpoint_*.pkl file.
    """
    path = Path(path_str)
    
    if path.is_dir():
        # Try to find checkpoints in this directory
        checkpoints = list(path.glob("checkpoint_*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint_*.pkl files found in {path}")
        
        # Sort by step number
        def get_step(p):
            try:
                return int(p.stem.split('_')[1])
            except (IndexError, ValueError):
                return -1
        
        latest_checkpoint = max(checkpoints, key=get_step)
        print(f"Found latest checkpoint: {latest_checkpoint}")
        path = latest_checkpoint
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
        
    print(f"Loaded checkpoint from {path}")
    if 'step' in checkpoint:
        print(f"  Step: {checkpoint['step']}")
    
    return checkpoint


@dataclass
class Args:
    # Environment
    env_type: str = "room4"
    env_file: str | None = None
    env_file_name: str | None = None
    max_episode_length: int = 1000

    # Data collection
    num_envs: int = 1000
    num_steps: int = 1000

    # Irreversible doors
    use_doors: bool = False
    num_doors: int = 5
    door_seed: int = 42

    # Model
    num_eigenvectors: int = 10

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 4096
    num_gradient_steps: int = 20000
    gamma: float = 0.2
    delta: float = 0.1

    # Episodic replay buffer
    max_time_offset: int | None = None

    # Augmented Lagrangian parameters
    duals_initial_val: float = 2.0
    barrier_coef: float = 2.0
    step_size_duals: float = 1.0

    # Logging and saving
    log_freq: int = 10000
    plot_freq: int = 10000
    save_freq: int = 10000
    checkpoint_freq: int = 50000
    save_model: bool = True
    plot_during_training: bool = False
    results_dir: str = "./results"

    # Resuming training
    resume_from: str | None = None

    # Misc
    seed: int = 42
    exp_name: str | None = None
    exp_number: int = 0


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


def create_gridworld_env(args: Args):
    """Create a gridworld environment from text file or example."""
    if args.env_type == 'file':
        if args.env_file is None and args.env_file_name is None:
            raise ValueError("Must specify either env_file or env_file_name when env_type='file'")

        env = create_environment_from_text(
            file_path=args.env_file,
            file_name=args.env_file_name,
            max_steps=args.max_episode_length,
            precision=32,
        )
    else:
        if args.env_type not in EXAMPLE_ENVIRONMENTS:
            raise ValueError(f"Unknown env_type: {args.env_type}")

        text_content = EXAMPLE_ENVIRONMENTS[args.env_type]
        env = create_environment_from_text(
            text_content=text_content,
            max_steps=args.max_episode_length,
            precision=32,
        )

    print(f"Loaded environment: {args.env_type}")
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Number of obstacles: {len(env.obstacles) if env.has_obstacles else 0}")

    from src.envs.door_gridworld import DoorGridWorldEnv
    if isinstance(env, DoorGridWorldEnv):
        print(f"  Environment has doors from file: {len(env.blocked_transitions)} blocked transitions")
    else:
        print(f"  Environment type: {type(env).__name__} (no file-defined doors)")

    return env


def main(args: Args):
    # Resume handling setup
    checkpoint_data = None
    start_step = 0
    metrics_history = []
    
    if args.resume_from is not None:
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING")
        
        try:
            checkpoint_data = load_checkpoint(args.resume_from)
            
            # Determine results directory from resume path
            resume_path = Path(args.resume_from)
            if resume_path.is_file():
                results_dir = resume_path.parent
            else:
                results_dir = resume_path
                
            print(f"Resuming in directory: {results_dir}")
            
            # Extract state
            start_step = checkpoint_data.get('step', 0) + 1
            metrics_history = checkpoint_data.get('metrics_history', [])
            
            # Merge args
            if 'args' in checkpoint_data:
                saved_args_dict = checkpoint_data['args']
                # List of keys we want to KEEP from the NEW args (training params)
                training_keys = [
                    'learning_rate', 'num_gradient_steps', 'log_freq', 
                    'plot_freq', 'checkpoint_freq', 'batch_size',
                    'save_model', 'plot_during_training'
                ]
                
                # Update saved args with current training params
                for key in training_keys:
                    if hasattr(args, key):
                        saved_args_dict[key] = getattr(args, key)
                
                # Update args object values from the merged dictionary
                # (We keep the original env params to ensure consistency)
                for key, value in saved_args_dict.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                
                print(f"Merged arguments from checkpoint (preserving new training params)")
            
            # Restore random state if available
            if 'rng_state' in checkpoint_data:
                random.setstate(checkpoint_data['rng_state'])
                np.random.set_state(checkpoint_data['np_rng_state'])
                print("Restored random seeds")
                
        except Exception as e:
            print(f"Error resuming: {e}")
            print("Falling back to fresh start...")
            checkpoint_data = None
            
        print(f"{'='*60}\n")

    # Set random seeds (if not restored)
    if checkpoint_data is None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    key = jax.random.PRNGKey(args.seed)
    encoder_key = key

    # Create environment
    env = create_gridworld_env(args)

    # Get canonical (free) states from base environment
    canonical_states = get_canonical_free_states(env)
    num_canonical = len(canonical_states)
    num_states = env.width * env.height  # Full state space including walls
    print(f"Number of free states: {num_canonical} (out of {num_states} total)")

    # Add doors if requested
    data_env = env
    door_config = None

    if args.use_doors:
        door_rng = np.random.default_rng(args.door_seed)
        door_config = create_random_doors(
            env=env,
            num_doors=args.num_doors,
            rng=door_rng,
        )
        data_env = create_door_gridworld_from_base(env, door_config)
        print(f"\nCreated {len(door_config['doors'])} irreversible doors:")
        for s_can, a_fwd, sp_can, a_rev in door_config['doors']:
            print(f"  Door: state {s_can} --({a_fwd})--> state {sp_can} (reverse {a_rev} blocked)")

    # Collect transition data
    print("\nCollecting transition data...")
    transition_counts_full, episodes, metrics = collect_transition_counts_and_episodes(
        env=data_env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_states=num_states,  # Pass full state space size
        seed=args.seed,
    )

    print(f"Collected {metrics['total_transitions']} transitions.")

    # Extract canonical state subspace from full transition counts
    transition_counts = transition_counts_full[jnp.ix_(canonical_states, jnp.arange(env.action_space), canonical_states)]

    print(f"\nTransition data:")
    print(f"  Number of canonical states: {num_canonical}")
    print(f"  Transition counts shape: {transition_counts.shape}")

    # Create full-to-canonical state mapping
    full_to_canonical = {int(full_idx): canon_idx for canon_idx, full_idx in enumerate(canonical_states)}

    # Initialize replay buffer
    max_valid_length = int(episodes['lengths'].max()) + 1
    replay_buffer = EpisodicReplayBuffer(
        max_episodes=args.num_envs,
        max_episode_length=max_valid_length,
        observation_type='canonical_state',
        seed=args.seed
    )

    # Convert and add episodes to buffer
    print("\nPopulating replay buffer with episodes...")
    replay_buffer.populate_with_tabular_episodes(
        canonical_states=canonical_states,
        episodes=episodes,
    )
    print(f"  Added {len(replay_buffer)} trajectory sequences to replay buffer")

    # Compute ground truth eigendecomposition
    print("\nBuilding transition matrix...")
    transition_matrix = get_transition_matrix(transition_counts)

    # Compute sampling distribution
    print("Computing empirical sampling distribution...")
    sampling_distribution = compute_sampling_distribution(transition_counts)
    sampling_probs = jnp.diag(sampling_distribution)
    print(f"  Sampling prob range: [{sampling_probs.min():.6f}, {sampling_probs.max():.6f}]")
    print(f"  Sampling prob std: {sampling_probs.std():.6f}")

    print(f"\nComputing non-symmetric Laplacian with gamma={args.gamma}, delta={args.delta}...")
    print(f"  L = (1+δ)I - (1-γ)P·SR_γ (matches geometric sampling with k >= 1)")
    laplacian = compute_nonsymmetric_laplacian(
        transition_matrix,
        gamma=args.gamma,
        delta=args.delta
    )

    eigendecomp = compute_eigendecomposition(
        laplacian,
        k=args.num_eigenvectors,
        ascending=True  # For Laplacians, we want smallest eigenvalues
    )

    gt_eigenvalues_real = eigendecomp['eigenvalues_real']
    gt_eigenvalues_imag = eigendecomp['eigenvalues_imag']
    gt_left_real = eigendecomp['left_eigenvectors_real']
    gt_left_imag = eigendecomp['left_eigenvectors_imag']
    gt_right_real = eigendecomp['right_eigenvectors_real']
    gt_right_imag = eigendecomp['right_eigenvectors_imag']

    print(f"\nSmallest {min(5, args.num_eigenvectors)} eigenvalues (complex):")
    print("  Eigenvalue (real + imag)")
    for i in range(min(5, args.num_eigenvectors)):
        ev_real = gt_eigenvalues_real[i]
        ev_imag = gt_eigenvalues_imag[i]
        print(f"  λ_{i}: {ev_real:.6f} + {ev_imag:.6f}i")

    # Setup results directory
    if checkpoint_data is None:
        if args.exp_name is None:
            exp_name = f"{args.env_type}__allo_direct__{args.exp_number}__{args.seed}__{int(time.time())}"
        else:
            exp_name = args.exp_name

        results_dir = Path(args.results_dir) / args.env_type / exp_name
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use directory from resume
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nResults directory: {results_dir}")

    # Save args
    with open(results_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Save ground truth data (overwrite even if resuming to be safe)
    np.save(results_dir / "gt_eigenvalues_real.npy", np.array(gt_eigenvalues_real))
    np.save(results_dir / "gt_eigenvalues_imag.npy", np.array(gt_eigenvalues_imag))
    np.save(results_dir / "gt_left_real.npy", np.array(gt_left_real))
    np.save(results_dir / "gt_left_imag.npy", np.array(gt_left_imag))
    np.save(results_dir / "gt_right_real.npy", np.array(gt_right_real))
    np.save(results_dir / "gt_right_imag.npy", np.array(gt_right_imag))
    np.save(results_dir / "sampling_distribution.npy", np.array(sampling_probs))

    # Create door markers for visualization
    door_markers = {}
    if door_config is not None and 'doors' in door_config:
        for s_canonical, a_forward, s_prime_canonical, a_reverse in door_config['doors']:
            s_full = int(canonical_states[s_canonical])
            s_prime_full = int(canonical_states[s_prime_canonical])
            door_markers[(s_full, a_forward)] = s_prime_full

    # Save visualization metadata (required by generate_plots_complex.py)
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

    # Visualize sampling distribution
    if args.plot_during_training and checkpoint_data is None:
        plot_sampling_distribution(
            sampling_probs=sampling_probs,
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            save_path=str(plots_dir / "sampling_distribution.png"),
            portals=door_markers if door_markers else None
        )

    # Initialize eigenvector matrices directly
    # Shape: (num_canonical, num_eigenvectors)
    print("\nInitializing eigenvector matrices...")

    if checkpoint_data is not None:
        initial_params = checkpoint_data['params']
        print("  Using restored parameters from checkpoint")
    else:
        # Initialize with small random values
        init_scale = 0.01
        initial_params = {
            'left_real': init_scale * jax.random.normal(encoder_key, (num_canonical, args.num_eigenvectors)),
            'left_imag': init_scale * jax.random.normal(jax.random.split(encoder_key)[0], (num_canonical, args.num_eigenvectors)),
            'right_real': init_scale * jax.random.normal(jax.random.split(encoder_key)[1], (num_canonical, args.num_eigenvectors)),
            'right_imag': init_scale * jax.random.normal(jax.random.split(jax.random.split(encoder_key)[1])[0], (num_canonical, args.num_eigenvectors)),
            'duals_real': args.duals_initial_val * jnp.ones((args.num_eigenvectors,)),
            'duals_imag': args.duals_initial_val * jnp.ones((args.num_eigenvectors,)),
        }
        print(f"  Initialized matrices with random values")

    # Create SGD optimizer for ALL parameters
    sgd_tx = optax.sgd(learning_rate=args.learning_rate)

    # Create train state
    # We'll use a dummy apply_fn since we don't have a model
    def dummy_apply(params, state_indices):
        """Extract eigenvector values for given state indices."""
        return {
            'left_real': params['left_real'][state_indices],
            'left_imag': params['left_imag'][state_indices],
            'right_real': params['right_real'][state_indices],
            'right_imag': params['right_imag'][state_indices],
        }

    train_state = TrainState.create(
        apply_fn=dummy_apply,
        params=initial_params,
        tx=sgd_tx,
    )
    
    # Restore optimizer state if resuming
    if checkpoint_data is not None and 'opt_state' in checkpoint_data:
        train_state = train_state.replace(opt_state=checkpoint_data['opt_state'])
        print("  Restored optimizer state from checkpoint")

    print(f"  Matrix shape: ({num_canonical}, {args.num_eigenvectors})")
    print(f"  Using SGD optimizer with learning rate: {args.learning_rate}")

    # Define JIT-compiled cosine similarity computation
    @jax.jit
    def compute_cosine_similarities(params, gt_left_real, gt_left_imag, gt_right_real, gt_right_imag, sampling_probs):
        """Compute cosine similarities between learned and ground truth eigenvectors."""
        learned_left_real = params['left_real']
        learned_left_imag = params['left_imag']
        learned_right_real = params['right_real']
        learned_right_imag = params['right_imag']

        num_eigenvectors = learned_left_real.shape[1]

        # Normalize learned eigenvectors similar to original implementation
        # For right eigenvectors: normalize by largest magnitude component
        right_sims = []
        for i in range(num_eigenvectors):
            u_real = learned_right_real[:, i]
            u_imag = learned_right_imag[:, i]
            v_real = gt_right_real[:, i]
            v_imag = gt_right_imag[:, i]

            # Find component with largest magnitude for learned
            magnitudes = jnp.sqrt(u_real**2 + u_imag**2)
            max_idx = jnp.argmax(magnitudes)
            scale_real = u_real[max_idx]
            scale_imag = u_imag[max_idx]
            scale_norm = jnp.sqrt(scale_real**2 + scale_imag**2)

            # Normalize: divide by (scale_real + i*scale_imag)
            # Division: (a + ib) / (c + id) = [(ac+bd) + i(bc-ad)] / (c²+d²)
            norm_real = (u_real * scale_real + u_imag * scale_imag) / (scale_norm**2 + 1e-10)
            norm_imag = (u_imag * scale_real - u_real * scale_imag) / (scale_norm**2 + 1e-10)

            # Complex inner product: <conj(u), v>
            inner_real = jnp.dot(norm_real, v_real) + jnp.dot(norm_imag, v_imag)
            inner_imag = jnp.dot(norm_real, v_imag) - jnp.dot(norm_imag, v_real)

            # Norms
            u_norm = jnp.sqrt(jnp.dot(norm_real, norm_real) + jnp.dot(norm_imag, norm_imag))
            v_norm = jnp.sqrt(jnp.dot(v_real, v_real) + jnp.dot(v_imag, v_imag))

            # Cosine similarity (absolute value of real part)
            cos_real = inner_real / (u_norm * v_norm + 1e-10)
            right_sims.append(jnp.abs(cos_real))

        # For left eigenvectors: weight by sampling distribution and take conjugate
        left_sims = []
        for i in range(num_eigenvectors):
            u_real = learned_left_real[:, i]
            u_imag = learned_left_imag[:, i]
            v_real = gt_left_real[:, i]
            v_imag = gt_left_imag[:, i]

            # Weight by sqrt of sampling distribution
            sqrt_d = jnp.sqrt(sampling_probs)
            u_real_weighted = u_real * sqrt_d
            u_imag_weighted = u_imag * sqrt_d
            v_real_weighted = v_real * sqrt_d
            v_imag_weighted = v_imag * sqrt_d

            # Find component with largest magnitude
            magnitudes = jnp.sqrt(u_real_weighted**2 + u_imag_weighted**2)
            max_idx = jnp.argmax(magnitudes)
            scale_real = u_real_weighted[max_idx]
            scale_imag = u_imag_weighted[max_idx]
            scale_norm = jnp.sqrt(scale_real**2 + scale_imag**2)

            # Normalize
            norm_real = (u_real_weighted * scale_real + u_imag_weighted * scale_imag) / (scale_norm**2 + 1e-10)
            norm_imag = (u_imag_weighted * scale_real - u_real_weighted * scale_imag) / (scale_norm**2 + 1e-10)

            # Take complex conjugate (flip imaginary sign)
            norm_real_conj = norm_real
            norm_imag_conj = -norm_imag

            # Complex inner product
            inner_real = jnp.dot(norm_real_conj, v_real_weighted) + jnp.dot(norm_imag_conj, v_imag_weighted)
            inner_imag = jnp.dot(norm_real_conj, v_imag_weighted) - jnp.dot(norm_imag_conj, v_real_weighted)

            # Norms
            u_norm = jnp.sqrt(jnp.dot(norm_real, norm_real) + jnp.dot(norm_imag, norm_imag))
            v_norm = jnp.sqrt(jnp.dot(v_real_weighted, v_real_weighted) + jnp.dot(v_imag_weighted, v_imag_weighted))

            # Cosine similarity
            cos_real = inner_real / (u_norm * v_norm + 1e-10)
            left_sims.append(jnp.abs(cos_real))

        return jnp.mean(jnp.array(left_sims)), jnp.mean(jnp.array(right_sims))

    # Define the update function
    @jax.jit
    def update_step(state, state_indices, next_state_indices, state_indices_2):
        """Single gradient update step."""

        def loss_fn(params):
            # Extract eigenvectors for sampled states
            psi_real = params['left_real'][state_indices]
            psi_imag = params['left_imag'][state_indices]
            phi_real = params['right_real'][state_indices]
            phi_imag = params['right_imag'][state_indices]

            psi_real_2 = params['left_real'][state_indices_2]
            psi_imag_2 = params['left_imag'][state_indices_2]
            phi_real_2 = params['right_real'][state_indices_2]
            phi_imag_2 = params['right_imag'][state_indices_2]

            next_psi_real = params['left_real'][next_state_indices]
            next_psi_imag = params['left_imag'][next_state_indices]
            next_phi_real = params['right_real'][next_state_indices]
            next_phi_imag = params['right_imag'][next_state_indices]

            # Get sizes
            d = args.num_eigenvectors
            n = phi_real.shape[0]

            # Get duals
            duals_real = params['duals_real']
            duals_imag = params['duals_imag']

            # Eigenvalue sum
            eigenvalue_sum_real = duals_real.sum()
            eigenvalue_sum_imag = duals_imag.sum()

            # Compute biorthogonality inner products: <ψ, φ>
            # Real part: Re(<ψ, φ>) = ψ_real^T φ_real + ψ_imag^T φ_imag
            inner_product_left_real_1 = (jnp.einsum('ij,ik->jk', psi_real, jax.lax.stop_gradient(phi_real)) +
                                   jnp.einsum('ij,ik->jk', psi_imag, jax.lax.stop_gradient(phi_imag))) / n
            inner_product_left_real_2 = (jnp.einsum('ij,ik->jk', psi_real_2, jax.lax.stop_gradient(phi_real_2)) +
                                   jnp.einsum('ij,ik->jk', psi_imag_2, jax.lax.stop_gradient(phi_imag_2))) / n
            # Imaginary part: Im(<ψ, φ>) = ψ_real^T φ_imag - ψ_imag^T φ_real
            inner_product_left_imag_1 = (jnp.einsum('ij,ik->jk', psi_real, jax.lax.stop_gradient(phi_imag)) -
                                   jnp.einsum('ij,ik->jk', psi_imag, jax.lax.stop_gradient(phi_real))) / n
            inner_product_left_imag_2 = (jnp.einsum('ij,ik->jk', psi_real_2, jax.lax.stop_gradient(phi_imag_2)) -
                                   jnp.einsum('ij,ik->jk', psi_imag_2, jax.lax.stop_gradient(phi_real_2))) / n

            # Same for right eigenvectors
            inner_product_right_real_1 = (jnp.einsum('ij,ik->jk', phi_real, jax.lax.stop_gradient(psi_real)) +
                                    jnp.einsum('ij,ik->jk', phi_imag, jax.lax.stop_gradient(psi_imag))) / n
            inner_product_right_real_2 = (jnp.einsum('ij,ik->jk', phi_real_2, jax.lax.stop_gradient(psi_real_2)) +
                                    jnp.einsum('ij,ik->jk', phi_imag_2, jax.lax.stop_gradient(psi_imag_2))) / n
            inner_product_right_imag_1 = (-jnp.einsum('ij,ik->jk', phi_real, jax.lax.stop_gradient(psi_imag)) +
                                    jnp.einsum('ij,ik->jk', phi_imag, jax.lax.stop_gradient(psi_real))) / n
            inner_product_right_imag_2 = (-jnp.einsum('ij,ik->jk', phi_real_2, jax.lax.stop_gradient(psi_imag_2)) +
                                    jnp.einsum('ij,ik->jk', phi_imag_2, jax.lax.stop_gradient(psi_real_2))) / n

            # Biorthogonality error matrices
            error_matrix_left_real_1 = jnp.tril(inner_product_left_real_1 - jnp.eye(d))
            error_matrix_left_real_2 = jnp.tril(inner_product_left_real_2 - jnp.eye(d))
            error_matrix_right_real_1 = jnp.tril(inner_product_right_real_1 - jnp.eye(d))
            error_matrix_right_real_2 = jnp.tril(inner_product_right_real_2 - jnp.eye(d))

            error_matrix_left_imag_1 = jnp.tril(inner_product_left_imag_1)
            error_matrix_left_imag_2 = jnp.tril(inner_product_left_imag_2)
            error_matrix_right_imag_1 = jnp.tril(inner_product_right_imag_1)
            error_matrix_right_imag_2 = jnp.tril(inner_product_right_imag_2)

            # Norm errors for left eigenvectors
            norm_errors_real = jnp.diag(error_matrix_left_real_1)
            norm_errors_imag = jnp.diag(error_matrix_left_imag_1)

            # Compute dual loss for left eigenvectors
            dual_loss_pos_real = -(jax.lax.stop_gradient(duals_real) * norm_errors_real).sum()
            dual_loss_neg_real = args.step_size_duals * (duals_real * jax.lax.stop_gradient(norm_errors_real)).sum()

            dual_loss_pos_imag = -(jax.lax.stop_gradient(duals_imag) * norm_errors_imag).sum()
            dual_loss_neg_imag = args.step_size_duals * (duals_imag * jax.lax.stop_gradient(norm_errors_imag)).sum()

            # Compute right eigenvector norms
            phi_norms_real_1 = jnp.log((phi_real**2 + phi_imag**2).mean(0)+1e-6).clip(max=10)
            phi_norms_real_2 = jnp.log((phi_real_2**2 + phi_imag_2**2).mean(0)+1e-6).clip(max=10)

            norm_errors_right_1 = phi_norms_real_1
            norm_errors_right_2 = phi_norms_real_2

            # Total dual loss
            dual_loss_pos = dual_loss_pos_real + dual_loss_pos_imag
            dual_loss_neg = dual_loss_neg_real + dual_loss_neg_imag

            # Compute barrier loss (for real part constraint)
            quadratic_error_matrix_left_real = 2 * error_matrix_left_real_1 * jax.lax.stop_gradient(error_matrix_left_real_2)
            quadratic_error_left_real = quadratic_error_matrix_left_real.sum()
            barrier_loss_left_real = args.barrier_coef * quadratic_error_left_real

            quadratic_error_matrix_right_real = 2 * error_matrix_right_real_1 * jax.lax.stop_gradient(error_matrix_right_real_2)
            quadratic_error_right_real = quadratic_error_matrix_right_real.sum()
            barrier_loss_right_real = args.barrier_coef * quadratic_error_right_real

            # Compute barrier loss (for imaginary part constraint)
            quadratic_error_matrix_left_imag = 2 * error_matrix_left_imag_1 * jax.lax.stop_gradient(error_matrix_left_imag_2)
            quadratic_error_left_imag = quadratic_error_matrix_left_imag.sum()
            barrier_loss_left_imag = args.barrier_coef * quadratic_error_left_imag

            quadratic_error_matrix_right_imag = 2 * error_matrix_right_imag_1 * jax.lax.stop_gradient(error_matrix_right_imag_2)
            quadratic_error_right_imag = quadratic_error_matrix_right_imag.sum()
            barrier_loss_right_imag = args.barrier_coef * quadratic_error_right_imag

            barrier_loss_left = barrier_loss_left_real + barrier_loss_left_imag
            barrier_loss_right = barrier_loss_right_real + barrier_loss_right_imag

            # Compute barrier loss for right eigenvector norm constraints
            quadratic_error_norm_right = 2 * norm_errors_right_1 * jax.lax.stop_gradient(norm_errors_right_2)
            barrier_loss_norm_right = args.barrier_coef * quadratic_error_norm_right.sum()

            # Separate barrier terms for logging
            barrier_biortho = barrier_loss_left + barrier_loss_right
            barrier_norm = barrier_loss_norm_right
            barrier_loss = barrier_biortho + barrier_norm

            # Compute graph drawing loss for complex eigenvectors
            graph_products_real = (
                jax.lax.stop_gradient(psi_real)
                * ((1+args.delta)*phi_real - next_phi_real)
            ).mean(0, keepdims=True)
            graph_products_imag = (
                jax.lax.stop_gradient(psi_imag)
                * ((1+args.delta)*phi_imag - next_phi_imag)
            ).mean(0, keepdims=True)
            graph_products_real_reverse = (
                jax.lax.stop_gradient(next_phi_real)
                * ((1+args.delta)*next_psi_real - psi_real)
                ).mean(0, keepdims=True)
            graph_products_imag_reverse = (
                jax.lax.stop_gradient(next_phi_imag)
                * ((1+args.delta)*next_psi_imag - psi_imag)
            ).mean(0, keepdims=True)

            graph_loss_direct = ((graph_products_real + graph_products_imag)**2).sum()
            graph_loss_reverse = ((graph_products_real_reverse + graph_products_imag_reverse)**2).sum()
            graph_loss = graph_loss_direct + graph_loss_reverse

            # Total loss
            allo = graph_loss + dual_loss_pos + barrier_loss  # Augmented Lagrangian objective
            negative_loss = dual_loss_neg
            total_loss = allo + negative_loss

            # Auxiliary metrics
            aux = {
                'allo': allo,
                'graph_loss': graph_loss,
                'dual_loss': dual_loss_pos,
                'dual_loss_neg': dual_loss_neg,
                'barrier_loss': barrier_loss,
                'barrier_biortho': barrier_biortho,
                'barrier_norm': barrier_norm,
                'approx_eigenvalue_sum_real': eigenvalue_sum_real,
                'approx_eigenvalue_sum_imag': eigenvalue_sum_imag,
            }

            # Add dual variables and errors to aux
            for i in range(min(11, args.num_eigenvectors)):
                aux[f'dual_real_{i}'] = duals_real[i]
                aux[f'dual_imag_{i}'] = duals_imag[i]
                aux[f'error_left_real_{i}'] = error_matrix_left_real_1[i, i]
                aux[f'error_left_imag_{i}'] = error_matrix_left_imag_1[i, i]
                aux[f'error_right_real_{i}'] = error_matrix_right_real_1[i, i]
                aux[f'error_right_imag_{i}'] = error_matrix_right_imag_1[i, i]
                aux[f'error_norm_right_{i}'] = norm_errors_right_1[i]

                for j in range(0, min(2, i)):
                    aux[f'error_left_real_{i}_{j}'] = error_matrix_left_real_1[i, j]
                    aux[f'error_left_imag_{i}_{j}'] = error_matrix_left_imag_1[i, j]
                    aux[f'error_right_real_{i}_{j}'] = error_matrix_right_real_1[i, j]
                    aux[f'error_right_imag_{i}_{j}'] = error_matrix_right_imag_1[i, j]

            # Compute total errors for monitoring
            total_error_left_real = jnp.abs(error_matrix_left_real_1).sum()
            total_error_left_imag = jnp.abs(error_matrix_left_imag_1).sum()
            total_error_right_real = jnp.abs(error_matrix_right_real_1).sum()
            total_error_right_imag = jnp.abs(error_matrix_right_imag_1).sum()

            aux['l1_constraint_error'] = total_error_left_real + total_error_left_imag + total_error_right_real + total_error_right_imag
            aux['total_norm_error'] = jnp.linalg.norm(error_matrix_left_real_1, 'fro') + jnp.linalg.norm(error_matrix_left_imag_1, 'fro') + \
                                       jnp.linalg.norm(error_matrix_right_real_1, 'fro') + jnp.linalg.norm(error_matrix_right_imag_1, 'fro')

            k_debug = min(2, args.num_eigenvectors)
            aux['total_two_component_error'] = jnp.abs(error_matrix_left_real_1[:k_debug, :k_debug]).sum() + \
                                                 jnp.abs(error_matrix_left_imag_1[:k_debug, :k_debug]).sum() + \
                                                 jnp.abs(error_matrix_right_real_1[:k_debug, :k_debug]).sum() + \
                                                 jnp.abs(error_matrix_right_imag_1[:k_debug, :k_debug]).sum()

            aux['distance_to_constraint_manifold'] = aux['total_norm_error']
            aux['distance_to_origin'] = jnp.linalg.norm(psi_real) + jnp.linalg.norm(psi_imag) + \
                                         jnp.linalg.norm(phi_real) + jnp.linalg.norm(phi_imag)

            return total_loss, aux

        # Compute loss and gradients
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Apply optimizer updates
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        # Create new state
        new_state = state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1
        )

        # Get grad norm
        grads_flat, _ = jax.tree_util.tree_flatten(grads)
        grads_vector = jnp.concatenate([jnp.ravel(g) for g in grads_flat])
        grad_norm = jnp.linalg.norm(grads_vector)
        aux['grad_norm'] = grad_norm

        return new_state, loss, aux

    # Training loop
    if checkpoint_data is None:
        print("\nStarting training...")
    else:
        print(f"\nResuming training from step {start_step}...")
        
    start_time = time.time()

    for step in tqdm(range(start_step, args.num_gradient_steps)):
        # Sample batches
        batch1 = replay_buffer.sample(args.batch_size, discount=args.gamma)
        batch2 = replay_buffer.sample(args.batch_size, discount=args.gamma)

        state_indices = jnp.array(batch1.obs)
        next_state_indices = jnp.array(batch1.next_obs)
        state_indices_2 = jnp.array(batch2.obs)

        # Update
        train_state, loss, aux = update_step(
            train_state,
            state_indices,
            next_state_indices,
            state_indices_2
        )

        # Log metrics
        if step % args.log_freq == 0:
            # Compute cosine similarities
            left_cosine_sim, right_cosine_sim = compute_cosine_similarities(
                train_state.params,
                gt_left_real, gt_left_imag,
                gt_right_real, gt_right_imag,
                sampling_probs
            )

            metrics_dict = {
                'gradient_step': step,
                'allo': float(aux['allo']),
                'graph_loss': float(aux['graph_loss']),
                'barrier_loss': float(aux['barrier_loss']),
                'barrier_biortho': float(aux['barrier_biortho']),
                'barrier_norm': float(aux['barrier_norm']),
                'dual_loss': float(aux['dual_loss']),
                'dual_loss_neg': float(aux['dual_loss_neg']),
                'l1_constraint_error': float(aux['l1_constraint_error']),
                'grad_norm': float(aux['grad_norm']),
                'left_cosine_sim_avg': float(left_cosine_sim),
                'right_cosine_sim_avg': float(right_cosine_sim),
            }
            metrics_history.append(metrics_dict)

        # Print metrics
        if step % args.log_freq == 0:
            tqdm.write(
                f"Step {step}: allo={aux['allo']:.4f}, graph={aux['graph_loss']:.4f}, "
                f"barrier_biortho={aux['barrier_biortho']:.4f}, barrier_norm={aux['barrier_norm']:.4f}, "
                f"l1_error={aux['l1_constraint_error']:.4f}, "
                f"left_sim={left_cosine_sim:.4f}, right_sim={right_cosine_sim:.4f}"
            )

        # Save checkpoint
        if args.save_model and step % args.checkpoint_freq == 0 and step > 0:
            checkpoint_path = results_dir / f"checkpoint_{step}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'params': train_state.params,
                    'opt_state': train_state.opt_state,
                    'step': step,
                    'metrics_history': metrics_history,
                    'args': vars(args),
                    'rng_state': random.getstate(),
                    'np_rng_state': np.random.get_state(),
                }, f)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_params = train_state.params

    if args.save_model:
        # Save with names expected by generate_plots_complex.py
        np.save(results_dir / "final_learned_left_real.npy", np.array(final_params['left_real']))
        np.save(results_dir / "final_learned_left_imag.npy", np.array(final_params['left_imag']))
        np.save(results_dir / "final_learned_right_real.npy", np.array(final_params['right_real']))
        np.save(results_dir / "final_learned_right_imag.npy", np.array(final_params['right_imag']))
        np.save(results_dir / "learned_duals_real.npy", np.array(final_params['duals_real']))
        np.save(results_dir / "learned_duals_imag.npy", np.array(final_params['duals_imag']))

    # Save metrics (required by generate_plots_complex.py)
    with open(results_dir / "metrics_history.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.2f}s")
    print(f"Results saved to: {results_dir}")

    # Visualize results if requested
    if args.plot_during_training:
        door_markers = {}
        if door_config is not None and 'doors' in door_config:
            for s_canonical, a_forward, s_prime_canonical, a_reverse in door_config['doors']:
                s_full = int(canonical_states[s_canonical])
                s_prime_full = int(canonical_states[s_prime_canonical])
                door_markers[(s_full, a_forward)] = s_prime_full

        # Create eigendecomposition dict for visualization
        learned_eigendecomp = {
            'eigenvalues': (final_params['duals_real'] + 1j * final_params['duals_imag']).astype(jnp.complex64),
            'eigenvalues_real': final_params['duals_real'],
            'eigenvalues_imag': final_params['duals_imag'],
            'right_eigenvectors_real': final_params['right_real'],
            'right_eigenvectors_imag': final_params['right_imag'],
            'left_eigenvectors_real': final_params['left_real'],
            'left_eigenvectors_imag': final_params['left_imag'],
        }

        # Visualize learned eigenvectors
        visualize_multiple_eigenvectors(
            eigenvector_indices=list(range(args.num_eigenvectors)),
            eigendecomposition=learned_eigendecomp,
            canonical_states=canonical_states,
            grid_width=env.width,
            grid_height=env.height,
            portals=door_markers if door_markers else None,
            eigenvector_type='left',
            component='real',
            ncols=min(4, args.num_eigenvectors),
            wall_color='gray',
            save_path=str(plots_dir / "learned_left_eigenvectors_real.png"),
            shared_colorbar=True
        )
        plt.close()

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
            save_path=str(plots_dir / "learned_right_eigenvectors_real.png"),
            shared_colorbar=True
        )
        plt.close()

    return results_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)