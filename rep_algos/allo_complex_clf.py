"""
Train a model to learn complex eigenvectors of non-symmetric Laplacians.
"""

import os
import sys
import random
import time
import json
import pickle
from pathlib import Path

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

from src.config.ded_clf import Args
from src.envs.gridworld import GridWorldEnv
from src.utils.envs import create_gridworld_env
from src.utils.interaction import (
    create_replay_buffer_only,
    collect_data_and_compute_eigenvectors,
)

from src.nets import CoordinateEncoder

from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint,
)
from src.utils.metrics import (
    compute_complex_cosine_similarities_with_normalization,
    compute_hitting_times_from_eigenvectors,
    normalize_eigenvectors_for_comparison,
)
from src.utils.plotting import (
    plot_learning_curves_one,
    plot_cosine_similarity_evolution,
    plot_eigenvector_comparison,
    plot_auxiliary_metrics,
    visualize_multiple_eigenvectors,
    visualize_source_vs_target_hitting_times,
)


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
        num_eigenvector_pairs=args.num_eigenvector_pairs,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )

    # Create optimizer
    encoder_tx = optax.adam(learning_rate=args.learning_rate)
    lambda_tx = optax.adam(learning_rate=args.ema_learning_rate)

    # Create masks for different parameter groups
    encoder_mask = {
        'encoder': True,
        'lambda_real': False,
        'lambda_imag': False,
    }
    other_mask = {
        'encoder': False,
        'lambda_real': True,
        'lambda_imag': True,
    }

    tx = optax.chain(
        optax.masked(encoder_tx, encoder_mask),
        optax.masked(lambda_tx, other_mask)
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

        encoder_initial_params = encoder.init(encoder_key, dummy_input)
        initial_params = {
            'encoder': encoder_initial_params,
            'lambda_real': jnp.ones((args.num_eigenvector_pairs,)),
            'lambda_imag': jnp.zeros((args.num_eigenvector_pairs,)),
        }

        encoder_state = TrainState.create(
            apply_fn=encoder.apply,
            params=initial_params,
            tx=tx,
        )

    encoder.apply = jax.jit(encoder.apply)
    is_ratio = 1 / (sampling_probs[:, None].clip(min=1e-8) * len(sampling_probs))  # Shape: (num_states, 1)

    def sg(z):
        """Stop gradient utility function."""
        return jax.lax.stop_gradient(z)

    # Define the update function
    @jax.jit
    def update_encoder(
        encoder_state: TrainState,
        state_coords_batch: jnp.ndarray,
        next_state_coords_batch: jnp.ndarray,
        state_coords_batch_2: jnp.ndarray,
        state_weighting: jnp.ndarray,
    ):
        def ip(a, b):
            """Weighted inner product that keeps dimensions."""
            weighted_a = state_weighting * a
            return jnp.mean(weighted_a * b, axis=0, keepdims=True)
        
        def multi_ip(a, b):
            """Weighted multiple inner products for multiple vectors."""
            weighted_a = state_weighting * a
            return jnp.einsum('ij,ik->jk', weighted_a, b) / a.shape[0]

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
            y_r = features_1['left_real']
            y_i = features_1['left_imag']

            next_x_r = next_features['right_real']
            next_x_i = next_features['right_imag']
            next_y_r = next_features['left_real']
            next_y_i = next_features['left_imag']

            # Compute current unnormalized eigenvalue estimates (Shape: (1,k))
            lambda_x_r = ip(x_r, next_x_r) + ip(x_i, next_x_i)
            lambda_x_i = ip(x_r, next_x_i) - ip(x_i, next_x_r)
            lambda_y_r = ip(y_r, next_y_r) + ip(y_i, next_y_i)
            lambda_y_i = ip(y_r, next_y_i) - ip(y_i, next_y_r)

            # EMA loss for eigenvalues
            ema_lambda_r = params['lambda_real'].reshape(1, -1)  # (Shape: (1,k))
            ema_lambda_i = params['lambda_imag'].reshape(1, -1)
            new_lambda_real = 0.5 * (lambda_x_r + lambda_y_r)
            new_lambda_imag = 0.5 * (lambda_x_i - lambda_y_i)
            lambda_loss_real = ((sg(new_lambda_real) - ema_lambda_r) ** 2).sum()
            lambda_loss_imag = ((sg(new_lambda_imag) - ema_lambda_i) ** 2).sum()
            lambda_loss = lambda_loss_real + lambda_loss_imag

            # Compute squared norms (Shape: (1,k))
            norm_x_r_sq = ip(x_r, x_r)
            norm_x_i_sq = ip(x_i, x_i)
            norm_x_sq = norm_x_r_sq + norm_x_i_sq

            norm_y_r_sq = ip(y_r, y_r)
            norm_y_i_sq = ip(y_i, y_i)
            norm_y_sq = norm_y_r_sq + norm_y_i_sq

            next_norm_y_r_sq = ip(next_y_r, next_y_r)
            next_norm_y_i_sq = ip(next_y_i, next_y_i)
            next_norm_y_sq = next_norm_y_r_sq + next_norm_y_i_sq

            # Compute graph losses (Shape: (1,k))
            f_x_real = - next_x_r + ema_lambda_r * x_r - ema_lambda_i * x_i  # (Shape: (n,k))
            f_x_imag = - next_x_i + ema_lambda_i * x_r + ema_lambda_r * x_i

            f_y_0_real = - y_r
            f_y_0_imag = - y_i
            f_y_residual_real = ema_lambda_r * y_r + ema_lambda_i * y_i
            f_y_residual_imag = -ema_lambda_i * y_r + ema_lambda_r * y_i

            graph_loss_x_real = ip(x_r, sg(f_x_real))  # (Shape: (1,k))
            graph_loss_x_imag = ip(x_i, sg(f_x_imag))
            graph_loss_x = graph_loss_x_real + graph_loss_x_imag

            graph_loss_y_real = (
                ip(next_y_r, sg(f_y_0_real))
                + ip(y_r, sg(f_y_residual_real))
            )
            graph_loss_y_imag = (
                ip(next_y_i, sg(f_y_0_imag))
                + ip(y_i, sg(f_y_residual_imag))
            )
            graph_loss_y = graph_loss_y_real + graph_loss_y_imag            
            graph_loss = (graph_loss_x + graph_loss_y).sum()  # Sum over all eigenvectors (Shape: ())

            # Compute chirality loss (Shape: ())
            x_i_normalized = x_i / jnp.sqrt(norm_x_sq)
            chirality_x = ip(sg(x_i_normalized**2), x_i_normalized)
            chirality_loss = ((chirality_x)**2).sum()

            # Compute Lyapunov functions and their gradients
            # 1. For x norm: <x_k,x_k> should be 1
            norm_x_error = norm_x_sq - 1  # (Shape: (1,k))
            V_x_norm = norm_x_error ** 2 / 2

            nabla_x_r_V_x_norm = 2 * norm_x_error * x_r  # (Shape: (n,k))
            nabla_x_i_V_x_norm = 2 * norm_x_error * x_i
            nabla_y_r_V_x_norm = jnp.zeros_like(y_r)
            nabla_y_i_V_x_norm = jnp.zeros_like(y_i)
            
            next_nabla_y_r_V_x_norm = jnp.zeros_like(next_y_r)
            next_nabla_y_i_V_x_norm = jnp.zeros_like(next_y_i)

            # 2. For y norm: <y_k,y_k> should be 1
            norm_y_error = norm_y_sq - 1  # (Shape: (1,k))
            V_y_norm = norm_y_error ** 2 / 2

            nabla_x_r_V_y_norm = jnp.zeros_like(x_r)  # (Shape: (n,k))
            nabla_x_i_V_y_norm = jnp.zeros_like(x_i)
            nabla_y_r_V_y_norm = 2 * norm_y_error * y_r
            nabla_y_i_V_y_norm = 2 * norm_y_error * y_i

            next_norm_y_error = next_norm_y_sq - 1  # (Shape: (1,k))

            next_nabla_y_r_V_y_norm = 2 * next_norm_y_error * next_y_r  #(Shape: (n,k))
            next_nabla_y_i_V_y_norm = 2 * next_norm_y_error * next_y_i

            # 3. For xy phase: <y_k,x_k> should be real
            phase_xy = ip(y_r, x_i) - ip(y_i, x_r)  # (Shape: (1,k))
            V_xy_phase = phase_xy ** 2 / 2

            nabla_x_r_V_xy_phase = - phase_xy * y_i  # (Shape: (n,k))
            nabla_x_i_V_xy_phase = phase_xy * y_r
            nabla_y_r_V_xy_phase = phase_xy * x_i
            nabla_y_i_V_xy_phase = - phase_xy * x_r

            next_phase_xy = ip(next_y_r, next_x_i) - ip(next_y_i, next_x_r)  # (Shape: (1,k))

            next_nabla_y_r_V_xy_phase = next_phase_xy * next_x_i  #(Shape: (n,k))
            next_nabla_y_i_V_xy_phase = - next_phase_xy * next_x_r

            # 4. For crossed terms: <y_j,x_k> should be 0 for j≠k
            cross_xryr =  multi_ip(x_r, y_r)
            cross_xiyi =  multi_ip(x_i, y_i)
            cross_xryi =  multi_ip(x_r, y_i)
            cross_xiryr =  multi_ip(x_i, y_r)

            corr_xy_real = cross_xryr + cross_xiyi  # (Shape: (k,k)) (First index: x, second index: y)
            corr_xy_imag = -cross_xryi + cross_xiryr

            corr_xy_real_lower = jnp.tril(corr_xy_real, k=-1)
            corr_xy_imag_lower = jnp.tril(corr_xy_imag, k=-1)

            corr_yx_real_lower = jnp.tril(corr_xy_real.T, k=-1)
            corr_yx_imag_lower = jnp.tril(corr_xy_imag.T, k=-1)

            V_xy_corr_real = jnp.sum(corr_xy_real_lower ** 2, -1).reshape(1,-1) / 2  #(Shape: (1,k))
            V_xy_corr_imag = jnp.sum(corr_xy_imag_lower ** 2, -1).reshape(1,-1) / 2
            V_yx_corr_real = jnp.sum(corr_yx_real_lower ** 2, -1).reshape(1,-1) / 2
            V_yx_corr_imag = jnp.sum(corr_yx_imag_lower ** 2, -1).reshape(1,-1) / 2
            V_xy_corr = V_xy_corr_real + V_xy_corr_imag + V_yx_corr_real + V_yx_corr_imag

            nabla_x_r_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', corr_xy_real_lower, y_r)  #(Shape: (n,k))
            nabla_x_i_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', corr_xy_real_lower, y_i)            

            nabla_x_r_V_xy_corr_imag = -2 * jnp.einsum('jk,ik->ij', corr_xy_imag_lower, y_i)
            nabla_x_i_V_xy_corr_imag = 2 * jnp.einsum('jk,ik->ij', corr_xy_imag_lower, y_r)
            
            nabla_y_r_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', corr_yx_real_lower, x_r)
            nabla_y_i_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', corr_yx_real_lower, x_i)
            
            nabla_y_r_V_xy_corr_imag = 2 * jnp.einsum('jk,ik->ij', corr_yx_imag_lower, x_i)
            nabla_y_i_V_xy_corr_imag = -2 * jnp.einsum('jk,ik->ij', corr_yx_imag_lower, x_r)

            next_corr_xy_real = multi_ip(next_x_r, next_y_r) + multi_ip(next_x_i, next_y_i)  # (Shape: (k,k)) (First index: x, second index: y)
            next_corr_xy_imag = -multi_ip(next_x_r, next_y_i) + multi_ip(next_x_i, next_y_r)

            next_corr_yx_real = jnp.tril(next_corr_xy_real.T, k=-1)
            next_corr_yx_imag = jnp.tril(next_corr_xy_imag.T, k=-1)

            next_nabla_y_r_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', next_corr_yx_real, next_x_r)  #(Shape: (n,k))
            next_nabla_y_i_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', next_corr_yx_real, next_x_i)
            next_nabla_y_r_V_xy_corr_imag = 2 * jnp.einsum('jk,ik->ij', next_corr_yx_imag, next_x_i)
            next_nabla_y_i_V_xy_corr_imag = -2 * jnp.einsum('jk,ik->ij', next_corr_yx_imag, next_x_r)

            # 5. Global
            V = V_x_norm + V_y_norm + V_xy_phase + V_xy_corr  # (Shape: (1,k))
            nabla_x_r_V = nabla_x_r_V_x_norm + nabla_x_r_V_y_norm + nabla_x_r_V_xy_phase + nabla_x_r_V_xy_corr_real + nabla_x_r_V_xy_corr_imag  # Shape: (n,k)
            nabla_x_i_V = nabla_x_i_V_x_norm + nabla_x_i_V_y_norm + nabla_x_i_V_xy_phase + nabla_x_i_V_xy_corr_real + nabla_x_i_V_xy_corr_imag
            nabla_y_r_V = nabla_y_r_V_x_norm + nabla_y_r_V_y_norm + nabla_y_r_V_xy_phase + nabla_y_r_V_xy_corr_real + nabla_y_r_V_xy_corr_imag
            nabla_y_i_V = nabla_y_i_V_x_norm + nabla_y_i_V_y_norm + nabla_y_i_V_xy_phase + nabla_y_i_V_xy_corr_real + nabla_y_i_V_xy_corr_imag

            next_nabla_y_r_V = next_nabla_y_r_V_x_norm + next_nabla_y_r_V_y_norm + next_nabla_y_r_V_xy_phase + next_nabla_y_r_V_xy_corr_real + next_nabla_y_r_V_xy_corr_imag
            next_nabla_y_i_V = next_nabla_y_i_V_x_norm + next_nabla_y_i_V_y_norm + next_nabla_y_i_V_xy_phase + next_nabla_y_i_V_xy_corr_real + next_nabla_y_i_V_xy_corr_imag

            norm_nabla_x_r_V_sq = ip(nabla_x_r_V, nabla_x_r_V)  # (Shape: (1,k))
            norm_nabla_x_i_V_sq = ip(nabla_x_i_V, nabla_x_i_V)
            norm_nabla_y_r_V_sq = ip(nabla_y_r_V, nabla_y_r_V)
            norm_nabla_y_i_V_sq = ip(nabla_y_i_V, nabla_y_i_V)
            norm_nabla_V_sq = (
                norm_nabla_x_r_V_sq + norm_nabla_x_i_V_sq 
                + norm_nabla_y_r_V_sq + norm_nabla_y_i_V_sq
            )

            # Compute Control Lyapunov Function (CLF) loss
            f_x_r_dot_nabla_V = ip(f_x_real, nabla_x_r_V)  # (Shape: (1,k))
            f_x_i_dot_nabla_V = ip(f_x_imag, nabla_x_i_V)

            f_y_r_dot_nabla_V = (
                ip(f_y_0_real, next_nabla_y_r_V)
                + ip(f_y_residual_real, nabla_y_r_V)
            )
            f_y_i_dot_nabla_V = (
                ip(f_y_0_imag, next_nabla_y_i_V)
                + ip(f_y_residual_imag, nabla_y_i_V)
            )

            f_dot_nabla_V = f_x_r_dot_nabla_V + f_x_i_dot_nabla_V + f_y_r_dot_nabla_V + f_y_i_dot_nabla_V

            clf_num = f_dot_nabla_V + args.lambda_x * V
            barrier = jnp.maximum(0, clf_num) / (norm_nabla_V_sq + 1e-8)

            u_x_r = barrier * nabla_x_r_V  # (Shape: (n,k))
            u_x_i = barrier * nabla_x_i_V
            u_y_r = barrier * nabla_y_r_V
            u_y_i = barrier * nabla_y_i_V

            clf_loss_x_r_k = ip(x_r, sg(u_x_r))  # (Shape: (1,k))
            clf_loss_x_i_k = ip(x_i, sg(u_x_i))
            clf_loss_y_r_k = ip(y_r, sg(u_y_r))
            clf_loss_y_i_k = ip(y_i, sg(u_y_i))

            clf_loss_x_r = clf_loss_x_r_k.sum()  # Sum over all eigenvectors (Shape: ())
            clf_loss_x_i = clf_loss_x_i_k.sum()
            clf_loss_y_r = clf_loss_y_r_k.sum()
            clf_loss_y_i = clf_loss_y_i_k.sum()

            clf_loss = clf_loss_x_r + clf_loss_x_i + clf_loss_y_r + clf_loss_y_i

            # Total loss
            total_loss = graph_loss + clf_loss + args.chirality_factor * chirality_loss + lambda_loss

            # Auxiliary metrics (average over eigenvector pairs for logging)
            aux = {
                'total_loss': total_loss,
                'graph_loss': graph_loss,
                'clf_loss': clf_loss,
                'chirality_loss': chirality_loss,
                'graph_loss_x_real': graph_loss_x_real.sum(),
                'graph_loss_x_imag': graph_loss_x_imag.sum(),
                'clf_loss_x_real': clf_loss_x_r,
                'clf_loss_x_imag': clf_loss_x_i,
                'graph_loss_y_real': graph_loss_y_real.sum(),
                'graph_loss_y_imag': graph_loss_y_imag.sum(),
                'clf_loss_y_real': clf_loss_y_r,
                'clf_loss_y_imag': clf_loss_y_i,
                'lambda_x_real': ema_lambda_r.mean(),
                'lambda_x_imag': ema_lambda_i.mean(),
                'new_lambda_x_real': lambda_x_r.mean(),
                'new_lambda_x_imag': lambda_x_i.mean(),
                'new_lambda_y_real': lambda_y_r.mean(),
                'new_lambda_y_imag': lambda_y_i.mean(),
                'V_x_norm': V_x_norm.mean(),
                'V_y_norm': V_y_norm.mean(),
                'V_xy_phase': V_xy_phase.mean(),
                'norm_x_sq': norm_x_sq.mean(),
                'norm_y_sq': norm_y_sq.mean(),
                'barrier': barrier.mean(),
            }

            return total_loss, aux

        # Compute loss and gradients
        (total_loss, aux), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        # Compute encoder gradient norm for clipping
        encoder_grads_flat, _ = jax.tree_util.tree_flatten(grads['encoder'])
        encoder_grads_vector = jnp.concatenate([jnp.ravel(g) for g in encoder_grads_flat])
        encoder_grad_norm = jnp.linalg.norm(encoder_grads_vector)
        aux['grad_norm'] = encoder_grad_norm

        # Clip encoder gradients (norm-based)
        max_norm = 1.0
        encoder_clip_factor = max_norm / jnp.maximum(encoder_grad_norm, max_norm)

        # Clip EMA gradients (element-wise, independent per estimator)
        grads = {
            'encoder': jax.tree_util.tree_map(lambda g: g * encoder_clip_factor, grads['encoder']),
            'lambda_real': jnp.clip(grads['lambda_real'], -1.0, 1.0),
            'lambda_imag': jnp.clip(grads['lambda_imag'], -1.0, 1.0),
        }

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
            'num_eigenvectors': args.num_eigenvector_pairs,
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
                eigenvector_indices=list(range(args.num_eigenvector_pairs)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='right',
                component='real',
                ncols=min(4, args.num_eigenvector_pairs),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_right_eigenvectors_real.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize right eigenvectors (imaginary part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvector_pairs)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='right',
                component='imag',
                ncols=min(4, args.num_eigenvector_pairs),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_right_eigenvectors_imag.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize left eigenvectors (real part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvector_pairs)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='left',
                component='real',
                ncols=min(4, args.num_eigenvector_pairs),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_left_eigenvectors_real.png"),
                shared_colorbar=True
            )
            plt.close()

            # Visualize left eigenvectors (imaginary part)
            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(args.num_eigenvector_pairs)),
                eigendecomposition=eigendecomp_viz,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                eigenvector_type='left',
                component='imag',
                ncols=min(4, args.num_eigenvector_pairs),
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
        warmup_state_weighting = is_ratio[warmup_indices]

        # Run one update to compile with the loaded state (discard result)
        warmup_state, warmup_loss, _ = update_encoder(
            encoder_state,
            warmup_coords,
            warmup_next_coords,
            warmup_coords,  # Use same batch for second set
            warmup_state_weighting,
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
        state_weighting = is_ratio[state_indices]

        # Update
        update_start = time.time()
        encoder_state, total_loss, metrics = update_encoder(
            encoder_state,
            coords_batch,
            next_coords_batch,
            coords_batch_2,
            state_weighting,
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
                gt_eigenvalues_real=gt_eigenvalues_real,
                gt_eigenvalues_imag=gt_eigenvalues_imag,
                sampling_probs=sampling_probs * is_ratio.squeeze(-1),
            )

            # Compute hitting times for learned eigenvectors and ground truth
            hitting_times = compute_hitting_times_from_eigenvectors(
                left_real=features_dict['left_real'],
                left_imag=features_dict['left_imag'],
                right_real=features_dict['right_real'],
                right_imag=features_dict['right_imag'],
                eigenvalues_real=encoder_state.params['lambda_real'],
                eigenvalues_imag=encoder_state.params['lambda_imag'],
                gamma=args.gamma,
                delta=args.delta,
                eigenvalue_type='kernel',
            )
            gt_hitting_times = compute_hitting_times_from_eigenvectors(
                left_real=gt_left_real,
                left_imag=gt_left_imag,
                right_real=gt_right_real,
                right_imag=gt_right_imag,
                eigenvalues_real=gt_eigenvalues_real,
                eigenvalues_imag=gt_eigenvalues_imag,
                gamma=args.gamma,
                delta=args.delta,
                eigenvalue_type='laplacian',
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
                graph_loss = metrics_dict['graph_loss']
                clf_loss = metrics_dict['clf_loss']
                chirality_loss = metrics_dict['chirality_loss']
                print(f"Step {gradient_step}: total_loss={total_loss.item():.4f}, "
                        f"graph_loss={graph_loss:.4f}, clf_loss={clf_loss:.4f}, "
                        f"chirality_loss={chirality_loss:.4f}, "
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
                sampling_probs=sampling_probs * is_ratio.squeeze(-1)
            )
            np.save(results_dir / "latest_learned_left_real_normalized.npy", np.array(normalized_features['left_real']))
            np.save(results_dir / "latest_learned_left_imag_normalized.npy", np.array(normalized_features['left_imag']))
            np.save(results_dir / "latest_learned_right_real_normalized.npy", np.array(normalized_features['right_real']))
            np.save(results_dir / "latest_learned_right_imag_normalized.npy", np.array(normalized_features['right_imag']))

            # Optionally create plots during training (slower)
            if args.plot_during_training:
                # Generate comparison plots with latest learned eigenvectors
                plot_eigenvector_comparison(
                    learned_left_real=np.array(features_dict['left_real']),
                    learned_left_imag=np.array(features_dict['left_imag']),
                    learned_right_real=np.array(features_dict['right_real']),
                    learned_right_imag=np.array(features_dict['right_imag']),
                    gt_left_real=np.array(gt_left_real),
                    gt_left_imag=np.array(gt_left_imag),
                    gt_right_real=np.array(gt_right_real),
                    gt_right_imag=np.array(gt_right_imag),
                    normalized_left_real=np.array(normalized_features['left_real']),
                    normalized_left_imag=np.array(normalized_features['left_imag']),
                    normalized_right_real=np.array(normalized_features['right_real']),
                    normalized_right_imag=np.array(normalized_features['right_imag']),
                    canonical_states=np.array(canonical_states),
                    grid_width=env.width,
                    grid_height=env.height,
                    save_dir=str(plots_dir),
                    door_markers=door_markers if door_markers else None,
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
        sampling_probs=sampling_probs * is_ratio.squeeze(-1)
    )
    np.save(results_dir / "final_learned_left_real_normalized.npy", np.array(final_normalized_features['left_real']))
    np.save(results_dir / "final_learned_left_imag_normalized.npy", np.array(final_normalized_features['left_imag']))
    np.save(results_dir / "final_learned_right_real_normalized.npy", np.array(final_normalized_features['right_real']))
    np.save(results_dir / "final_learned_right_imag_normalized.npy", np.array(final_normalized_features['right_imag']))

    # Generate final visualization plots (always run at end of training)
    print("\n" + "="*60)
    print("Generating final visualizations...")
    print("="*60)

    # 1. Learning curves
    plot_learning_curves_one(metrics_history, str(plots_dir / "learning_curves.png"))

    # 2. Cosine similarity evolution
    plot_cosine_similarity_evolution(metrics_history, str(plots_dir / "cosine_similarity_evolution.png"))

    # 3. Auxiliary metrics
    plot_auxiliary_metrics(metrics_history, str(plots_dir / "auxiliary_metrics.png"))

    # 4. Eigenvector comparison plots (Ground Truth vs Raw Learned vs Normalized)
    plot_eigenvector_comparison(
        learned_left_real=np.array(final_features_dict['left_real']),
        learned_left_imag=np.array(final_features_dict['left_imag']),
        learned_right_real=np.array(final_features_dict['right_real']),
        learned_right_imag=np.array(final_features_dict['right_imag']),
        gt_left_real=np.array(gt_left_real),
        gt_left_imag=np.array(gt_left_imag),
        gt_right_real=np.array(gt_right_real),
        gt_right_imag=np.array(gt_right_imag),
        normalized_left_real=np.array(final_normalized_features['left_real']),
        normalized_left_imag=np.array(final_normalized_features['left_imag']),
        normalized_right_real=np.array(final_normalized_features['right_real']),
        normalized_right_imag=np.array(final_normalized_features['right_imag']),
        canonical_states=np.array(canonical_states),
        grid_width=env.width,
        grid_height=env.height,
        save_dir=str(plots_dir),
        door_markers=door_markers if door_markers else None,
    )

    # 5. Hitting times visualization (Ground Truth vs Learned)
    print("Generating hitting times visualizations...")

    # Compute final hitting times from learned eigenvectors
    final_hitting_times = compute_hitting_times_from_eigenvectors(
        left_real=final_features_dict['left_real'],
        left_imag=final_features_dict['left_imag'],
        right_real=final_features_dict['right_real'],
        right_imag=final_features_dict['right_imag'],
        eigenvalues_real=encoder_state.params['lambda_real'],
        eigenvalues_imag=encoder_state.params['lambda_imag'],
        gamma=args.gamma,
        delta=args.delta,
        eigenvalue_type='kernel',
    )

    # Compute ground truth hitting times
    final_gt_hitting_times = compute_hitting_times_from_eigenvectors(
        left_real=gt_left_real,
        left_imag=gt_left_imag,
        right_real=gt_right_real,
        right_imag=gt_right_imag,
        eigenvalues_real=gt_eigenvalues_real,
        eigenvalues_imag=gt_eigenvalues_imag,
        gamma=args.gamma,
        delta=args.delta,
        eigenvalue_type='laplacian',
    )

    # Select states to visualize (evenly spaced across state space)
    num_states_to_viz = min(6, num_states)
    viz_state_indices = np.linspace(0, num_states - 1, num_states_to_viz, dtype=int).tolist()

    # Visualize learned hitting times
    visualize_source_vs_target_hitting_times(
        state_indices=viz_state_indices,
        hitting_time_matrix=np.array(final_hitting_times),
        canonical_states=np.array(canonical_states),
        grid_width=env.width,
        grid_height=env.height,
        portals=door_markers if door_markers else None,
        ncols=min(6, num_states_to_viz),
        save_path=str(plots_dir / "hitting_times_learned.png"),
        log_scale=True,
        shared_colorbar=True,
    )
    plt.close()

    # Visualize ground truth hitting times
    visualize_source_vs_target_hitting_times(
        state_indices=viz_state_indices,
        hitting_time_matrix=np.array(final_gt_hitting_times),
        canonical_states=np.array(canonical_states),
        grid_width=env.width,
        grid_height=env.height,
        portals=door_markers if door_markers else None,
        ncols=min(6, num_states_to_viz),
        save_path=str(plots_dir / "hitting_times_ground_truth.png"),
        log_scale=True,
        shared_colorbar=True,
    )
    plt.close()

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"\nGenerated plots in {plots_dir}:")
    print("  - learning_curves.png: Training losses, Lyapunov functions, norms")
    print("  - cosine_similarity_evolution.png: Left/right eigenvector similarity over time")
    print("  - auxiliary_metrics.png: Eigenvalue estimates, norms, Lyapunov functions")
    if args.num_eigenvector_pairs == 1:
        print("  - comparison_right_real.png: Right eigenvector (real) comparison")
        print("  - comparison_right_imag.png: Right eigenvector (imag) comparison")
        print("  - comparison_left_real.png: Left eigenvector (real) comparison")
        print("  - comparison_left_imag.png: Left eigenvector (imag) comparison")
    else:
        print(f"  - comparison_right_real_i.png: Right eigenvector i (real) comparison (i=0..{args.num_eigenvector_pairs-1})")
        print(f"  - comparison_right_imag_i.png: Right eigenvector i (imag) comparison (i=0..{args.num_eigenvector_pairs-1})")
        print(f"  - comparison_left_real_i.png: Left eigenvector i (real) comparison (i=0..{args.num_eigenvector_pairs-1})")
        print(f"  - comparison_left_imag_i.png: Left eigenvector i (imag) comparison (i=0..{args.num_eigenvector_pairs-1})")
    print("  - hitting_times_learned.png: Hitting times from learned eigenvectors")
    print("  - hitting_times_ground_truth.png: Hitting times from ground truth eigenvectors")

    print(f"\nAll results saved to: {results_dir}")

    return encoder_state, results_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    learn_eigenvectors(args)
