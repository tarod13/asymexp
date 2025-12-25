"""
Train a model to learn eigenvectors of the symmetrized transition matrix.

This script uses (x,y) coordinates as inputs to the network and learns
the eigenvectors through an augmented Lagrangian optimization approach.
"""

import os
import random
import time
from typing import Dict

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tyro
from tqdm import tqdm
from flax.training.train_state import TrainState
import flax.linen as nn

from src.envs.gridworld import GridWorldEnv
from src.data_collection import collect_transition_counts

import wandb
import certifi

# Setup certificates and WandB
ca_cert_path = certifi.where()
os.environ['WANDB_CA_CERTS'] = ca_cert_path
os.environ['WANDB_API_KEY'] = '83c25550226f8a86fdd4874026d2c0804cd3dc05'
os.environ['WANDB_ENTITY'] = 'tarod13'


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
    normalize: bool = True
) -> jnp.ndarray:
    """
    Build symmetrized transition matrix from counts.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]
        smoothing: Small value to add for numerical stability
        normalize: Whether to row-normalize to get proper transition probabilities

    Returns:
        Symmetrized transition matrix of shape [num_states, num_states]
    """
    # Sum over actions if needed
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)  # [num_states, num_states]
    else:
        transition_matrix = transition_counts

    # Symmetrize
    transition_matrix = (transition_matrix + transition_matrix.T) / 2.0

    # Add smoothing factor
    transition_matrix = transition_matrix + smoothing

    # Normalize if requested (with iterations for better symmetry)
    if normalize:
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
        Dictionary containing:
            - eigenvalues: Shape [k] (real-valued, sorted descending)
            - eigenvectors: Shape [num_states, k] (column vectors)
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

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


# Training arguments
class Args:
    # Environment
    env_name: str = "gridworld"
    grid_width: int = 10
    grid_height: int = 10
    max_episode_length: int = 100

    # Data collection
    num_envs: int = 1000
    num_steps: int = 100

    # Model
    num_eigenvectors: int = 10
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_gradient_steps: int = 10000
    gamma: float = 0.99

    # Augmented Lagrangian parameters
    duals_initial_val: float = 0.1
    barrier_initial_val: float = 1.0
    max_barrier_coefs: float = 100.0
    step_size_duals: float = 0.01
    step_size_duals_I: float = 0.0
    integral_decay: float = 0.99
    init_dual_diag: bool = False

    # Regularization
    graph_epsilon: float = 0.01
    graph_variance_scale: float = 0.1
    perturbation_type: str = 'exponential'  # 'exponential', 'squared', 'squared-null-grad', 'none'
    turn_off_above_threshold: bool = False
    cum_error_threshold: float = 0.1

    # Logging
    track: bool = True
    wandb_project_name: str = "asymexp_allo"
    wandb_mode: str = "online"
    wandb_dir: str = "./wandb"
    log_freq: int = 100
    save_freq: int = 1000

    # Saving
    save_model: bool = True

    # Misc
    seed: int = 42
    exp_name: str = None
    exp_number: int = 0


def create_gridworld_env(args: Args):
    """Create a simple gridworld environment."""
    env = GridWorldEnv(
        width=args.grid_width,
        height=args.grid_height,
        max_steps=args.max_episode_length,
        precision=32,
    )
    return env


def state_idx_to_xy(state_idx: int, width: int) -> tuple:
    """Convert state index to (x,y) coordinates."""
    y = state_idx // width
    x = state_idx % width
    return x, y


def collect_data_and_compute_eigenvectors(env, args: Args):
    """
    Collect transition data and compute ground truth eigenvectors.

    Returns:
        transition_matrix: Symmetrized transition matrix
        eigendecomp: Dictionary with eigenvalues and eigenvectors
        state_coords: Array of (x,y) coordinates for each state
    """
    print("Collecting transition data...")
    num_states = env.width * env.height

    # Collect transition counts
    transition_counts, metrics = collect_transition_counts(
        env=env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_states=num_states,
        seed=args.seed,
    )

    print(f"Collected {metrics['total_transitions']} transitions")

    # Build symmetrized transition matrix
    print("Building symmetrized transition matrix...")
    transition_matrix = get_symmetrized_transition_matrix(
        transition_counts,
        smoothing=1e-5,
        normalize=True,
    )

    # Compute eigendecomposition
    print(f"Computing eigendecomposition (top {args.num_eigenvectors} eigenvectors)...")
    eigendecomp = compute_symmetrized_eigendecomposition(
        transition_matrix,
        k=args.num_eigenvectors,
    )

    print(f"Top {min(5, args.num_eigenvectors)} eigenvalues:")
    for i in range(min(5, args.num_eigenvectors)):
        print(f"  λ_{i}: {eigendecomp['eigenvalues'][i]:.6f}")

    # Create state coordinate mapping
    state_coords = []
    for state_idx in range(num_states):
        x, y = state_idx_to_xy(state_idx, env.width)
        state_coords.append([x, y])
    state_coords = jnp.array(state_coords, dtype=jnp.float32)

    return transition_matrix, eigendecomp, state_coords


def learn_eigenvectors(args):
    """Main training loop to learn eigenvectors."""

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]

    # Set up run
    run_name = f"{args.env_name}__{args.exp_name}__{args.exp_number}__{args.seed}__{int(time.time())}"

    if args.track:
        if args.wandb_mode == 'offline':
            os.environ['WANDB_MODE'] = 'offline'
        run = wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            save_code=True,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
        )
    else:
        run = None

    # Create results directory
    if args.save_model:
        path = f'./results/models/{args.env_name}/placeholder.pkl'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, encoder_key = jax.random.split(rng_key, 2)

    # Create environment and collect data
    env = create_gridworld_env(args)
    transition_matrix, eigendecomp, state_coords = collect_data_and_compute_eigenvectors(env, args)

    gt_eigenvalues = eigendecomp['eigenvalues']
    gt_eigenvectors = eigendecomp['eigenvectors']

    print(f"\nState coordinates shape: {state_coords.shape}")
    print(f"Ground truth eigenvectors shape: {gt_eigenvectors.shape}")

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

    # Precompute transition probabilities for sampling
    transition_probs = transition_matrix  # Already normalized

    for gradient_step in tqdm(range(args.num_gradient_steps)):
        # Sample random batches
        rng_key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(rng_key, 5)

        # Sample initial states uniformly
        state_indices = jax.random.randint(subkey1, (args.batch_size,), 0, num_states)

        # Sample next states according to transition probabilities
        # For each state, sample next state from its transition distribution
        next_state_indices = []
        for i in range(args.batch_size):
            state_idx = int(state_indices[i])
            # Sample next state from transition distribution
            next_idx = jax.random.categorical(subkey2, jnp.log(transition_probs[state_idx] + 1e-10))
            next_state_indices.append(next_idx)
            subkey2, _ = jax.random.split(subkey2)
        next_state_indices = jnp.array(next_state_indices)

        # Sample another batch for the second orthogonality check
        state_indices_2 = jax.random.randint(subkey3, (args.batch_size,), 0, num_states)

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
            if args.track:
                log_dict = {
                    "SPS": int(gradient_step / (time.time() - start_time)),
                    "gradient_step": gradient_step,
                    "allo": allo.item(),
                }
                for k, v in metrics.items():
                    log_dict[f"{k}"] = v.item()
                run.log(log_dict)

            if gradient_step % (args.log_freq * 10) == 0:
                print(f"Step {gradient_step}: loss={allo.item():.4f}, "
                      f"total_error={metrics['total_error'].item():.4f}")

    print("\nTraining complete!")

    # Save model
    if args.save_model:
        import pickle
        save_path = f'./results/models/{args.env_name}/{run_name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({
                'params': encoder_state.params,
                'args': vars(args),
            }, f)
        print(f"Model saved to {save_path}")

    if args.track:
        run.finish()

    return encoder_state, run


if __name__ == "__main__":
    args = tyro.cli(Args)
    learn_eigenvectors(args)
