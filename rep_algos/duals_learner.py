"""
Dual-variable CLF learner for complex eigenvectors.

This learner extends the CLF approach with explicit dual variables for each constraint.
It also uses the conjugate relationship for eigenvalues: λ_y^I = -λ_x^I

Key differences from base CLF:
- Additional dual variables for norm, phase, and orthogonality constraints
- Three optimizers: encoder, lambda, and dual variables
- Conjugate eigenvalue relationship built in

Usage:
    from rep_algos.shared_training import learn_eigenvectors
    from rep_algos import duals_learner
    from src.config.ded_clf import Args

    args = tyro.cli(Args)
    learn_eigenvectors(args, duals_learner)
"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def init_params(encoder_initial_params, args):
    """
    Initialize parameters for the duals learner.

    Args:
        encoder_initial_params: Initial parameters for the encoder network
        args: Training arguments

    Returns:
        Dictionary containing initial parameters
    """
    return {
        'encoder': encoder_initial_params,
        'lambda_real': jnp.ones((1, args.num_eigenvector_pairs)),
        'lambda_imag': jnp.zeros((1, args.num_eigenvector_pairs)),
        'dual_x_norm': jnp.zeros((1, args.num_eigenvector_pairs)),
        'dual_y_norm': jnp.zeros((1, args.num_eigenvector_pairs)),
        'dual_xy_phase': jnp.zeros((1, args.num_eigenvector_pairs)),
        'dual_x_orth_real': jnp.zeros((args.num_eigenvector_pairs, args.num_eigenvector_pairs)),
        'dual_x_orth_imag': jnp.zeros((args.num_eigenvector_pairs, args.num_eigenvector_pairs)),
        'dual_y_orth_real': jnp.zeros((args.num_eigenvector_pairs, args.num_eigenvector_pairs)),
        'dual_y_orth_imag': jnp.zeros((args.num_eigenvector_pairs, args.num_eigenvector_pairs)),
    }


def get_optimizer_masks(args):
    """
    Get optimizer masks for different parameter groups.
    Note: This learner uses create_optimizer() instead.

    Returns:
        Tuple of (encoder_mask, other_mask)
    """
    # Not used when create_optimizer is provided, but kept for compatibility
    encoder_mask = {
        'encoder': True,
        'lambda_real': False,
        'lambda_imag': False,
        'dual_x_norm': False,
        'dual_y_norm': False,
        'dual_xy_phase': False,
        'dual_x_orth_real': False,
        'dual_x_orth_imag': False,
        'dual_y_orth_real': False,
        'dual_y_orth_imag': False,
    }
    other_mask = {
        'encoder': False,
        'lambda_real': True,
        'lambda_imag': True,
        'dual_x_norm': True,
        'dual_y_norm': True,
        'dual_xy_phase': True,
        'dual_x_orth_real': True,
        'dual_x_orth_imag': True,
        'dual_y_orth_real': True,
        'dual_y_orth_imag': True,
    }
    return encoder_mask, other_mask


def create_optimizer(args):
    """
    Create custom optimizer with three parameter groups.

    Args:
        args: Training arguments

    Returns:
        Combined optimizer chain
    """
    encoder_tx = optax.adam(learning_rate=args.learning_rate)
    lambda_tx = optax.adam(learning_rate=args.ema_learning_rate)
    dual_tx = optax.adam(learning_rate=args.dual_learning_rate)

    encoder_mask = {
        'encoder': True,
        'lambda_real': False,
        'lambda_imag': False,
        'dual_x_norm': False,
        'dual_y_norm': False,
        'dual_xy_phase': False,
        'dual_x_orth_real': False,
        'dual_x_orth_imag': False,
        'dual_y_orth_real': False,
        'dual_y_orth_imag': False,
    }
    lambda_mask = {
        'encoder': False,
        'lambda_real': True,
        'lambda_imag': True,
        'dual_x_norm': False,
        'dual_y_norm': False,
        'dual_xy_phase': False,
        'dual_x_orth_real': False,
        'dual_x_orth_imag': False,
        'dual_y_orth_real': False,
        'dual_y_orth_imag': False,
    }
    dual_mask = {
        'encoder': False,
        'lambda_real': False,
        'lambda_imag': False,
        'dual_x_norm': True,
        'dual_y_norm': True,
        'dual_xy_phase': True,
        'dual_x_orth_real': True,
        'dual_x_orth_imag': True,
        'dual_y_orth_real': True,
        'dual_y_orth_imag': True,
    }

    return optax.chain(
        optax.masked(encoder_tx, encoder_mask),
        optax.masked(lambda_tx, lambda_mask),
        optax.masked(dual_tx, dual_mask),
    )


def create_update_function(encoder, args):
    """
    Create the JIT-compiled update function for the duals learner.

    Args:
        encoder: The encoder network
        args: Training arguments

    Returns:
        JIT-compiled update function
    """

    def sg(z):
        """Stop gradient utility function."""
        return jax.lax.stop_gradient(z)

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

            # EMA of eigenvalue estimates with conjugate relationship
            ema_lambda_x_r = params['lambda_real']  # (Shape: (1,k))
            ema_lambda_x_i = params['lambda_imag']
            ema_lambda_y_r = params['lambda_real']  # Shared eigenvalues for left/right
            ema_lambda_y_i = -params['lambda_imag']  # Conjugate relationship!

            new_lambda_real = 0.5 * (lambda_x_r + lambda_y_r)
            new_lambda_imag = 0.5 * (lambda_x_i - lambda_y_i)
            lambda_loss_real = ((sg(new_lambda_real) - ema_lambda_x_r) ** 2).sum()
            lambda_loss_imag = ((sg(new_lambda_imag) - ema_lambda_x_i) ** 2).sum()
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
            f_x_real = - next_x_r + ema_lambda_x_r * x_r - ema_lambda_x_i * x_i  # (Shape: (n,k))
            f_x_imag = - next_x_i + ema_lambda_x_i * x_r + ema_lambda_x_r * x_i

            f_y_0_real = - y_r
            f_y_0_imag = - y_i
            f_y_residual_real = ema_lambda_y_r * y_r - ema_lambda_y_i * y_i
            f_y_residual_imag = ema_lambda_y_i * y_r + ema_lambda_y_r * y_i

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

            # Compute Lyapunov functions and their gradients with dual variables
            # 1. For x norm
            dual_x_norm = params['dual_x_norm']  # (Shape: (1,k))
            norm_x_error = norm_x_sq - 1  # (Shape: (1,k))
            V_x_norm = norm_x_error ** 2 / 2

            nabla_x_r_V_x_norm = 2 * (dual_x_norm + args.barrier * norm_x_error) * x_r  # (Shape: (n,k))
            nabla_x_i_V_x_norm = 2 * (dual_x_norm + args.barrier * norm_x_error) * x_i
            nabla_y_r_V_x_norm = jnp.zeros_like(y_r)
            nabla_y_i_V_x_norm = jnp.zeros_like(y_i)

            next_nabla_y_r_V_x_norm = jnp.zeros_like(next_y_r)
            next_nabla_y_i_V_x_norm = jnp.zeros_like(next_y_i)

            # 2. For y norm
            dual_y_norm = params['dual_y_norm']  # (Shape: (1,k))
            norm_y_error = norm_y_sq - 1  # (Shape: (1,k))
            V_y_norm = norm_y_error ** 2 / 2

            nabla_x_r_V_y_norm = jnp.zeros_like(x_r)  # (Shape: (n,k))
            nabla_x_i_V_y_norm = jnp.zeros_like(x_i)
            nabla_y_r_V_y_norm = 2 * (dual_y_norm + args.barrier * norm_y_error) * y_r
            nabla_y_i_V_y_norm = 2 * (dual_y_norm + args.barrier * norm_y_error) * y_i

            next_norm_y_error = next_norm_y_sq - 1  # (Shape: (1,k))

            next_nabla_y_r_V_y_norm = 2 * next_norm_y_error * next_y_r  # (Shape: (n,k))
            next_nabla_y_i_V_y_norm = 2 * next_norm_y_error * next_y_i

            # 3. For xy phase
            dual_xy_phase = params['dual_xy_phase']  # (Shape: (1,k))
            phase_xy = ip(y_r, x_i) - ip(y_i, x_r)  # (Shape: (1,k))
            V_xy_phase = phase_xy ** 2 / 2

            nabla_x_r_V_xy_phase = -(dual_xy_phase + args.barrier * phase_xy) * y_i  # (Shape: (n,k))
            nabla_x_i_V_xy_phase = (dual_xy_phase + args.barrier * phase_xy) * y_r
            nabla_y_r_V_xy_phase = (dual_xy_phase + args.barrier * phase_xy) * x_i
            nabla_y_i_V_xy_phase = -(dual_xy_phase + args.barrier * phase_xy) * x_r

            next_phase_xy = ip(next_y_r, next_x_i) - ip(next_y_i, next_x_r)  # (Shape: (1,k))

            next_nabla_y_r_V_xy_phase = (dual_xy_phase + args.barrier * next_phase_xy) * next_x_i  # (Shape: (n,k))
            next_nabla_y_i_V_xy_phase = -(dual_xy_phase + args.barrier * next_phase_xy) * next_x_r

            # 4. For crossed terms
            dual_x_corr_real = jnp.tril(params['dual_x_orth_real'], k=-1)  # (Shape: (k,k))
            dual_x_corr_imag = jnp.tril(params['dual_x_orth_imag'], k=-1)
            dual_y_corr_real = jnp.tril(params['dual_y_orth_real'], k=-1)
            dual_y_corr_imag = jnp.tril(params['dual_y_orth_imag'], k=-1)

            corr_xy_real = multi_ip(x_r, y_r) + multi_ip(x_i, y_i)  # (Shape: (k,k))
            corr_xy_imag = multi_ip(x_r, y_i) - multi_ip(x_i, y_r)

            corr_xy_real_lower = jnp.tril(corr_xy_real, k=-1)
            corr_xy_imag_lower = jnp.tril(corr_xy_imag, k=-1)

            corr_yx_real_lower = jnp.tril(corr_xy_real.T, k=-1)
            corr_yx_imag_lower = jnp.tril(corr_xy_imag.T, k=-1)

            V_xy_corr_real = jnp.sum(corr_xy_real_lower ** 2, -1).reshape(1, -1) / 2  # (Shape: (1,k))
            V_xy_corr_imag = jnp.sum(corr_xy_imag_lower ** 2, -1).reshape(1, -1) / 2
            V_yx_corr_real = jnp.sum(corr_yx_real_lower ** 2, -1).reshape(1, -1) / 2
            V_yx_corr_imag = jnp.sum(corr_yx_imag_lower ** 2, -1).reshape(1, -1) / 2
            V_xy_corr = V_xy_corr_real + V_xy_corr_imag + V_yx_corr_real + V_yx_corr_imag

            nabla_x_r_V_xy_corr_real = jnp.einsum('jk,ik->ij', dual_x_corr_real + args.barrier * corr_xy_real_lower, y_r)  # (Shape: (n,k))
            nabla_x_i_V_xy_corr_real = jnp.einsum('jk,ik->ij', dual_x_corr_real + args.barrier * corr_xy_real_lower, y_i)

            nabla_x_r_V_xy_corr_imag = jnp.einsum('jk,ik->ij', dual_x_corr_imag + args.barrier * corr_xy_imag_lower, y_i)
            nabla_x_i_V_xy_corr_imag = -jnp.einsum('jk,ik->ij', dual_x_corr_imag + args.barrier * corr_xy_imag_lower, y_r)

            nabla_y_r_V_xy_corr_real = jnp.einsum('jk,ik->ij', dual_y_corr_real + args.barrier * corr_yx_real_lower, x_r)
            nabla_y_i_V_xy_corr_real = jnp.einsum('jk,ik->ij', dual_y_corr_real + args.barrier * corr_yx_real_lower, x_i)

            nabla_y_r_V_xy_corr_imag = -jnp.einsum('jk,ik->ij', dual_y_corr_imag + args.barrier * corr_yx_imag_lower, x_i)
            nabla_y_i_V_xy_corr_imag = jnp.einsum('jk,ik->ij', dual_y_corr_imag + args.barrier * corr_yx_imag_lower, x_r)

            next_corr_xy_real = multi_ip(next_x_r, next_y_r) + multi_ip(next_x_i, next_y_i)  # (Shape: (k,k))
            next_corr_xy_imag = multi_ip(next_x_r, next_y_i) - multi_ip(next_x_i, next_y_r)

            next_corr_yx_real = jnp.tril(next_corr_xy_real.T, k=-1)
            next_corr_yx_imag = jnp.tril(next_corr_xy_imag.T, k=-1)

            next_nabla_y_r_V_xy_corr_real = jnp.einsum('jk,ik->ij', dual_y_corr_real + args.barrier * next_corr_yx_real, next_x_r)  # (Shape: (n,k))
            next_nabla_y_i_V_xy_corr_real = jnp.einsum('jk,ik->ij', dual_y_corr_real + args.barrier * next_corr_yx_real, next_x_i)
            next_nabla_y_r_V_xy_corr_imag = -jnp.einsum('jk,ik->ij', dual_y_corr_imag + args.barrier * next_corr_yx_imag, next_x_i)
            next_nabla_y_i_V_xy_corr_imag = jnp.einsum('jk,ik->ij', dual_y_corr_imag + args.barrier * next_corr_yx_imag, next_x_r)

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

            # Use gradient direction (without barrier scaling for duals approach)
            u_x_r = 1 * nabla_x_r_V  # (Shape: (n,k))
            u_x_i = 1 * nabla_x_i_V
            u_y_r = 1 * nabla_y_r_V
            u_y_i = 1 * nabla_y_i_V

            clf_loss_x_r_k = ip(x_r, sg(u_x_r))  # (Shape: (1,k))
            clf_loss_x_i_k = ip(x_i, sg(u_x_i))
            clf_loss_y_r_k = ip(y_r, sg(u_y_r))
            clf_loss_y_i_k = ip(y_i, sg(u_y_i))

            clf_loss_x_r = clf_loss_x_r_k.sum()  # Sum over all eigenvectors (Shape: ())
            clf_loss_x_i = clf_loss_x_i_k.sum()
            clf_loss_y_r = clf_loss_y_r_k.sum()
            clf_loss_y_i = clf_loss_y_i_k.sum()

            clf_loss = clf_loss_x_r + clf_loss_x_i + clf_loss_y_r + clf_loss_y_i

            # Compute dual losses (ascent on dual variables)
            dual_x_norm_loss = -(dual_x_norm * sg(norm_x_error)).sum()
            dual_y_norm_loss = -(dual_y_norm * sg(norm_y_error)).sum()
            dual_xy_phase_loss = -(dual_xy_phase * sg(phase_xy)).sum()
            dual_x_orth_real_loss = -jnp.sum(dual_x_corr_real * sg(corr_xy_real_lower))
            dual_x_orth_imag_loss = -jnp.sum(dual_x_corr_imag * sg(corr_xy_imag_lower))
            dual_y_orth_real_loss = -jnp.sum(dual_y_corr_real * sg(corr_yx_real_lower))
            dual_y_orth_imag_loss = -jnp.sum(dual_y_corr_imag * sg(corr_yx_imag_lower))
            dual_loss = (
                dual_x_norm_loss + dual_y_norm_loss + dual_xy_phase_loss
                + dual_x_orth_real_loss + dual_x_orth_imag_loss
                + dual_y_orth_real_loss + dual_y_orth_imag_loss
            )

            # Total loss
            total_loss = graph_loss + clf_loss + args.chirality_factor * chirality_loss + lambda_loss + dual_loss

            # Auxiliary metrics (average over eigenvector pairs for logging)
            aux = {
                'total_loss': total_loss,
                'graph_loss': graph_loss,
                'clf_loss': clf_loss,
                'chirality_loss': chirality_loss,
                'dual_loss': dual_loss,
                'graph_loss_x_real': graph_loss_x_real.sum(),
                'graph_loss_x_imag': graph_loss_x_imag.sum(),
                'clf_loss_x_real': clf_loss_x_r,
                'clf_loss_x_imag': clf_loss_x_i,
                'graph_loss_y_real': graph_loss_y_real.sum(),
                'graph_loss_y_imag': graph_loss_y_imag.sum(),
                'clf_loss_y_real': clf_loss_y_r,
                'clf_loss_y_imag': clf_loss_y_i,
                'lambda_x_real': ema_lambda_x_r.mean(),
                'lambda_x_imag': ema_lambda_x_i.mean(),
                'lambda_y_real': ema_lambda_y_r.mean(),
                'lambda_y_imag': ema_lambda_y_i.mean(),
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

        # Get grad norm
        grads_flat, _ = jax.tree_util.tree_flatten(grads)
        grads_vector = jnp.concatenate([jnp.ravel(g) for g in grads_flat])
        grad_norm = jnp.linalg.norm(grads_vector)
        aux['grad_norm'] = grad_norm

        # Clip gradients if necessary
        max_norm = 1.0
        grads = jax.tree_util.tree_map(
            lambda g: g * (max_norm / jnp.maximum(grad_norm, max_norm)), grads
        )

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

    return update_encoder


# Entry point for running this learner directly
if __name__ == "__main__":
    import tyro
    from src.config.ded_clf import Args
    from rep_algos.shared_training import learn_eigenvectors
    import rep_algos.duals_learner as duals_learner

    args = tyro.cli(Args)
    args.exp_name = "duals"
    learn_eigenvectors(args, duals_learner)
