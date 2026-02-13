"""
CLF (Control Lyapunov Function) learner with multiple constraint error approximation modes.

Supports 4 constraint approximation modes:
- "ema": EMA approximation (current approach)
- "two_batch": Unbiased with two independent batches
- "single_batch": Biased with single batch (reuse batch1 for constraints)
- "same_episodes": Intermediate bias - two batches from same episodes

Usage:
    from rep_algos.shared_training import learn_eigenvectors
    from rep_algos import clf_learner_multi_mode
    from src.config.ded_clf import Args

    args = tyro.cli(Args)
    args.constraint_mode = "two_batch"  # or "ema", "single_batch", "same_episodes"
    learn_eigenvectors(args, clf_learner_multi_mode)
"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def init_params(encoder_initial_params, args):
    """
    Initialize parameters for the CLF learner.

    Args:
        encoder_initial_params: Initial parameters for the encoder network
        args: Training arguments

    Returns:
        Dictionary containing initial parameters
    """
    k = args.num_eigenvector_pairs
    params = {
        'encoder': encoder_initial_params,

        # EMA estimates for eigenvalues (always needed)
        'lambda_real': jnp.ones((k,)),
        'lambda_imag': jnp.zeros((k,)),
    }

    # Add EMA constraint estimators only for EMA mode
    if args.constraint_mode == "ema":
        params.update({
            'norm_x_ema': jnp.zeros((k,)),
            'norm_y_ema': jnp.zeros((k,)),
            'phase_xy_ema': jnp.zeros((k,)),
            'corr_xy_real_ema': jnp.zeros((k, k)),
            'corr_xy_imag_ema': jnp.zeros((k, k)),
        })

    return params


def get_optimizer_masks(args):
    """
    Get optimizer masks for different parameter groups.

    Returns:
        Tuple of (encoder_mask, other_mask)
    """
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

    # Add EMA masks only for EMA mode
    if args.constraint_mode == "ema":
        encoder_mask.update({
            'norm_x_ema': False,
            'norm_y_ema': False,
            'phase_xy_ema': False,
            'corr_xy_real_ema': False,
            'corr_xy_imag_ema': False,
        })
        other_mask.update({
            'norm_x_ema': True,
            'norm_y_ema': True,
            'phase_xy_ema': True,
            'corr_xy_real_ema': True,
            'corr_xy_imag_ema': True,
        })

    return encoder_mask, other_mask


def create_update_function(encoder, args):
    """
    Create the JIT-compiled update function for the CLF learner.

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
        next_state_coords_batch_2: jnp.ndarray,
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
            # ----------------------------------------------------------------
            # 1. Forward Pass & Feature Extraction
            # ----------------------------------------------------------------
            encoder_params = params['encoder']
            features_1 = encoder.apply(encoder_params, state_coords_batch)[0]
            features_2 = encoder.apply(encoder_params, state_coords_batch_2)[0]
            next_features = encoder.apply(encoder_params, next_state_coords_batch)[0]
            next_features_2 = encoder.apply(encoder_params, next_state_coords_batch_2)[0]

            # Extract right/left eigenvectors (Batch 1)
            x_r, x_i = features_1['right_real'], features_1['right_imag']
            y_r, y_i = features_1['left_real'], features_1['left_imag']

            # Extract right/left eigenvectors (Batch 2)
            x2_r, x2_i = features_2['right_real'], features_2['right_imag']
            y2_r, y2_i = features_2['left_real'], features_2['left_imag']

            # Extract next step vectors (Batch 1)
            next_x_r, next_x_i = next_features['right_real'], next_features['right_imag']
            next_y_r, next_y_i = next_features['left_real'], next_features['left_imag']

            # Extract next step vectors (Batch 2)
            next_x2_r, next_x2_i = next_features_2['right_real'], next_features_2['right_imag']
            next_y2_r, next_y2_i = next_features_2['left_real'], next_features_2['left_imag']

            # ----------------------------------------------------------------
            # 2. Extract Parameters
            # ----------------------------------------------------------------
            ema_lambda_r = params['lambda_real'].reshape(1, -1)
            ema_lambda_i = params['lambda_imag'].reshape(1, -1)

            # ----------------------------------------------------------------
            # 3. Update Lambda (Eigenvalue) Estimators
            # ----------------------------------------------------------------
            
            # Compute unnormalized Rayleigh quotients for x and y (Shape: (1,k))
            lambda_x_r = ip(x_r, next_x_r) + ip(x_i, next_x_i)
            lambda_x_i = ip(x_r, next_x_i) - ip(x_i, next_x_r)
            lambda_y_r = ip(y_r, next_y_r) + ip(y_i, next_y_i)
            lambda_y_i = ip(y_r, next_y_i) - ip(y_i, next_y_r)

            # Compute new lambda estimates by averaging x and y contributions
            new_lambda_real = 0.5 * (lambda_x_r + lambda_y_r)
            new_lambda_imag = 0.5 * (lambda_x_i - lambda_y_i)
            
            # EMA loss for eigenvalues
            lambda_loss_real = ((sg(new_lambda_real) - ema_lambda_r) ** 2).sum()
            lambda_loss_imag = ((sg(new_lambda_imag) - ema_lambda_i) ** 2).sum()
            lambda_loss = lambda_loss_real + lambda_loss_imag

            # ----------------------------------------------------------------
            # 4. Compute Constraints & Update EMA Estimators
            # ----------------------------------------------------------------
            # Norms
            norm_x_sq = ip(x_r, x_r) + ip(x_i, x_i)
            norm_x2_sq = ip(x2_r, x2_r) + ip(x2_i, x2_i)
            norm_y_sq = ip(y_r, y_r) + ip(y_i, y_i)
            norm_y2_sq = ip(y2_r, y2_r) + ip(y2_i, y2_i)
            next_norm_y_sq = ip(next_y_r, next_y_r) + ip(next_y_i, next_y_i)
            next_norm_y2_sq = ip(next_y2_r, next_y2_r) + ip(next_y2_i, next_y2_i)

            # EMA Loss for Norms (only for EMA mode)
            if args.constraint_mode == "ema":
                loss_norm_x_ema = ((sg(0.5 * (norm_x_sq + norm_x2_sq)) - params['norm_x_ema'].reshape(1, -1)) ** 2).sum()
                loss_norm_y_ema = ((sg(0.5 * (norm_y_sq + norm_y2_sq)) - params['norm_y_ema'].reshape(1, -1)) ** 2).sum()
            else:
                loss_norm_x_ema = 0.0
                loss_norm_y_ema = 0.0

            # Phase
            phase_xy = ip(y_r, x_i) - ip(y_i, x_r)
            phase_xy2 = ip(y2_r, x2_i) - ip(y2_i, x2_r)

            # EMA Loss for Phase (only for EMA mode)
            if args.constraint_mode == "ema":
                loss_phase_xy_ema = ((sg(0.5 * (phase_xy + phase_xy2)) - params['phase_xy_ema'].reshape(1, -1)) ** 2).sum()
            else:
                loss_phase_xy_ema = 0.0

            # Correlations (Crossed terms)
            # Batch 1
            cross_xryr = multi_ip(x_r, y_r)
            cross_xiyi = multi_ip(x_i, y_i)
            cross_xryi = multi_ip(x_r, y_i)
            cross_xiryr = multi_ip(x_i, y_r)
            
            corr_xy_real = cross_xryr + cross_xiyi
            corr_xy_imag = -cross_xryi + cross_xiryr

            # Batch 2
            cross_xryr2 = multi_ip(x2_r, y2_r)
            cross_xiyi2 = multi_ip(x2_i, y2_i)
            cross_xryi2 = multi_ip(x2_r, y2_i)
            cross_xiryr2 = multi_ip(x2_i, y2_r)

            corr_xy2_real = cross_xryr2 + cross_xiyi2
            corr_xy2_imag = -cross_xryi2 + cross_xiryr2

            # EMA Loss for Correlations (only for EMA mode)
            if args.constraint_mode == "ema":
                loss_corr_xy_real = ((sg(0.5 * (corr_xy_real + corr_xy2_real)) - params['corr_xy_real_ema']) ** 2).sum()
                loss_corr_xy_imag = ((sg(0.5 * (corr_xy_imag + corr_xy2_imag)) - params['corr_xy_imag_ema']) ** 2).sum()
            else:
                loss_corr_xy_real = 0.0
                loss_corr_xy_imag = 0.0

            # Collect all estimator losses
            constraint_estimator_loss = (
                loss_norm_x_ema + loss_norm_y_ema +
                loss_phase_xy_ema +
                loss_corr_xy_real + loss_corr_xy_imag
            )

             # ----------------------------------------------------------------
            # 5. Graph Loss (Dynamics)
            # ----------------------------------------------------------------
            f_x_real = - next_x_r + ema_lambda_r * x_r - ema_lambda_i * x_i
            f_x_imag = - next_x_i + ema_lambda_i * x_r + ema_lambda_r * x_i

            f_y_0_real = - y_r
            f_y_0_imag = - y_i
            f_y_residual_real = ema_lambda_r * y_r + ema_lambda_i * y_i
            f_y_residual_imag = -ema_lambda_i * y_r + ema_lambda_r * y_i

            graph_loss_x_real = ip(x_r, sg(f_x_real))
            graph_loss_x_imag = ip(x_i, sg(f_x_imag))
            graph_loss_x = graph_loss_x_real + graph_loss_x_imag

            graph_loss_y_real = ip(next_y_r, sg(f_y_0_real)) + ip(y_r, sg(f_y_residual_real))
            graph_loss_y_imag = ip(next_y_i, sg(f_y_0_imag)) + ip(y_i, sg(f_y_residual_imag))
            graph_loss_y = graph_loss_y_real + graph_loss_y_imag

            graph_loss = (graph_loss_x + graph_loss_y).sum()

            # Chirality Loss
            x_i_normalized = x_i / jnp.sqrt(norm_x_sq + 1e-6)
            chirality_x = ip(sg(x_i_normalized**2), x_i_normalized)
            chirality_loss = ((chirality_x)**2).sum()

            # ----------------------------------------------------------------
            # 6. Lyapunov Functions & Gradients
            # ----------------------------------------------------------------

            # 1. For x norm: <x_k,x_k> should be 1
            norm_x_error = norm_x_sq - 1  # (Shape: (1,k))
            norm_x2_error = norm_x2_sq - 1

            # Select coefficient for gradients based on constraint mode
            if args.constraint_mode == "ema":
                coef_x_norm = params['norm_x_ema'].reshape(1, -1) - 1  # EMA-based
                V_x_norm = coef_x_norm ** 2 / 2
            elif args.constraint_mode == "single_batch":
                coef_x_norm = norm_x_error  # Batch1 error
                V_x_norm = coef_x_norm ** 2 / 2
            else:  # two_batch, same_episodes
                coef_x_norm = norm_x2_error  # Batch2 error
                V_x_norm = coef_x_norm ** 2 / 2

            nabla_x_r_V_x_norm = 2 * coef_x_norm * x_r
            nabla_x_i_V_x_norm = 2 * coef_x_norm * x_i
            nabla_y_r_V_x_norm = jnp.zeros_like(y_r)
            nabla_y_i_V_x_norm = jnp.zeros_like(y_i)

            next_nabla_y_r_V_x_norm = jnp.zeros_like(next_y_r)
            next_nabla_y_i_V_x_norm = jnp.zeros_like(next_y_i)

            # 2. For y norm: <y_k,y_k> should be 1
            norm_y_error = norm_y_sq - 1  # (Shape: (1,k))
            norm_y2_error = norm_y2_sq - 1
            next_norm_y_error = next_norm_y_sq - 1
            next_norm_y2_error = next_norm_y2_sq - 1

            if args.constraint_mode == "ema":
                coef_y_norm = params['norm_y_ema'].reshape(1, -1) - 1  # EMA-based
                V_y_norm = coef_y_norm ** 2 / 2
            elif args.constraint_mode == "single_batch":
                coef_y_norm = norm_y_error  # Batch1 error
                V_y_norm = coef_y_norm ** 2 / 2
            else:  # two_batch, same_episodes
                coef_y_norm = norm_y2_error  # Batch2 error
                V_y_norm = coef_y_norm ** 2 / 2

            nabla_x_r_V_y_norm = jnp.zeros_like(x_r)
            nabla_x_i_V_y_norm = jnp.zeros_like(x_i)
            nabla_y_r_V_y_norm = 2 * coef_y_norm * y_r
            nabla_y_i_V_y_norm = 2 * coef_y_norm * y_i
            next_nabla_y_r_V_y_norm = 2 * coef_y_norm * next_y_r
            next_nabla_y_i_V_y_norm = 2 * coef_y_norm * next_y_i

            # 3. For xy phase: <y_k,x_k> should be real
            phase_xy = ip(y_r, x_i) - ip(y_i, x_r)  # (Shape: (1,k))
            phase_xy2 = ip(y2_r, x2_i) - ip(y2_i, x2_r)
            next_phase_xy = ip(next_y_r, next_x_i) - ip(next_y_i, next_x_r)
            next_phase_xy2 = ip(next_y2_r, next_x2_i) - ip(next_y2_i, next_x2_r)

            if args.constraint_mode == "ema":
                coef_xy_phase = params['phase_xy_ema'].reshape(1, -1)  # EMA-based
                V_xy_phase = coef_xy_phase ** 2 / 2
            elif args.constraint_mode == "single_batch":
                coef_xy_phase = phase_xy  # Batch1 error
                V_xy_phase = coef_xy_phase ** 2 / 2
            else:  # two_batch, same_episodes
                coef_xy_phase = phase_xy2  # Batch2 error
                V_xy_phase = coef_xy_phase ** 2 / 2

            nabla_x_r_V_xy_phase = - coef_xy_phase * y_i
            nabla_x_i_V_xy_phase = coef_xy_phase * y_r
            nabla_y_r_V_xy_phase = coef_xy_phase * x_i
            nabla_y_i_V_xy_phase = - coef_xy_phase * x_r
            next_nabla_y_r_V_xy_phase = coef_xy_phase * next_x_i
            next_nabla_y_i_V_xy_phase = - coef_xy_phase * next_x_r

            # 4. For crossed terms: <y_j,x_k> should be 0 for j!=k
            cross_xryr = multi_ip(x_r, y_r)
            cross_xiyi = multi_ip(x_i, y_i)
            cross_xryi = multi_ip(x_r, y_i)
            cross_xiryr = multi_ip(x_i, y_r)

            corr_xy_real = cross_xryr + cross_xiyi  # (Shape: (k,k)) (First index: x, second index: y)
            corr_xy_imag = -cross_xryi + cross_xiryr

            corr_xy_real_lower = jnp.tril(corr_xy_real, k=-1)
            corr_xy_imag_lower = jnp.tril(corr_xy_imag, k=-1)

            corr_yx_real_lower = jnp.tril(corr_xy_real.T, k=-1)
            corr_yx_imag_lower = jnp.tril(corr_xy_imag.T, k=-1)

            cross_xryr2 = multi_ip(x2_r, y2_r)
            cross_xiyi2 = multi_ip(x2_i, y2_i)
            cross_xryi2 = multi_ip(x2_r, y2_i)
            cross_xiryr2 = multi_ip(x2_i, y2_r)

            cross_xy2_real = cross_xryr2 + cross_xiyi2
            cross_xy2_imag = -cross_xryi2 + cross_xiryr2

            corr_xy2_real_lower = jnp.tril(cross_xy2_real, k=-1)
            corr_xy2_imag_lower = jnp.tril(cross_xy2_imag, k=-1)

            corr_yx2_real_lower = jnp.tril(cross_xy2_real.T, k=-1)
            corr_yx2_imag_lower = jnp.tril(cross_xy2_imag.T, k=-1)

            # Select correlation coefficients based on mode
            if args.constraint_mode == "ema":
                coef_corr_xy_real_lower = jnp.tril(params['corr_xy_real_ema'], k=-1)  # EMA-based
                coef_corr_xy_imag_lower = jnp.tril(params['corr_xy_imag_ema'], k=-1)
                coef_corr_yx_real_lower = jnp.tril(params['corr_xy_real_ema'].T, k=-1)
                coef_corr_yx_imag_lower = jnp.tril(params['corr_xy_imag_ema'].T, k=-1)
            elif args.constraint_mode == "single_batch":
                coef_corr_xy_real_lower = corr_xy_real_lower  # Batch1
                coef_corr_xy_imag_lower = corr_xy_imag_lower
                coef_corr_yx_real_lower = corr_yx_real_lower
                coef_corr_yx_imag_lower = corr_yx_imag_lower
            else:  # two_batch, same_episodes
                coef_corr_xy_real_lower = corr_xy2_real_lower  # Batch2
                coef_corr_xy_imag_lower = corr_xy2_imag_lower
                coef_corr_yx_real_lower = corr_yx2_real_lower
                coef_corr_yx_imag_lower = corr_yx2_imag_lower

            V_xy_corr_real = jnp.sum(coef_corr_xy_real_lower ** 2, -1).reshape(1, -1) / 2  # (Shape: (1,k))
            V_xy_corr_imag = jnp.sum(coef_corr_xy_imag_lower ** 2, -1).reshape(1, -1) / 2
            V_yx_corr_real = jnp.sum(coef_corr_yx_real_lower ** 2, -1).reshape(1, -1) / 2
            V_yx_corr_imag = jnp.sum(coef_corr_yx_imag_lower ** 2, -1).reshape(1, -1) / 2
            V_xy_corr = V_xy_corr_real + V_xy_corr_imag + V_yx_corr_real + V_yx_corr_imag

            nabla_x_r_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', coef_corr_xy_real_lower, y_r)
            nabla_x_i_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', coef_corr_xy_real_lower, y_i)
            nabla_x_r_V_xy_corr_imag = -2 * jnp.einsum('jk,ik->ij', coef_corr_xy_imag_lower, y_i)
            nabla_x_i_V_xy_corr_imag = 2 * jnp.einsum('jk,ik->ij', coef_corr_xy_imag_lower, y_r)
            nabla_y_r_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower, x_r)
            nabla_y_i_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower, x_i)
            nabla_y_r_V_xy_corr_imag = 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower, x_i)
            nabla_y_i_V_xy_corr_imag = -2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower, x_r)
            next_nabla_y_r_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower, next_x_r)
            next_nabla_y_i_V_xy_corr_real = 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower, next_x_i)
            next_nabla_y_r_V_xy_corr_imag = 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower, next_x_i)
            next_nabla_y_i_V_xy_corr_imag = -2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower, next_x_r)

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
            total_loss = (
                graph_loss 
                + clf_loss 
                + args.chirality_factor * chirality_loss 
                + lambda_loss 
                + constraint_estimator_loss
            )

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

        # Gradient clipping
        max_norm = 1.0

        if args.use_global_grad_clip:
            # Global gradient clipping (original behavior)
            # Clip all parameters (encoder + lambda + EMA) together using global norm
            grads_flat, grads_tree = jax.tree_util.tree_flatten(grads)
            grads_vector = jnp.concatenate([jnp.ravel(g) for g in grads_flat])
            grad_norm = jnp.linalg.norm(grads_vector)
            aux['grad_norm'] = grad_norm

            clip_factor = max_norm / jnp.maximum(grad_norm, max_norm)
            grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)

        else:
            # Separate gradient clipping (new behavior)
            # Clip encoder gradients by norm, other parameters element-wise
            encoder_grads_flat, _ = jax.tree_util.tree_flatten(grads['encoder'])
            encoder_grads_vector = jnp.concatenate([jnp.ravel(g) for g in encoder_grads_flat])
            encoder_grad_norm = jnp.linalg.norm(encoder_grads_vector)
            aux['grad_norm'] = encoder_grad_norm

            encoder_clip_factor = max_norm / jnp.maximum(encoder_grad_norm, max_norm)

            clipped_grads = {
                'encoder': jax.tree_util.tree_map(lambda g: g * encoder_clip_factor, grads['encoder']),
                'lambda_real': jnp.clip(grads['lambda_real'], -1.0, 1.0),
                'lambda_imag': jnp.clip(grads['lambda_imag'], -1.0, 1.0),
            }

            # Add EMA gradient clipping only for EMA mode
            if args.constraint_mode == "ema":
                clipped_grads.update({
                    'norm_x_ema': jnp.clip(grads['norm_x_ema'], -1.0, 1.0),
                    'norm_y_ema': jnp.clip(grads['norm_y_ema'], -1.0, 1.0),
                    'phase_xy_ema': jnp.clip(grads['phase_xy_ema'], -1.0, 1.0),
                    'corr_xy_real_ema': jnp.clip(grads['corr_xy_real_ema'], -1.0, 1.0),
                    'corr_xy_imag_ema': jnp.clip(grads['corr_xy_imag_ema'], -1.0, 1.0),
                })

            grads = clipped_grads

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
    import rep_algos.clf_learner as clf_learner

    args = tyro.cli(Args)
    args.exp_name = "clf"
    learn_eigenvectors(args, clf_learner)
