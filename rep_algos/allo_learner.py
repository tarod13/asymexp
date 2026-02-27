"""
ALLO (Asymmetric Laplacian Learning Optimization) learner module.

This module wraps the CLF-based ALLO algorithm from allo_complex_clf.py as a
proper learner module compatible with the shared_training.py infrastructure.

The ALLO algorithm learns complex eigenvectors of the non-symmetric Laplacian
using Control Lyapunov Functions (CLF) for norm/biorthogonality constraints.

Unlike clf_learner_multi_mode.py, ALLO does NOT use the second batch's
next-state coordinates — it enforces constraints using the first batch only,
with a graph loss that separates eigenvalue estimation from CLF enforcement.

Usage:
    from rep_algos.shared_training import learn_eigenvectors
    from rep_algos import allo_learner
    from src.config.ded_clf import Args
    import tyro

    args = tyro.cli(Args)
    learn_eigenvectors(args, allo_learner)
"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def init_params(encoder_initial_params, args):
    """
    Initialize parameters for the ALLO learner.

    Returns a dict with encoder weights and EMA eigenvalue estimates.
    """
    k = args.num_eigenvector_pairs
    return {
        'encoder': encoder_initial_params,
        'lambda_real': jnp.ones((k,)),
        'lambda_imag': jnp.zeros((k,)),
    }


def get_optimizer_masks(args):
    """
    Return (encoder_mask, other_mask) for the two optimizer groups.
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
    return encoder_mask, other_mask


def create_update_function(encoder, args):
    """
    Create the JIT-compiled update function for the ALLO learner.

    The returned function accepts 6 positional arguments to match the
    shared_training.py calling convention:
        encoder_state, coords_batch, next_coords_batch,
        coords_batch_2, next_coords_batch_2, state_weighting

    Note: next_coords_batch_2 is accepted but not used by this algorithm.
    """

    def sg(z):
        return jax.lax.stop_gradient(z)

    @jax.jit
    def update_encoder(
        encoder_state: TrainState,
        state_coords_batch: jnp.ndarray,
        next_state_coords_batch: jnp.ndarray,
        state_coords_batch_2: jnp.ndarray,
        next_state_coords_batch_2: jnp.ndarray,   # accepted, unused
        state_weighting: jnp.ndarray,
    ):
        def ip(a, b):
            """Weighted inner product; returns shape (1, k)."""
            return jnp.mean(state_weighting * a * b, axis=0, keepdims=True)

        def multi_ip(a, b):
            """Weighted outer inner-product; returns shape (k, k)."""
            return jnp.einsum('ij,ik->jk', state_weighting * a, b) / a.shape[0]

        def encoder_loss(params):
            encoder_params = params['encoder']

            # Forward passes
            features_1    = encoder.apply(encoder_params, state_coords_batch)[0]
            next_features = encoder.apply(encoder_params, next_state_coords_batch)[0]

            # Unpack eigenvector components
            x_r = features_1['right_real']   # (n, k)
            x_i = features_1['right_imag']
            y_r = features_1['left_real']
            y_i = features_1['left_imag']

            next_x_r = next_features['right_real']
            next_x_i = next_features['right_imag']
            next_y_r = next_features['left_real']
            next_y_i = next_features['left_imag']

            # ---- EMA eigenvalue estimates ----
            ema_lambda_r = params['lambda_real'].reshape(1, -1)   # (1, k)
            ema_lambda_i = params['lambda_imag'].reshape(1, -1)

            # Rayleigh quotients for updating EMA targets
            lambda_x_r = ip(x_r, next_x_r) + ip(x_i, next_x_i)
            lambda_x_i = ip(x_r, next_x_i) - ip(x_i, next_x_r)
            lambda_y_r = ip(y_r, next_y_r) + ip(y_i, next_y_i)
            lambda_y_i = ip(y_r, next_y_i) - ip(y_i, next_y_r)

            new_lambda_real = 0.5 * (lambda_x_r + lambda_y_r)
            new_lambda_imag = 0.5 * (lambda_x_i - lambda_y_i)
            lambda_loss = (
                ((sg(new_lambda_real) - ema_lambda_r) ** 2).sum()
                + ((sg(new_lambda_imag) - ema_lambda_i) ** 2).sum()
            )

            # ---- Squared norms ----
            norm_x_sq = ip(x_r, x_r) + ip(x_i, x_i)          # (1, k)
            norm_y_sq = ip(y_r, y_r) + ip(y_i, y_i)
            next_norm_y_sq = ip(next_y_r, next_y_r) + ip(next_y_i, next_y_i)

            # ---- Graph losses (eigenvalue equations) ----
            f_x_real = -next_x_r + ema_lambda_r * x_r - ema_lambda_i * x_i   # (n, k)
            f_x_imag = -next_x_i + ema_lambda_i * x_r + ema_lambda_r * x_i

            f_y_0_real      =  -y_r
            f_y_0_imag      =  -y_i
            f_y_res_real    =   ema_lambda_r * y_r + ema_lambda_i * y_i
            f_y_res_imag    =  -ema_lambda_i * y_r + ema_lambda_r * y_i

            graph_loss_x = (ip(x_r, sg(f_x_real)) + ip(x_i, sg(f_x_imag))).sum()
            graph_loss_y = (
                ip(next_y_r, sg(f_y_0_real)) + ip(y_r, sg(f_y_res_real))
                + ip(next_y_i, sg(f_y_0_imag)) + ip(y_i, sg(f_y_res_imag))
            ).sum()
            graph_loss = graph_loss_x + graph_loss_y

            # ---- Chirality loss (encourages non-zero imaginary part) ----
            x_i_norm = x_i / jnp.sqrt(norm_x_sq)
            chirality_loss = (ip(sg(x_i_norm ** 2), x_i_norm) ** 2).sum()

            # ---- Lyapunov functions ----
            # 1. x norm: <x_k, x_k> = 1
            norm_x_err = norm_x_sq - 1                             # (1, k)
            V_x_norm   = norm_x_err ** 2 / 2

            nabla_xr_Vx = 2 * norm_x_err * x_r
            nabla_xi_Vx = 2 * norm_x_err * x_i
            nabla_yr_Vx = jnp.zeros_like(y_r)
            nabla_yi_Vx = jnp.zeros_like(y_i)
            next_nabla_yr_Vx = jnp.zeros_like(next_y_r)
            next_nabla_yi_Vx = jnp.zeros_like(next_y_i)

            # 2. y norm: <y_k, y_k> = 1
            norm_y_err      = norm_y_sq - 1
            next_norm_y_err = next_norm_y_sq - 1
            V_y_norm        = norm_y_err ** 2 / 2

            nabla_xr_Vy = jnp.zeros_like(x_r)
            nabla_xi_Vy = jnp.zeros_like(x_i)
            nabla_yr_Vy = 2 * norm_y_err * y_r
            nabla_yi_Vy = 2 * norm_y_err * y_i
            next_nabla_yr_Vy = 2 * next_norm_y_err * next_y_r
            next_nabla_yi_Vy = 2 * next_norm_y_err * next_y_i

            # 3. xy phase: Im(<y_k, x_k>) = 0
            phase_xy = ip(y_r, x_i) - ip(y_i, x_r)               # (1, k)
            V_xy_phase = phase_xy ** 2 / 2

            nabla_xr_Vp  = -phase_xy * y_i
            nabla_xi_Vp  =  phase_xy * y_r
            nabla_yr_Vp  =  phase_xy * x_i
            nabla_yi_Vp  = -phase_xy * x_r

            next_phase_xy = ip(next_y_r, next_x_i) - ip(next_y_i, next_x_r)
            next_nabla_yr_Vp =  next_phase_xy * next_x_i
            next_nabla_yi_Vp = -next_phase_xy * next_x_r

            # 4. Off-diagonal: <y_j, x_k> = 0 for j ≠ k
            cxy_r = multi_ip(x_r, y_r) + multi_ip(x_i, y_i)      # (k, k)
            cxy_i = -multi_ip(x_r, y_i) + multi_ip(x_i, y_r)

            cxy_r_lo = jnp.tril(cxy_r, k=-1)
            cxy_i_lo = jnp.tril(cxy_i, k=-1)
            cyx_r_lo = jnp.tril(cxy_r.T, k=-1)
            cyx_i_lo = jnp.tril(cxy_i.T, k=-1)

            V_c = (
                jnp.sum(cxy_r_lo ** 2, -1).reshape(1, -1)
                + jnp.sum(cxy_i_lo ** 2, -1).reshape(1, -1)
                + jnp.sum(cyx_r_lo ** 2, -1).reshape(1, -1)
                + jnp.sum(cyx_i_lo ** 2, -1).reshape(1, -1)
            ) / 2

            nabla_xr_Vc = 2 * jnp.einsum('jk,ik->ij', cxy_r_lo, y_r)
            nabla_xi_Vc = (
                2 * jnp.einsum('jk,ik->ij', cxy_r_lo, y_i)
                - 2 * jnp.einsum('jk,ik->ij', cxy_i_lo, y_i)
            )
            nabla_xr_Vc_imag_part = -2 * jnp.einsum('jk,ik->ij', cxy_i_lo, y_i)
            nabla_xi_Vc_imag_part =  2 * jnp.einsum('jk,ik->ij', cxy_i_lo, y_r)

            nabla_xr_Vc = (
                2 * jnp.einsum('jk,ik->ij', cxy_r_lo, y_r)
                + nabla_xr_Vc_imag_part
            )
            nabla_xi_Vc = (
                2 * jnp.einsum('jk,ik->ij', cxy_r_lo, y_i)
                + nabla_xi_Vc_imag_part
            )
            nabla_yr_Vc = (
                2 * jnp.einsum('jk,ik->ij', cyx_r_lo, x_r)
                + 2 * jnp.einsum('jk,ik->ij', cyx_i_lo, x_i)
            )
            nabla_yi_Vc = (
                2 * jnp.einsum('jk,ik->ij', cyx_r_lo, x_i)
                - 2 * jnp.einsum('jk,ik->ij', cyx_i_lo, x_r)
            )

            next_cxy_r = multi_ip(next_x_r, next_y_r) + multi_ip(next_x_i, next_y_i)
            next_cxy_i = -multi_ip(next_x_r, next_y_i) + multi_ip(next_x_i, next_y_r)
            next_cyx_r_lo = jnp.tril(next_cxy_r.T, k=-1)
            next_cyx_i_lo = jnp.tril(next_cxy_i.T, k=-1)

            next_nabla_yr_Vc = (
                2 * jnp.einsum('jk,ik->ij', next_cyx_r_lo, next_x_r)
                + 2 * jnp.einsum('jk,ik->ij', next_cyx_i_lo, next_x_i)
            )
            next_nabla_yi_Vc = (
                2 * jnp.einsum('jk,ik->ij', next_cyx_r_lo, next_x_i)
                - 2 * jnp.einsum('jk,ik->ij', next_cyx_i_lo, next_x_r)
            )

            # ---- Total Lyapunov function V ----
            V = V_x_norm + V_y_norm + V_xy_phase + V_c            # (1, k)

            nabla_xr_V = nabla_xr_Vx + nabla_xr_Vy + nabla_xr_Vp + nabla_xr_Vc
            nabla_xi_V = nabla_xi_Vx + nabla_xi_Vy + nabla_xi_Vp + nabla_xi_Vc
            nabla_yr_V = nabla_yr_Vx + nabla_yr_Vy + nabla_yr_Vp + nabla_yr_Vc
            nabla_yi_V = nabla_yi_Vx + nabla_yi_Vy + nabla_yi_Vp + nabla_yi_Vc

            next_nabla_yr_V = (
                next_nabla_yr_Vx + next_nabla_yr_Vy
                + next_nabla_yr_Vp + next_nabla_yr_Vc
            )
            next_nabla_yi_V = (
                next_nabla_yi_Vx + next_nabla_yi_Vy
                + next_nabla_yi_Vp + next_nabla_yi_Vc
            )

            norm_nabla_V_sq = (
                ip(nabla_xr_V, nabla_xr_V)
                + ip(nabla_xi_V, nabla_xi_V)
                + ip(nabla_yr_V, nabla_yr_V)
                + ip(nabla_yi_V, nabla_yi_V)
            )

            # ---- CLF loss: enforce dV/dt <= -lambda_x * V ----
            f_xr_dot_nabla = ip(f_x_real, nabla_xr_V)
            f_xi_dot_nabla = ip(f_x_imag, nabla_xi_V)
            f_yr_dot_nabla = (
                ip(f_y_0_real, next_nabla_yr_V)
                + ip(f_y_res_real, nabla_yr_V)
            )
            f_yi_dot_nabla = (
                ip(f_y_0_imag, next_nabla_yi_V)
                + ip(f_y_res_imag, nabla_yi_V)
            )

            f_dot_nabla = (
                f_xr_dot_nabla + f_xi_dot_nabla
                + f_yr_dot_nabla + f_yi_dot_nabla
            )

            clf_num = f_dot_nabla + args.lambda_x * V
            barrier = jnp.maximum(0.0, clf_num) / (norm_nabla_V_sq + 1e-8)

            u_xr = barrier * nabla_xr_V
            u_xi = barrier * nabla_xi_V
            u_yr = barrier * nabla_yr_V
            u_yi = barrier * nabla_yi_V

            clf_loss = (
                ip(x_r, sg(u_xr)) + ip(x_i, sg(u_xi))
                + ip(y_r, sg(u_yr)) + ip(y_i, sg(u_yi))
            ).sum()

            total_loss = (
                graph_loss
                + clf_loss
                + args.chirality_factor * chirality_loss
                + lambda_loss
            )

            aux = {
                'total_loss':        total_loss,
                'graph_loss':        graph_loss,
                'clf_loss':          clf_loss,
                'chirality_loss':    chirality_loss,
                'lambda_x_real':     ema_lambda_r.mean(),
                'lambda_x_imag':     ema_lambda_i.mean(),
                'new_lambda_x_real': lambda_x_r.mean(),
                'new_lambda_x_imag': lambda_x_i.mean(),
                'V_x_norm':          V_x_norm.mean(),
                'V_y_norm':          V_y_norm.mean(),
                'V_xy_phase':        V_xy_phase.mean(),
                'V_xy_corr':         V_c.mean(),
                'norm_x_sq':         norm_x_sq.mean(),
                'norm_y_sq':         norm_y_sq.mean(),
                'barrier':           barrier.mean(),
            }
            return total_loss, aux

        (total_loss, aux), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        # Gradient norm and clipping (encoder)
        encoder_grads_flat, _ = jax.tree_util.tree_flatten(grads['encoder'])
        encoder_grad_norm = jnp.linalg.norm(
            jnp.concatenate([jnp.ravel(g) for g in encoder_grads_flat])
        )
        aux['grad_norm'] = encoder_grad_norm

        max_norm = 1.0
        clip_factor = max_norm / jnp.maximum(encoder_grad_norm, max_norm)

        clipped_grads = {
            'encoder':      jax.tree_util.tree_map(lambda g: g * clip_factor, grads['encoder']),
            'lambda_real':  jnp.clip(grads['lambda_real'], -1.0, 1.0),
            'lambda_imag':  jnp.clip(grads['lambda_imag'], -1.0, 1.0),
        }

        updates, new_opt_state = encoder_state.tx.update(
            clipped_grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        new_state = encoder_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=encoder_state.step + 1,
        )
        return new_state, total_loss, aux

    return update_encoder
