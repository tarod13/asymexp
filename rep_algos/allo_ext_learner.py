"""
ALLO-Ext learner: ALLO with externally-updated duals and barrier coefficient.

Structural differences from allo_learner.py:
  - Dual variables and the barrier coefficient are NOT updated by the optimizer.
    Instead they are updated after the gradient step using EMA-smoothed error
    estimates, matching the `additional_update_step` logic in
    laplacian_dual_dynamics/al.py.
  - The loss therefore contains only the encoder-facing terms:
        graph_loss  +  stop_grad(duals) * error_matrix  +  stop_grad(barrier) * quad_err
    There is no `dual_loss_neg` or `barrier_neg` in the computation graph.
  - Two EMA accumulators (error_ema, quad_error_ema) track smoothed estimates
    of the linear and quadratic constraint errors across steps.

External update rules (applied after each gradient step):
  - error_ema      += update_rate   * (fresh_error      - error_ema)
  - quad_error_ema += q_update_rate * (fresh_quad_error - quad_error_ema)
  - duals          += lr_duals * tril(error_ema)        [clipped]
  - barrier        += lr_barrier * mean(clip(quad_error_ema, min=0))  [clipped]

Usage:
    python train.py allo_ext --env_file_name GridRoom-s-1 ...
"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def init_params(encoder_initial_params, args):
    k = args.num_eigenvector_pairs
    barrier_init = getattr(args, 'barrier_initial_val', 0.5)
    duals_init   = getattr(args, 'duals_initial_val',   0.0)

    return {
        'encoder':        encoder_initial_params,
        'lambda_real':    jnp.zeros((k,)),
        'lambda_imag':    jnp.zeros((k,)),
        # Augmented-Lagrangian variables (updated externally)
        'duals':          jnp.tril(jnp.full((k, k), duals_init)),
        'barrier_coefs':  jnp.array([[barrier_init]]),
        # EMA accumulators for error estimates
        'error_ema':      jnp.zeros((k, k)),
        'quad_error_ema': jnp.zeros((k, k)),
    }


def create_optimizer(args):
    """Only the encoder is updated by Adam; everything else is managed manually."""
    adam = optax.adam(args.learning_rate)
    encoder_mask = {
        'encoder':        True,
        'lambda_real':    False,
        'lambda_imag':    False,
        'duals':          False,
        'barrier_coefs':  False,
        'error_ema':      False,
        'quad_error_ema': False,
    }
    return optax.masked(adam, encoder_mask)


def get_optimizer_masks(args):
    """Fallback masks — only consulted if create_optimizer is not found."""
    encoder_mask = {
        'encoder':        True,
        'lambda_real':    False,
        'lambda_imag':    False,
        'duals':          False,
        'barrier_coefs':  False,
        'error_ema':      False,
        'quad_error_ema': False,
    }
    other_mask = {k: not v for k, v in encoder_mask.items()}
    return encoder_mask, other_mask


def create_update_function(encoder, args):
    k                  = args.num_eigenvector_pairs
    max_barrier        = getattr(args, 'max_barrier_coefs',    2.0)
    min_barrier        = getattr(args, 'min_barrier_coefs',    0.0)
    lr_barrier         = getattr(args, 'lr_barrier_coefs',     1.0)
    lr_duals           = getattr(args, 'lr_duals',             0.0001)
    min_duals          = getattr(args, 'min_duals',           -100.0)
    max_duals          = getattr(args, 'max_duals',            100.0)
    error_update_rate  = getattr(args, 'error_update_rate',    1.0)
    q_error_update_rate = getattr(args, 'q_error_update_rate', 0.1)
    graph_epsilon      = getattr(args, 'graph_epsilon',        0.01)
    graph_var_scale    = getattr(args, 'graph_variance_scale', 0.1)
    perturb_type       = getattr(args, 'perturbation_type',    'none')

    @jax.jit
    def update_encoder(
        encoder_state: TrainState,
        state_coords_batch:       jnp.ndarray,
        next_state_coords_batch:  jnp.ndarray,
        state_coords_batch_2:     jnp.ndarray,
        next_state_coords_batch_2: jnp.ndarray,   # accepted but unused by ALLO-Ext
        state_weighting:          jnp.ndarray,
    ):
        n = state_coords_batch.shape[0]

        def encoder_loss(params):
            encoder_params = params['encoder']
            phi      = encoder.apply(encoder_params, state_coords_batch)[0]['right_real']
            phi_2    = encoder.apply(encoder_params, state_coords_batch_2)[0]['right_real']
            next_phi = encoder.apply(encoder_params, next_state_coords_batch)[0]['right_real']

            dual_variables = params['duals']
            barrier_coefs  = params['barrier_coefs']

            # Weighted inner-product matrices (estimate E_target[φ φᵀ])
            ip1 = jnp.einsum('ij,ik->jk', state_weighting * phi,   jax.lax.stop_gradient(phi))   / n
            ip2 = jnp.einsum('ij,ik->jk', state_weighting * phi_2, jax.lax.stop_gradient(phi_2)) / n

            error_matrix_1 = jnp.tril(ip1 - jnp.eye(k))
            error_matrix_2 = jnp.tril(ip2 - jnp.eye(k))

            # Graph-drawing loss (with optional perturbation)
            phi_centered  = phi - jnp.mean(phi, axis=0, keepdims=True)
            phi_variances = (phi_centered ** 2).mean(0, keepdims=True)
            delta_var     = jnp.exp(-1.0 / graph_var_scale)

            if perturb_type == 'squared-null-grad':
                graph_pert = graph_epsilon * (
                    ((phi_centered - 1) ** 2).mean(0, keepdims=True)
                ).clip(0, 1)
            elif perturb_type == 'squared':
                graph_pert = graph_epsilon * (
                    ((jnp.absolute(phi_centered - 1) + delta_var) ** 2
                     - delta_var ** 2).mean(0, keepdims=True)
                ).clip(0, 1)
            elif perturb_type == 'exponential':
                graph_pert = graph_epsilon * (
                    jnp.exp(-phi_variances / graph_var_scale) - delta_var
                ).clip(0, 1)
            else:
                graph_pert = jnp.zeros_like(phi_variances)
            graph_pert = graph_pert.at[0, 0].set(0.0)

            diff       = phi - next_phi
            graph_loss = ((diff ** 2).mean(0, keepdims=True) + graph_pert).sum()

            # Error matrices (matching generalized_augmented.py convention).
            error_matrix      = 0.5 * (error_matrix_1 + error_matrix_2)
            quad_error_matrix = error_matrix_1 * error_matrix_2

            # Encoder-side Lagrangian terms — duals and barrier are stop-grad'd
            # so no gradient flows back into them through the optimizer.
            # dual_loss uses the averaged error matrix, matching al.py line 73.
            dual_loss    = (jax.lax.stop_gradient(dual_variables) * error_matrix).sum()
            quad_err     = quad_error_matrix.sum()
            barrier_loss = jax.lax.stop_gradient(barrier_coefs[0, 0]) * quad_err

            loss = graph_loss + dual_loss + barrier_loss

            norm_phi  = (phi ** 2).mean(0, keepdims=True)
            total_err = jnp.absolute(error_matrix_1).sum()
            aux = {
                'total_loss':              loss,
                'graph_loss':              graph_loss,
                'dual_loss':               dual_loss,
                'barrier_loss':            barrier_loss,
                'barrier_coef':            barrier_coefs[0, 0],
                'total_error':             total_err,
                'distance_to_origin':      norm_phi.sum(),
                'approx_eigenvalue_sum':   -0.5 * jnp.diag(dual_variables).sum(),
            }
            return loss, (error_matrix, quad_error_matrix, aux)

        (loss, (error_matrix, quad_error_matrix, aux)), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        # ── External updates (outside the optimizer) ──────────────────────────

        # EMA update for error estimates.
        # On the first step (accumulator is zero) use rate=1 to initialise cleanly.
        def ema_update(old, new, rate):
            is_init = jnp.isclose(jnp.linalg.norm(old), 0.0, atol=1e-13)
            effective_rate = is_init + (1.0 - is_init) * rate
            return old + effective_rate * (new - old)

        new_error_ema      = ema_update(new_params['error_ema'],      error_matrix,      error_update_rate)
        new_quad_error_ema = ema_update(new_params['quad_error_ema'], quad_error_matrix, q_error_update_rate)

        # Dual update: duals += lr_duals * tril(error_ema)
        updated_duals = new_params['duals'] + lr_duals * jnp.tril(new_error_ema)
        updated_duals = jnp.clip(updated_duals, min_duals, max_duals)
        updated_duals = jnp.tril(updated_duals)

        # Barrier update: only positive quadratic errors drive growth (conservative).
        barrier_update   = lr_barrier * jnp.clip(new_quad_error_ema, a_min=0.0).mean()
        updated_barrier  = jnp.clip(new_params['barrier_coefs'] + barrier_update, min_barrier, max_barrier)

        new_params['error_ema']      = new_error_ema
        new_params['quad_error_ema'] = new_quad_error_ema
        new_params['duals']          = updated_duals
        new_params['barrier_coefs']  = updated_barrier
        new_params['lambda_real']    = -0.5 * jnp.diag(updated_duals)
        new_params['lambda_imag']    = jnp.zeros_like(new_params['lambda_imag'])

        # Overwrite with post-update value so the logged barrier_coef reflects
        # what will be used in the *next* gradient step.
        aux['barrier_coef'] = updated_barrier[0, 0]

        # Grad norm for monitoring
        encoder_grads_flat, _ = jax.tree_util.tree_flatten(grads['encoder'])
        aux['grad_norm'] = jnp.linalg.norm(
            jnp.concatenate([jnp.ravel(g) for g in encoder_grads_flat])
        )

        new_state = encoder_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=encoder_state.step + 1,
        )
        return new_state, loss, aux

    return update_encoder
