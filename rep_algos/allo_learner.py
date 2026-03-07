"""
ALLO (Asymmetric Laplacian Learning Optimization) learner module.

Wraps the augmented-Lagrangian algorithm from allo.py as a proper learner
module compatible with shared_training.py.

ALLO learns a single set of real eigenvectors φ_k of the symmetrized
Laplacian, which serve as BOTH left and right eigenvectors (since the
symmetrized Laplacian has real eigenvectors with left = right).

The encoder's `right_real` head is used as φ_k; the other heads (`left_real`,
`left_imag`, `right_imag`) receive no gradient from the ALLO loss and are
ignored at evaluation time.  run_reward_shaping.py is expected to load the
ALLO model with  left = right = right_real  and  imag = 0.

Eigenvalue estimates are derived from the dual diagonal:
    λ_k  ≈  −0.5 · duals[k, k]

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
    """Initialise parameters for the ALLO learner."""
    k = args.num_eigenvector_pairs

    duals_init   = getattr(args, 'duals_initial_val',   -2.0)
    barrier_init = getattr(args, 'barrier_initial_val',  0.5)
    init_diag    = getattr(args, 'init_dual_diag',       False)

    if init_diag:
        duals = jnp.tril(jnp.full((k, k), duals_init))
    else:
        duals = jnp.eye(k) * duals_init

    return {
        'encoder':        encoder_initial_params,
        # Eigenvalue estimates (derived from dual diagonal; used by shared_training
        # for hitting-time visualisation and by run_reward_shaping.py)
        'lambda_real':    jnp.zeros((k,)),
        'lambda_imag':    jnp.zeros((k,)),
        # ALLO augmented-Lagrangian variables
        'duals':          duals,                           # (k, k) lower-triangular
        'barrier_coefs':  jnp.array([[barrier_init]]),    # (1, 1) shared scalar
        'error_integral': jnp.zeros((k, k)),              # (k, k) integral term
    }


def create_optimizer(args):
    """
    Two-group optimizer:
      • encoder      → Adam  (args.learning_rate)
      • duals + barrier_coefs → SGD  (args.step_size_duals)
      • lambda_real/imag, error_integral → zero updates (managed manually)
    """
    step_duals = getattr(args, 'step_size_duals', 1.0)

    adam = optax.adam(args.learning_rate)
    sgd  = optax.sgd(step_duals)

    encoder_mask = {
        'encoder': True,
        'lambda_real': False, 'lambda_imag': False,
        'duals': False, 'barrier_coefs': False, 'error_integral': False,
    }
    duals_mask = {
        'encoder': False,
        'lambda_real': False, 'lambda_imag': False,
        'duals': True, 'barrier_coefs': True, 'error_integral': False,
    }

    return optax.chain(
        optax.masked(adam, encoder_mask),
        optax.masked(sgd,  duals_mask),
    )


def get_optimizer_masks(args):
    """Fallback masks — only consulted if create_optimizer is not found."""
    encoder_mask = {
        'encoder': True,
        'lambda_real': False, 'lambda_imag': False,
        'duals': False, 'barrier_coefs': False, 'error_integral': False,
    }
    other_mask = {
        'encoder': False,
        'lambda_real': False, 'lambda_imag': False,
        'duals': True, 'barrier_coefs': True, 'error_integral': False,
    }
    return encoder_mask, other_mask


def create_update_function(encoder, args):
    """
    Create the JIT-compiled ALLO update function.

    The returned function matches the 6-argument signature expected by
    shared_training.py.  `next_state_coords_batch_2` and `state_weighting`
    are accepted but not used — ALLO's loss is unweighted by design.
    """
    k               = args.num_eigenvector_pairs
    max_barrier     = getattr(args, 'max_barrier_coefs',    0.5)
    step_duals      = getattr(args, 'step_size_duals',      1.0)
    step_duals_I    = getattr(args, 'step_size_duals_I',    0.0)
    integral_decay  = getattr(args, 'integral_decay',       0.99)
    graph_epsilon   = getattr(args, 'graph_epsilon',        0.01)
    graph_var_scale = getattr(args, 'graph_variance_scale', 0.1)
    perturb_type    = getattr(args, 'perturbation_type',    'none')

    @jax.jit
    def update_encoder(
        encoder_state: TrainState,
        state_coords_batch: jnp.ndarray,
        next_state_coords_batch: jnp.ndarray,
        state_coords_batch_2: jnp.ndarray,
        next_state_coords_batch_2: jnp.ndarray,  # accepted, not used by ALLO
        state_weighting: jnp.ndarray,            # accepted, not used by ALLO
    ):
        n = state_coords_batch.shape[0]

        def check_previous_entries_below_threshold(matrix, threshold):
            below_threshold = (jnp.abs(matrix) < threshold).astype(jnp.float32)
            row_all_below = jnp.prod(below_threshold, axis=1, keepdims=True)
            cumulative_results = jnp.cumprod(row_all_below, axis=0)
            result_with_zero = jnp.ones(
                (matrix.shape[0] + 1, 1), dtype=cumulative_results.dtype)
            result_with_zero = result_with_zero.at[1:, :].set(cumulative_results)
            return result_with_zero[:-1, :]

        def encoder_loss(params):
            encoder_params = params['encoder']

            # Forward passes — use right_real as the single ALLO representation φ
            phi      = encoder.apply(encoder_params, state_coords_batch)[0]['right_real']
            phi_2    = encoder.apply(encoder_params, state_coords_batch_2)[0]['right_real']
            next_phi = encoder.apply(encoder_params, next_state_coords_batch)[0]['right_real']

            d = k

            # Dual variables
            dual_variables    = params['duals']
            barrier_coefs     = params['barrier_coefs']
            diagonal_duals    = jnp.diag(dual_variables)
            eigenvalue_sum    = -0.5 * diagonal_duals.sum()

            # Inner-product matrices (unweighted — ALLO's original formulation)
            ip1 = jnp.einsum('ij,ik->jk', phi,   jax.lax.stop_gradient(phi))   / n
            ip2 = jnp.einsum('ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n

            error_matrix_1 = jnp.tril(ip1 - jnp.eye(d))
            error_matrix_2 = jnp.tril(ip2 - jnp.eye(d))

            # Dual (augmented Lagrangian) losses
            error_integral = params['error_integral']
            dual_loss_pos  = (
                jax.lax.stop_gradient(dual_variables) * error_matrix_1
            ).sum()
            dual_loss_P    = jax.lax.stop_gradient(step_duals   * error_matrix_1)
            dual_loss_I    = step_duals_I * jax.lax.stop_gradient(error_integral)
            dual_loss_neg  = -(dual_variables * (dual_loss_P + dual_loss_I)).sum()

            # Barrier (penalty) losses
            quad_err      = (
                2 * error_matrix_1 * jax.lax.stop_gradient(error_matrix_2)
            ).sum()
            barrier_pos   = jax.lax.stop_gradient(barrier_coefs[0, 0]) * quad_err
            barrier_neg   = (
                -barrier_coefs[0, 0] * jax.lax.stop_gradient(jnp.absolute(quad_err))
            )

            # Graph perturbation (variance-based degeneracy prevention)
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

            # Graph-drawing loss
            diff       = phi - next_phi
            graph_loss = (0.5 * (diff ** 2).mean(0, keepdims=True) + graph_pert).sum()

            # Total ALLO loss (unchanged from allo.py)
            positive_loss = graph_loss + dual_loss_pos + barrier_pos
            negative_loss = dual_loss_neg + barrier_neg
            allo_loss     = positive_loss + negative_loss

            # Metrics
            norm_phi   = (phi ** 2).mean(0, keepdims=True)
            total_err  = jnp.absolute(error_matrix_1).sum()
            dist_manif = jnp.tril(error_matrix_1 ** 2).sum()
            total_two  = jnp.absolute(
                error_matrix_1[:, :min(2, d)]
            ).sum()

            aux = {
                'total_loss':                    allo_loss,
                'graph_loss':                    graph_loss,
                'dual_loss':                     dual_loss_pos,
                'dual_loss_neg':                 dual_loss_neg,
                'barrier_loss':                  barrier_pos,
                'approx_eigenvalue_sum':         eigenvalue_sum,
                'barrier_coef':                  barrier_coefs[0, 0],
                'total_error':                   total_err,
                'total_two_component_error':     total_two,
                'distance_to_constraint_manifold': dist_manif,
                'distance_to_origin':            norm_phi.sum(),
            }

            return allo_loss, (error_matrix_1, aux)

        # Gradients
        (allo_loss, (error_matrix, aux)), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        # Apply optimizer
        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        # Custom updates (outside optimizer, exactly as in allo.py)
        new_params['error_integral'] = (
            integral_decay * new_params['error_integral'] + error_matrix
        )
        new_params['barrier_coefs'] = jnp.clip(
            new_params['barrier_coefs'], 0.0, max_barrier
        )
        # Derive eigenvalue estimates from dual diagonal
        new_params['lambda_real'] = -0.5 * jnp.diag(new_params['duals'])
        new_params['lambda_imag'] = jnp.zeros_like(new_params['lambda_imag'])

        # Grad norm for monitoring
        grads_flat, _ = jax.tree_util.tree_flatten(grads)
        aux['grad_norm'] = jnp.linalg.norm(
            jnp.concatenate([jnp.ravel(g) for g in grads_flat])
        )

        new_state = encoder_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=encoder_state.step + 1,
        )
        return new_state, allo_loss, aux

    return update_encoder
