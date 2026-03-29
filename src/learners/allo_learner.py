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

Eigenvalue estimates are kernel eigenvalues λ_M computed via geometric Rayleigh
quotient: λ_k ≈ E_D[φ_k(s) · φ_k(s')] where s' is drawn geometrically from s
with parameter γ.  This matches the 'kernel' eigenvalue_type used by
shared_training.py for hitting-time computation.

Usage:
    from src.learners.shared_training import learn_eigenvectors
    from src.learners import allo_learner
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
        duals = jnp.eye(k) * duals_init                        # diagonal only
    else:
        duals = jnp.tril(jnp.full((k, k), duals_init))        # full lower-triangular (default)

    # When episode-centric sampling is used with random_wind, eigenvalue estimates are
    # stored per wind bucket so each MDP's spectral structure is tracked separately.
    use_buckets = (
        getattr(args, 'sample_episodes', False)
        and getattr(args, 'random_wind', False)
    )
    B = getattr(args, 'num_wind_buckets', 20) if use_buckets else None
    lambda_real = jnp.zeros((B, k)) if use_buckets else jnp.zeros((k,))
    lambda_imag = jnp.zeros((B, k)) if use_buckets else jnp.zeros((k,))

    return {
        'encoder':        encoder_initial_params,
        # Eigenvalue estimates (monitored via geometric Rayleigh quotient; used by
        # shared_training for hitting-time visualisation).
        # Shape (k,) normally; (num_wind_buckets, k) when sample_episodes+random_wind.
        'lambda_real':    lambda_real,
        'lambda_imag':    lambda_imag,
        # ALLO augmented-Lagrangian variables
        'duals':          duals,                           # (k, k) lower-triangular
        'barrier_coefs':  jnp.array([[barrier_init]]),    # (1, 1) shared scalar
        'error_integral': jnp.zeros((k, k)),              # (k, k) integral term
    }


def create_optimizer(args):
    """
    Two-group optimizer matching allo.py exactly:
      • encoder             → Adam  (args.learning_rate)
      • duals + barrier_coefs → SGD  (args.learning_rate)
      • lambda_real/imag, error_integral → zero updates (managed manually)

    Note: step_size_duals is a multiplier *inside* the loss (dual_loss_P),
    not an optimizer learning rate.  Using it as the SGD lr would produce
    an effective dual step of step_size_duals² instead of learning_rate ×
    step_size_duals, causing catastrophically large dual updates.
    """
    adam = optax.adam(args.learning_rate)
    sgd  = optax.sgd(args.learning_rate)  # matches allo.py

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
    shared_training.py.  `next_state_coords_batch_2` is accepted but not
    used.  `state_weighting` is applied inside the inner products so that
    all three sampling modes (rejection, weighted, none) work correctly.

    When args.sample_episodes=True and args.random_wind=True the batch is
    assumed to be ordered episode-major (output of replay_buffer.sample_episodes).
    The loss is computed per-episode via jax.vmap and averaged, and
    lambda_real is updated per wind-bucket using a scatter-mean.
    """
    k               = args.num_eigenvector_pairs
    max_barrier     = getattr(args, 'max_barrier_coefs',    0.5)
    step_duals      = getattr(args, 'step_size_duals',      1.0)
    step_duals_I    = getattr(args, 'step_size_duals_I',    0.0)
    integral_decay  = getattr(args, 'integral_decay',       0.99)
    graph_epsilon   = getattr(args, 'graph_epsilon',        0.01)
    graph_var_scale = getattr(args, 'graph_variance_scale', 0.1)
    perturb_type    = getattr(args, 'perturbation_type',    'none')

    # Episode-centric sampling settings (captured as Python-level constants).
    use_episodic = (
        getattr(args, 'sample_episodes', False)
        and getattr(args, 'random_wind', False)
    )
    N = getattr(args, 'num_sampled_episodes', 1) if use_episodic else 1
    T = (args.batch_size // N) if use_episodic else args.batch_size
    B = getattr(args, 'num_wind_buckets', 20)  # number of wind buckets

    def _allo_loss_body(params, phi, phi_2, next_phi, n, ep_weighting):
        """Core ALLO loss computation for a batch of n transitions."""
        d = k

        dual_variables = params['duals']
        barrier_coefs  = params['barrier_coefs']
        diagonal_duals = jnp.diag(dual_variables)
        eigenvalue_sum = -0.5 * diagonal_duals.sum()

        ip1 = jnp.einsum('ij,ik->jk', ep_weighting * phi,   jax.lax.stop_gradient(phi))   / n
        ip2 = jnp.einsum('ij,ik->jk', ep_weighting * phi_2, jax.lax.stop_gradient(phi_2)) / n

        error_matrix_1 = jnp.tril(ip1 - jnp.eye(d))
        error_matrix_2 = jnp.tril(ip2 - jnp.eye(d))

        error_integral = params['error_integral']
        dual_loss_pos  = (jax.lax.stop_gradient(dual_variables) * error_matrix_1).sum()
        dual_loss_P    = jax.lax.stop_gradient(step_duals * error_matrix_1)
        dual_loss_I    = step_duals_I * jax.lax.stop_gradient(error_integral)
        dual_loss_neg  = -(dual_variables * (dual_loss_P + dual_loss_I)).sum()

        quad_err    = (2 * error_matrix_1 * jax.lax.stop_gradient(error_matrix_2)).sum()
        barrier_pos = jax.lax.stop_gradient(barrier_coefs[0, 0]) * quad_err
        barrier_neg = -barrier_coefs[0, 0] * jax.lax.stop_gradient(jnp.absolute(quad_err))

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
        graph_loss = (0.5 * (diff ** 2).mean(0, keepdims=True) + graph_pert).sum()

        positive_loss = graph_loss + dual_loss_pos + barrier_pos
        negative_loss = dual_loss_neg + barrier_neg
        allo_loss     = positive_loss + negative_loss

        norm_phi   = (phi ** 2).mean(0, keepdims=True)
        total_err  = jnp.absolute(error_matrix_1).sum()
        dist_manif = jnp.tril(error_matrix_1 ** 2).sum()
        total_two  = jnp.absolute(error_matrix_1[:, :min(2, d)]).sum()

        rayleigh_quotient = jnp.einsum(
            'ij,ij->j',
            ep_weighting * jax.lax.stop_gradient(phi),
            jax.lax.stop_gradient(next_phi),
        ) / n

        aux = {
            'total_loss':                      allo_loss,
            'graph_loss':                      graph_loss,
            'dual_loss':                       dual_loss_pos,
            'dual_loss_neg':                   dual_loss_neg,
            'barrier_loss':                    barrier_pos,
            'approx_eigenvalue_sum':           eigenvalue_sum,
            'barrier_coef':                    barrier_coefs[0, 0],
            'total_error':                     total_err,
            'total_two_component_error':       total_two,
            'distance_to_constraint_manifold': dist_manif,
            'distance_to_origin':              norm_phi.sum(),
            'rayleigh_quotient':               rayleigh_quotient,  # (k,)
        }
        return allo_loss, error_matrix_1, aux

    @jax.jit
    def update_encoder(
        encoder_state: TrainState,
        state_coords_batch: jnp.ndarray,
        next_state_coords_batch: jnp.ndarray,
        state_coords_batch_2: jnp.ndarray,
        next_state_coords_batch_2: jnp.ndarray,  # accepted, not used by ALLO
        state_weighting: jnp.ndarray,
    ):
        if use_episodic:
            # ── Episode-centric path ──────────────────────────────────────────
            # Reshape flat (N*T, d) batches into (N, T, d) episode groups.
            obs_dim = state_coords_batch.shape[-1]
            ep_coords    = state_coords_batch.reshape(N, T, obs_dim)
            ep_n_coords  = next_state_coords_batch.reshape(N, T, obs_dim)
            ep_coords_2  = state_coords_batch_2.reshape(N, T, obs_dim)
            ep_weights   = state_weighting.reshape(N, T, 1)

            def per_episode_loss(params, ec, enc, ec2, ew):
                """Compute ALLO loss for a single episode (shapes (T, d) / (T, 1))."""
                ep_params = params['encoder']
                phi      = encoder.apply(ep_params, ec)[0]['right_real']    # (T, k)
                phi_2    = encoder.apply(ep_params, ec2)[0]['right_real']   # (T, k)
                next_phi = encoder.apply(ep_params, enc)[0]['right_real']   # (T, k)

                # Compute bucket index from wind (last coord dim, same for all T steps).
                ep_wind      = ec[0, -1]
                ep_bucket    = jnp.clip(
                    jnp.floor((ep_wind + 1.0) / 2.0 * B).astype(jnp.int32),
                    0, B - 1,
                )

                loss, err_mat, ep_aux = _allo_loss_body(params, phi, phi_2, next_phi, T, ew)
                ep_aux['ep_bucket'] = ep_bucket  # scalar — for post-step scatter update
                return loss, err_mat, ep_aux

            def vmapped_loss(params):
                losses, err_mats, ep_auxes = jax.vmap(
                    per_episode_loss, in_axes=(None, 0, 0, 0, 0)
                )(params, ep_coords, ep_n_coords, ep_coords_2, ep_weights)
                total_loss = losses.mean()
                error_matrix = err_mats.mean(axis=0)   # (k, k) averaged constraint error
                return total_loss, (error_matrix, ep_auxes)

            (allo_loss, (error_matrix, ep_auxes)), grads = jax.value_and_grad(
                vmapped_loss, has_aux=True
            )(encoder_state.params)

            # Average per-episode metrics for logging.
            aux = jax.tree_util.tree_map(lambda x: x.mean(axis=0), ep_auxes)
            # Preserve per-episode quantities needed for the bucketed lambda update.
            per_ep_rq      = ep_auxes['rayleigh_quotient']  # (N, k)
            ep_buckets     = ep_auxes['ep_bucket']           # (N,) int32

        else:
            # ── Standard flat-batch path ──────────────────────────────────────
            n = state_coords_batch.shape[0]
            ep_params = encoder_state.params

            def encoder_loss(params):
                encoder_params = params['encoder']
                phi      = encoder.apply(encoder_params, state_coords_batch)[0]['right_real']
                phi_2    = encoder.apply(encoder_params, state_coords_batch_2)[0]['right_real']
                next_phi = encoder.apply(encoder_params, next_state_coords_batch)[0]['right_real']
                loss, err_mat, loss_aux = _allo_loss_body(
                    params, phi, phi_2, next_phi, n, state_weighting
                )
                return loss, (err_mat, loss_aux)

            (allo_loss, (error_matrix, aux)), grads = jax.value_and_grad(
                encoder_loss, has_aux=True
            )(encoder_state.params)

        # ── Optimizer step ────────────────────────────────────────────────────
        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        # ── Custom post-optimizer updates ─────────────────────────────────────
        new_params['error_integral'] = (
            integral_decay * new_params['error_integral'] + error_matrix
        )
        new_params['barrier_coefs'] = jnp.clip(
            new_params['barrier_coefs'], 0.0, max_barrier
        )

        if use_episodic:
            # Bucketed lambda_real update: scatter per-episode Rayleigh quotients
            # into the corresponding wind bucket, averaging within each bucket.
            # Buckets not seen in this batch retain their current estimate.
            bucket_sum   = jnp.zeros((B, k)).at[ep_buckets].add(per_ep_rq)
            bucket_count = jnp.zeros(B).at[ep_buckets].add(1.0)
            bucket_mean  = bucket_sum / jnp.maximum(bucket_count[:, None], 1.0)
            new_params['lambda_real'] = jnp.where(
                bucket_count[:, None] > 0,
                bucket_mean,
                new_params['lambda_real'],
            )
            new_params['lambda_imag'] = jnp.zeros_like(new_params['lambda_imag'])
        else:
            # Standard: overwrite with this batch's Rayleigh quotient estimate.
            new_params['lambda_real'] = aux['rayleigh_quotient']
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


def get_eigenvalues(params):
    """Extract eigenvalue estimates from params.

    When sample_episodes+random_wind bucketing is active, lambda arrays have
    shape (num_wind_buckets, k). Returns their mean over buckets so downstream
    evaluation code always receives shape (k,).
    """
    lr = params['lambda_real']
    li = params['lambda_imag']
    if lr.ndim == 2:
        lr, li = lr.mean(axis=0), li.mean(axis=0)
    return lr, li
