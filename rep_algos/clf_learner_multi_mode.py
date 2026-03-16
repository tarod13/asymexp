"""
CLF (Control Lyapunov Function) learner with multiple constraint error approximation modes.

Supports 4 constraint approximation modes:
- "ema": EMA approximation (current approach)
- "two_batch": Unbiased with two independent batches
- "single_batch": Biased with single batch (reuse batch1 for constraints)
- "same_episodes": Intermediate bias - two batches from same episodes

Supports 4 constraint enforcement methods (args.constraint_enforcement_method):
- "clf": CLF controller — QP-style correction u = barrier·∇V, loss += ip(feat, sg(u))
- "barrier": Increasing barrier — single scalar penalty sg(barrier_coef)·V,
              with barrier_coef updated externally after each gradient step
- "granular_barrier": Increasing barrier — per-constraint penalties sg(bc[i])·V[i],
              with individual coefficients updated externally after each gradient step
- "complex_allo": Complex augmented-Lagrangian — dual variables multiplying each
              linear constraint error + scalar barrier sg(barrier_coef)·V (identical
              to "barrier"); both updated externally after each gradient step

Usage:
    from rep_algos.shared_training import learn_eigenvectors
    from rep_algos import clf_learner_multi_mode
    from src.config.ded_clf import Args

    args = tyro.cli(Args)
    args.constraint_mode = "two_batch"  # or "ema", "single_batch", "same_episodes"
    args.constraint_enforcement_method = "clf"  # or "barrier"
    learn_eigenvectors(args, clf_learner_multi_mode)
"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: Lyapunov components
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lyapunov_terms(
    x_r, x_i, y_r, y_i,
    x2_r, x2_i, y2_r, y2_i,
    next_x_r, next_x_i, next_y_r, next_y_i,
    next_x2_r, next_x2_i, next_y2_r, next_y2_i,
    params, args, ip, multi_ip,
):
    """Compute the Lyapunov function V, its gradient ∇V, and the constraint
    estimator loss — all shared between the CLF and barrier enforcement methods.

    Args:
        x_r, x_i, y_r, y_i:                      Batch-1 current features
        x2_r, x2_i, y2_r, y2_i:                  Batch-2 current features
        next_x_r, next_x_i, next_y_r, next_y_i:  Batch-1 next-step features
        next_x2_r, next_x2_i, next_y2_r, next_y2_i: Batch-2 next-step features
        params:   Full parameter dict
        args:     Training arguments (constraint_mode, eigenvalue_estimation_method, …)
        ip:       Weighted inner product (closure over state_weighting)
        multi_ip: Multi-vector weighted inner product (closure over state_weighting)

    Returns:
        V:                        shape (1, k) — per-eigenvector Lyapunov values
        nabla_V:                  dict with keys x_r, x_i, y_r, y_i,
                                  next_y_r, next_y_i — gradients of V
        constraint_estimator_loss: scalar — EMA tracking loss (non-zero for "ema" mode)
        V_components:             dict with V_x_norm, V_y_norm, V_xy_phase,
                                  V_xy_corr_matrix (shape k×k), norm_x_sq, norm_y_sq
    """

    # ------------------------------------------------------------------
    # Norms
    # ------------------------------------------------------------------
    norm_x_sq = ip(x_r, x_r) + ip(x_i, x_i)
    norm_y_sq = ip(y_r, y_r) + ip(y_i, y_i)
    norm_x2_sq = ip(x2_r, x2_r) + ip(x2_i, x2_i)
    norm_y2_sq = ip(y2_r, y2_r) + ip(y2_i, y2_i)
    next_norm_y_sq = ip(next_y_r, next_y_r) + ip(next_y_i, next_y_i)
    next_norm_y2_sq = ip(next_y2_r, next_y2_r) + ip(next_y2_i, next_y2_i)

    # EMA loss for norms (only for "ema" mode)
    if args.constraint_mode == "ema":
        loss_norm_x_ema = (
            (jax.lax.stop_gradient(0.5 * (norm_x_sq + norm_x2_sq))
             - params['norm_x_ema'].reshape(1, -1)) ** 2
        ).sum()
        loss_norm_y_ema = (
            (jax.lax.stop_gradient(0.5 * (norm_y_sq + norm_y2_sq))
             - params['norm_y_ema'].reshape(1, -1)) ** 2
        ).sum()
    else:
        loss_norm_x_ema = 0.0
        loss_norm_y_ema = 0.0

    # ------------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------------
    phase_xy = ip(y_r, x_i) - ip(y_i, x_r)
    phase_xy2 = ip(y2_r, x2_i) - ip(y2_i, x2_r)
    next_phase_xy = ip(next_y_r, next_x_i) - ip(next_y_i, next_x_r)
    next_phase_xy2 = ip(next_y2_r, next_x2_i) - ip(next_y2_i, next_x2_r)

    # EMA loss for phase (only for "ema" mode)
    if args.constraint_mode == "ema":
        loss_phase_xy_ema = (
            (jax.lax.stop_gradient(0.5 * (phase_xy + phase_xy2))
             - params['phase_xy_ema'].reshape(1, -1)) ** 2
        ).sum()
    else:
        loss_phase_xy_ema = 0.0

    # ------------------------------------------------------------------
    # Correlations (crossed terms)
    # ------------------------------------------------------------------
    # When use_sg_ip=True, stop_gradient is applied to the smaller-indexed
    # eigenvector in each inner product, matching the ALLO-Ext convention:
    #   multi_ip(a, b)[j, k] = <a_j, b_k>
    # For tril(., k=-1) the column index k < j is always the smaller index.
    # XY: smaller index is on the y-side (second arg) → sg(y).
    # YX: these are computed separately (not as .T of XY) so the smaller
    #     index is on the x-side (second arg of multi_ip(y, sg(x))).
    use_sg_ip = getattr(args, 'use_sg_ip', True)
    _sg = jax.lax.stop_gradient if use_sg_ip else (lambda z: z)

    # Batch 1 — current
    corr_xy_real = multi_ip(x_r, _sg(y_r)) + multi_ip(x_i, _sg(y_i))
    corr_xy_imag = -multi_ip(x_r, _sg(y_i)) + multi_ip(x_i, _sg(y_r))
    if use_sg_ip:
        corr_yx_real_b1 = multi_ip(y_r, _sg(x_r)) + multi_ip(y_i, _sg(x_i))
        corr_yx_imag_b1 = -multi_ip(y_i, _sg(x_r)) + multi_ip(y_r, _sg(x_i))

    # Batch 2 — current
    cross_xy2_real = multi_ip(x2_r, _sg(y2_r)) + multi_ip(x2_i, _sg(y2_i))
    cross_xy2_imag = -multi_ip(x2_r, _sg(y2_i)) + multi_ip(x2_i, _sg(y2_r))
    if use_sg_ip:
        corr_yx_real_b2 = multi_ip(y2_r, _sg(x2_r)) + multi_ip(y2_i, _sg(x2_i))
        corr_yx_imag_b2 = -multi_ip(y2_i, _sg(x2_r)) + multi_ip(y2_r, _sg(x2_i))

    # EMA loss for correlations (only for "ema" mode)
    if args.constraint_mode == "ema":
        loss_corr_xy_real = (
            (jax.lax.stop_gradient(0.5 * (corr_xy_real + cross_xy2_real))
             - params['corr_xy_real_ema']) ** 2
        ).sum()
        loss_corr_xy_imag = (
            (jax.lax.stop_gradient(0.5 * (corr_xy_imag + cross_xy2_imag))
             - params['corr_xy_imag_ema']) ** 2
        ).sum()
    else:
        loss_corr_xy_real = 0.0
        loss_corr_xy_imag = 0.0

    constraint_estimator_loss = (
        loss_norm_x_ema + loss_norm_y_ema
        + loss_phase_xy_ema
        + loss_corr_xy_real + loss_corr_xy_imag
    )

    # ------------------------------------------------------------------
    # Select constraint coefficients based on constraint_mode
    # ------------------------------------------------------------------

    # 1. x-norm coefficient
    norm_x_error = norm_x_sq - 1
    norm_x2_error = norm_x2_sq - 1
    if args.constraint_mode == "ema":
        coef_x_norm = params['norm_x_ema'].reshape(1, -1) - 1
    elif args.constraint_mode == "single_batch":
        coef_x_norm = norm_x_error
    else:  # two_batch, same_episodes
        coef_x_norm = norm_x2_error
    V_x_norm = coef_x_norm ** 2 / 2

    # 2. y-norm coefficient (current and next)
    norm_y_error = norm_y_sq - 1
    norm_y2_error = norm_y2_sq - 1
    next_norm_y_error = next_norm_y_sq - 1
    next_norm_y2_error = next_norm_y2_sq - 1
    if args.constraint_mode == "ema":
        coef_y_norm_current = params['norm_y_ema'].reshape(1, -1) - 1
        coef_y_norm_next = params['norm_y_ema'].reshape(1, -1) - 1
    elif args.constraint_mode == "single_batch":
        coef_y_norm_current = norm_y_error
        coef_y_norm_next = next_norm_y_error
    else:  # two_batch, same_episodes
        coef_y_norm_current = norm_y2_error
        coef_y_norm_next = next_norm_y2_error
    V_y_norm = coef_y_norm_current ** 2 / 2

    # 3. xy-phase coefficient (current and next)
    if args.constraint_mode == "ema":
        coef_xy_phase_current = params['phase_xy_ema'].reshape(1, -1)
        coef_xy_phase_next = params['phase_xy_ema'].reshape(1, -1)
    elif args.constraint_mode == "single_batch":
        coef_xy_phase_current = phase_xy
        coef_xy_phase_next = next_phase_xy
    else:  # two_batch, same_episodes
        coef_xy_phase_current = phase_xy2
        coef_xy_phase_next = next_phase_xy2
    V_xy_phase = coef_xy_phase_current ** 2 / 2

    # 4. Crossed-term correlation coefficients
    corr_xy_real_lower = jnp.tril(corr_xy_real, k=-1)
    corr_xy_imag_lower = jnp.tril(corr_xy_imag, k=-1)
    if use_sg_ip:
        corr_yx_real_lower = jnp.tril(corr_yx_real_b1, k=-1)
        corr_yx_imag_lower = jnp.tril(corr_yx_imag_b1, k=-1)
    else:
        corr_yx_real_lower = jnp.tril(corr_xy_real.T, k=-1)
        corr_yx_imag_lower = jnp.tril(corr_xy_imag.T, k=-1)

    corr_xy2_real_lower = jnp.tril(cross_xy2_real, k=-1)
    corr_xy2_imag_lower = jnp.tril(cross_xy2_imag, k=-1)
    if use_sg_ip:
        corr_yx2_real_lower = jnp.tril(corr_yx_real_b2, k=-1)
        corr_yx2_imag_lower = jnp.tril(corr_yx_imag_b2, k=-1)
    else:
        corr_yx2_real_lower = jnp.tril(cross_xy2_real.T, k=-1)
        corr_yx2_imag_lower = jnp.tril(cross_xy2_imag.T, k=-1)

    # Next-state correlations (batch 1 and 2)
    next_corr_xy_real = multi_ip(next_x_r, _sg(next_y_r)) + multi_ip(next_x_i, _sg(next_y_i))
    next_corr_xy_imag = -multi_ip(next_x_r, _sg(next_y_i)) + multi_ip(next_x_i, _sg(next_y_r))
    next_corr_xy_real_lower = jnp.tril(next_corr_xy_real, k=-1)
    next_corr_xy_imag_lower = jnp.tril(next_corr_xy_imag, k=-1)
    if use_sg_ip:
        next_corr_yx_real = multi_ip(next_y_r, _sg(next_x_r)) + multi_ip(next_y_i, _sg(next_x_i))
        next_corr_yx_imag = -multi_ip(next_y_i, _sg(next_x_r)) + multi_ip(next_y_r, _sg(next_x_i))
        next_corr_yx_real_lower = jnp.tril(next_corr_yx_real, k=-1)
        next_corr_yx_imag_lower = jnp.tril(next_corr_yx_imag, k=-1)
    else:
        next_corr_yx_real_lower = jnp.tril(next_corr_xy_real.T, k=-1)
        next_corr_yx_imag_lower = jnp.tril(next_corr_xy_imag.T, k=-1)

    next_cross_xy2_real = multi_ip(next_x2_r, _sg(next_y2_r)) + multi_ip(next_x2_i, _sg(next_y2_i))
    next_cross_xy2_imag = -multi_ip(next_x2_r, _sg(next_y2_i)) + multi_ip(next_x2_i, _sg(next_y2_r))
    next_corr_xy2_real_lower = jnp.tril(next_cross_xy2_real, k=-1)
    next_corr_xy2_imag_lower = jnp.tril(next_cross_xy2_imag, k=-1)
    if use_sg_ip:
        next_corr_yx2_real = multi_ip(next_y2_r, _sg(next_x2_r)) + multi_ip(next_y2_i, _sg(next_x2_i))
        next_corr_yx2_imag = -multi_ip(next_y2_i, _sg(next_x2_r)) + multi_ip(next_y2_r, _sg(next_x2_i))
        next_corr_yx2_real_lower = jnp.tril(next_corr_yx2_real, k=-1)
        next_corr_yx2_imag_lower = jnp.tril(next_corr_yx2_imag, k=-1)
    else:
        next_corr_yx2_real_lower = jnp.tril(next_cross_xy2_real.T, k=-1)
        next_corr_yx2_imag_lower = jnp.tril(next_cross_xy2_imag.T, k=-1)

    if args.constraint_mode == "ema":
        coef_corr_xy_real_lower_current = jnp.tril(params['corr_xy_real_ema'], k=-1)
        coef_corr_xy_imag_lower_current = jnp.tril(params['corr_xy_imag_ema'], k=-1)
        coef_corr_yx_real_lower_current = jnp.tril(params['corr_xy_real_ema'].T, k=-1)
        coef_corr_yx_imag_lower_current = jnp.tril(params['corr_xy_imag_ema'].T, k=-1)
        coef_corr_xy_real_lower_next = coef_corr_xy_real_lower_current
        coef_corr_xy_imag_lower_next = coef_corr_xy_imag_lower_current
        coef_corr_yx_real_lower_next = coef_corr_yx_real_lower_current
        coef_corr_yx_imag_lower_next = coef_corr_yx_imag_lower_current
    elif args.constraint_mode == "single_batch":
        coef_corr_xy_real_lower_current = corr_xy_real_lower
        coef_corr_xy_imag_lower_current = corr_xy_imag_lower
        coef_corr_yx_real_lower_current = corr_yx_real_lower
        coef_corr_yx_imag_lower_current = corr_yx_imag_lower
        coef_corr_xy_real_lower_next = next_corr_xy_real_lower
        coef_corr_xy_imag_lower_next = next_corr_xy_imag_lower
        coef_corr_yx_real_lower_next = next_corr_yx_real_lower
        coef_corr_yx_imag_lower_next = next_corr_yx_imag_lower
    else:  # two_batch, same_episodes
        coef_corr_xy_real_lower_current = corr_xy2_real_lower
        coef_corr_xy_imag_lower_current = corr_xy2_imag_lower
        coef_corr_yx_real_lower_current = corr_yx2_real_lower
        coef_corr_yx_imag_lower_current = corr_yx2_imag_lower
        coef_corr_xy_real_lower_next = next_corr_xy2_real_lower
        coef_corr_xy_imag_lower_next = next_corr_xy2_imag_lower
        coef_corr_yx_real_lower_next = next_corr_yx2_real_lower
        coef_corr_yx_imag_lower_next = next_corr_yx2_imag_lower

    # Per-(i,j) correlation penalty matrix (shape k×k, lower-tri only).
    # Each entry sums the squared real and imaginary parts of both xy and yx
    # cross-correlations for the same (i,j) pair.
    V_xy_corr_matrix = (
        coef_corr_xy_real_lower_current ** 2
        + coef_corr_xy_imag_lower_current ** 2
        + coef_corr_yx_real_lower_current ** 2
        + coef_corr_yx_imag_lower_current ** 2
    ) / 2
    V_xy_corr = jnp.sum(V_xy_corr_matrix, -1).reshape(1, -1)

    # ------------------------------------------------------------------
    # Assemble V and ∇V
    # ------------------------------------------------------------------
    V = V_x_norm + V_y_norm + V_xy_phase + V_xy_corr  # shape (1, k)

    nabla_x_r_V = (
        2 * coef_x_norm * x_r
        + (-coef_xy_phase_current * y_i)
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_xy_real_lower_current, y_r)
        + (-2 * jnp.einsum('jk,ik->ij', coef_corr_xy_imag_lower_current, y_i))
    )
    nabla_x_i_V = (
        2 * coef_x_norm * x_i
        + coef_xy_phase_current * y_r
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_xy_real_lower_current, y_i)
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_xy_imag_lower_current, y_r)
    )
    nabla_y_r_V = (
        2 * coef_y_norm_current * y_r
        + coef_xy_phase_current * x_i
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower_current, x_r)
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower_current, x_i)
    )
    nabla_y_i_V = (
        2 * coef_y_norm_current * y_i
        + (-coef_xy_phase_current * x_r)
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower_current, x_i)
        + (-2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower_current, x_r))
    )
    next_nabla_y_r_V = (
        2 * coef_y_norm_next * next_y_r
        + coef_xy_phase_next * next_x_i
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower_next, next_x_r)
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower_next, next_x_i)
    )
    next_nabla_y_i_V = (
        2 * coef_y_norm_next * next_y_i
        + (-coef_xy_phase_next * next_x_r)
        + 2 * jnp.einsum('jk,ik->ij', coef_corr_yx_real_lower_next, next_x_i)
        + (-2 * jnp.einsum('jk,ik->ij', coef_corr_yx_imag_lower_next, next_x_r))
    )

    nabla_V = {
        'x_r': nabla_x_r_V,
        'x_i': nabla_x_i_V,
        'y_r': nabla_y_r_V,
        'y_i': nabla_y_i_V,
        'next_y_r': next_nabla_y_r_V,
        'next_y_i': next_nabla_y_i_V,
    }

    V_components = {
        'V_x_norm': V_x_norm,
        'V_y_norm': V_y_norm,
        'V_xy_phase': V_xy_phase,
        'V_xy_corr_matrix': V_xy_corr_matrix,
        'norm_x_sq': norm_x_sq,
        'norm_y_sq': norm_y_sq,
        # Raw (unsquared) constraint errors — used by complex_allo dual loss.
        # Already computed above; zero extra cost.
        'c_x_norm':       coef_x_norm,                        # shape (1, k)
        'c_y_norm':       coef_y_norm_current,                # shape (1, k)
        'c_xy_phase':     coef_xy_phase_current,              # shape (1, k)
        'c_xy_corr_real': coef_corr_xy_real_lower_current,    # shape (k, k)
        'c_xy_corr_imag': coef_corr_xy_imag_lower_current,    # shape (k, k)
        'c_yx_corr_real': coef_corr_yx_real_lower_current,    # shape (k, k)
        'c_yx_corr_imag': coef_corr_yx_imag_lower_current,    # shape (k, k)
    }

    return V, nabla_V, constraint_estimator_loss, V_components


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement strategy factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_clf_enforcement_fn(args):
    """Factory returning the CLF controller enforcement function.

    The returned callable computes a QP-style corrective control:
        barrier = max(0, f·∇V + λ·V) / (‖∇V‖² + ε)
        u       = barrier · ∇V
        loss   += ip(feat, sg(u))

    This is a Python-level closure: the method choice is resolved at
    create_update_function() time, so no runtime branching appears inside
    the JIT-compiled body.
    """
    lambda_x = args.lambda_x

    def clf_enforcement(V, nabla_V, f_vectors, features, params, ip, sg, V_components):
        """
        Args:
            V:            shape (1, k)
            nabla_V:      dict — x_r, x_i, y_r, y_i, next_y_r, next_y_i
            f_vectors:    dict — x_real, x_imag, y_0_real, y_0_imag,
                                 y_res_real, y_res_imag
            features:     dict — x_r, x_i, y_r, y_i  (batch-1 current)
            params:       full param dict (unused by CLF enforcement)
            ip:           weighted inner product fn
            sg:           stop_gradient alias
            V_components: dict — V_x_norm, V_y_norm, V_xy_phase, V_xy_corr_matrix, …
                          (unused by CLF enforcement; accepted for API compatibility)
        Returns:
            (clf_loss scalar, aux dict)
        """
        x_r = features['x_r']
        x_i = features['x_i']
        y_r = features['y_r']
        y_i = features['y_i']

        nabla_x_r_V = nabla_V['x_r']
        nabla_x_i_V = nabla_V['x_i']
        nabla_y_r_V = nabla_V['y_r']
        nabla_y_i_V = nabla_V['y_i']
        next_nabla_y_r_V = nabla_V['next_y_r']
        next_nabla_y_i_V = nabla_V['next_y_i']

        f_x_real = f_vectors['x_real']
        f_x_imag = f_vectors['x_imag']
        f_y_0_real = f_vectors['y_0_real']
        f_y_0_imag = f_vectors['y_0_imag']
        f_y_res_real = f_vectors['y_res_real']
        f_y_res_imag = f_vectors['y_res_imag']

        # ‖∇V‖²
        norm_nabla_V_sq = (
            ip(nabla_x_r_V, nabla_x_r_V)
            + ip(nabla_x_i_V, nabla_x_i_V)
            + ip(nabla_y_r_V, nabla_y_r_V)
            + ip(nabla_y_i_V, nabla_y_i_V)
        )

        # f · ∇V  (Lie derivative of V along the dynamics residual)
        f_dot_nabla_V = (
            ip(f_x_real, nabla_x_r_V)
            + ip(f_x_imag, nabla_x_i_V)
            + ip(f_y_0_real, next_nabla_y_r_V)
            + ip(f_y_res_real, nabla_y_r_V)
            + ip(f_y_0_imag, next_nabla_y_i_V)
            + ip(f_y_res_imag, nabla_y_i_V)
        )

        clf_num = f_dot_nabla_V + lambda_x * V
        barrier = jnp.maximum(0, clf_num) / (norm_nabla_V_sq + 1e-8)

        u_x_r = barrier * nabla_x_r_V
        u_x_i = barrier * nabla_x_i_V
        u_y_r = barrier * nabla_y_r_V
        u_y_i = barrier * nabla_y_i_V

        clf_loss_x_r = ip(x_r, sg(u_x_r)).sum()
        clf_loss_x_i = ip(x_i, sg(u_x_i)).sum()
        clf_loss_y_r = ip(y_r, sg(u_y_r)).sum()
        clf_loss_y_i = ip(y_i, sg(u_y_i)).sum()

        clf_loss = clf_loss_x_r + clf_loss_x_i + clf_loss_y_r + clf_loss_y_i

        enforcement_aux = {
            'clf_loss': clf_loss,
            'clf_loss_x_real': clf_loss_x_r,
            'clf_loss_x_imag': clf_loss_x_i,
            'clf_loss_y_real': clf_loss_y_r,
            'clf_loss_y_imag': clf_loss_y_i,
            'barrier': barrier.mean(),
        }
        return clf_loss, enforcement_aux

    return clf_enforcement


def _make_barrier_enforcement_fn(args):
    """Factory returning the single-coefficient barrier enforcement function.

    The returned callable applies a direct quadratic penalty:
        loss += sg(barrier_coef) · V.sum()

    barrier_coef is held in params but is stop_grad'd inside the loss so
    the optimizer never updates it.  It is instead updated externally after
    each gradient step (see _external_barrier_update).
    """
    def barrier_enforcement(V, nabla_V, f_vectors, features, params, ip, sg, V_components):
        """
        Args:
            V:      shape (1, k)
            params: must contain 'barrier_coef': scalar array
            (all other args accepted for API compatibility but unused)
        Returns:
            (barrier_loss scalar, aux dict)
        """
        barrier_coef = params['barrier_coef']
        barrier_loss = sg(barrier_coef) * V.sum()

        enforcement_aux = {
            'barrier_loss': barrier_loss,
            'barrier_coef': barrier_coef,
            # Expose V_mean so update_encoder can compute the external update
            'V_mean': V.mean(),
        }
        return barrier_loss, enforcement_aux

    return barrier_enforcement


def _make_granular_barrier_enforcement_fn(args):
    """Factory returning the granular per-constraint barrier enforcement function.

    The returned callable applies a direct quadratic penalty with one
    independent coefficient per constraint instance:

        loss += Σ_i  sg(bc['x_norm'][i])       · V_x_norm[i]
              + Σ_i  sg(bc['y_norm'][i])       · V_y_norm[i]
              + Σ_i  sg(bc['xy_phase'][i])     · V_xy_phase[i]
              + Σ_{i>j} sg(bc['xy_corr'][i,j]) · V_xy_corr_matrix[i,j]

    barrier_coefs is held in params but is stop_grad'd inside the loss so
    the optimizer never updates it.  It is instead updated externally after
    each gradient step (see _external_granular_barrier_update).
    """
    def granular_barrier_enforcement(V, nabla_V, f_vectors, features, params, ip, sg, V_components):
        """
        Args:
            V:            shape (1, k) — total Lyapunov value per eigenvector
            V_components: dict — must contain V_x_norm (1,k), V_y_norm (1,k),
                          V_xy_phase (1,k), V_xy_corr_matrix (k,k)
            params:       must contain 'barrier_coefs': dict with per-constraint
                          coefficient arrays (x_norm, y_norm, xy_phase, xy_corr)
            (all other args accepted for API compatibility but unused)
        Returns:
            (barrier_loss scalar, aux dict)
        """
        bc = params['barrier_coefs']
        V_x_norm      = V_components['V_x_norm']          # shape (1, k)
        V_y_norm      = V_components['V_y_norm']          # shape (1, k)
        V_xy_phase    = V_components['V_xy_phase']        # shape (1, k)
        V_xy_corr_mat = V_components['V_xy_corr_matrix']  # shape (k, k)

        barrier_loss = (
            (sg(bc['x_norm']).reshape(1, -1)     * V_x_norm).sum()
            + (sg(bc['y_norm']).reshape(1, -1)   * V_y_norm).sum()
            + (sg(bc['xy_phase']).reshape(1, -1) * V_xy_phase).sum()
            + (sg(bc['xy_corr'])                 * V_xy_corr_mat).sum()
        )

        enforcement_aux = {
            'barrier_loss': barrier_loss,
            # Per-type mean coefficients for logging (scalars)
            'barrier_coef_x_norm':   bc['x_norm'].mean(),
            'barrier_coef_y_norm':   bc['y_norm'].mean(),
            'barrier_coef_xy_phase': bc['xy_phase'].mean(),
            'barrier_coef_xy_corr':  bc['xy_corr'].mean(),
            # Per-constraint violation signals for the external update
            'V_bc_x_norm':   V_x_norm.reshape(-1),   # shape (k,)
            'V_bc_y_norm':   V_y_norm.reshape(-1),   # shape (k,)
            'V_bc_xy_phase': V_xy_phase.reshape(-1), # shape (k,)
            'V_bc_xy_corr':  V_xy_corr_mat,          # shape (k, k)
        }
        return barrier_loss, enforcement_aux

    return granular_barrier_enforcement


def _make_complex_allo_enforcement_fn(args):
    """Factory returning the complex ALLO enforcement function.

    Combines two mechanisms:
      1. Dual variables multiplying each linear constraint error — mirrors the
         augmented-Lagrangian structure of ALLO-Ext, but over the full complex
         bi-orthogonality constraint set.
      2. A scalar barrier identical to the "barrier" enforcement method:
             sg(barrier_coef) · V.sum()

    The dual variables and barrier_coef are stop-grad'd inside the loss and
    updated externally after each gradient step via _external_complex_allo_update.

    The raw constraint errors (c_*) come from V_components, which are computed
    in _compute_lyapunov_terms with use_sg_ip=True (default) so that stop
    gradients are applied to the smaller-indexed eigenvector in each
    bi-orthogonality inner product — breaking ordering symmetries and making
    ordered eigenvectors the stable equilibrium.
    """
    def complex_allo_enforcement(V, nabla_V, f_vectors, features, params, ip, sg, V_components):
        """
        Args:
            V:            shape (1, k) — per-eigenvector Lyapunov values
            V_components: must contain c_x_norm, c_y_norm, c_xy_phase (shape (1,k))
                          and c_xy_corr_real/imag, c_yx_corr_real/imag (shape (k,k))
            params:       must contain 'duals' (dict) and 'barrier_coef' (scalar)
            (all other args accepted for API compatibility)
        Returns:
            (enforcement_loss scalar, aux dict)
        """
        duals = params['duals']

        # Linear constraint errors (raw, not squared)
        c_x_norm      = V_components['c_x_norm']       # (1, k)
        c_y_norm      = V_components['c_y_norm']        # (1, k)
        c_xy_phase    = V_components['c_xy_phase']      # (1, k)
        c_xy_corr_real = V_components['c_xy_corr_real'] # (k, k) lower-tri
        c_xy_corr_imag = V_components['c_xy_corr_imag'] # (k, k) lower-tri
        c_yx_corr_real = V_components['c_yx_corr_real'] # (k, k) lower-tri
        c_yx_corr_imag = V_components['c_yx_corr_imag'] # (k, k) lower-tri

        # Dual loss: sg(dual) * constraint_error for each constraint type.
        # Duals are stop-grad'd so no gradient flows back into them through
        # the optimizer; they are updated externally (augmented-Lagrangian style).
        dual_loss = (
            (sg(duals['x_norm']).reshape(1, -1)      * c_x_norm).sum()
            + (sg(duals['y_norm']).reshape(1, -1)    * c_y_norm).sum()
            + (sg(duals['xy_phase']).reshape(1, -1)  * c_xy_phase).sum()
            + (sg(duals['xy_corr_real'])              * c_xy_corr_real).sum()
            + (sg(duals['xy_corr_imag'])              * c_xy_corr_imag).sum()
            + (sg(duals['yx_corr_real'])              * c_yx_corr_real).sum()
            + (sg(duals['yx_corr_imag'])              * c_yx_corr_imag).sum()
        )

        # Barrier loss: identical to _make_barrier_enforcement_fn.
        barrier_coef = params['barrier_coef']
        barrier_loss = sg(barrier_coef) * V.sum()

        enforcement_loss = dual_loss + barrier_loss

        enforcement_aux = {
            'dual_loss':    dual_loss,
            'barrier_loss': barrier_loss,
            'barrier_coef': barrier_coef,
            'V_mean':       V.mean(),
            # Raw constraint errors forwarded for external dual update
            'c_x_norm':       c_x_norm.reshape(-1),
            'c_y_norm':       c_y_norm.reshape(-1),
            'c_xy_phase':     c_xy_phase.reshape(-1),
            'c_xy_corr_real': c_xy_corr_real,
            'c_xy_corr_imag': c_xy_corr_imag,
            'c_yx_corr_real': c_yx_corr_real,
            'c_yx_corr_imag': c_yx_corr_imag,
        }
        return enforcement_loss, enforcement_aux

    return complex_allo_enforcement


def _external_barrier_update(params, V_mean, args):
    """Update barrier_coef after the gradient step (called inside update_encoder).

    Only active when constraint_enforcement_method == "barrier".  Increases
    barrier_coef whenever mean constraint violation V_mean > 0, mirroring the
    augmented-Lagrangian update pattern.

    Args:
        params:  post-optimizer param dict (mutable)
        V_mean:  scalar — mean Lyapunov value from the forward pass
        args:    training arguments

    Returns:
        Updated params dict.
    """
    min_barrier = getattr(args, 'min_barrier_coefs', 0.0)
    max_barrier = getattr(args, 'max_barrier_coefs', 10.0)
    lr_barrier  = getattr(args, 'lr_barrier_coefs',  0.01)

    barrier_update = lr_barrier * jnp.maximum(0.0, V_mean)
    params['barrier_coef'] = jnp.clip(
        params['barrier_coef'] + barrier_update,
        min_barrier, max_barrier,
    )
    return params


def _external_granular_barrier_update(params, V_bc_signals, args):
    """Update barrier_coefs after the gradient step (called inside update_encoder).

    Only active when constraint_enforcement_method == "granular_barrier".
    Increases each per-constraint coefficient independently whenever its
    violation signal > 0, mirroring the augmented-Lagrangian update pattern.

    Args:
        params:        post-optimizer param dict (mutable)
        V_bc_signals:  dict with keys x_norm (k,), y_norm (k,), xy_phase (k,),
                       xy_corr (k,k) — per-constraint violation signals
        args:          training arguments

    Returns:
        Updated params dict.
    """
    min_barrier = getattr(args, 'min_barrier_coefs', 0.0)
    max_barrier = getattr(args, 'max_barrier_coefs', 10.0)
    lr_barrier  = getattr(args, 'lr_barrier_coefs',  0.01)

    bc = params['barrier_coefs']
    for key in ('x_norm', 'y_norm', 'xy_phase', 'xy_corr'):
        bc[key] = jnp.clip(
            bc[key] + lr_barrier * jnp.maximum(0.0, V_bc_signals[key]),
            min_barrier, max_barrier,
        )
    params['barrier_coefs'] = bc
    return params


def _external_complex_allo_update(params, aux, args):
    """Update dual variables and barrier_coef after the gradient step.

    Only active when constraint_enforcement_method == "complex_allo".

    Dual update (augmented-Lagrangian rule):
        duals[key] += lr_duals * constraint_error[key]
        duals[key]  = clip(duals[key], min_duals, max_duals)

    Barrier update: delegates to the existing _external_barrier_update so the
    barrier component is treated identically to the "barrier" method.

    Args:
        params: post-optimizer param dict (mutable)
        aux:    enforcement_aux dict returned by the loss (contains c_* errors
                and V_mean)
        args:   training arguments

    Returns:
        Updated params dict.
    """
    lr_duals  = getattr(args, 'lr_duals',   0.001)
    min_duals = getattr(args, 'min_duals', -100.0)
    max_duals = getattr(args, 'max_duals',  100.0)

    duals = params['duals']
    for key in ('x_norm', 'y_norm', 'xy_phase',
                'xy_corr_real', 'xy_corr_imag',
                'yx_corr_real', 'yx_corr_imag'):
        # Reshape aux errors to match duals shape (aux flattens x_norm/y_norm/xy_phase)
        err = aux[f'c_{key}']
        if key in ('x_norm', 'y_norm', 'xy_phase'):
            err = err.reshape(duals[key].shape)
        duals[key] = jnp.clip(duals[key] + lr_duals * err, min_duals, max_duals)
    params['duals'] = duals

    # Barrier update — identical logic to the "barrier" method
    params = _external_barrier_update(params, aux['V_mean'], args)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: eigenvalue target computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lambda_targets(
    x_r, x_i, y_r, y_i,
    next_x_r, next_x_i, next_y_r, next_y_i,
    norm_x_sq_est, norm_y_sq_est,
    eigenvalue_estimation_method, normalize_eigenvalue_targets,
    ip,
):
    """Compute per-eigenvector lambda targets for the EMA eigenvalue estimators.

    Args:
        x_r, x_i:                  Batch-1 current right-eigenvector features
        y_r, y_i:                  Batch-1 current left-eigenvector features
        next_x_r, next_x_i:        Batch-1 next-step right-eigenvector features
        next_y_r, next_y_i:        Batch-1 next-step left-eigenvector features
        norm_x_sq_est:             Squared norm of x, shape (1, k) — pre-computed
                                   by caller (also used downstream for graph loss)
        norm_y_sq_est:             Squared norm of y, shape (1, k) — same
        eigenvalue_estimation_method: One of 'separate', 'average', 'two_sided'
        normalize_eigenvalue_targets: Whether to normalize Rayleigh quotients by
                                   state norms (ignored for 'two_sided')
        ip:                        Weighted inner-product closure

    Returns:
        Tuple (lambda_x_r, lambda_x_i, lambda_y_r, lambda_y_i), each shape (1, k).
        For 'two_sided', lambda_x == lambda_y (same common estimate for both).
    """
    if eigenvalue_estimation_method == 'two_sided':
        # Bi-orthogonal complex division: lambda = <y, next_x> / <y, x>
        # normalize_eigenvalue_targets is ignored; the division denominator
        # already normalises the estimate.

        # Numerator (A = <y, next_x>)
        num_r = ip(y_r, next_x_r) + ip(y_i, next_x_i)
        num_i = ip(y_r, next_x_i) - ip(y_i, next_x_r)
        # Denominator (B = <y, x>)
        den_r = ip(y_r, x_r) + ip(y_i, x_i)
        den_i = ip(y_r, x_i) - ip(y_i, x_r)
        # Denominator Magnitude Squared (|B|^2)
        den_mag_sq = jnp.maximum(den_r**2 + den_i**2, 1e-6)
        # Complex Division
        lambda_r = (num_r * den_r + num_i * den_i) / den_mag_sq
        lambda_i = (num_i * den_r - num_r * den_i) / den_mag_sq

        # Assign common lambda to both x and y so downstream EMA loss works
        # without modification.
        return lambda_r, lambda_i, lambda_r, lambda_i

    else:
        # 'separate' and 'average': Rayleigh quotient numerators
        rq_x_r = ip(x_r, next_x_r) + ip(x_i, next_x_i)
        rq_x_i = ip(x_r, next_x_i) - ip(x_i, next_x_r)
        rq_y_r = ip(y_r, next_y_r) + ip(y_i, next_y_i)
        rq_y_i = ip(y_r, next_y_i) - ip(y_i, next_y_r)

        if normalize_eigenvalue_targets:
            norm_x_clip = jnp.maximum(norm_x_sq_est, 1e-6)
            norm_y_clip = jnp.maximum(norm_y_sq_est, 1e-6)
            return (
                rq_x_r / norm_x_clip,
                rq_x_i / norm_x_clip,
                rq_y_r / norm_y_clip,
                rq_y_i / norm_y_clip,
            )
        else:
            return rq_x_r, rq_x_i, rq_y_r, rq_y_i


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

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
    }

    # EMA estimates for eigenvalues
    if args.eigenvalue_estimation_method != 'average':
        # Separate eigenvalue estimates for x (right) and y (left) eigenvectors
        # Used by 'separate' and 'two_sided' methods
        params.update({
            'lambda_x_real': jnp.ones((k,)),
            'lambda_x_imag': jnp.zeros((k,)),
            'lambda_y_real': jnp.ones((k,)),
            'lambda_y_imag': jnp.zeros((k,)),
        })
    else:
        # Shared eigenvalue estimate (averaged from x and y Rayleigh quotients)
        params.update({
            'lambda_real': jnp.ones((k,)),
            'lambda_imag': jnp.zeros((k,)),
        })

    # Add EMA constraint estimators only for EMA constraint_mode
    if args.constraint_mode == "ema":
        params.update({
            'norm_x_ema': jnp.zeros((k,)),
            'norm_y_ema': jnp.zeros((k,)),
            'phase_xy_ema': jnp.zeros((k,)),
            'corr_xy_real_ema': jnp.zeros((k, k)),
            'corr_xy_imag_ema': jnp.zeros((k, k)),
        })

    # Add barrier coefficient(s) for increasing-barrier enforcement methods
    enforcement_method = getattr(args, 'constraint_enforcement_method', 'clf')
    if enforcement_method == "barrier":
        barrier_init = getattr(args, 'barrier_initial_val', 0.5)
        params['barrier_coef'] = jnp.array(barrier_init)
    elif enforcement_method == "granular_barrier":
        barrier_init = getattr(args, 'barrier_initial_val', 0.5)
        params['barrier_coefs'] = {
            'x_norm':   jnp.full((k,), barrier_init),
            'y_norm':   jnp.full((k,), barrier_init),
            'xy_phase': jnp.full((k,), barrier_init),
            'xy_corr':  jnp.full((k, k), barrier_init),
        }
    elif enforcement_method == "complex_allo":
        barrier_init = getattr(args, 'barrier_initial_val', 0.5)
        duals_init   = getattr(args, 'duals_initial_val',   0.0)
        params['barrier_coef'] = jnp.array(barrier_init)
        params['duals'] = {
            'x_norm':       jnp.full((k,),    duals_init),
            'y_norm':       jnp.full((k,),    duals_init),
            'xy_phase':     jnp.full((k,),    duals_init),
            'xy_corr_real': jnp.full((k, k),  duals_init),
            'xy_corr_imag': jnp.full((k, k),  duals_init),
            'yx_corr_real': jnp.full((k, k),  duals_init),
            'yx_corr_imag': jnp.full((k, k),  duals_init),
        }

    return params


def get_optimizer_masks(args):
    """
    Get optimizer masks for different parameter groups.

    Returns:
        Tuple of (encoder_mask, other_mask)
    """
    encoder_mask = {'encoder': True}
    other_mask = {'encoder': False}

    if args.eigenvalue_estimation_method != 'average':
        encoder_mask.update({
            'lambda_x_real': False,
            'lambda_x_imag': False,
            'lambda_y_real': False,
            'lambda_y_imag': False,
        })
        other_mask.update({
            'lambda_x_real': True,
            'lambda_x_imag': True,
            'lambda_y_real': True,
            'lambda_y_imag': True,
        })
    else:
        encoder_mask.update({
            'lambda_real': False,
            'lambda_imag': False,
        })
        other_mask.update({
            'lambda_real': True,
            'lambda_imag': True,
        })

    # EMA masks only for EMA constraint_mode
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

    # Barrier coefficient(s) are managed externally — excluded from optimizer
    enforcement_method = getattr(args, 'constraint_enforcement_method', 'clf')
    if enforcement_method == "barrier":
        encoder_mask['barrier_coef'] = False
        other_mask['barrier_coef'] = True
    elif enforcement_method == "granular_barrier":
        _bc_mask_false = {'x_norm': False, 'y_norm': False, 'xy_phase': False, 'xy_corr': False}
        _bc_mask_true  = {'x_norm': True,  'y_norm': True,  'xy_phase': True,  'xy_corr': True}
        encoder_mask['barrier_coefs'] = _bc_mask_false
        other_mask['barrier_coefs'] = _bc_mask_true
    elif enforcement_method == "complex_allo":
        # barrier_coef and duals are managed purely externally — excluded from
        # both optimizer groups (stop-grad'd in loss → zero gradients anyway).
        _duals_mask_false = {
            'x_norm': False, 'y_norm': False, 'xy_phase': False,
            'xy_corr_real': False, 'xy_corr_imag': False,
            'yx_corr_real': False, 'yx_corr_imag': False,
        }
        encoder_mask['barrier_coef'] = False
        other_mask['barrier_coef']   = False
        encoder_mask['duals'] = _duals_mask_false
        other_mask['duals']   = _duals_mask_false

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
    # ── Python-time method selection (resolved once, never branched at JAX
    #    trace time, so fully JIT-safe) ────────────────────────────────────
    enforcement_method = getattr(args, 'constraint_enforcement_method', 'clf')
    if enforcement_method == "clf":
        _enforce_constraints = _make_clf_enforcement_fn(args)
    elif enforcement_method == "barrier":
        _enforce_constraints = _make_barrier_enforcement_fn(args)
    elif enforcement_method == "granular_barrier":
        _enforce_constraints = _make_granular_barrier_enforcement_fn(args)
    elif enforcement_method == "complex_allo":
        _enforce_constraints = _make_complex_allo_enforcement_fn(args)
    else:
        raise ValueError(
            f"Unknown constraint_enforcement_method: {enforcement_method!r}. "
            "Choose 'clf', 'barrier', 'granular_barrier', or 'complex_allo'."
        )

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
            if args.eigenvalue_estimation_method != 'average':
                ema_lambda_x_r = params['lambda_x_real'].reshape(1, -1)
                ema_lambda_x_i = params['lambda_x_imag'].reshape(1, -1)
                ema_lambda_y_r = params['lambda_y_real'].reshape(1, -1)
                ema_lambda_y_i = params['lambda_y_imag'].reshape(1, -1)
            else:
                ema_lambda_r = params['lambda_real'].reshape(1, -1)
                ema_lambda_i = params['lambda_imag'].reshape(1, -1)
                ema_lambda_x_r = ema_lambda_r
                ema_lambda_x_i = ema_lambda_i
                ema_lambda_y_r = ema_lambda_r
                ema_lambda_y_i = ema_lambda_i

            # ----------------------------------------------------------------
            # 3. Update Lambda (Eigenvalue) Estimators
            # ----------------------------------------------------------------

            # Compute norms (Shape: (1,k)) — used unconditionally by graph loss
            # normalisation and chirality loss downstream.
            norm_x_sq_est = ip(x_r, x_r) + ip(x_i, x_i)
            norm_y_sq_est = ip(y_r, y_r) + ip(y_i, y_i)

            lambda_x_r, lambda_x_i, lambda_y_r, lambda_y_i = _compute_lambda_targets(
                x_r, x_i, y_r, y_i,
                next_x_r, next_x_i, next_y_r, next_y_i,
                norm_x_sq_est, norm_y_sq_est,
                args.eigenvalue_estimation_method,
                args.normalize_eigenvalue_targets,
                ip,
            )

            # EMA loss for eigenvalues
            if args.eigenvalue_estimation_method == 'average':
                new_lambda_real = 0.5 * (lambda_x_r + lambda_y_r)
                new_lambda_imag = 0.5 * (lambda_x_i + lambda_y_i)
                lambda_loss = (
                    ((sg(new_lambda_real) - ema_lambda_r) ** 2).sum()
                    + ((sg(new_lambda_imag) - ema_lambda_i) ** 2).sum()
                )
            else:
                # 'separate' and 'two_sided' both use disentangled x/y EMA params
                lambda_loss = (
                    ((sg(lambda_x_r) - ema_lambda_x_r) ** 2).sum()
                    + ((sg(lambda_x_i) - ema_lambda_x_i) ** 2).sum()
                    + ((sg(lambda_y_r) - ema_lambda_y_r) ** 2).sum()
                    + ((sg(lambda_y_i) - ema_lambda_y_i) ** 2).sum()
                )

            # ----------------------------------------------------------------
            # 4. Lyapunov function V and ∇V (shared between enforcement methods)
            # ----------------------------------------------------------------
            V, nabla_V, constraint_estimator_loss, V_components = _compute_lyapunov_terms(
                x_r, x_i, y_r, y_i,
                x2_r, x2_i, y2_r, y2_i,
                next_x_r, next_x_i, next_y_r, next_y_i,
                next_x2_r, next_x2_i, next_y2_r, next_y2_i,
                params, args, ip, multi_ip,
            )

            # ----------------------------------------------------------------
            # 5. Graph Loss (Dynamics) + Chirality
            # ----------------------------------------------------------------
            f_x_real = - next_x_r + ema_lambda_x_r * x_r - ema_lambda_x_i * x_i
            f_x_imag = - next_x_i + ema_lambda_x_i * x_r + ema_lambda_x_r * x_i

            f_y_0_real = - y_r
            f_y_0_imag = - y_i
            f_y_res_real = ema_lambda_y_r * y_r + ema_lambda_y_i * y_i
            f_y_res_imag = -ema_lambda_y_i * y_r + ema_lambda_y_r * y_i

            graph_loss_x_real = ip(x_r, sg(f_x_real))
            graph_loss_x_imag = ip(x_i, sg(f_x_imag))
            graph_loss_x = graph_loss_x_real + graph_loss_x_imag

            graph_loss_y_real = ip(next_y_r, sg(f_y_0_real)) + ip(y_r, sg(f_y_res_real))
            graph_loss_y_imag = ip(next_y_i, sg(f_y_0_imag)) + ip(y_i, sg(f_y_res_imag))
            graph_loss_y = graph_loss_y_real + graph_loss_y_imag

            if args.norm_graph_loss:
                # Divide each per-eigenvector component by its 2-norm to normalise the loss
                # scale. norm_x_sq_est / norm_y_sq_est have shape (1, k), matching graph_loss_x/y.
                norm_x_clip = jnp.maximum(norm_x_sq_est, 1e-6)
                norm_y_clip = jnp.maximum(norm_y_sq_est, 1e-6)
                graph_loss_x = graph_loss_x / sg(norm_x_clip)
                graph_loss_y = graph_loss_y / sg(norm_y_clip)

            graph_loss = (graph_loss_x + graph_loss_y).sum()

            x_i_normalized = x_i / jnp.sqrt(norm_x_sq_est + 1e-6)
            chirality_x = ip(sg(x_i_normalized**2), x_i_normalized)
            chirality_loss = ((chirality_x)**2).sum()

            # ----------------------------------------------------------------
            # 6. Constraint Enforcement (pluggable strategy)
            # ----------------------------------------------------------------
            f_vectors = {
                'x_real':   f_x_real,
                'x_imag':   f_x_imag,
                'y_0_real': f_y_0_real,
                'y_0_imag': f_y_0_imag,
                'y_res_real': f_y_res_real,
                'y_res_imag': f_y_res_imag,
            }
            features = {'x_r': x_r, 'x_i': x_i, 'y_r': y_r, 'y_i': y_i}

            enforcement_loss, enforcement_aux = _enforce_constraints(
                V, nabla_V, f_vectors, features, params, ip, sg, V_components,
            )

            # ----------------------------------------------------------------
            # 7. Total Loss
            # ----------------------------------------------------------------
            total_loss = (
                graph_loss
                + enforcement_loss
                + args.chirality_factor * chirality_loss
                + lambda_loss
                + constraint_estimator_loss
            )

            # Auxiliary metrics
            aux = {
                'total_loss': total_loss,
                'graph_loss': graph_loss,
                'chirality_loss': chirality_loss,
                'graph_loss_x_real': graph_loss_x_real.sum(),
                'graph_loss_x_imag': graph_loss_x_imag.sum(),
                'graph_loss_y_real': graph_loss_y_real.sum(),
                'graph_loss_y_imag': graph_loss_y_imag.sum(),
                'ema_lambda_x_real': ema_lambda_x_r.mean(),
                'ema_lambda_x_imag': ema_lambda_x_i.mean(),
                'ema_lambda_y_real': ema_lambda_y_r.mean(),
                'ema_lambda_y_imag': ema_lambda_y_i.mean(),
                'new_lambda_x_real': lambda_x_r.mean(),
                'new_lambda_x_imag': lambda_x_i.mean(),
                'new_lambda_y_real': lambda_y_r.mean(),
                'new_lambda_y_imag': lambda_y_i.mean(),
                **V_components,
                **enforcement_aux,
            }

            return total_loss, aux

        # Compute loss and gradients
        (total_loss, aux), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)

        # Gradient clipping
        max_norm = 1.0

        if args.use_global_grad_clip:
            grads_flat, grads_tree = jax.tree_util.tree_flatten(grads)
            grads_vector = jnp.concatenate([jnp.ravel(g) for g in grads_flat])
            grad_norm = jnp.linalg.norm(grads_vector)
            aux['grad_norm'] = grad_norm

            clip_factor = max_norm / jnp.maximum(grad_norm, max_norm)
            grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)

        else:
            encoder_grads_flat, _ = jax.tree_util.tree_flatten(grads['encoder'])
            encoder_grads_vector = jnp.concatenate([jnp.ravel(g) for g in encoder_grads_flat])
            encoder_grad_norm = jnp.linalg.norm(encoder_grads_vector)
            aux['grad_norm'] = encoder_grad_norm

            encoder_clip_factor = max_norm / jnp.maximum(encoder_grad_norm, max_norm)

            clipped_grads = {
                'encoder': jax.tree_util.tree_map(lambda g: g * encoder_clip_factor, grads['encoder']),
            }
            if args.eigenvalue_estimation_method != 'average':
                clipped_grads.update({
                    'lambda_x_real': jnp.clip(grads['lambda_x_real'], -1.0, 1.0),
                    'lambda_x_imag': jnp.clip(grads['lambda_x_imag'], -1.0, 1.0),
                    'lambda_y_real': jnp.clip(grads['lambda_y_real'], -1.0, 1.0),
                    'lambda_y_imag': jnp.clip(grads['lambda_y_imag'], -1.0, 1.0),
                })
            else:
                clipped_grads.update({
                    'lambda_real': jnp.clip(grads['lambda_real'], -1.0, 1.0),
                    'lambda_imag': jnp.clip(grads['lambda_imag'], -1.0, 1.0),
                })

            if args.constraint_mode == "ema":
                clipped_grads.update({
                    'norm_x_ema': jnp.clip(grads['norm_x_ema'], -1.0, 1.0),
                    'norm_y_ema': jnp.clip(grads['norm_y_ema'], -1.0, 1.0),
                    'phase_xy_ema': jnp.clip(grads['phase_xy_ema'], -1.0, 1.0),
                    'corr_xy_real_ema': jnp.clip(grads['corr_xy_real_ema'], -1.0, 1.0),
                    'corr_xy_imag_ema': jnp.clip(grads['corr_xy_imag_ema'], -1.0, 1.0),
                })

            # Barrier params are stop_grad'd in the loss — their gradients are
            # always zero, but must be included in clipped_grads to keep the
            # pytree structure consistent with encoder_state.params.
            if enforcement_method == "barrier":
                clipped_grads['barrier_coef'] = grads['barrier_coef']
            elif enforcement_method == "granular_barrier":
                clipped_grads['barrier_coefs'] = grads['barrier_coefs']
            elif enforcement_method == "complex_allo":
                clipped_grads['barrier_coef'] = grads['barrier_coef']
                clipped_grads['duals'] = grads['duals']

            grads = clipped_grads

        # Apply optimizer updates
        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        # ── External update for barrier coefficient(s) (barrier methods only) ─
        # Python-level branches: resolved at trace time, invisible to XLA.
        if enforcement_method == "barrier":
            new_params = _external_barrier_update(new_params, aux['V_mean'], args)
            aux['barrier_coef'] = new_params['barrier_coef']
        elif enforcement_method == "granular_barrier":
            V_bc_signals = {
                'x_norm':   aux['V_bc_x_norm'],
                'y_norm':   aux['V_bc_y_norm'],
                'xy_phase': aux['V_bc_xy_phase'],
                'xy_corr':  aux['V_bc_xy_corr'],
            }
            new_params = _external_granular_barrier_update(new_params, V_bc_signals, args)
            # Overwrite aux with post-update coef means for logging
            for _key in ('x_norm', 'y_norm', 'xy_phase', 'xy_corr'):
                aux[f'barrier_coef_{_key}'] = new_params['barrier_coefs'][_key].mean()
        elif enforcement_method == "complex_allo":
            new_params = _external_complex_allo_update(new_params, aux, args)
            aux['barrier_coef'] = new_params['barrier_coef']

        # Create new state
        new_encoder_state = encoder_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=encoder_state.step + 1
        )

        return new_encoder_state, total_loss, aux

    return update_encoder


def get_eigenvalues(params):
    """
    Extract eigenvalue estimates from params.

    Supports both shared ('average') and disentangled ('separate', 'two_sided')
    eigenvalue_estimation_method parameterizations. When disentangled,
    returns the average of x and y estimates as the canonical eigenvalue.
    """
    if 'lambda_real' in params:
        return params['lambda_real'], params['lambda_imag']
    else:
        # Disentangled: both lambda_x_imag and lambda_y_imag store +λ_i (same convention),
        # so a simple average is correct for both real and imaginary parts.
        lambda_real = 0.5 * (params['lambda_x_real'] + params['lambda_y_real'])
        lambda_imag = 0.5 * (params['lambda_x_imag'] + params['lambda_y_imag'])
        return lambda_real, lambda_imag


# Entry point for running this learner directly
if __name__ == "__main__":
    import tyro
    from src.config.ded_clf import Args
    from rep_algos.shared_training import learn_eigenvectors
    import rep_algos.clf_learner as clf_learner

    args = tyro.cli(Args)
    args.exp_name = "clf"
    learn_eigenvectors(args, clf_learner)
