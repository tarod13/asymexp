"""
Tabular Q-learning with optional potential-based reward shaping.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.gridworld import GridWorldState


# ===========================================================================
# Potential function
# ===========================================================================

def build_potential(
    hitting_times: np.ndarray,
    goal_idx: int,
    clamp_negatives: bool = False,
    potential_mode: str = "negative",
    potential_temp: float = 1.0,
) -> np.ndarray:
    """
    Build the potential Φ(s) used for reward shaping.

    Hitting times are normalised to [0, 100] before the transformation.
    Non-finite values are replaced with the column maximum before normalisation.
    If clamp_negatives=True, negative values are clamped to 0 before normalisation.

    potential_mode controls the transformation applied to the normalised
    hitting time h ∈ [0, 100]:
      "negative"    : Φ(s) = −h                          (default; closer = higher)
      "inverse"     : Φ(s) = 1 / (h / temp + 1e-5)
      "exp-negative": Φ(s) = exp(−h / temp)
    """
    h = hitting_times[:, goal_idx].copy()

    finite_mask = np.isfinite(h)
    if finite_mask.sum() == 0:
        return np.zeros(len(h), dtype=np.float32)

    h = np.where(
        finite_mask,
        h,
        float(h[finite_mask].max()) if finite_mask.any() else 0.0
    )

    if clamp_negatives:
        non_negative_mask = (h >= 0)
        h = np.where(
            non_negative_mask,
            h,
            float(h[non_negative_mask].max()) if non_negative_mask.any() else 0.0
        )

    h = h - np.min(h, axis=0, keepdims=True)  # shift so min is zero (prevents large negative potentials when normalising by max)
    h_max = np.max(h, axis=0, keepdims=True)  # max absolute value across states
    h = 100 * h / h_max.clip(1e-8)  # normalize to [0, 100]

    if potential_mode == "negative":
        return -h.astype(np.float32)
    elif potential_mode == "inverse":
        return (1.0 / (h / potential_temp + 1e-5)).astype(np.float32)
    elif potential_mode == "exp-negative":
        return np.exp(-h / potential_temp).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown potential_mode '{potential_mode}'. "
            "Expected one of: 'negative', 'inverse', 'exp-negative'."
        )


def build_all_potentials(
    hitting_times: np.ndarray,
    clamp_negatives: bool = False,
    potential_mode: str = "negative",
    potential_temp: float = 1.0,
) -> np.ndarray:
    """
    Build the full potential matrix Φ[s, g] for every possible goal g.

    Returns an [N, N] float32 array where column g equals
    build_potential(hitting_times, g, ...).  Pre-computing this once lets
    per-seed potentials be retrieved with a simple column slice.
    """
    N = hitting_times.shape[0]
    F = np.empty((N, N), dtype=np.float32)
    for g in range(N):
        F[:, g] = build_potential(
            hitting_times, g,
            clamp_negatives=clamp_negatives,
            potential_mode=potential_mode,
            potential_temp=potential_temp,
        )
    return F


# ===========================================================================
# Tabular Q-learning
# ===========================================================================

def run_q_learning(
    env,
    canonical_states: np.ndarray,
    goal_per_seed: np.ndarray,              # [num_seeds] int — fixed goal per task
    eval_starts_per_seed: np.ndarray,       # [num_seeds] int — fixed eval start per task
    total_steps: int,
    num_seeds: int = 1,
    max_steps_per_episode: int = 500,
    num_eval_episodes: int = 30,
    potential_per_seed: np.ndarray | None = None,  # [num_seeds, N] float32 or None
    train_start_per_seed: np.ndarray | None = None,  # [num_seeds] int or None → random
    min_goal_distance: int = 0,
    shaping_coef: float = 0.0,
    gamma: float = 0.99,
    lr: float = 0.1,
    epsilon: float = 0.1,
    seed: int = 0,
    eval_interval: int = 100_000,
    n_step_td: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run tabular ε-greedy Q-learning with one independent Q-table per seed.

    Each seed has a fixed goal.  Training runs for exactly total_steps
    environment steps.  Every eval_interval steps the greedy policy is
    evaluated for num_eval_episodes episodes from the fixed eval start.

    Returns
    -------
    eval_sr      : [num_chunks, num_seeds]  float  — success rate
    eval_len_all : [num_chunks, num_seeds]  float  — mean episode length (all)
    eval_len_suc : [num_chunks, num_seeds]  float  — mean episode length (successful only, NaN if none)
    eval_steps   : [num_chunks]             int    — training step at each checkpoint
    """
    num_states  = len(canonical_states)
    num_actions = 4

    full_to_can = np.full(env.width * env.height, -1, dtype=np.int32)
    for i, s in enumerate(canonical_states):
        full_to_can[int(s)] = i
    full_to_can_jax      = jnp.array(full_to_can,           dtype=jnp.int32)
    canonical_states_jax = jnp.array(canonical_states,      dtype=jnp.int32)
    eval_starts_jax      = jnp.array(eval_starts_per_seed,  dtype=jnp.int32)
    goal_jax             = jnp.array(goal_per_seed,         dtype=jnp.int32)

    if potential_per_seed is None:
        potential_jax = jnp.zeros((num_seeds, num_states), jnp.float32)
    else:
        potential_jax = jnp.array(potential_per_seed, jnp.float32)

    if train_start_per_seed is None:
        train_start_jax = jnp.full((num_seeds,), -1, dtype=jnp.int32)
    else:
        train_start_jax = jnp.array(train_start_per_seed, dtype=jnp.int32)

    # Valid-starts lookup table for distance-constrained random resets.
    if min_goal_distance > 0:
        coords = np.array(
            [(int(canonical_states[s]) % env.width,
              int(canonical_states[s]) // env.width)
             for s in range(num_states)]
        )
        taxi = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
        valid_starts_lists = []
        for g in range(num_states):
            vs = [s for s in range(num_states)
                  if s != g and taxi[s, g] >= min_goal_distance]
            if not vs:
                vs = [s for s in range(num_states) if s != g]
            valid_starts_lists.append(vs)
        max_valid = max(len(v) for v in valid_starts_lists)
        table  = np.zeros((num_states, max_valid), dtype=np.int32)
        counts = np.zeros(num_states, dtype=np.int32)
        for g, vs in enumerate(valid_starts_lists):
            table[g, :len(vs)] = vs
            counts[g] = len(vs)
        valid_table_jax  = jnp.array(table)
        valid_counts_jax = jnp.array(counts)
    else:
        valid_table_jax  = None
        valid_counts_jax = None

    # ── n-step return constants ────────────────────────────────────────
    n          = n_step_td
    k_indices  = jnp.arange(n)
    gammas_jax = jnp.array([gamma ** k for k in range(n)], dtype=jnp.float32)
    gamma_n    = float(gamma ** n)

    # ── Training chunk ────────────────────────────────────────────────
    def run_chunk(carry, chunk_key):
        def run_step(step_carry, step_key):
            Q, s, done, step_in_ep, goal, potential, fixed_start, buf_s, buf_a, buf_r, buf_ptr, buf_len = step_carry

            reset_key, eps_key, rand_key, env_key, tie_key = jax.random.split(step_key, 5)
            at_reset = step_in_ep == 0

            if valid_table_jax is not None:
                k = jax.random.randint(reset_key, (), 0, valid_counts_jax[goal])
                new_start_random = valid_table_jax[goal, k]
            else:
                new_start_raw = jax.random.randint(reset_key, (), 0, num_states)
                new_start_random = jnp.where(
                    new_start_raw == goal,
                    (goal + 1) % num_states,
                    new_start_raw,
                )
            new_start = jnp.where(fixed_start >= 0, fixed_start, new_start_random)
            s       = jnp.where(at_reset, new_start, s)
            done    = jnp.where(at_reset, jnp.array(False), done)
            buf_len = jnp.where(at_reset, jnp.zeros((), jnp.int32), buf_len)
            buf_ptr = jnp.where(at_reset, jnp.zeros((), jnp.int32), buf_ptr)

            use_random = jax.random.uniform(eps_key) < epsilon
            q_s    = Q[s]
            is_max = (q_s == q_s.max()).astype(jnp.float32)
            greedy_a = jax.random.choice(tie_key, num_actions, p=is_max / is_max.sum())
            random_a = jax.random.randint(rand_key, (), 0, num_actions)
            a        = jnp.where(use_random, random_a, greedy_a)

            full_idx  = canonical_states_jax[s]
            env_state = GridWorldState(
                position=jnp.stack(
                    [full_idx % env.width, full_idx // env.width]
                ).astype(env.dtype),
                terminal=jnp.array(False),
                steps=jnp.array(0, dtype=env.dtype),
            )
            next_env_state, _, _, _ = env.step(env_key, env_state, a)
            next_full = env.get_state_representation(next_env_state)
            can_idx   = full_to_can_jax[next_full]
            s_prime   = jnp.where(can_idx >= 0, can_idx, s)

            reached = s_prime == goal
            r = jnp.where(reached, 1.0, 0.0)
            r = r + shaping_coef * (gamma * potential[s_prime] - potential[s])

            # ── Write transition to n-step buffer ─────────────────────
            buf_s   = buf_s.at[buf_ptr].set(s)
            buf_a   = buf_a.at[buf_ptr].set(a)
            buf_r   = buf_r.at[buf_ptr].set(r)
            new_buf_ptr = (buf_ptr + 1) % n
            new_buf_len = jnp.minimum(buf_len + 1, n)
            # oldest valid entry in the circular buffer
            oldest = (new_buf_ptr - new_buf_len + n) % n

            # Bootstrap value: 0 at terminal, Q-max otherwise
            timeout       = step_in_ep == max_steps_per_episode - 1
            episode_end   = reached | timeout
            bootstrap_val = Q[s_prime].max() * jnp.where(reached, 0.0, 1.0)

            # ── Normal n-step update (buffer full, mid-episode) ────────
            # G = Σ_{k=0}^{n-1} γ^k · r_{oldest+k}  +  γ^n · bootstrap
            G_normal = (
                jnp.dot(gammas_jax, buf_r[(oldest + k_indices) % n])
                + gamma_n * bootstrap_val
            )
            do_normal = (new_buf_len == n) & ~episode_end
            Q = Q.at[buf_s[oldest], buf_a[oldest]].add(
                lr * jnp.where(do_normal, G_normal - Q[buf_s[oldest], buf_a[oldest]], 0.0)
            )

            # ── Episode-end flush: update all remaining buffer entries ─
            # Entry i (0 = oldest) gets a variable-length return to terminal.
            def flush_entry(i, Q):
                entry_idx = (oldest + i) % n
                entry_s   = buf_s[entry_idx]
                entry_a   = buf_a[entry_idx]
                b_indices = (entry_idx + k_indices) % n
                mask      = k_indices < (new_buf_len - i)
                remaining = jnp.maximum(new_buf_len - i, 0)
                G_flush   = (
                    jnp.dot(gammas_jax * mask, buf_r[b_indices])
                    + jnp.pow(gamma, remaining) * bootstrap_val
                )
                delta = G_flush - Q[entry_s, entry_a]
                Q = Q.at[entry_s, entry_a].add(
                    lr * jnp.where(i < new_buf_len, delta, 0.0)
                )
                return Q

            Q = jax.lax.cond(
                episode_end,
                lambda Q: jax.lax.fori_loop(0, n, flush_entry, Q),
                lambda Q: Q,
                Q,
            )

            # ── Reset buffer at episode end ────────────────────────────
            new_buf_len_out = jnp.where(episode_end, jnp.zeros((), jnp.int32), new_buf_len)
            new_buf_ptr_out = jnp.where(episode_end, jnp.zeros((), jnp.int32), new_buf_ptr)

            new_done        = done | reached
            new_s           = jnp.where(new_done, s, s_prime)
            next_step_in_ep = jnp.where(
                new_done | (step_in_ep == max_steps_per_episode - 1),
                jnp.zeros((), jnp.int32),
                step_in_ep + 1,
            )
            return (Q, new_s, new_done, next_step_in_ep, goal, potential, fixed_start,
                    buf_s, buf_a, buf_r, new_buf_ptr_out, new_buf_len_out), None

        step_keys = jax.random.split(chunk_key, eval_interval)
        new_carry, _ = jax.lax.scan(run_step, carry, step_keys)
        return new_carry

    vmapped_chunk = jax.jit(jax.vmap(run_chunk))

    # ── Evaluation ────────────────────────────────────────────────────
    def eval_one_seed(Q, eval_start, goal):
        """Greedy rollout; returns (success, length) per episode."""
        def eval_one_episode(episode_key):
            def eval_step(carry, step_key):
                s, done = carry
                env_key, tie_key = jax.random.split(step_key)
                q_s    = Q[s]
                is_max = (q_s == q_s.max()).astype(jnp.float32)
                a      = jax.random.choice(tie_key, num_actions, p=is_max / is_max.sum())

                full_idx  = canonical_states_jax[s]
                env_state = GridWorldState(
                    position=jnp.stack(
                        [full_idx % env.width, full_idx // env.width]
                    ).astype(env.dtype),
                    terminal=jnp.array(False),
                    steps=jnp.array(0, dtype=env.dtype),
                )
                next_env_state, _, _, _ = env.step(env_key, env_state, a)
                next_full = env.get_state_representation(next_env_state)
                can_idx   = full_to_can_jax[next_full]
                s_prime   = jnp.where(can_idx >= 0, can_idx, s)

                reached  = s_prime == goal
                new_done = done | reached
                new_s    = jnp.where(new_done, s, s_prime)
                return (new_s, new_done), reached

            step_keys = jax.random.split(episode_key, max_steps_per_episode)
            _, reached_steps = jax.lax.scan(
                eval_step,
                (eval_start, jnp.array(False)),
                step_keys,
            )  # reached_steps: [max_steps_per_episode] bool
            success = reached_steps.any()
            length  = jnp.where(
                success,
                jnp.argmax(reached_steps) + 1,
                max_steps_per_episode,
            )
            return success, length

        episode_keys = jax.random.split(jax.random.PRNGKey(0), num_eval_episodes)
        successes, lengths = jax.vmap(eval_one_episode)(episode_keys)
        return successes, lengths  # each [num_eval_episodes]

    vmapped_eval = jax.jit(jax.vmap(eval_one_seed))

    # ── Main loop ─────────────────────────────────────────────────────
    num_chunks = max(1, total_steps // eval_interval)

    seed_keys       = jax.random.split(jax.random.PRNGKey(seed), num_seeds)
    seed_chunk_keys = jax.vmap(lambda k: jax.random.split(k, num_chunks))(seed_keys)

    carry = (
        jnp.zeros((num_seeds, num_states, num_actions), jnp.float32),  # Q
        jnp.zeros((num_seeds,), jnp.int32),   # s
        jnp.zeros((num_seeds,), jnp.bool_),   # done
        jnp.zeros((num_seeds,), jnp.int32),   # step_in_ep
        goal_jax,
        potential_jax,
        train_start_jax,
        jnp.zeros((num_seeds, n), jnp.int32),    # buf_s
        jnp.zeros((num_seeds, n), jnp.int32),    # buf_a
        jnp.zeros((num_seeds, n), jnp.float32),  # buf_r
        jnp.zeros((num_seeds,),   jnp.int32),    # buf_ptr
        jnp.zeros((num_seeds,),   jnp.int32),    # buf_len
    )

    all_eval_sr      = []
    all_eval_len_all = []
    all_eval_len_suc = []
    all_eval_steps   = []

    def _fmt(sec):
        m, s = divmod(int(sec), 60)
        return f"{m}m{s:02d}s" if m else f"{s}s"

    t0 = time.monotonic()
    for chunk_i in range(num_chunks):
        carry = vmapped_chunk(carry, seed_chunk_keys[:, chunk_i])

        successes_np, lengths_np = (
            np.array(x) for x in vmapped_eval(carry[0], eval_starts_jax, goal_jax)
        )  # each [num_seeds, num_eval_episodes]

        eval_sr_np      = successes_np.mean(axis=1)
        eval_len_all_np = lengths_np.mean(axis=1)
        suc_lengths     = np.where(successes_np, lengths_np.astype(float), np.nan)
        with np.errstate(all="ignore"):
            eval_len_suc_np = np.nanmean(suc_lengths, axis=1)

        all_eval_sr.append(eval_sr_np)
        all_eval_len_all.append(eval_len_all_np)
        all_eval_len_suc.append(eval_len_suc_np)
        step_count = (chunk_i + 1) * eval_interval
        all_eval_steps.append(step_count)

        elapsed   = time.monotonic() - t0
        avg_chunk = elapsed / (chunk_i + 1)
        eta       = avg_chunk * (num_chunks - chunk_i - 1)
        suc_disp  = (f"{np.nanmean(eval_len_suc_np):.0f}"
                     if not np.all(np.isnan(eval_len_suc_np)) else "—")
        print(f"    step {step_count}/{total_steps}:"
              f"  eval_sr={eval_sr_np.mean():.2f}"
              f"  eval_len_all={eval_len_all_np.mean():.0f}"
              f"  eval_len_suc={suc_disp}"
              f"  elapsed {_fmt(elapsed)}  eta {_fmt(eta)}"
              f"  ({avg_chunk:.1f}s/chunk)")

    return (
        np.stack(all_eval_sr,      axis=0),          # [num_chunks, num_seeds]
        np.stack(all_eval_len_all, axis=0),          # [num_chunks, num_seeds]
        np.stack(all_eval_len_suc, axis=0),          # [num_chunks, num_seeds]
        np.array(all_eval_steps,   dtype=np.int32),  # [num_chunks]
        np.array(carry[0]),                          # [num_seeds, num_states, num_actions]
    )
