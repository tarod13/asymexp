"""
Reward shaping experiment using hitting times from a pre-trained Laplacian model.

The hitting time h(s, g) is the expected number of steps to reach state g from
state s under the random-walk policy.  We turn it into a potential function:

    F(s) = -h(s, g)          # closer to goal → higher potential

and add the standard potential-based shaping bonus to every transition s → s':

    Φ(s, s') = γ_rl · F(s') - F(s)
             = h(s, g) - γ_rl · h(s', g)

This bonus is added to the environment reward:

    r_shaped = r_env + β · Φ(s, s')

By construction (Ng, Harada & Russell, 1999) this shaping is policy-invariant:
it cannot change the set of optimal policies.

Pipeline
--------
1. Load a pre-trained Laplacian encoder from a results directory.
2. Compute the pairwise hitting-time matrix H[i, j] = h(i, j).
3. Select a goal state and build the potential F(s) = −h(s, goal).
4. Run tabular Q-learning with:
     (a) baseline  – sparse reward only (+1.0 on reaching the goal)
     (b) shaped    – sparse reward + β · Φ(s, s')
5. Plot learning curves (steps-per-episode and success rate).

Usage
-----
    python experiments/reward_shaping/run_reward_shaping.py \\
        --model_dir ./results/file/my_run \\
        --goal_state 42 \\
        --shaping_coef 0.1 \\
        --num_episodes 3000 \\
        --num_seeds 5
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make sure project root is importable regardless of working directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.envs.gridworld import GridWorldState
from src.utils.envs import create_gridworld_env, get_canonical_free_states, get_env_transition_markers
from src.utils.metrics import compute_hitting_times_from_eigenvectors
from src.utils.laplacian import compute_laplacian, compute_eigendecomposition
from src.utils.plotting import visualize_source_vs_target_hitting_times


# ===========================================================================
# Loading utilities
# ===========================================================================

def load_model(model_dir: Path, use_gt: bool = False) -> dict:
    """
    Load eigenvectors and eigenvalues from a results directory produced by
    train_lap_rep.py.

    Returns a dict with keys:
        training_args   – original training hyper-parameters (dict)
        canonical_states – np.ndarray of free-state full indices
        left_real / left_imag / right_real / right_imag  – [N, K] float arrays
        eigenvalues_real / eigenvalues_imag              – [K] float arrays
        eigenvalue_type  – 'kernel' (learned) or 'laplacian' (GT)
    """
    model_dir = Path(model_dir)

    with open(model_dir / "args.json") as f:
        training_args = json.load(f)

    with open(model_dir / "viz_metadata.pkl", "rb") as f:
        viz_metadata = pickle.load(f)
    canonical_states = np.array(viz_metadata["canonical_states"])

    if use_gt:
        left_real  = np.load(model_dir / "gt_left_real.npy")
        left_imag  = np.load(model_dir / "gt_left_imag.npy")
        right_real = np.load(model_dir / "gt_right_real.npy")
        right_imag = np.load(model_dir / "gt_right_imag.npy")
        eig_real   = np.load(model_dir / "gt_eigenvalues_real.npy")
        eig_imag   = np.load(model_dir / "gt_eigenvalues_imag.npy")
        eigenvalue_type = "laplacian"
    else:
        # Raw learned eigenvectors (adjoint left eigenvectors, as used during training)
        left_real  = np.load(model_dir / "final_learned_left_real.npy")
        left_imag  = np.load(model_dir / "final_learned_left_imag.npy")
        right_real = np.load(model_dir / "final_learned_right_real.npy")
        right_imag = np.load(model_dir / "final_learned_right_imag.npy")
        # Eigenvalue estimates stored inside the final model checkpoint
        ckpt_path = model_dir / "models" / "final_model.pkl"
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        eig_real = np.array(ckpt["params"]["lambda_real"])
        eig_imag = np.array(ckpt["params"]["lambda_imag"])
        eigenvalue_type = "kernel"

    return dict(
        training_args=training_args,
        canonical_states=canonical_states,
        left_real=left_real,
        left_imag=left_imag,
        right_real=right_real,
        right_imag=right_imag,
        eigenvalues_real=eig_real,
        eigenvalues_imag=eig_imag,
        eigenvalue_type=eigenvalue_type,
    )


# ===========================================================================
# Transition table
# ===========================================================================

def build_transition_table(
    env,
    canonical_states: np.ndarray,
) -> np.ndarray:
    """
    Precompute the deterministic transition table next_state[s, a].

    next_state[s, a] = canonical index of the state reached by taking
    action a from canonical state s.

    Priority order (highest first):
      1. Portal: teleport to destination state
      2. Door: blocked transition stays in place
      3. Normal physics: boundary/obstacle → stay, else move

    Actions: 0=up, 1=right, 2=down, 3=left
    Coordinate convention: full_idx = y * width + x
    Action effects: up=(0,-1), right=(+1,0), down=(0,+1), left=(-1,0)
    """
    num_canonical = len(canonical_states)
    full_to_canonical = {int(s): i for i, s in enumerate(canonical_states)}

    # (dx, dy) for actions 0..3
    action_effects = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # Blocked transitions (doors)
    # env.asymmetric_transitions stores (state_full, action) -> prob, where
    # prob is the probability the transition SUCCEEDS (0 = fully blocked).
    # The env step blocks exactly (state_full, action), so use the keys directly.
    blocked: set[tuple[int, int]] = set()
    if env.has_doors:
        for (state_full, action), prob in env.asymmetric_transitions.items():
            if prob == 0.0:  # hard door — deterministically blocked
                blocked.add((int(state_full), int(action)))

    # Portal overrides: (source_full, action) -> dest_full
    portals: dict[tuple[int, int], int] = {}
    if env.has_portals:
        for (state_full, action), dest_full in env.portals.items():
            portals[(int(state_full), int(action))] = int(dest_full)

    next_state = np.empty((num_canonical, 4), dtype=np.int32)

    for s_idx in range(num_canonical):
        full_idx = int(canonical_states[s_idx])
        y, x = divmod(full_idx, env.width)

        for a, (dx, dy) in enumerate(action_effects):
            # 1. Portal override
            if (full_idx, a) in portals:
                dest_full = portals[(full_idx, a)]
                next_state[s_idx, a] = full_to_canonical.get(dest_full, s_idx)
                continue

            # 2. Irreversible door — stay in place
            if (full_idx, a) in blocked:
                next_state[s_idx, a] = s_idx
                continue

            # 3. Normal physics
            nx, ny = x + dx, y + dy
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                next_state[s_idx, a] = s_idx
                continue

            next_full = ny * env.width + nx
            if next_full not in full_to_canonical:
                next_state[s_idx, a] = s_idx
                continue

            next_state[s_idx, a] = full_to_canonical[next_full]

    return next_state


# ===========================================================================
# Potential function
# ===========================================================================

def build_potential(hitting_times: np.ndarray, goal_idx: int, clamp_negatives: bool = False) -> np.ndarray:
    """
    Build the potential F(s) = −h(s, goal) used for reward shaping.

    Hitting times are normalised by the abs-max finite value.
    Non-finite values are replaced with 0 (neutral potential).
    If clamp_negatives=True, negative values are also clamped to 0 before
    normalisation.
    """
    h = hitting_times[:, goal_idx].copy()

    finite_mask = np.isfinite(h)
    if finite_mask.sum() == 0:
        return np.zeros(len(h), dtype=np.float32)

    if clamp_negatives:
        h = np.where(finite_mask & (h >= 0), h, 0.0)
        finite_mask = h > 0
    else:
        h = np.where(finite_mask, h, 0.0)

    h_ref = float(np.abs(h).max())
    if h_ref < 1e-8:
        h_ref = 1e-8

    h = h / h_ref  # normalise

    return -h.astype(np.float32)   # higher potential = closer to goal


def build_all_potentials(hitting_times: np.ndarray, clamp_negatives: bool = False) -> np.ndarray:
    """
    Build the full potential matrix F[s, g] for every possible goal g.

    Returns an [N, N] float32 array where column g equals
    build_potential(hitting_times, g).  Pre-computing this once lets
    per-seed potentials be retrieved with a simple column slice.
    """
    N = hitting_times.shape[0]
    F = np.empty((N, N), dtype=np.float32)
    for g in range(N):
        F[:, g] = build_potential(hitting_times, g, clamp_negatives=clamp_negatives)
    return F


# ===========================================================================
# Tabular Q-learning
# ===========================================================================

def run_q_learning(
    env,
    canonical_states: np.ndarray,
    goal_per_seed: np.ndarray,              # [num_seeds] int — fixed goal per task
    eval_starts_per_seed: np.ndarray,       # [num_seeds] int — fixed eval start per task
    num_episodes: int,
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
    log_interval_steps: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run tabular ε-greedy Q-learning with one independent Q-table per seed.

    Each seed corresponds to one fixed task (goal).  There is no knowledge
    transfer across seeds — they are fully independent.

    All seeds are run in parallel via jax.vmap; the step loop within each
    chunk is a jax.lax.scan call (JIT-compiled, no Python overhead).

    Training
    --------
    Each episode keeps the fixed goal for the seed and draws a fresh random
    start state, so the agent learns to navigate to its goal from anywhere.

    Evaluation
    ----------
    After every log_interval_steps environment steps the current greedy
    policy (epsilon=0) is tested from eval_starts_per_seed toward each
    seed's goal for num_eval_episodes episodes.  The same starts and goals
    are used for every condition, giving a fair, reproducible comparison.

    Returns
    -------
    train_steps       : [num_seeds, num_episodes]   int
    train_reached     : [num_seeds, num_episodes]   bool
    eval_success_rate : [num_chunks, num_seeds]      float
    eval_episodes     : [num_chunks]                 int
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

    # Potential per seed: zeros for baseline (shaping_coef=0 → no effect)
    if potential_per_seed is None:
        potential_jax = jnp.zeros((num_seeds, num_states), jnp.float32)
    else:
        potential_jax = jnp.array(potential_per_seed, jnp.float32)  # [num_seeds, N]

    # Fixed training start per seed: -1 means random, >=0 means fixed.
    if train_start_per_seed is None:
        train_start_jax = jnp.full((num_seeds,), -1, dtype=jnp.int32)
    else:
        train_start_jax = jnp.array(train_start_per_seed, dtype=jnp.int32)

    # Valid-starts lookup table for distance-constrained random resets.
    # valid_starts_table[g, :num_valid[g]] lists all canonical indices that are
    # at least min_goal_distance (Manhattan) away from goal g and != g.
    # Closed over in run_step so JAX never traces through the if-branch at runtime.
    if min_goal_distance > 0:
        coords = np.array(
            [(int(canonical_states[s]) % env.width,
              int(canonical_states[s]) // env.width)
             for s in range(num_states)]
        )  # [N, 2]
        taxi = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)  # [N, N]
        valid_starts_lists = []
        for g in range(num_states):
            vs = [s for s in range(num_states)
                  if s != g and taxi[s, g] >= min_goal_distance]
            if not vs:                        # fallback: any non-goal state
                vs = [s for s in range(num_states) if s != g]
            valid_starts_lists.append(vs)
        max_valid = max(len(v) for v in valid_starts_lists)
        table  = np.zeros((num_states, max_valid), dtype=np.int32)
        counts = np.zeros(num_states, dtype=np.int32)
        for g, vs in enumerate(valid_starts_lists):
            table[g, :len(vs)] = vs
            counts[g] = len(vs)
        valid_table_jax  = jnp.array(table)   # [N, max_valid]
        valid_counts_jax = jnp.array(counts)  # [N]
    else:
        valid_table_jax  = None
        valid_counts_jax = None

    # ── Training ──────────────────────────────────────────────────────
    def run_chunk(carry, chunk_key):
        def run_step(step_carry, step_key):
            Q, s, done, step_in_ep, goal, potential, fixed_start = step_carry

            reset_key, eps_key, rand_key, env_key, tie_key = jax.random.split(step_key, 5)
            at_reset = step_in_ep == 0

            # At episode start: random or fixed start, fixed goal for this seed.
            if valid_table_jax is not None:
                # Sample uniformly from pre-filtered starts (distance constraint).
                k = jax.random.randint(
                    reset_key, (), 0, valid_counts_jax[goal])
                new_start_random = valid_table_jax[goal, k]
            else:
                new_start_raw = jax.random.randint(reset_key, (), 0, num_states)
                new_start_random = jnp.where(
                    new_start_raw == goal,
                    (goal + 1) % num_states,
                    new_start_raw,
                )
            # Use fixed_start when >= 0; otherwise use the (possibly constrained) random.
            new_start = jnp.where(fixed_start >= 0, fixed_start, new_start_random)
            s    = jnp.where(at_reset, new_start, s)
            done = jnp.where(at_reset, jnp.array(False), done)

            # ε-greedy action with random tie-breaking
            use_random = jax.random.uniform(eps_key) < epsilon
            q_s    = Q[s]
            is_max = (q_s == q_s.max()).astype(jnp.float32)
            greedy_a   = jax.random.choice(tie_key, num_actions, p=is_max / is_max.sum())
            random_a   = jax.random.randint(rand_key, (), 0, num_actions)
            a          = jnp.where(use_random, random_a, greedy_a)

            # Environment step
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
            r       = jnp.where(reached, 1.0, 0.0)
            r       = r + shaping_coef * (
                gamma * potential[s_prime] - potential[s]
            )

            # Q update — always learn, including on the terminal transition
            target = jnp.where(reached, r, r + gamma * Q[s_prime].max())
            Q = Q.at[s, a].add(lr * (target - Q[s, a]))

            new_done        = done | reached
            new_s           = jnp.where(new_done, s, s_prime)
            # Reset episode counter immediately on done so the next step
            # triggers at_reset and starts a fresh episode.
            next_step_in_ep = jnp.where(
                new_done | (step_in_ep == max_steps_per_episode - 1),
                jnp.zeros((), jnp.int32),
                step_in_ep + 1,
            )
            episode_end = new_done | (step_in_ep == max_steps_per_episode - 1)
            return (Q, new_s, new_done, next_step_in_ep, goal, potential, fixed_start), (new_done, episode_end)

        step_keys = jax.random.split(chunk_key, chunk_steps)
        new_carry, (reached_flat, ep_end_flat) = jax.lax.scan(run_step, carry, step_keys)
        # Return flat per-step flags; the main loop converts them to per-episode
        # stats in numpy.  Episodes now have variable lengths (they restart
        # immediately on done), so a fixed reshape is no longer valid.
        return new_carry, (reached_flat, ep_end_flat)

    vmapped_chunk = jax.jit(jax.vmap(run_chunk))

    # ── Evaluation ────────────────────────────────────────────────────
    def eval_one_seed(Q, eval_start, goal):
        """Greedy rollout for num_eval_episodes episodes. Q: [N, A]."""
        def eval_step(carry, step_key):
            s, done, step_in_ep = carry
            at_reset = step_in_ep == 0
            s    = jnp.where(at_reset, eval_start, s)
            done = jnp.where(at_reset, jnp.array(False), done)

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

            reached = s_prime == goal
            new_done = done | reached
            new_s    = jnp.where(new_done, s, s_prime)
            # Restart immediately on done so the next step triggers at_reset.
            next_step_in_ep = jnp.where(
                new_done | (step_in_ep == max_steps_per_episode - 1),
                jnp.zeros((), jnp.int32),
                step_in_ep + 1,
            )
            episode_end = new_done | (step_in_ep == max_steps_per_episode - 1)
            return (new_s, new_done, next_step_in_ep), (new_done, episode_end)

        eval_steps = num_eval_episodes * max_steps_per_episode
        eval_keys  = jax.random.split(jax.random.PRNGKey(0), eval_steps)
        _, (done_flat, ep_end_flat) = jax.lax.scan(
            eval_step,
            (eval_start, jnp.array(False), jnp.zeros((), jnp.int32)),
            eval_keys,
        )
        return done_flat, ep_end_flat

    # vmap over seeds; each seed has its own Q-table, eval start, and goal
    vmapped_eval = jax.jit(jax.vmap(eval_one_seed))

    # ── Main loop ─────────────────────────────────────────────────────
    chunk_steps = log_interval_steps
    num_chunks  = max(1, (num_episodes * max_steps_per_episode) // chunk_steps)

    seed_keys       = jax.random.split(jax.random.PRNGKey(seed), num_seeds)
    seed_chunk_keys = jax.vmap(lambda k: jax.random.split(k, num_chunks))(seed_keys)

    carry = (
        jnp.zeros((num_seeds, num_states, num_actions), jnp.float32),  # Q
        jnp.zeros((num_seeds,), jnp.int32),   # s
        jnp.zeros((num_seeds,), jnp.bool_),   # done
        jnp.zeros((num_seeds,), jnp.int32),   # step_in_ep
        goal_jax,                              # goal (fixed per seed)
        potential_jax,                         # potential (fixed per seed)
        train_start_jax,                       # fixed train start (-1 = random)
    )

    # Per-seed accumulators: lists of 1-D arrays, one entry per chunk.
    seed_steps_acc   = [[] for _ in range(num_seeds)]
    seed_reached_acc = [[] for _ in range(num_seeds)]
    all_eval_sr, eval_ep_indices = [], []
    cumulative_eps = np.zeros(num_seeds, dtype=np.int64)

    def _fmt(s):
        m, s = divmod(int(s), 60)
        return f"{m}m{s:02d}s" if m else f"{s}s"

    ep_width = len(str(num_episodes))
    t0 = time.monotonic()
    for chunk_i in range(num_chunks):
        carry, (reached_flat, ep_end_flat) = vmapped_chunk(carry, seed_chunk_keys[:, chunk_i])
        reached_np = np.array(reached_flat)  # [num_seeds, chunk_steps]
        ep_end_np  = np.array(ep_end_flat)   # [num_seeds, chunk_steps]

        # Convert flat per-step flags to per-episode stats for each seed.
        # Cap each seed at num_episodes to keep the step budget from inflating
        # the episode count when episodes terminate early.
        chunk_reached_rates = np.zeros(num_seeds, dtype=float)
        chunk_active        = np.zeros(num_seeds, dtype=bool)
        chunk_suc_steps = []
        chunk_n_reached = 0
        for si in range(num_seeds):
            remaining = num_episodes - cumulative_eps[si]
            if remaining <= 0:
                continue
            ep_ends = np.where(ep_end_np[si])[0]
            if len(ep_ends) == 0:
                continue
            ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
            steps_s   = (ep_ends - ep_starts + 1).astype(np.int32)
            reached_s = reached_np[si, ep_ends]
            # Discard episodes beyond the budget for this seed.
            n_add     = min(len(ep_ends), remaining)
            steps_s   = steps_s[:n_add]
            reached_s = reached_s[:n_add]
            seed_steps_acc[si].append(steps_s)
            seed_reached_acc[si].append(reached_s)
            cumulative_eps[si] += n_add
            chunk_reached_rates[si] = reached_s.mean()
            chunk_active[si]        = True
            chunk_n_reached += int(reached_s.sum())
            chunk_suc_steps.extend(steps_s[reached_s.astype(bool)])
        train_avg_steps = np.mean(chunk_suc_steps) if chunk_suc_steps else float("nan")

        done_all_np, ep_end_all_np = (
            np.array(x) for x in vmapped_eval(carry[0], eval_starts_jax, goal_jax)
        )  # each [num_seeds, eval_steps]
        eval_sr_np     = np.zeros(num_seeds)
        eval_steps_np  = np.full(num_seeds, np.nan)
        for si in range(num_seeds):
            ep_ends_e = np.where(ep_end_all_np[si])[0]
            if len(ep_ends_e) == 0:
                continue
            ep_starts_e = np.concatenate([[0], ep_ends_e[:-1] + 1])
            lengths_e   = ep_ends_e - ep_starts_e + 1
            reached_e   = done_all_np[si, ep_ends_e]
            n = min(len(ep_ends_e), num_eval_episodes)
            reached_e, lengths_e = reached_e[:n], lengths_e[:n]
            eval_sr_np[si]    = reached_e.mean()
            suc_len = lengths_e[reached_e.astype(bool)]
            if len(suc_len):
                eval_steps_np[si] = suc_len.mean()
        all_eval_sr.append(eval_sr_np)
        eval_ep_indices.append(int(cumulative_eps.mean()))

        avg_q   = float(np.array(carry[0]).mean())
        elapsed   = time.monotonic() - t0
        avg_chunk = elapsed / (chunk_i + 1)
        eta       = avg_chunk * (num_chunks - chunk_i - 1)
        ep_disp   = int(cumulative_eps.mean())
        train_steps_disp = f"{train_avg_steps:.0f}" if not np.isnan(train_avg_steps) else "—"
        eval_steps_disp  = (f"{np.nanmean(eval_steps_np):.0f}"
                            if not np.all(np.isnan(eval_steps_np)) else "—")
        print(f"    ep ~{ep_disp:{ep_width}d}/{num_episodes}:"
              f"  train_sr={chunk_reached_rates[chunk_active].mean():.2f}"
              f"  train_seeds={chunk_active.sum()}/{num_seeds}"
              f"  train_n={chunk_n_reached}"
              f"  train_steps_goal={train_steps_disp}"
              f"  eval_sr={eval_sr_np.mean():.2f}"
              f"  eval_steps_goal={eval_steps_disp}"
              f"  avg_Q={avg_q:.4f}"
              f"  elapsed {_fmt(elapsed)}  eta {_fmt(eta)}"
              f"  ({avg_chunk:.1f}s/chunk)")

        if cumulative_eps.min() >= num_episodes:
            break

    # Concatenate per-seed episode lists and pad seeds to equal length.
    all_steps_final   = [
        np.concatenate(seed_steps_acc[si])   if seed_steps_acc[si]   else np.zeros(0, dtype=np.int32)
        for si in range(num_seeds)
    ]
    all_reached_final = [
        np.concatenate(seed_reached_acc[si]) if seed_reached_acc[si] else np.zeros(0, dtype=bool)
        for si in range(num_seeds)
    ]
    max_eps = max((len(s) for s in all_steps_final), default=0)
    train_steps   = np.full((num_seeds, max_eps), max_steps_per_episode, dtype=np.int32)
    train_reached = np.zeros((num_seeds, max_eps), dtype=bool)
    for si in range(num_seeds):
        n = len(all_steps_final[si])
        train_steps[si, :n]   = all_steps_final[si]
        train_reached[si, :n] = all_reached_final[si]

    return (
        train_steps,                                  # [num_seeds, total_episodes]
        train_reached,                                # [num_seeds, total_episodes]
        np.stack(all_eval_sr,       axis=0),          # [num_chunks, num_seeds]
        np.array(eval_ep_indices,   dtype=np.int32),  # [num_chunks]
    )


# ===========================================================================
# Plotting
# ===========================================================================

def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average with edge padding."""
    if window <= 1 or len(x) < window:
        return x.astype(float)
    kernel = np.ones(window) / window
    x_padded = np.pad(x.astype(float), (window - 1, 0), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


def plot_results(
    results: dict,
    output_path: Path,
    window: int = 100,
    max_steps: int = 500,
) -> None:
    """
    Save a three-panel figure:
      Left   – training steps per episode (lower = better)
      Centre – training success rate (higher = better)
      Right  – evaluation success rate on fixed (start, goal) pairs (higher = better)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, data) in enumerate(results.items()):
        c = colours[idx % len(colours)]
        steps_arr   = data["steps"].astype(float)    # [seeds, episodes]
        reached_arr = data["reached"].astype(float)  # [seeds, episodes]
        eval_sr     = data["eval_sr"]                # [chunks, seeds]
        eval_ep     = data["eval_episodes"]          # [chunks]

        n_ep = steps_arr.shape[1]
        x    = np.arange(n_ep)

        for arr, ax, ylabel in [
            (steps_arr,   axes[0], f"Steps per episode (max {max_steps})"),
            (reached_arr, axes[1], "Success rate"),
        ]:
            mean  = _smooth(arr.mean(axis=0), window)
            std_s = _smooth(arr.std(axis=0),  window)
            ax.plot(x, mean, label=label, color=c, linewidth=1.5)
            ax.fill_between(x, mean - std_s, mean + std_s, color=c, alpha=0.15)
            ax.set_xlabel("Training episode", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linewidth=0.4, alpha=0.5)

        # Eval: mean and std over seeds
        eval_mean = eval_sr.mean(axis=1)   # [chunks]
        eval_std  = eval_sr.std(axis=1)    # [chunks]
        axes[2].plot(eval_ep, eval_mean, label=label, color=c,
                     linewidth=1.5, marker="o", markersize=4)
        axes[2].fill_between(eval_ep, eval_mean - eval_std, eval_mean + eval_std,
                             color=c, alpha=0.15)

    axes[0].set_title("Training: steps per episode  (↓ better)", fontsize=11)
    axes[1].set_title("Training: success rate  (↑ better)", fontsize=11)
    axes[2].set_title("Evaluation: success rate on fixed pairs  (↑ better)", fontsize=11)
    axes[1].set_ylim(-0.05, 1.10)
    axes[2].set_ylim(-0.05, 1.10)
    axes[2].set_xlabel("Training episode", fontsize=10)
    axes[2].set_ylabel("Success rate", fontsize=10)
    axes[2].grid(True, linewidth=0.4, alpha=0.5)

    for ax in axes:
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle(
        f"Reward shaping with Laplacian hitting times\n"
        f"(training smoothing window = {window} episodes)",
        fontsize=12,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


def plot_hitting_times_grid(
    hitting_times: np.ndarray,
    canonical_states: np.ndarray,
    env,
    output_dir: Path,
    ncols: int = 8,
) -> None:
    """
    Save grid-overlaid hitting-time maps for every canonical state.
    Each state appears as both target (times TO it) and source (times FROM it).
    Produces 4 PNGs: linear/log × shared/independent color scale.
    """
    door_markers = get_env_transition_markers(env)
    all_indices = list(range(len(canonical_states)))
    output_dir.mkdir(parents=True, exist_ok=True)

    for log_scale in (False, True):
        suffix = "_log" if log_scale else ""
        for shared in (True, False):
            scale_str = "shared" if shared else "independent"
            fname = output_dir / f"hitting_times{suffix}_{scale_str}_scale.png"
            visualize_source_vs_target_hitting_times(
                state_indices=all_indices,
                hitting_time_matrix=hitting_times,
                canonical_states=canonical_states,
                grid_width=env.width,
                grid_height=env.height,
                portals=door_markers if door_markers else None,
                ncols=ncols,
                save_path=str(fname),
                log_scale=log_scale,
                shared_colorbar=shared,
            )
            plt.close()
            print(f"  Hitting-times plot → {fname}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward shaping with Laplacian hitting times."
    )
    parser.add_argument(
        "--env", required=True,
        help="Environment name (e.g. 'GridRoom-4-Doors') passed as env_file_name "
             "to create_gridworld_env with env_type='file'.",
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Results directory for the complex representation (train_lap_rep.py). "
             "Required to run the 'shaped (complex)' condition; omit for baseline only.",
    )
    parser.add_argument(
        "--allo_model_dir", default=None,
        help="Results directory for the ALLO representation (train_allo_rep.py). "
             "When provided a third condition 'shaped (allo)' is added.",
    )
    parser.add_argument(
        "--num_eval_episodes", type=int, default=30,
        help="Greedy episodes per eval pair at each checkpoint (default: 30).",
    )
    parser.add_argument(
        "--shaping_coef", type=float, default=0.1,
        help="β: weight for the potential-based shaping bonus (default: 0.1).",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3000,
        help="Number of Q-learning episodes per seed (default: 3000).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Max steps per episode before declaring failure (default: 500).",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5,
        help="Number of independent seeds to average over (default: 5).",
    )
    parser.add_argument(
        "--gamma_rl", type=float, default=0.99,
        help="RL discount factor γ (default: 0.99).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Q-learning step size α (default: 0.1).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="ε-greedy exploration rate (default: 0.1).",
    )
    parser.add_argument(
        "--log_interval", type=int, default=250_000,
        help="Print progress and run eval every this many environment steps (default: 250000).",
    )
    parser.add_argument(
        "--eval_seed", type=int, default=0,
        help="Seed for sampling random goal states (default: 0).",
    )
    parser.add_argument(
        "--use_gt", action="store_true",
        help="Use ground-truth Laplacian eigenvectors. When set, eigenvectors are "
             "computed directly from the environment transition matrix (no --model_dir "
             "needed). Requires --num_eigenvectors.",
    )
    parser.add_argument(
        "--num_eigenvectors", type=int, default=None,
        help="Number of eigenvectors to use for the ground-truth Laplacian "
             "(required when --use_gt is set).",
    )
    parser.add_argument(
        "--gt_gamma", type=float, default=0.95,
        help="Discount factor γ used in the GT Laplacian L = (1+δ)I - (1-γ)P·SR_γ "
             "(default: 0.95). Only used when --use_gt is set.",
    )
    parser.add_argument(
        "--gt_delta", type=float, default=0.1,
        help="Eigenvalue-shift δ used in the GT Laplacian (default: 0.1). "
             "Only used when --use_gt is set.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write outputs (default: <model_dir>/reward_shaping/).",
    )
    parser.add_argument(
        "--start_state", type=str, default=None,
        help="Fixed starting state as 'x,y' grid coordinates (e.g. '3,2'). "
             "If provided, every training and evaluation episode begins from this state. "
             "If the cell is blocked or out of bounds a warning is printed and random "
             "starts are used instead.",
    )
    parser.add_argument(
        "--min_goal_distance", type=int, default=0,
        help="Minimum taxi (Manhattan) distance from the fixed starting state to any "
             "sampled goal (default: 0, no constraint). Requires a valid --start_state; "
             "otherwise this flag is ignored with a warning.",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        choices=["baseline", "complex", "allo"],
        help="Single-method mode for distributed execution: which condition to run. "
             "Must be paired with --seed_idx.  The script saves a partial result file "
             "and skips the combined plot; use analyze_reward_shaping.py afterwards.",
    )
    parser.add_argument(
        "--seed_idx", type=int, default=None,
        help="Which seed to run in single-method mode (0-indexed, up to --num_seeds-1). "
             "Must be paired with --method.",
    )
    args = parser.parse_args()

    if args.use_gt and args.num_eigenvectors is None:
        parser.error("--use_gt requires --num_eigenvectors.")

    single_seed_mode = args.seed_idx is not None
    if single_seed_mode and args.method is None:
        parser.error("--seed_idx requires --method.")
    if single_seed_mode and args.seed_idx >= args.num_seeds:
        parser.error(
            f"--seed_idx {args.seed_idx} is out of range for --num_seeds {args.num_seeds}."
        )
    if single_seed_mode and args.method == "allo" and args.allo_model_dir is None:
        parser.error("--method=allo requires --allo_model_dir.")

    model_dir  = Path(args.model_dir) if args.model_dir else None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif model_dir is not None:
        output_dir = model_dir / "reward_shaping"
    else:
        output_dir = Path("reward_shaping")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load pre-trained Laplacian model (complex representation)
    # ------------------------------------------------------------------
    # When --use_gt, eigenvectors are computed from the environment after it is
    # built (section 2b), so we skip disk loading here entirely.
    # In single-seed mode skip eigenvectors when not needed (saves memory).
    _load_complex = (not args.use_gt) and (model_dir is not None) and (
        args.method is None or args.method == "complex"
    )
    if _load_complex:
        print(f"\n{'='*60}")
        print(f"Loading complex representation model from: {model_dir}")
        print(f"{'='*60}")
        model_data = load_model(model_dir)
        evtype     = model_data["eigenvalue_type"]
        print(f"  Eigenvector source : learned (normalized)")
        print(f"  Eigenvalue type    : {evtype}")
        print(f"  Eigenvectors (K)   : {model_data['eigenvalues_real'].shape[0]}")
    else:
        model_data = None
        evtype     = None

    # Optionally load ALLO representation model.
    # In single-seed mode skip it when not needed (saves time and memory).
    allo_model_data = None
    _load_allo = (not args.use_gt) and args.allo_model_dir is not None and (
        args.method is None or args.method == "allo"
    )
    if _load_allo:
        allo_model_dir = Path(args.allo_model_dir)
        print(f"\n{'='*60}")
        print(f"Loading ALLO representation model from: {allo_model_dir}")
        print(f"{'='*60}")
        allo_model_data = load_model(allo_model_dir)
        # ALLO learns a single real representation φ_k (symmetric Laplacian).
        # φ_k serves as both left and right eigenvectors; imag parts are zero.
        allo_model_data["left_real"]  = allo_model_data["right_real"]
        allo_model_data["left_imag"]  = np.zeros_like(allo_model_data["right_real"])
        allo_model_data["right_imag"] = np.zeros_like(allo_model_data["right_real"])
        allo_model_data["eigenvalues_imag"] = np.zeros_like(
            allo_model_data["eigenvalues_real"])
        allo_evtype = allo_model_data["eigenvalue_type"]
        print(f"  Eigenvector source : learned (normalized)")
        print(f"  Eigenvalue type    : {allo_evtype}")
        print(f"  Eigenvectors (K)   : {allo_model_data['eigenvalues_real'].shape[0]}")
        print(f"  (left = right = right_real, imag = 0 for symmetric Laplacian)")

    # ------------------------------------------------------------------
    # 2. Build environment
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Building environment: {args.env}")
    print(f"{'='*60}")

    ta = SimpleNamespace(
        env_type="file",
        env_file=None,
        env_file_name=args.env,
        max_episode_length=args.max_steps,
        windy=False,
        wind=0.0,
    )
    env = create_gridworld_env(ta)
    canonical_states = get_canonical_free_states(env)
    num_canonical    = len(canonical_states)
    print(f"  Canonical states   : {num_canonical}")

    # ------------------------------------------------------------------
    # 2b. Validate optional fixed starting state; build eligible goal set.
    # ------------------------------------------------------------------
    fixed_start_canonical = None   # canonical index, or None → random starts
    fixed_start_coords    = None   # (sx, sy) tuple

    if args.start_state is not None:
        try:
            sx, sy = [int(v.strip()) for v in args.start_state.split(",")]
        except Exception:
            print(
                f"WARNING: could not parse --start_state '{args.start_state}'. "
                "Expected 'x,y' format. Falling back to random starts.",
                file=sys.stderr,
            )
        else:
            start_full = int(sy * env.width + sx)
            can_lookup = {int(s): i for i, s in enumerate(canonical_states)}
            if start_full in can_lookup:
                fixed_start_canonical = can_lookup[start_full]
                fixed_start_coords    = (sx, sy)
                print(f"\n  Fixed start state  : ({sx},{sy})"
                      f"  [full={start_full}, canonical={fixed_start_canonical}]")
            else:
                print(
                    f"WARNING: starting state ({sx},{sy}) is blocked or out of bounds. "
                    "Falling back to random starts.",
                    file=sys.stderr,
                )

    # Build the set of eligible goal canonical indices.
    # When a fixed start is given, goals that are too close to it (or equal to it)
    # are excluded here.  When there is no fixed start, the distance guarantee is
    # enforced per-seed at eval-start sampling time (section 4).
    eligible_goals: list[int] = list(range(num_canonical))
    if fixed_start_canonical is not None:
        # Never let the goal equal the fixed starting state.
        eligible_goals = [ci for ci in eligible_goals if ci != fixed_start_canonical]
        if args.min_goal_distance > 0:
            sx, sy = fixed_start_coords
            filtered = [
                ci for ci in eligible_goals
                if (abs(int(canonical_states[ci]) % env.width - sx)
                    + abs(int(canonical_states[ci]) // env.width - sy))
                >= args.min_goal_distance
            ]
            if not filtered:
                print(
                    f"WARNING: no goals satisfy min_goal_distance="
                    f"{args.min_goal_distance}. Using all non-start states.",
                    file=sys.stderr,
                )
            else:
                eligible_goals = filtered
            print(f"  Min goal distance  : {args.min_goal_distance}"
                  f"  ({len(eligible_goals)}/{num_canonical} eligible goals)")

    # ------------------------------------------------------------------
    # 2c. Compute ground-truth eigenvectors from the environment (--use_gt)
    # ------------------------------------------------------------------
    _need_gt = args.use_gt and (args.method is None or args.method == "complex")
    if _need_gt:
        print(f"\n{'='*60}")
        print("Computing ground-truth eigenvectors from environment ...")
        print(f"{'='*60}")
        next_state = build_transition_table(env, canonical_states)
        N = len(canonical_states)
        # Uniform random-walk: each of the 4 actions equally likely.
        # Soft doors (0 < prob < 1) add a stay-in-place mass of 0.25*(1-prob)
        # on top of the prob-weighted forward mass.
        P = np.zeros((N, N), dtype=np.float64)
        asym = env.asymmetric_transitions if env.has_doors else {}
        for a in range(4):
            for s in range(N):
                full_s = int(canonical_states[s])
                door_prob = asym.get((full_s, a), 1.0)
                dest = next_state[s, a]
                P[s, dest] += 0.25 * door_prob
                P[s, s]    += 0.25 * (1.0 - door_prob)
        laplacian = compute_laplacian(jnp.array(P), gamma=args.gt_gamma, delta=args.gt_delta)
        eig = compute_eigendecomposition(laplacian, k=args.num_eigenvectors, ascending=True)
        model_data = dict(
            training_args={"gamma": args.gt_gamma, "delta": args.gt_delta},
            left_real=np.array(eig["left_eigenvectors_real"]),
            left_imag=np.array(eig["left_eigenvectors_imag"]),
            right_real=np.array(eig["right_eigenvectors_real"]),
            right_imag=np.array(eig["right_eigenvectors_imag"]),
            eigenvalues_real=np.array(eig["eigenvalues_real"]),
            eigenvalues_imag=np.array(eig["eigenvalues_imag"]),
            eigenvalue_type="laplacian",
        )
        evtype = "laplacian"
        print(f"  Eigenvector source : ground truth (from environment)")
        print(f"  Eigenvalue type    : laplacian")
        print(f"  Eigenvectors (K)   : {args.num_eigenvectors}")
        print(f"  gt_gamma           : {args.gt_gamma}")
        print(f"  gt_delta           : {args.gt_delta}")

    # ------------------------------------------------------------------
    # 3. Compute hitting times (complex representation)
    # ------------------------------------------------------------------
    hitting_times = None
    if model_data is not None:
        print(f"\n{'='*60}")
        print("Computing hitting times (complex representation) ...")
        print(f"{'='*60}")

        hitting_times = np.array(
            compute_hitting_times_from_eigenvectors(
                left_real        = model_data["left_real"],
                left_imag        = model_data["left_imag"],
                right_real       = model_data["right_real"],
                right_imag       = model_data["right_imag"],
                eigenvalues_real = model_data["eigenvalues_real"],
                eigenvalues_imag = model_data["eigenvalues_imag"],
                gamma            = model_data["training_args"].get("gamma", 0.95),
                delta            = model_data["training_args"].get("delta", 0.1),
                eigenvalue_type  = evtype,
            )
        )

        finite = hitting_times[np.isfinite(hitting_times) & (hitting_times >= 0)]
        print(f"  Shape              : {hitting_times.shape}")
        print(f"  Finite values      : {len(finite)} / {hitting_times.size}"
              f"  ({len(finite)/hitting_times.size:.1%})")
        if len(finite) > 0:
            print(f"  Range              : [{finite.min():.2f}, {finite.max():.2f}]")

        np.save(output_dir / "hitting_times.npy", hitting_times)

        plot_hitting_times_grid(
            hitting_times,
            np.array(canonical_states),
            env,
            output_dir / ("hitting_times_gt_plots" if args.use_gt else "hitting_times_complex_plots"),
        )

    # Compute hitting times for ALLO representation if provided
    allo_hitting_times = None
    if allo_model_data is not None:
        print(f"\n{'='*60}")
        print("Computing hitting times (ALLO representation) ...")
        print(f"{'='*60}")
        allo_hitting_times = np.array(
            compute_hitting_times_from_eigenvectors(
                left_real        = allo_model_data["left_real"],
                left_imag        = allo_model_data["left_imag"],
                right_real       = allo_model_data["right_real"],
                right_imag       = allo_model_data["right_imag"],
                eigenvalues_real = allo_model_data["eigenvalues_real"],
                eigenvalues_imag = allo_model_data["eigenvalues_imag"],
                gamma            = allo_model_data["training_args"].get("gamma", 0.95),
                delta            = allo_model_data["training_args"].get("delta", 0.1),
                eigenvalue_type  = allo_model_data["eigenvalue_type"],
            )
        )
        allo_finite = allo_hitting_times[
            np.isfinite(allo_hitting_times) & (allo_hitting_times >= 0)
        ]
        print(f"  Shape              : {allo_hitting_times.shape}")
        print(f"  Finite values      : {len(allo_finite)} / {allo_hitting_times.size}"
              f"  ({len(allo_finite)/allo_hitting_times.size:.1%})")
        if len(allo_finite) > 0:
            print(f"  Range              : [{allo_finite.min():.2f}, {allo_finite.max():.2f}]")
        np.save(output_dir / "hitting_times_allo.npy", allo_hitting_times)
        plot_hitting_times_grid(
            allo_hitting_times,
            np.array(canonical_states),
            env,
            output_dir / "hitting_times_allo_plots",
        )

    # ------------------------------------------------------------------
    # 4. Sample one fixed (goal, eval_start) per seed for evaluation.
    #    Each seed is an independent task with a fixed goal; training
    #    uses random starts.  The same tasks are used for every condition.
    # ------------------------------------------------------------------
    eval_rng = np.random.default_rng(args.eval_seed)
    goal_per_seed = eval_rng.choice(
        eligible_goals, size=args.num_seeds,
        replace=args.num_seeds > len(eligible_goals),
    ).astype(np.int32)

    if fixed_start_canonical is not None:
        # Every seed starts from the same fixed cell.
        eval_starts_per_seed = np.full(args.num_seeds, fixed_start_canonical, dtype=np.int32)
        train_start_per_seed = np.full(args.num_seeds, fixed_start_canonical, dtype=np.int32)
    else:
        # When min_goal_distance > 0, enforce it on the (eval_start, goal) pair.
        eval_starts_per_seed_list = []
        for g in goal_per_seed:
            g_full = int(canonical_states[int(g)])
            gx, gy = g_full % env.width, g_full // env.width
            candidates = [
                s for s in range(num_canonical)
                if s != int(g) and (
                    args.min_goal_distance <= 0
                    or (abs(int(canonical_states[s]) % env.width - gx)
                        + abs(int(canonical_states[s]) // env.width - gy))
                    >= args.min_goal_distance
                )
            ]
            eval_starts_per_seed_list.append(eval_rng.choice(candidates))
        eval_starts_per_seed = np.array(eval_starts_per_seed_list, dtype=np.int32)
        train_start_per_seed = None

    print(f"\n  Eval seed          : {args.eval_seed}")
    print(f"  Tasks (num_seeds={args.num_seeds}):")
    for i, (st, g) in enumerate(zip(eval_starts_per_seed, goal_per_seed)):
        stf = int(canonical_states[int(st)])
        gf  = int(canonical_states[int(g)])
        sty, stx = divmod(stf, env.width)
        gy,  gx  = divmod(gf,  env.width)
        print(f"    seed {i:2d}  eval_start ({stx},{sty}) → goal ({gx},{gy})"
              f"  [canonical {int(st)} → {int(g)}]")

    # Single-seed mode: narrow to the one requested seed BEFORE building
    # potentials so the potential arrays are already [1, N] shaped.
    if single_seed_mode:
        si = args.seed_idx
        goal_per_seed        = goal_per_seed[si : si + 1]
        eval_starts_per_seed = eval_starts_per_seed[si : si + 1]
        if train_start_per_seed is not None:
            train_start_per_seed = train_start_per_seed[si : si + 1]

    # Build per-seed potential vectors by slicing the full F[s, g] matrix.
    complex_potential_per_seed = None
    if hitting_times is not None:
        complex_potential_matrix   = build_all_potentials(hitting_times)           # [N, N]
        complex_potential_per_seed = complex_potential_matrix[:, goal_per_seed].T  # [n, N]
    allo_potential_per_seed = None
    if allo_hitting_times is not None:
        allo_potential_matrix  = build_all_potentials(allo_hitting_times)      # [N, N]
        allo_potential_per_seed = allo_potential_matrix[:, goal_per_seed].T    # [n, N]

    # Number of seeds to actually run (1 in single-seed mode).
    num_q_seeds = len(goal_per_seed)

    # ------------------------------------------------------------------
    # 5. Run Q-learning: baseline vs shaped (complex) [vs shaped (allo)]
    # ------------------------------------------------------------------
    conditions = {
        "baseline": dict(potential_per_seed=None, shaping_coef=0.0),
    }
    if complex_potential_per_seed is not None:
        conditions[f"shaped β={args.shaping_coef} (complex)"] = dict(
            potential_per_seed=complex_potential_per_seed, shaping_coef=args.shaping_coef
        )
    if allo_potential_per_seed is not None:
        conditions[f"shaped β={args.shaping_coef} (allo)"] = dict(
            potential_per_seed=allo_potential_per_seed, shaping_coef=args.shaping_coef
        )

    # Filter to the requested condition when --method is given.
    if args.method is not None:
        _method_to_key = {
            "baseline": "baseline",
            "complex":  f"shaped β={args.shaping_coef} (complex)",
            "allo":     f"shaped β={args.shaping_coef} (allo)",
        }
        target_key = _method_to_key[args.method]
        if target_key not in conditions:
            print(
                f"ERROR: Condition '{target_key}' is not available. "
                "(Did you forget --allo_model_dir for method=allo?)",
                file=sys.stderr,
            )
            sys.exit(1)
        conditions = {target_key: conditions[target_key]}

    results = {}
    win = min(200, args.num_episodes)
    for cond_name, cond_kwargs in conditions.items():
        print(f"\n{'='*60}")
        print(f"Q-learning: {cond_name}")
        print(f"{'='*60}")

        steps, reached, eval_sr, eval_ep = run_q_learning(
            env                  = env,
            canonical_states     = canonical_states,
            goal_per_seed        = goal_per_seed,
            eval_starts_per_seed = eval_starts_per_seed,
            train_start_per_seed = train_start_per_seed,
            min_goal_distance    = args.min_goal_distance,
            num_episodes         = args.num_episodes,
            num_seeds            = num_q_seeds,
            max_steps_per_episode= args.max_steps,
            num_eval_episodes    = args.num_eval_episodes,
            gamma                = args.gamma_rl,
            lr                   = args.lr,
            epsilon              = args.epsilon,
            seed                 = args.seed_idx if single_seed_mode else 0,
            log_interval_steps   = args.log_interval,
            **cond_kwargs,
        )

        for seed_i in range(num_q_seeds):
            final_train_sr = reached[seed_i, -win:].mean()
            final_eval_sr  = float(eval_sr[-1, seed_i])
            print(f"  seed {seed_i}:  train_sr={final_train_sr:.2f}"
                  f"  eval_sr={final_eval_sr:.2f}  (last {win} train eps)")

        results[cond_name] = dict(
            steps         = steps,    # [num_q_seeds, num_episodes]
            reached       = reached,
            eval_sr       = eval_sr,  # [num_chunks, num_q_seeds]
            eval_episodes = eval_ep,  # [num_chunks]
        )

    # ------------------------------------------------------------------
    # 6. Save and plot
    # ------------------------------------------------------------------

    if single_seed_mode:
        # Save a partial result file; combined analysis is done later by
        # experiments/reward_shaping/analyze_reward_shaping.py.
        cond_name   = list(results.keys())[0]
        partial_dir = output_dir / "partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        fname = f"results_{args.method}_{args.seed_idx:04d}.pkl"
        with open(partial_dir / fname, "wb") as f:
            pickle.dump(
                dict(
                    method     = args.method,
                    cond_name  = cond_name,
                    seed_idx   = args.seed_idx,
                    results    = results,
                    goal       = int(goal_per_seed[0]),
                    eval_start = int(eval_starts_per_seed[0]),
                    args       = vars(args),
                ),
                f,
            )
        print(f"\nPartial result saved → {partial_dir / fname}")
        print(f"\nDone.  Partial output written to {partial_dir / fname}")
        return

    print(f"\n{'='*60}")
    print("Saving results ...")
    print(f"{'='*60}")

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(
            dict(
                results              = results,
                args                 = vars(args),
                goal_per_seed        = goal_per_seed.tolist(),
                eval_starts_per_seed = eval_starts_per_seed.tolist(),
                has_allo_model       = allo_model_data is not None,
            ),
            f,
        )
    print(f"  results.pkl saved  → {output_dir / 'results.pkl'}")

    plot_results(
        results,
        output_dir / "learning_curves.png",
        window   = min(100, args.num_episodes // 10),
        max_steps= args.max_steps,
    )

    print(f"\nDone.  All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
