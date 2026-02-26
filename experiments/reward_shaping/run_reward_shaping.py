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
from src.utils.envs import create_gridworld_env
from src.utils.metrics import compute_hitting_times_from_eigenvectors


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
        # Normalized learned eigenvectors (biorthogonal, unit norm)
        left_real  = np.load(model_dir / "final_learned_left_real_normalized.npy")
        left_imag  = np.load(model_dir / "final_learned_left_imag_normalized.npy")
        right_real = np.load(model_dir / "final_learned_right_real_normalized.npy")
        right_imag = np.load(model_dir / "final_learned_right_imag_normalized.npy")
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
    # env.asymmetric_transitions stores (dest_state, reverse_action) -> prob.
    # Reconstruct (source_state, forward_action) pairs for the blocked set.
    blocked: set[tuple[int, int]] = set()
    if env.has_doors:
        action_reverse = {0: 2, 1: 3, 2: 0, 3: 1}
        action_delta   = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        for (dest_full, rev_action) in env.asymmetric_transitions:
            dest_full    = int(dest_full)
            rev_action   = int(rev_action)
            fwd_action   = action_reverse[rev_action]
            dx, dy       = action_delta[rev_action]
            dest_y, dest_x = divmod(dest_full, env.width)
            source_x = dest_x + dx
            source_y = dest_y + dy
            if 0 <= source_x < env.width and 0 <= source_y < env.height:
                source_full = source_y * env.width + source_x
                blocked.add((source_full, fwd_action))

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

def build_potential(hitting_times: np.ndarray, goal_idx: int) -> np.ndarray:
    """
    Build the potential F(s) = −h(s, goal) used for reward shaping.

    Hitting times are normalised by the 90th-percentile finite value so that
    the shaping coefficient β is independent of the grid size.  Non-finite
    values (disconnected states, numerical issues) are clamped.
    """
    h = hitting_times[:, goal_idx].copy()

    finite_mask = np.isfinite(h) & (h >= 0)
    if finite_mask.sum() == 0:
        return np.zeros(len(h), dtype=np.float32)

    h_ref = float(np.percentile(h[finite_mask], 90))
    if h_ref < 1e-8:
        h_ref = float(h[finite_mask].max()) + 1e-8

    # Clamp non-finite values
    h = np.where(np.isfinite(h) & (h >= 0), h, h_ref)
    h = h / h_ref  # normalise to ≈[0, 1]

    return -h.astype(np.float32)   # higher potential = closer to goal


# ===========================================================================
# Tabular Q-learning
# ===========================================================================

def run_q_learning(
    env,
    canonical_states: np.ndarray,
    goal_idx: int,
    num_episodes: int,
    num_seeds: int = 1,
    max_steps_per_episode: int = 500,
    potential: np.ndarray | None = None,
    shaping_coef: float = 0.0,
    gamma: float = 0.99,
    lr: float = 0.1,
    epsilon: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run tabular ε-greedy Q-learning on a goal-reaching task using the real env.

    All seeds are run in parallel via jax.vmap; the episode and step loops
    within each seed are jax.lax.scan calls (JIT-compiled, no Python overhead).

    Task
    ----
    Reach *goal_idx* from a random non-goal canonical state.
    Sparse reward: +1.0 on reaching the goal, 0 otherwise.

    Shaping (optional)
    ------------------
    Potential-based bonus: β · (γ · F(s') − F(s))
    where F is precomputed from hitting times.

    Returns
    -------
    steps_per_episode : [num_seeds, num_episodes] int
    reached_goal      : [num_seeds, num_episodes] bool
    """
    num_states  = len(canonical_states)
    num_actions = 4

    # Inverse mapping: full grid index -> canonical index (-1 for non-canonical states)
    full_to_can = np.full(env.width * env.height, -1, dtype=np.int32)
    for i, s in enumerate(canonical_states):
        full_to_can[int(s)] = i
    full_to_can_jax      = jnp.array(full_to_can,      dtype=jnp.int32)
    canonical_states_jax = jnp.array(canonical_states, dtype=jnp.int32)
    non_goal_jax         = jnp.array(
        [s for s in range(num_states) if s != goal_idx], dtype=jnp.int32
    )
    potential_jax = (
        jnp.array(potential, dtype=jnp.float32) if potential is not None else None
    )

    def run_step(carry, step_key):
        Q, s, done = carry

        eps_key, rand_key, env_key = jax.random.split(step_key, 3)

        # ε-greedy action selection
        use_random = jax.random.uniform(eps_key) < epsilon
        greedy_a   = jnp.argmax(Q[s])
        random_a   = jax.random.randint(rand_key, (), 0, num_actions)
        a          = jnp.where(use_random, random_a, greedy_a)

        # Step through the real environment
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

        reached = s_prime == goal_idx
        r       = jnp.where(reached, 1.0, 0.0)
        if potential_jax is not None:
            r = r + shaping_coef * (gamma * potential_jax[s_prime] - potential_jax[s])

        # Q-learning update — skipped if episode was already done
        target = jnp.where(reached, r, r + gamma * Q[s_prime].max())
        Q = jax.lax.cond(
            done,
            lambda Q: Q,
            lambda Q: Q.at[s, a].add(lr * (target - Q[s, a])),
            Q,
        )

        new_done = done | reached
        new_s    = jnp.where(done, s, s_prime)
        return (Q, new_s, new_done), new_done

    def run_episode(Q, ep_key):
        ep_key, start_key = jax.random.split(ep_key)
        s         = non_goal_jax[jax.random.randint(start_key, (), 0, len(non_goal_jax))]
        step_keys = jax.random.split(ep_key, max_steps_per_episode)

        (Q, _, _), done_flags = jax.lax.scan(
            run_step, (Q, s, jnp.array(False)), step_keys,
        )

        # steps = index of first True + 1; max_steps_per_episode if never reached
        any_done = jnp.any(done_flags)
        steps    = jnp.where(any_done, jnp.argmax(done_flags) + 1, max_steps_per_episode)
        return Q, (steps, any_done)

    def run_all(key):
        Q       = jnp.zeros((num_states, num_actions), dtype=jnp.float32)
        ep_keys = jax.random.split(key, num_episodes)
        _, (steps_per_ep, reached) = jax.lax.scan(run_episode, Q, ep_keys)
        return steps_per_ep, reached

    seed_keys                = jax.random.split(jax.random.PRNGKey(seed), num_seeds)
    steps_per_ep, reached    = jax.jit(jax.vmap(run_all))(seed_keys)
    return np.array(steps_per_ep, dtype=np.int32), np.array(reached)


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
    Save a two-panel figure:
      Left  – steps per episode (lower = better, clipped at max_steps on failure)
      Right – success rate       (higher = better)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, data) in enumerate(results.items()):
        c = colours[idx % len(colours)]
        steps_arr   = data["steps"].astype(float)   # [seeds, episodes]
        reached_arr = data["reached"].astype(float)  # [seeds, episodes]

        n_ep = steps_arr.shape[1]
        x    = np.arange(n_ep)

        for arr, ax, ylabel in [
            (steps_arr,   axes[0], f"Steps per episode (max {max_steps})"),
            (reached_arr, axes[1], "Success rate"),
        ]:
            mean = _smooth(arr.mean(axis=0), window)
            std  = arr.std(axis=0)
            std_s = _smooth(std, window)

            ax.plot(x, mean, label=label, color=c, linewidth=1.5)
            ax.fill_between(x, mean - std_s, mean + std_s, color=c, alpha=0.15)
            ax.set_xlabel("Episode", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[0].set_title("Steps per episode  (↓ better)", fontsize=11)
    axes[1].set_title("Success rate  (↑ better)", fontsize=11)
    axes[1].set_ylim(-0.05, 1.10)

    for ax in axes:
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle(
        f"Reward shaping with Laplacian hitting times\n"
        f"(smoothing window = {window} episodes)",
        fontsize=12,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward shaping with Laplacian hitting times."
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Results directory produced by train_lap_rep.py.",
    )
    parser.add_argument(
        "--goal_state", type=int, default=None,
        help="Canonical-state index of the goal (default: last canonical state).",
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
        "--use_gt", action="store_true",
        help="Use ground-truth eigenvectors instead of the learned ones.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write outputs (default: <model_dir>/reward_shaping/).",
    )
    args = parser.parse_args()

    model_dir  = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "reward_shaping"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load pre-trained Laplacian model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading Laplacian model from: {model_dir}")
    print(f"{'='*60}")
    model_data     = load_model(model_dir, use_gt=args.use_gt)
    training_args  = model_data["training_args"]
    canonical_states = model_data["canonical_states"]
    num_canonical  = len(canonical_states)
    evtype         = model_data["eigenvalue_type"]

    print(f"  Eigenvector source : {'ground truth' if args.use_gt else 'learned (normalized)'}")
    print(f"  Eigenvalue type    : {evtype}")
    print(f"  Canonical states   : {num_canonical}")
    print(f"  Eigenvectors (K)   : {model_data['eigenvalues_real'].shape[0]}")

    # ------------------------------------------------------------------
    # 2. Build environment
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Building environment ...")
    print(f"{'='*60}")

    # Reconstruct the environment from the saved training args.
    # We nullify env_file (an absolute path from the training machine) so
    # create_gridworld_env falls back to env_file_name which is portable.
    ta = SimpleNamespace(**training_args)
    ta.env_file = None
    env = create_gridworld_env(ta)

    # ------------------------------------------------------------------
    # 3. Compute hitting times
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Computing hitting times ...")
    print(f"{'='*60}")

    hitting_times = np.array(
        compute_hitting_times_from_eigenvectors(
            left_real        = model_data["left_real"],
            left_imag        = model_data["left_imag"],
            right_real       = model_data["right_real"],
            right_imag       = model_data["right_imag"],
            eigenvalues_real = model_data["eigenvalues_real"],
            eigenvalues_imag = model_data["eigenvalues_imag"],
            gamma            = training_args.get("gamma", 0.95),
            delta            = training_args.get("delta", 0.1),
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

    # ------------------------------------------------------------------
    # 4. Select goal state
    # ------------------------------------------------------------------
    if args.goal_state is not None:
        goal_idx = args.goal_state
        if not (0 <= goal_idx < num_canonical):
            raise ValueError(
                f"--goal_state {goal_idx} is out of range [0, {num_canonical-1}]"
            )
    else:
        # Default: last canonical state (often in the far corner of the grid)
        goal_idx = num_canonical - 1

    goal_full = int(canonical_states[goal_idx])
    goal_y, goal_x = divmod(goal_full, env.width)
    print(f"\n  Goal               : canonical index {goal_idx}"
          f" → grid ({goal_x}, {goal_y})")

    potential = build_potential(hitting_times, goal_idx)

    # ------------------------------------------------------------------
    # 5. Run Q-learning: baseline vs shaped
    # ------------------------------------------------------------------
    evtype_label = "GT" if args.use_gt else "learned"
    conditions = {
        "baseline":
            dict(potential=None,      shaping_coef=0.0),
        f"shaped β={args.shaping_coef} ({evtype_label})":
            dict(potential=potential, shaping_coef=args.shaping_coef),
    }

    results = {}
    win = min(200, args.num_episodes)
    for cond_name, cond_kwargs in conditions.items():
        print(f"\n{'='*60}")
        print(f"Q-learning: {cond_name}")
        print(f"{'='*60}")

        steps, reached = run_q_learning(
            env                  = env,
            canonical_states     = canonical_states,
            goal_idx             = goal_idx,
            num_episodes         = args.num_episodes,
            num_seeds            = args.num_seeds,
            max_steps_per_episode= args.max_steps,
            gamma                = args.gamma_rl,
            lr                   = args.lr,
            epsilon              = args.epsilon,
            seed                 = 0,
            **cond_kwargs,
        )  # steps, reached: [num_seeds, num_episodes]

        for seed_i in range(args.num_seeds):
            final_sr = reached[seed_i, -win:].mean()
            print(f"  seed {seed_i}:  final success rate = {final_sr:.2f}"
                  f"  (last {win} episodes)")

        results[cond_name] = dict(
            steps   = steps,    # [num_seeds, num_episodes]
            reached = reached,
        )

    # ------------------------------------------------------------------
    # 6. Save and plot
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Saving results ...")
    print(f"{'='*60}")

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(
            dict(
                results   = results,
                args      = vars(args),
                goal_idx  = goal_idx,
                goal_pos  = (goal_x, goal_y),
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
