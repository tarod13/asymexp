"""
Reward shaping experiment — local (non-distributed) entry point.

Runs all available conditions (baseline, complex, gt, allo) across all seeds
in a single process.  Results are saved directly to <output_dir>/results.pkl
and <output_dir>/learning_curves.png.

Use this script for development and debugging.  For large-scale cluster runs
use run_reward_shaping_cluster.py with scripts/run_reward_shaping_array.sh.

Conditions produced depend on the flags supplied:

  --model_dir only                 → baseline + complex
  --model_dir --use_gt             → baseline + complex + gt
  --use_gt (no --model_dir)        → baseline + gt  (computed from env)
  --allo_model_dir                 → adds allo to any of the above
  (no flags)                       → baseline only

Usage
-----
    # Baseline only
    python experiments/reward_shaping/run_reward_shaping_local.py \\
        --env GridRoom-4-Doors --num_seeds 3

    # Baseline + complex shaped
    python experiments/reward_shaping/run_reward_shaping_local.py \\
        --env GridRoom-4-Doors \\
        --model_dir ./results/file/my_run \\
        --num_seeds 3

    # Baseline + complex + gt + allo
    python experiments/reward_shaping/run_reward_shaping_local.py \\
        --env GridRoom-4-Doors \\
        --model_dir ./results/file/my_run \\
        --allo_model_dir ./results/file/my_allo_run \\
        --use_gt --num_seeds 3

    # Ground-truth only (computed from environment)
    python experiments/reward_shaping/run_reward_shaping_local.py \\
        --env GridRoom-4-Doors \\
        --use_gt --num_seeds 3
"""

import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.rl.loading import (
    load_model, compute_gt_model_data, truncate_model_eigenvectors,
)
from src.rl.qlearning import build_all_potentials, run_q_learning
from src.rl.plotting import plot_results, plot_hitting_times_grid
from src.utils.envs import create_gridworld_env, get_canonical_free_states
from src.utils.metrics import compute_hitting_times_from_eigenvectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward shaping – local sweep over all conditions and seeds."
    )
    parser.add_argument(
        "--env", required=True,
        help="Environment name (e.g. 'GridRoom-4-Doors') passed as env_file_name "
             "to create_gridworld_env with env_type='file'.",
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Results directory for the complex representation (train_lap_rep.py). "
             "When provided, adds a 'shaped (complex)' condition. "
             "With --use_gt also adds a 'shaped (gt)' condition.",
    )
    parser.add_argument(
        "--allo_model_dir", default=None,
        help="Results directory for the ALLO representation (train_allo_rep.py). "
             "When provided, adds a 'shaped (allo)' condition.",
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
        "--total_steps", type=int, default=1_500_000,
        help="Total environment steps per seed (default: 1500000).",
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
        "--eval_interval", type=int, default=100_000,
        help="Run evaluation every this many environment steps (default: 100000).",
    )
    parser.add_argument(
        "--eval_seed", type=int, default=0,
        help="Seed for sampling random goal states (default: 0).",
    )
    parser.add_argument(
        "--use_gt", action="store_true",
        help="Include a ground-truth Laplacian condition.  GT eigenvectors are "
             "loaded from --model_dir if the gt_*.npy files are present there; "
             "otherwise they are computed analytically from the environment.",
    )
    parser.add_argument(
        "--num_eigenvectors", type=int, default=None,
        help="Number of eigenvector pairs to use for hitting-time computation "
             "across ALL methods (complex, gt, allo).  Defaults to all available "
             "pairs.  If the requested count exceeds what is available, the "
             "maximum available count is used and a warning is printed.",
    )
    parser.add_argument(
        "--gt_gamma", type=float, default=0.95,
        help="Discount factor γ used in the GT Laplacian (default: 0.95).",
    )
    parser.add_argument(
        "--gt_delta", type=float, default=0.1,
        help="Eigenvalue-shift δ used in the GT Laplacian (default: 0.1).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write outputs (default: <model_dir>/reward_shaping/).",
    )
    parser.add_argument(
        "--start_state", type=str, default=None,
        help="Fixed starting state as 'x,y' grid coordinates (e.g. '3,2'). "
             "If provided, every training and evaluation episode begins from this state.",
    )
    parser.add_argument(
        "--min_goal_distance", type=int, default=0,
        help="Minimum taxi (Manhattan) distance from the fixed starting state to any "
             "sampled goal (default: 0). Requires a valid --start_state.",
    )
    parser.add_argument(
        "--checkpoint_prefix", type=str, default="final_",
        help="Prefix for learned-eigenvector .npy files and the model checkpoint "
             "(default: 'final_'). Use e.g. 'latest_' to load in-progress checkpoints.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif model_dir is not None:
        output_dir = model_dir / "reward_shaping"
    else:
        output_dir = Path("reward_shaping")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load pre-trained models
    # ------------------------------------------------------------------

    # --- Complex (learned) representation ---
    learned_model_data = None
    if model_dir is not None:
        print(f"\n{'='*60}")
        print(f"Loading complex (learned) representation from: {model_dir}")
        print(f"{'='*60}")
        learned_model_data = load_model(
            model_dir, use_gt=False,
            checkpoint_prefix=args.checkpoint_prefix,
        )
        print(f"  Eigenvalue type    : {learned_model_data['eigenvalue_type']}")
        print(f"  Eigenvectors (K)   : {learned_model_data['eigenvalues_real'].shape[0]}")

    # --- Ground-truth representation ---
    gt_model_data = None
    if args.use_gt:
        if model_dir is not None:
            # Try loading saved GT eigenvectors from the model directory first.
            gt_files = [
                model_dir / "gt_left_real.npy",
                model_dir / "gt_left_imag.npy",
                model_dir / "gt_right_real.npy",
                model_dir / "gt_right_imag.npy",
                model_dir / "gt_eigenvalues_real.npy",
                model_dir / "gt_eigenvalues_imag.npy",
            ]
            if all(f.exists() for f in gt_files):
                print(f"\n{'='*60}")
                print(f"Loading GT eigenvectors from training artifacts: {model_dir}")
                print(f"{'='*60}")
                gt_model_data = load_model(
                    model_dir, use_gt=True,
                    checkpoint_prefix=args.checkpoint_prefix,
                )
                print(f"  Eigenvalue type    : {gt_model_data['eigenvalue_type']}")
                print(f"  Eigenvectors (K)   : {gt_model_data['eigenvalues_real'].shape[0]}")
            else:
                print(
                    f"WARNING: GT eigenvector files not found in {model_dir}; "
                    "computing ground-truth eigenvectors from environment samples.",
                    file=sys.stderr,
                )
                gt_model_data = None  # triggers env-based computation below

    # --- ALLO representation ---
    allo_model_data = None
    if args.allo_model_dir is not None:
        allo_model_dir = Path(args.allo_model_dir)
        print(f"\n{'='*60}")
        print(f"Loading ALLO representation from: {allo_model_dir}")
        print(f"{'='*60}")
        allo_model_data = load_model(
            allo_model_dir,
            checkpoint_prefix=args.checkpoint_prefix,
        )
        allo_model_data["left_real"]  = allo_model_data["right_real"]
        allo_model_data["left_imag"]  = np.zeros_like(allo_model_data["right_real"])
        allo_model_data["right_imag"] = np.zeros_like(allo_model_data["right_real"])
        allo_model_data["eigenvalues_imag"] = np.zeros_like(
            allo_model_data["eigenvalues_real"])
        print(f"  Eigenvalue type    : {allo_model_data['eigenvalue_type']}")
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
    fixed_start_canonical = None
    fixed_start_coords    = None

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

    eligible_goals: list[int] = list(range(num_canonical))
    if fixed_start_canonical is not None:
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
    # 2c. GT fallback: compute analytically from environment if needed
    # ------------------------------------------------------------------
    if args.use_gt and gt_model_data is None:
        print(f"\n{'='*60}")
        print("Computing ground-truth eigenvectors from environment ...")
        print(f"{'='*60}")
        gt_model_data = compute_gt_model_data(
            env, canonical_states,
            gamma=args.gt_gamma, delta=args.gt_delta,
            num_eigenvectors=args.num_eigenvectors,
        )
        resolved_k = gt_model_data["eigenvalues_real"].shape[0]
        print(f"  Eigenvector source : ground truth (computed from environment)")
        print(f"  Eigenvalue type    : laplacian")
        print(f"  Eigenvectors (K)   : {resolved_k}")
        print(f"  gt_gamma           : {args.gt_gamma}")
        print(f"  gt_delta           : {args.gt_delta}")

    # ------------------------------------------------------------------
    # 3. Apply eigenvector truncation, then compute hitting times
    # ------------------------------------------------------------------
    hitting_times      = None
    gt_hitting_times   = None
    allo_hitting_times = None

    if learned_model_data is not None:
        learned_model_data = truncate_model_eigenvectors(
            learned_model_data, args.num_eigenvectors
        )
        print(f"\n{'='*60}")
        print("Computing hitting times (complex representation) ...")
        print(f"{'='*60}")
        hitting_times = np.array(
            compute_hitting_times_from_eigenvectors(
                left_real        = learned_model_data["left_real"],
                left_imag        = learned_model_data["left_imag"],
                right_real       = learned_model_data["right_real"],
                right_imag       = learned_model_data["right_imag"],
                eigenvalues_real = learned_model_data["eigenvalues_real"],
                eigenvalues_imag = learned_model_data["eigenvalues_imag"],
                gamma            = learned_model_data["training_args"].get("gamma", 0.95),
                delta            = learned_model_data["training_args"].get("delta", 0.1),
                eigenvalue_type  = learned_model_data["eigenvalue_type"],
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
            np.array(learned_model_data["canonical_states"]),
            env,
            output_dir / "hitting_times_complex_plots",
        )

    if gt_model_data is not None:
        gt_model_data = truncate_model_eigenvectors(
            gt_model_data, args.num_eigenvectors
        )
        print(f"\n{'='*60}")
        print("Computing hitting times (ground-truth representation) ...")
        print(f"{'='*60}")
        gt_hitting_times = np.array(
            compute_hitting_times_from_eigenvectors(
                left_real        = gt_model_data["left_real"],
                left_imag        = gt_model_data["left_imag"],
                right_real       = gt_model_data["right_real"],
                right_imag       = gt_model_data["right_imag"],
                eigenvalues_real = gt_model_data["eigenvalues_real"],
                eigenvalues_imag = gt_model_data["eigenvalues_imag"],
                gamma            = gt_model_data["training_args"].get("gamma", 0.95),
                delta            = gt_model_data["training_args"].get("delta", 0.1),
                eigenvalue_type  = gt_model_data["eigenvalue_type"],
            )
        )
        gt_finite = gt_hitting_times[np.isfinite(gt_hitting_times) & (gt_hitting_times >= 0)]
        print(f"  Shape              : {gt_hitting_times.shape}")
        print(f"  Finite values      : {len(gt_finite)} / {gt_hitting_times.size}"
              f"  ({len(gt_finite)/gt_hitting_times.size:.1%})")
        if len(gt_finite) > 0:
            print(f"  Range              : [{gt_finite.min():.2f}, {gt_finite.max():.2f}]")
        np.save(output_dir / "hitting_times_gt.npy", gt_hitting_times)
        plot_hitting_times_grid(
            gt_hitting_times,
            np.array(gt_model_data["canonical_states"]),
            env,
            output_dir / "hitting_times_gt_plots",
        )

    if allo_model_data is not None:
        allo_model_data = truncate_model_eigenvectors(
            allo_model_data, args.num_eigenvectors
        )
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
            np.array(allo_model_data["canonical_states"]),
            env,
            output_dir / "hitting_times_allo_plots",
        )

    # ------------------------------------------------------------------
    # 4. Sample one fixed (goal, eval_start) per seed
    # ------------------------------------------------------------------
    eval_rng = np.random.default_rng(args.eval_seed)
    goal_per_seed = eval_rng.choice(
        eligible_goals, size=args.num_seeds,
        replace=args.num_seeds > len(eligible_goals),
    ).astype(np.int32)

    if fixed_start_canonical is not None:
        eval_starts_per_seed = np.full(args.num_seeds, fixed_start_canonical, dtype=np.int32)
        train_start_per_seed = np.full(args.num_seeds, fixed_start_canonical, dtype=np.int32)
    else:
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

    # ------------------------------------------------------------------
    # 5. Build conditions dict
    # ------------------------------------------------------------------
    complex_potential_per_seed = None
    if hitting_times is not None:
        complex_potential_per_seed = build_all_potentials(hitting_times)[:, goal_per_seed].T

    gt_potential_per_seed = None
    if gt_hitting_times is not None:
        gt_potential_per_seed = build_all_potentials(gt_hitting_times)[:, goal_per_seed].T

    allo_potential_per_seed = None
    if allo_hitting_times is not None:
        allo_potential_per_seed = build_all_potentials(allo_hitting_times)[:, goal_per_seed].T

    conditions = {
        "baseline": dict(potential_per_seed=None, shaping_coef=0.0),
    }
    if complex_potential_per_seed is not None:
        conditions[f"shaped β={args.shaping_coef} (complex)"] = dict(
            potential_per_seed=complex_potential_per_seed, shaping_coef=args.shaping_coef
        )
    if gt_potential_per_seed is not None:
        conditions[f"shaped β={args.shaping_coef} (gt)"] = dict(
            potential_per_seed=gt_potential_per_seed, shaping_coef=args.shaping_coef
        )
    if allo_potential_per_seed is not None:
        conditions[f"shaped β={args.shaping_coef} (allo)"] = dict(
            potential_per_seed=allo_potential_per_seed, shaping_coef=args.shaping_coef
        )

    # ------------------------------------------------------------------
    # 6. Run Q-learning for every condition
    # ------------------------------------------------------------------
    results = {}
    for cond_name, cond_kwargs in conditions.items():
        print(f"\n{'='*60}")
        print(f"Q-learning: {cond_name}")
        print(f"{'='*60}")

        eval_sr, eval_len_all, eval_len_suc, eval_steps = run_q_learning(
            env                  = env,
            canonical_states     = canonical_states,
            goal_per_seed        = goal_per_seed,
            eval_starts_per_seed = eval_starts_per_seed,
            train_start_per_seed = train_start_per_seed,
            min_goal_distance    = args.min_goal_distance,
            total_steps          = args.total_steps,
            num_seeds            = args.num_seeds,
            max_steps_per_episode= args.max_steps,
            num_eval_episodes    = args.num_eval_episodes,
            gamma                = args.gamma_rl,
            lr                   = args.lr,
            epsilon              = args.epsilon,
            seed                 = 0,
            eval_interval        = args.eval_interval,
            **cond_kwargs,
        )

        for seed_i in range(args.num_seeds):
            final_sr      = float(eval_sr[-1, seed_i])
            final_len_all = float(eval_len_all[-1, seed_i])
            final_len_suc = eval_len_suc[-1, seed_i]
            suc_disp = f"{final_len_suc:.1f}" if not np.isnan(final_len_suc) else "—"
            print(f"  seed {seed_i}:  eval_sr={final_sr:.2f}"
                  f"  eval_len_all={final_len_all:.0f}"
                  f"  eval_len_suc={suc_disp}")

        results[cond_name] = dict(
            eval_sr      = eval_sr,
            eval_len_all = eval_len_all,
            eval_len_suc = eval_len_suc,
            eval_steps   = eval_steps,
        )

    # ------------------------------------------------------------------
    # 7. Save and plot
    # ------------------------------------------------------------------
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

    plot_results(results, output_dir / "learning_curves.png")

    print(f"\nDone.  All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
