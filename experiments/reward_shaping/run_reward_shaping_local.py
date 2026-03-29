"""
Reward shaping experiment — local (non-distributed) entry point.

Runs up to 7 conditions sequentially (baseline, complex, gt_truncated,
gt_full, allo_hitting_time, allo_squared_diff, allo_weighted_squared_diff)
across all seeds in a single process.  Results are saved to
<output_dir>/results.pkl and <output_dir>/learning_curves.png.

Use this script for development and debugging.  For large-scale cluster runs
use run_reward_shaping_cluster.py with scripts/run_reward_shaping_array.sh.

Which conditions are produced depends on the flags supplied:

  --model_dir only          → baseline + complex
  --model_dir --use_gt      → baseline + complex + gt_truncated + gt_full
                              (gt_full skipped with warning if full files absent)
  --use_gt (no model_dir)   → baseline + gt_truncated  (GT computed from env)
  --allo_model_dir          → adds allo_hitting_time + allo_squared_diff +
                              allo_weighted_squared_diff
  (no flags)                → baseline only

Usage
-----
    # Baseline + complex + all GT + all ALLO
    python experiments/reward_shaping/run_reward_shaping_local.py \\
        --env GridRoom-4-Doors \\
        --model_dir ./results/file/my_run \\
        --allo_model_dir ./results/file/my_allo_run \\
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
from src.rl.qlearning import build_all_potentials, compute_allo_distance_matrices, run_q_learning
from src.rl.debug_viz import run_step_by_step_debug
from src.rl.plotting import plot_results, plot_hitting_times_grid
from src.utils.envs import create_gridworld_env, get_canonical_free_states, get_env_transition_markers, get_portal_tile_sets
from src.utils.plotting import plot_potential_vs_value
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
        "--total_steps", type=int, default=1_000_000,
        help="Total environment steps per seed (default: 1000000).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=250,
        help="Max steps per episode before declaring failure (default: 250).",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=100,
        help="Number of independent seeds to average over (default: 100).",
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
        "--epsilon", type=float, default=0.05,
        help="ε-greedy exploration rate (default: 0.05).",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1_000,
        help="Run evaluation every this many environment steps (default: 1000).",
    )
    parser.add_argument(
        "--eval_seed", type=int, default=0,
        help="Seed for sampling random goal states (default: 0).",
    )
    parser.add_argument(
        "--use_gt", action="store_true", default=True,
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
        "--start_state", type=str, default="1,1",
        help="Fixed starting state as 'x,y' grid coordinates (e.g. '3,2'). "
             "If provided, every training and evaluation episode begins from this state.",
    )
    parser.add_argument(
        "--goal_state", type=str, default=None,
        help="Fixed goal state as 'x,y' grid coordinates (e.g. '15,15'). "
             "If provided, overrides random goal sampling and forces num_seeds=1.",
    )
    parser.add_argument(
        "--min_goal_distance", type=int, default=8,
        help="Minimum taxi (Manhattan) distance from the fixed starting state to any "
             "sampled goal (default: 8). Requires a valid --start_state.",
    )
    parser.add_argument(
        "--checkpoint_prefix", type=str, default="final_",
        help="Prefix for learned-eigenvector .npy files and the model checkpoint "
             "(default: 'final_'). Use e.g. 'latest_' to load in-progress checkpoints.",
    )
    parser.add_argument(
        "--n_step_td", type=int, default=1,
        help="Number of steps for n-step Q-learning returns (default: 1).",
    )
    parser.add_argument(
        "--no_hitting_times_plot", action="store_true", default=True,
        help="Skip generating hitting-times visualizations (default: skip them).",
    )
    parser.add_argument(
        "--step_by_step_visual", action="store_true",
        help="Run a single debugging episode (1 seed, 1 episode) and save a "
             "per-step figure to <output_dir>/debug_frames/.  Skips normal "
             "Q-learning, results saving, and learning-curve plots.",
    )
    parser.add_argument(
        "--potential_mode", type=str, default="inverse-power",
        choices=["negative", "inverse", "inverse-power", "inverse-log", "pos-exp"],
        help="Transformation applied to normalised hitting times to produce Φ(s). "
             "'negative': Φ=−h, 'inverse': Φ=1/(h/τ+δ), "
             "'inverse-power': Φ=1/((h/τ)^p+δ) (default, p=--potential_power), "
             "'inverse-log': Φ=1/(log1p(h/τ)+δ), 'pos-exp': Φ=b^h (b=--potential_base).",
    )
    parser.add_argument(
        "--potential_temp", type=float, default=1.0,
        help="Temperature τ used by 'inverse', 'inverse-power', and 'inverse-log' "
             "potential modes (default: 1.0).",
    )
    parser.add_argument(
        "--potential_delta", type=float, default=1.0,
        help="Denominator offset δ for 'inverse', 'inverse-power', and 'inverse-log' modes. "
             "Caps the maximum potential at 1/δ; default 1.0 gives max=1.0.",
    )
    parser.add_argument(
        "--potential_power", type=float, default=0.5,
        help="Exponent p for 'inverse-power' mode (default: 0.5; 0.5 recovers inverse-sqrt).",
    )
    parser.add_argument(
        "--potential_base", type=float, default=0.99,
        help="Base b for 'pos-exp' mode: Φ(s)=b^h (default: 0.99; must be in (0,1) for decay).",
    )
    args = parser.parse_args()

    # --goal_state forces a single deterministic trajectory; override counts.
    if args.goal_state is not None:
        if args.num_seeds != 1:
            print(
                f"  NOTE: --goal_state provided; overriding --num_seeds "
                f"from {args.num_seeds} → 1.",
                file=sys.stderr,
            )
            args.num_seeds = 1

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

    # --- Ground-truth representations (truncated and full, evaluated separately) ---
    # gt_model_data_trunc : uses gt_* prefix files (or env-computed), always truncated
    # gt_model_data_full  : uses full_* prefix / precomputed file, NOT truncated;
    #                       set to None with a warning if files are absent.
    gt_model_data_trunc = None
    gt_model_data_full  = None
    if args.use_gt:
        # ── Truncated GT ──────────────────────────────────────────────────────
        if model_dir is not None:
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
                print(f"Loading GT eigenvectors (truncated) from: {model_dir}")
                print(f"{'='*60}")
                gt_model_data_trunc = load_model(
                    model_dir, use_gt=True,
                    checkpoint_prefix=args.checkpoint_prefix,
                )
                print(f"  Eigenvalue type    : {gt_model_data_trunc['eigenvalue_type']}")
                print(f"  Eigenvectors (K)   : {gt_model_data_trunc['eigenvalues_real'].shape[0]}")
            else:
                print(
                    f"WARNING: gt_* eigenvector files not found in {model_dir}; "
                    "computing ground-truth eigenvectors from environment.",
                    file=sys.stderr,
                )
                # gt_model_data_trunc stays None → triggers env-based fallback later

        # ── Full GT ───────────────────────────────────────────────────────────
        if model_dir is not None:
            full_files = [
                model_dir / "full_left_real.npy",
                model_dir / "full_left_imag.npy",
                model_dir / "full_right_real.npy",
                model_dir / "full_right_imag.npy",
                model_dir / "full_eigenvalues_real.npy",
                model_dir / "full_eigenvalues_imag.npy",
            ]
            precomp_full_ht = model_dir / "full_ideal_gt_hitting_times.npy"
            if precomp_full_ht.exists() or all(f.exists() for f in full_files):
                print(f"\n{'='*60}")
                print(f"Loading full (un-truncated) GT eigenvectors from: {model_dir}")
                print(f"{'='*60}")
                if precomp_full_ht.exists():
                    print(f"  (pre-computed hitting times found: {precomp_full_ht.name})")
                    gt_model_data_full = "precomputed"  # sentinel; handled in Section 3
                else:
                    gt_model_data_full = load_model(
                        model_dir, use_gt=True, gt_prefix="full_",
                        checkpoint_prefix=args.checkpoint_prefix,
                    )
                    print(f"  Eigenvalue type    : {gt_model_data_full['eigenvalue_type']}")
                    print(f"  Eigenvectors (K)   : {gt_model_data_full['eigenvalues_real'].shape[0]}")
            else:
                print(
                    "WARNING: full_* eigenvector files and "
                    "full_ideal_gt_hitting_times.npy not found in "
                    f"{model_dir}; skipping gt_full condition.",
                    file=sys.stderr,
                )

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
    # 2b. Validate optional fixed starting/goal states; build eligible goal set.
    # ------------------------------------------------------------------
    can_lookup = {int(s): i for i, s in enumerate(canonical_states)}

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

    fixed_goal_canonical = None
    fixed_goal_coords    = None

    if args.goal_state is not None:
        try:
            gx, gy = [int(v.strip()) for v in args.goal_state.split(",")]
        except Exception:
            print(
                f"WARNING: could not parse --goal_state '{args.goal_state}'. "
                "Expected 'x,y' format. Falling back to random goal sampling.",
                file=sys.stderr,
            )
        else:
            goal_full = int(gy * env.width + gx)
            if goal_full in can_lookup:
                fixed_goal_canonical = can_lookup[goal_full]
                fixed_goal_coords    = (gx, gy)
                print(f"\n  Fixed goal state   : ({gx},{gy})"
                      f"  [full={goal_full}, canonical={fixed_goal_canonical}]")
            else:
                print(
                    f"WARNING: goal state ({gx},{gy}) is blocked or out of bounds. "
                    "Falling back to random goal sampling.",
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
    if args.use_gt and gt_model_data_trunc is None:
        print(f"\n{'='*60}")
        print("Computing ground-truth eigenvectors from environment ...")
        print(f"{'='*60}")
        gt_model_data_trunc = compute_gt_model_data(
            env, canonical_states,
            gamma=args.gt_gamma, delta=args.gt_delta,
            num_eigenvectors=args.num_eigenvectors,
        )
        resolved_k = gt_model_data_trunc["eigenvalues_real"].shape[0]
        print(f"  Eigenvector source : ground truth (computed from environment)")
        print(f"  Eigenvalue type    : laplacian")
        print(f"  Eigenvectors (K)   : {resolved_k}")
        print(f"  gt_gamma           : {args.gt_gamma}")
        print(f"  gt_delta           : {args.gt_delta}")

    # ------------------------------------------------------------------
    # 3. Apply eigenvector truncation, then compute distance matrices
    # ------------------------------------------------------------------

    def _log_matrix(tag, mat):
        """Print shape and finite-value statistics for a distance matrix."""
        finite = mat[np.isfinite(mat)]
        print(f"  Shape              : {mat.shape}")
        print(f"  Finite values      : {len(finite)} / {mat.size}"
              f"  ({len(finite)/mat.size:.1%})")
        if len(finite) > 0:
            print(f"  Range              : [{finite.min():.2f}, {finite.max():.2f}]")

    def _compute_ht(md, label):
        """Compute hitting-time matrix from a model_data dict."""
        return np.array(
            compute_hitting_times_from_eigenvectors(
                left_real        = md["left_real"],
                left_imag        = md["left_imag"],
                right_real       = md["right_real"],
                right_imag       = md["right_imag"],
                eigenvalues_real = md["eigenvalues_real"],
                eigenvalues_imag = md["eigenvalues_imag"],
                gamma            = md["training_args"].get("gamma", 0.95),
                delta            = md["training_args"].get("delta", 0.1),
                eigenvalue_type  = md["eigenvalue_type"],
            )
        )

    # ── 3a. Complex (learned) ─────────────────────────────────────────────────
    hitting_times = None
    if learned_model_data is not None:
        learned_model_data = truncate_model_eigenvectors(
            learned_model_data, args.num_eigenvectors
        )
        print(f"\n{'='*60}")
        print("Computing hitting times (complex representation) ...")
        print(f"{'='*60}")
        hitting_times = _compute_ht(learned_model_data, "complex")
        _log_matrix("complex", hitting_times)
        np.save(output_dir / "hitting_times.npy", hitting_times)
        if not args.no_hitting_times_plot:
            plot_hitting_times_grid(
                hitting_times,
                np.array(learned_model_data["canonical_states"]),
                env,
                output_dir / "hitting_times_complex_plots",
            )

    # ── 3b. GT truncated ─────────────────────────────────────────────────────
    # The gt_* files were already truncated to num_eigenvector_pairs during
    # training.  Do NOT apply a second truncation — use exactly what was loaded.
    gt_hitting_times_trunc = None
    if gt_model_data_trunc is not None:
        gt_model_data_trunc = truncate_model_eigenvectors(
            gt_model_data_trunc, None
        )
        print(f"\n{'='*60}")
        print("Computing hitting times (GT truncated) ...")
        print(f"{'='*60}")
        gt_hitting_times_trunc = _compute_ht(gt_model_data_trunc, "gt_truncated")
        _log_matrix("gt_truncated", gt_hitting_times_trunc)
        np.save(output_dir / "hitting_times_gt_truncated.npy", gt_hitting_times_trunc)
        if not args.no_hitting_times_plot:
            plot_hitting_times_grid(
                gt_hitting_times_trunc,
                np.array(gt_model_data_trunc["canonical_states"]),
                env,
                output_dir / "hitting_times_gt_truncated_plots",
            )

    # ── 3c. GT full ───────────────────────────────────────────────────────────
    gt_hitting_times_full = None
    if gt_model_data_full is not None:
        precomp_full_ht = (
            model_dir / "full_ideal_gt_hitting_times.npy"
            if model_dir is not None else None
        )
        if gt_model_data_full == "precomputed":
            print(f"\n{'='*60}")
            print("Loading pre-computed GT hitting times (gt_full, fast path) ...")
            print(f"{'='*60}")
            gt_hitting_times_full = np.load(precomp_full_ht)
            print(f"  Source             : {precomp_full_ht}")
            _log_matrix("gt_full", gt_hitting_times_full)
        else:
            gt_model_data_full = truncate_model_eigenvectors(gt_model_data_full, None)
            print(f"\n{'='*60}")
            print("Computing hitting times (GT full, no truncation) ...")
            print(f"{'='*60}")
            gt_hitting_times_full = _compute_ht(gt_model_data_full, "gt_full")
            _log_matrix("gt_full", gt_hitting_times_full)
        np.save(output_dir / "hitting_times_gt_full.npy", gt_hitting_times_full)
        if not args.no_hitting_times_plot and gt_model_data_full != "precomputed":
            plot_hitting_times_grid(
                gt_hitting_times_full,
                np.array(gt_model_data_full["canonical_states"]),
                env,
                output_dir / "hitting_times_gt_full_plots",
            )

    # ── 3d. ALLO: hitting time, squared diff, weighted squared diff ───────────
    allo_hitting_times      = None
    allo_sq_diff            = None
    allo_weighted_sq_diff   = None
    if allo_model_data is not None:
        allo_model_data = truncate_model_eigenvectors(
            allo_model_data, args.num_eigenvectors
        )
        print(f"\n{'='*60}")
        print("Computing ALLO distance matrices ...")
        print(f"{'='*60}")
        allo_hitting_times = _compute_ht(allo_model_data, "allo_hitting_time")
        print("  [allo_hitting_time]")
        _log_matrix("allo_hitting_time", allo_hitting_times)
        np.save(output_dir / "hitting_times_allo.npy", allo_hitting_times)

        allo_sq_diff, allo_weighted_sq_diff = compute_allo_distance_matrices(allo_model_data)
        print("  [allo_squared_diff]")
        _log_matrix("allo_squared_diff", allo_sq_diff)
        print("  [allo_weighted_squared_diff]")
        _log_matrix("allo_weighted_squared_diff", allo_weighted_sq_diff)
        np.save(output_dir / "allo_sq_diff.npy",          allo_sq_diff)
        np.save(output_dir / "allo_weighted_sq_diff.npy", allo_weighted_sq_diff)
        if not args.no_hitting_times_plot:
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
    if fixed_goal_canonical is not None:
        goal_per_seed = np.array([fixed_goal_canonical], dtype=np.int32)
    else:
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

    if fixed_goal_canonical is not None:
        stf = int(canonical_states[int(eval_starts_per_seed[0])])
        sty, stx = divmod(stf, env.width)
        gfx, gfy = fixed_goal_coords
        print(f"\n  Evaluating single deterministic trajectory: "
              f"Start ({stx},{sty}) -> Goal ({gfx},{gfy})")

    # ------------------------------------------------------------------
    # 5. Build conditions dict (up to 7 entries, insertion order = palette order)
    # ------------------------------------------------------------------
    def _make_potential(dist_matrix):
        """Convert a distance matrix → per-seed potential slice [num_seeds, N]."""
        return build_all_potentials(
            dist_matrix,
            potential_mode=args.potential_mode,
            potential_temp=args.potential_temp,
            potential_delta=args.potential_delta,
            potential_power=args.potential_power,
            potential_base=args.potential_base,
            gamma=args.gamma_rl,
        )[:, goal_per_seed].T

    b = args.shaping_coef
    conditions = {
        "baseline": dict(potential_per_seed=None, shaping_coef=0.0),
    }
    if hitting_times is not None:
        conditions[f"shaped β={b} (complex)"] = dict(
            potential_per_seed=_make_potential(hitting_times), shaping_coef=b,
        )
    if gt_hitting_times_trunc is not None:
        conditions[f"shaped β={b} (gt truncated)"] = dict(
            potential_per_seed=_make_potential(gt_hitting_times_trunc), shaping_coef=b,
        )
    if gt_hitting_times_full is not None:
        conditions[f"shaped β={b} (gt full)"] = dict(
            potential_per_seed=_make_potential(gt_hitting_times_full), shaping_coef=b,
        )
    if allo_hitting_times is not None:
        conditions[f"shaped β={b} (allo hitting time)"] = dict(
            potential_per_seed=_make_potential(allo_hitting_times), shaping_coef=b,
        )
    if allo_sq_diff is not None:
        conditions[f"shaped β={b} (allo squared diff)"] = dict(
            potential_per_seed=_make_potential(allo_sq_diff), shaping_coef=b,
        )
    if allo_weighted_sq_diff is not None:
        conditions[f"shaped β={b} (allo weighted diff)"] = dict(
            potential_per_seed=_make_potential(allo_weighted_sq_diff), shaping_coef=b,
        )

    # ------------------------------------------------------------------
    # 5b. Step-by-step visual debug (bypasses all normal training/plotting)
    # ------------------------------------------------------------------
    if args.step_by_step_visual:
        print(f"\n{'='*60}")
        print("STEP-BY-STEP VISUAL DEBUG MODE")
        print(f"{'='*60}")

        full_to_can_dbg = np.full(env.width * env.height, -1, dtype=np.int32)
        for i, s in enumerate(canonical_states):
            full_to_can_dbg[int(s)] = i

        dbg_door_markers = get_env_transition_markers(env)
        dbg_portal_sources, dbg_portal_ends = get_portal_tile_sets(env)

        for cond_name, cond_kwargs in conditions.items():
            potential_per_seed = cond_kwargs.get("potential_per_seed")
            shaping_coef_cond  = cond_kwargs.get("shaping_coef", 0.0)

            if potential_per_seed is not None:
                potential_seed0 = np.array(potential_per_seed[0], dtype=np.float32)
            else:
                potential_seed0 = np.zeros(num_canonical, dtype=np.float32)

            # Build a filesystem-safe subdirectory name from the condition label
            safe_name = (
                cond_name
                .replace(" ", "_")
                .replace("=", "")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "-")
            )
            cond_dir = output_dir / "debug_frames" / safe_name

            print(f"\n  --- Condition: {cond_name} ---")
            print(f"  Goal (canonical)  : {goal_per_seed[0]}")
            print(f"  Start (canonical) : {eval_starts_per_seed[0]}")
            print(f"  Output dir        : {cond_dir}")

            run_step_by_step_debug(
                env              = env,
                canonical_states = canonical_states,
                full_to_can      = full_to_can_dbg,
                potential        = potential_seed0,
                goal_canonical   = int(goal_per_seed[0]),
                start_canonical  = int(eval_starts_per_seed[0]),
                gamma            = args.gamma_rl,
                lr               = args.lr,
                epsilon          = args.epsilon,
                shaping_coef     = shaping_coef_cond,
                max_steps        = args.max_steps,
                debug_dir        = cond_dir,
                n_step_td        = args.n_step_td,
                portals          = dbg_door_markers if dbg_door_markers else None,
                portal_sources   = dbg_portal_sources if dbg_portal_sources else None,
                portal_ends      = dbg_portal_ends   if dbg_portal_ends   else None,
                potential_mode   = args.potential_mode,
                potential_temp   = args.potential_temp,
                potential_power  = args.potential_power,
                potential_base   = args.potential_base,
            )

        print(f"\nDone.  All debug frames written to {output_dir / 'debug_frames'}/")
        return

    # ------------------------------------------------------------------
    # 6. Run Q-learning for every condition
    # ------------------------------------------------------------------
    results = {}
    for cond_name, cond_kwargs in conditions.items():
        print(f"\n{'='*60}")
        print(f"Q-learning: {cond_name}")
        print(f"{'='*60}")

        eval_sr, eval_len_all, eval_len_suc, eval_steps, Q_final = run_q_learning(
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
            n_step_td            = args.n_step_td,
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
            eval_sr           = eval_sr,
            eval_len_all      = eval_len_all,
            eval_len_suc      = eval_len_suc,
            eval_steps        = eval_steps,
            Q_final           = Q_final,
            potential_per_seed= cond_kwargs.get("potential_per_seed"),
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

    # ------------------------------------------------------------------
    # 8. Potential vs value plots (one figure per condition)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Plotting potential vs value ...")
    print(f"{'='*60}")

    pv_dir = output_dir / "potential_vs_value"
    pv_dir.mkdir(exist_ok=True)

    door_markers = get_env_transition_markers(env)
    num_canonical = len(canonical_states)

    # Extract baseline V(s) to use as a reference row in shaped-condition plots.
    baseline_V = None
    if "baseline" in results:
        baseline_V = results["baseline"]["Q_final"].max(axis=-1)  # [num_seeds, num_canonical]

    for cond_name, cond_data in results.items():
        # Baseline has no potential — skip; its values appear as row 2 in other plots.
        if cond_name == "baseline":
            continue

        Q_fin = cond_data["Q_final"]            # [num_seeds, num_states, num_actions]
        pot   = cond_data["potential_per_seed"]  # [num_seeds, num_states] or None

        V_per_seed   = Q_fin.max(axis=-1)                          # [num_seeds, num_canonical]
        pot_per_seed = np.array(pot, dtype=np.float32)             # always non-None here

        safe_name = cond_name.replace(" ", "_").replace("=", "").replace("/", "_")
        save_path = str(pv_dir / f"{safe_name}_potential_vs_value.png")

        plot_potential_vs_value(
            canonical_states        = canonical_states,
            grid_width              = env.width,
            grid_height             = env.height,
            potential_per_seed      = pot_per_seed,
            value_per_seed          = V_per_seed,
            goal_per_seed           = goal_per_seed,
            baseline_value_per_seed = baseline_V,
            cond_name               = cond_name,
            portals                 = door_markers if door_markers else None,
            save_path               = save_path,
        )

    print(f"\nDone.  All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
