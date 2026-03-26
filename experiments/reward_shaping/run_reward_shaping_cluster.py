"""
Reward shaping experiment — cluster (distributed) entry point.

Runs a single (method, seed) pair as part of a SLURM array job.
Partial results are saved to <output_dir>/partial/results_<method>_<seed>.pkl
and later aggregated by experiments/reward_shaping/analyze_reward_shaping.py.

Usage (called by scripts/run_reward_shaping_array.sh)
------
    python experiments/reward_shaping/run_reward_shaping_cluster.py \\
        --env GridRoom-4-Doors \\
        --model_dir ./results/file/my_run \\
        --method complex --seed_idx 2 \\
        --num_seeds 5 --total_steps 1500000
"""

import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.rl.loading import load_model, compute_gt_model_data, truncate_model_eigenvectors
from src.rl.qlearning import build_all_potentials, compute_allo_distance_matrices, run_q_learning
from src.utils.envs import create_gridworld_env, get_canonical_free_states
from src.utils.metrics import compute_hitting_times_from_eigenvectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward shaping – single (method, seed) run for cluster array jobs."
    )
    parser.add_argument(
        "--env", required=True,
        help="Environment name (e.g. 'GridRoom-4-Doors') passed as env_file_name "
             "to create_gridworld_env with env_type='file'.",
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Results directory for the complex representation (train_lap_rep.py). "
             "Required for 'complex' and 'gt' methods.",
    )
    parser.add_argument(
        "--allo_model_dir", default=None,
        help="Results directory for the ALLO representation (train_allo_rep.py). "
             "Required for the 'allo' method.",
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
        help="Total number of seeds in this sweep (used to validate --seed_idx).",
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
        help="Use ground-truth Laplacian eigenvectors (implied by --method gt).",
    )
    parser.add_argument(
        "--num_eigenvectors", type=int, default=None,
        help="Number of eigenvector pairs to use for hitting-time computation "
             "across ALL methods.  Defaults to all available pairs.  If the "
             "requested count exceeds what is available, the maximum is used "
             "and a warning is printed.",
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
        "--goal_state", type=str, default=None,
        help="Fixed goal state as 'x,y' grid coordinates (e.g. '15,15'). "
             "If provided, overrides random goal sampling and forces num_seeds=1.",
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
    parser.add_argument(
        "--method", type=str, required=True,
        choices=[
            "baseline", "complex",
            "gt_truncated", "gt_full",
            "allo_hitting_time", "allo_squared_diff", "allo_weighted_squared_diff",
            # Legacy aliases kept for backward-compatibility
            "gt", "allo",
        ],
        help="Which condition to run.  gt/allo are legacy aliases for "
             "gt_truncated/allo_hitting_time.",
    )
    parser.add_argument(
        "--seed_idx", type=int, required=True,
        help="Which seed to run (0-indexed, up to --num_seeds-1).",
    )
    parser.add_argument(
        "--n_step_td", type=int, default=1,
        help="Number of steps for n-step Q-learning returns (default: 1).",
    )
    parser.add_argument(
        "--potential_mode", type=str, default="negative",
        choices=["negative", "inverse", "exp-negative", "inverse-sqrt"],
        help="Transformation applied to normalised hitting times to produce Φ(s). "
             "'negative': Φ=−h (default), 'inverse': Φ=1/(h/τ+δ), "
             "'exp-negative': Φ=exp(−h/τ), 'inverse-sqrt': Φ=1/(√(h/τ)+δ).",
    )
    parser.add_argument(
        "--potential_temp", type=float, default=1.0,
        help="Temperature τ used by 'inverse', 'exp-negative', and 'inverse-sqrt' "
             "potential modes (default: 1.0).",
    )
    parser.add_argument(
        "--potential_delta", type=float, default=1.0,
        help="Denominator offset δ for 'inverse' and 'inverse-sqrt' modes. "
             "Caps the maximum potential at 1/δ; default 1.0 gives max=1.0.",
    )
    args = parser.parse_args()

    # Resolve legacy aliases
    if args.method == "gt":
        args.method = "gt_truncated"
    if args.method == "allo":
        args.method = "allo_hitting_time"

    # gt_* methods imply --use_gt
    if args.method in ("gt_truncated", "gt_full"):
        args.use_gt = True

    # --goal_state forces a single deterministic trajectory; override counts.
    if args.goal_state is not None:
        if args.num_seeds != 1:
            print(
                f"  NOTE: --goal_state provided; overriding --num_seeds "
                f"from {args.num_seeds} → 1.",
                file=sys.stderr,
            )
            args.num_seeds = 1
        args.seed_idx = 0

    if args.seed_idx >= args.num_seeds:
        parser.error(
            f"--seed_idx {args.seed_idx} is out of range for --num_seeds {args.num_seeds}."
        )
    _allo_methods = ("allo_hitting_time", "allo_squared_diff", "allo_weighted_squared_diff")
    if args.method in _allo_methods and args.allo_model_dir is None:
        parser.error(f"--method={args.method} requires --allo_model_dir.")

    model_dir = Path(args.model_dir) if args.model_dir else None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif model_dir is not None:
        output_dir = model_dir / "reward_shaping"
    else:
        output_dir = Path("reward_shaping")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load pre-trained model (skip if not needed for this method)
    # ------------------------------------------------------------------
    model_data = None
    evtype     = None

    _allo_methods = ("allo_hitting_time", "allo_squared_diff", "allo_weighted_squared_diff")

    if (model_dir is not None) and (args.method in ("complex", "gt_truncated", "gt_full")):
        if args.use_gt and args.method == "gt_truncated":
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
                model_data = load_model(
                    model_dir, use_gt=True,
                    checkpoint_prefix=args.checkpoint_prefix,
                )
                evtype = model_data["eigenvalue_type"]
                print(f"  Eigenvector source : gt truncated (saved during training)")
                print(f"  Eigenvalue type    : {evtype}")
                print(f"  Eigenvectors (K)   : {model_data['eigenvalues_real'].shape[0]}")
            else:
                print(
                    f"WARNING: gt_* eigenvector files not found in {model_dir}; "
                    "will compute GT from environment.",
                    file=sys.stderr,
                )
                # model_data stays None → GT fallback in section 2c

        elif args.use_gt and args.method == "gt_full":
            full_files = [
                model_dir / "full_left_real.npy",
                model_dir / "full_left_imag.npy",
                model_dir / "full_right_real.npy",
                model_dir / "full_right_imag.npy",
                model_dir / "full_eigenvalues_real.npy",
                model_dir / "full_eigenvalues_imag.npy",
            ]
            precomp_full_ht = model_dir / "full_ideal_gt_hitting_times.npy"
            if precomp_full_ht.exists():
                print(f"\n{'='*60}")
                print("Found pre-computed full GT hitting times — will load in section 3.")
                print(f"{'='*60}")
                # model_data left as None; handled via sentinel in section 3
            elif all(f.exists() for f in full_files):
                print(f"\n{'='*60}")
                print(f"Loading full GT eigenvectors from: {model_dir}")
                print(f"{'='*60}")
                model_data = load_model(
                    model_dir, use_gt=True, gt_prefix="full_",
                    checkpoint_prefix=args.checkpoint_prefix,
                )
                evtype = model_data["eigenvalue_type"]
                print(f"  Eigenvector source : gt full (saved during training)")
                print(f"  Eigenvalue type    : {evtype}")
                print(f"  Eigenvectors (K)   : {model_data['eigenvalues_real'].shape[0]}")
            else:
                print(
                    "WARNING: full_ideal_gt_hitting_times.npy and full_* eigenvector "
                    f"files not found in {model_dir}. Skipping gt_full condition.",
                    file=sys.stderr,
                )
                sys.exit(0)   # graceful skip — not an error

        else:  # "complex"
            print(f"\n{'='*60}")
            print(f"Loading complex (learned) representation from: {model_dir}")
            print(f"{'='*60}")
            model_data = load_model(
                model_dir, use_gt=False,
                checkpoint_prefix=args.checkpoint_prefix,
            )
            evtype = model_data["eigenvalue_type"]
            print(f"  Eigenvector source : learned")
            print(f"  Eigenvalue type    : {evtype}")
            print(f"  Eigenvectors (K)   : {model_data['eigenvalues_real'].shape[0]}")

    allo_model_data = None
    if args.method in _allo_methods:
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
    # 2c. GT fallback: compute from environment if no model_dir given
    # ------------------------------------------------------------------
    _need_gt_fallback = (
        args.use_gt
        and model_data is None
        and args.method in ("complex", "gt_truncated")
    )
    if _need_gt_fallback:
        print(f"\n{'='*60}")
        print("Computing ground-truth eigenvectors from environment ...")
        print(f"{'='*60}")
        model_data = compute_gt_model_data(
            env, canonical_states,
            gamma=args.gt_gamma, delta=args.gt_delta,
            num_eigenvectors=args.num_eigenvectors,
        )
        evtype = "laplacian"
        print(f"  Eigenvector source : ground truth (computed from environment)")
        print(f"  Eigenvalue type    : laplacian")
        print(f"  Eigenvectors (K)   : {model_data['eigenvalues_real'].shape[0]}")

    # ------------------------------------------------------------------
    # 3. Compute distance matrix for this method
    # ------------------------------------------------------------------
    def _compute_ht(md, label):
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

    def _log(mat):
        finite = mat[np.isfinite(mat) & (mat >= 0)]
        print(f"  Shape              : {mat.shape}")
        print(f"  Finite values      : {len(finite)} / {mat.size}"
              f"  ({len(finite)/mat.size:.1%})")
        if len(finite) > 0:
            print(f"  Range              : [{finite.min():.2f}, {finite.max():.2f}]")

    # The single distance matrix used by this task (shape [N, N])
    dist_matrix = None

    if args.method == "complex":
        if model_data is not None:
            model_data = truncate_model_eigenvectors(model_data, args.num_eigenvectors)
            print(f"\n{'='*60}")
            print("Computing hitting times (complex) ...")
            print(f"{'='*60}")
            dist_matrix = _compute_ht(model_data, "complex")
            _log(dist_matrix)

    elif args.method == "gt_truncated":
        if model_data is not None:
            # gt_* files were already truncated at training time — use as-is.
            model_data = truncate_model_eigenvectors(model_data, None)
            print(f"\n{'='*60}")
            print("Computing hitting times (GT truncated) ...")
            print(f"{'='*60}")
            dist_matrix = _compute_ht(model_data, "gt_truncated")
            _log(dist_matrix)

    elif args.method == "gt_full":
        precomp_full_ht = (
            model_dir / "full_ideal_gt_hitting_times.npy"
            if model_dir is not None else None
        )
        if precomp_full_ht is not None and precomp_full_ht.exists():
            print(f"\n{'='*60}")
            print("Loading pre-computed GT hitting times (gt_full) ...")
            print(f"{'='*60}")
            dist_matrix = np.load(precomp_full_ht)
            _log(dist_matrix)
        elif model_data is not None:
            model_data = truncate_model_eigenvectors(model_data, None)
            print(f"\n{'='*60}")
            print("Computing hitting times (GT full) ...")
            print(f"{'='*60}")
            dist_matrix = _compute_ht(model_data, "gt_full")
            _log(dist_matrix)
        else:
            print("WARNING: no GT full data available; skipping.", file=sys.stderr)
            sys.exit(0)

    elif args.method == "allo_hitting_time":
        allo_model_data = truncate_model_eigenvectors(allo_model_data, args.num_eigenvectors)
        print(f"\n{'='*60}")
        print("Computing hitting times (ALLO) ...")
        print(f"{'='*60}")
        dist_matrix = _compute_ht(allo_model_data, "allo_hitting_time")
        _log(dist_matrix)

    elif args.method == "allo_squared_diff":
        allo_model_data = truncate_model_eigenvectors(allo_model_data, args.num_eigenvectors)
        print(f"\n{'='*60}")
        print("Computing ALLO squared-diff distance matrix ...")
        print(f"{'='*60}")
        dist_matrix, _ = compute_allo_distance_matrices(allo_model_data)
        _log(dist_matrix)

    elif args.method == "allo_weighted_squared_diff":
        allo_model_data = truncate_model_eigenvectors(allo_model_data, args.num_eigenvectors)
        print(f"\n{'='*60}")
        print("Computing ALLO weighted-squared-diff distance matrix ...")
        print(f"{'='*60}")
        _, dist_matrix = compute_allo_distance_matrices(allo_model_data)
        _log(dist_matrix)

    # ------------------------------------------------------------------
    # 4. Sample one fixed (goal, eval_start) per seed; narrow to requested seed
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

    # Narrow to the requested seed
    si = args.seed_idx
    goal_per_seed        = goal_per_seed[si : si + 1]
    eval_starts_per_seed = eval_starts_per_seed[si : si + 1]
    if train_start_per_seed is not None:
        train_start_per_seed = train_start_per_seed[si : si + 1]

    # ------------------------------------------------------------------
    # 5. Build condition and run Q-learning
    # ------------------------------------------------------------------
    b = args.shaping_coef

    _method_to_label = {
        "baseline":                  "baseline",
        "complex":                   f"shaped β={b} (complex)",
        "gt_truncated":              f"shaped β={b} (gt truncated)",
        "gt_full":                   f"shaped β={b} (gt full)",
        "allo_hitting_time":         f"shaped β={b} (allo hitting time)",
        "allo_squared_diff":         f"shaped β={b} (allo squared diff)",
        "allo_weighted_squared_diff":f"shaped β={b} (allo weighted diff)",
    }
    cond_name = _method_to_label[args.method]

    if args.method == "baseline":
        potential_per_seed = None
        shaping_coef_use   = 0.0
    else:
        if dist_matrix is None:
            print(
                f"ERROR: No distance matrix computed for method '{args.method}'. "
                "Check that the required --model_dir / --allo_model_dir files exist.",
                file=sys.stderr,
            )
            sys.exit(1)
        pot_matrix = build_all_potentials(
            dist_matrix,
            potential_mode=args.potential_mode,
            potential_temp=args.potential_temp,
            potential_delta=args.potential_delta,
            gamma=args.gamma_rl,
        )
        potential_per_seed = pot_matrix[:, goal_per_seed].T
        shaping_coef_use   = b

    cond_kwargs = dict(potential_per_seed=potential_per_seed, shaping_coef=shaping_coef_use)

    print(f"\n{'='*60}")
    print(f"Q-learning: {cond_name}  (seed {si})")
    print(f"{'='*60}")

    eval_sr, eval_len_all, eval_len_suc, eval_steps, _Q_final = run_q_learning(
        env                  = env,
        canonical_states     = canonical_states,
        goal_per_seed        = goal_per_seed,
        eval_starts_per_seed = eval_starts_per_seed,
        train_start_per_seed = train_start_per_seed,
        min_goal_distance    = args.min_goal_distance,
        total_steps          = args.total_steps,
        num_seeds            = 1,
        max_steps_per_episode= args.max_steps,
        num_eval_episodes    = args.num_eval_episodes,
        gamma                = args.gamma_rl,
        lr                   = args.lr,
        epsilon              = args.epsilon,
        seed                 = args.seed_idx,
        eval_interval        = args.eval_interval,
        n_step_td            = args.n_step_td,
        **cond_kwargs,
    )

    final_sr      = float(eval_sr[-1, 0])
    final_len_all = float(eval_len_all[-1, 0])
    final_len_suc = eval_len_suc[-1, 0]
    suc_disp = f"{final_len_suc:.1f}" if not np.isnan(final_len_suc) else "—"
    print(f"  seed {si}:  eval_sr={final_sr:.2f}"
          f"  eval_len_all={final_len_all:.0f}"
          f"  eval_len_suc={suc_disp}")

    results = {
        cond_name: dict(
            eval_sr      = eval_sr,
            eval_len_all = eval_len_all,
            eval_len_suc = eval_len_suc,
            eval_steps   = eval_steps,
        )
    }

    # ------------------------------------------------------------------
    # 6. Save partial result
    # ------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
