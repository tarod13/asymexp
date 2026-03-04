"""Full training + reward-shaping pipeline.

Steps
-----
1. Train the ALLO representation          (train_allo_rep.py)
2. Train the complex representation        (train_lap_rep.py)
3. Run reward-shaping Q-learning comparing
     baseline | shaped-ALLO | shaped-complex
   and generate a single visualisation.

Usage
-----
  python scripts/run_pipeline.py [options]

Options (all optional, with defaults shown):
  --seed           SEED        Random seed for both training runs  [42]
  --steps          N           Gradient steps for each training run [100000]
  --results_dir    DIR         Where to save training outputs       [./results]
  --env_file_name  ENV         Environment name                     [GridRoom-4-Doors]
  --shaping_coef   BETA        Reward-shaping coefficient           [0.1]
  --num_episodes   N           Q-learning episodes per seed         [3000]
  --num_seeds      N           Number of Q-learning seeds           [5]
  --skip_allo                  Skip allo training (reuse existing)
  --skip_complex               Skip complex training (reuse existing)
  --allo_dir       DIR         Pre-existing allo model dir (sets --skip_allo)
  --complex_dir    DIR         Pre-existing complex model dir (sets --skip_complex)

Examples
--------
  # Full pipeline from scratch
  python scripts/run_pipeline.py

  # Quick smoke-test with fewer gradient steps and episodes
  python scripts/run_pipeline.py --steps 5000 --num_episodes 500 --num_seeds 2

  # Reuse pre-trained models, just re-run reward shaping
  python scripts/run_pipeline.py \\
      --allo_dir    ./results/file/my_allo_run \\
      --complex_dir ./results/file/my_complex_run
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def latest_dir(base: Path, fragment: str) -> Path | None:
    """Return the most recently modified subdirectory whose name contains fragment."""
    matches = [d for d in base.iterdir() if d.is_dir() and fragment in d.name]
    if not matches:
        return None
    return max(matches, key=lambda d: d.stat().st_mtime)


def run(cmd: list[str], env: dict | None = None) -> None:
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def sbatch(cmd: list[str], env: dict) -> str:
    """Submit a SLURM job and return the job ID."""
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip().split()[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training + reward-shaping pipeline.")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--steps",         type=int,   default=100_000)
    parser.add_argument("--results_dir",   type=str,   default="./results")
    parser.add_argument("--env_file_name", type=str,   default="GridRoom-4-Doors")
    parser.add_argument("--shaping_coef",  type=float, default=0.1)
    parser.add_argument("--num_episodes",  type=int,   default=30000)
    parser.add_argument("--max_steps",     type=int,   default=500,
                        help="Max steps per episode in Q-learning (default: 500).")
    parser.add_argument("--num_seeds",     type=int,   default=5)
    parser.add_argument("--skip_allo",          action="store_true")
    parser.add_argument("--skip_complex",       action="store_true")
    parser.add_argument("--allo_dir",           type=str,   default="")
    parser.add_argument("--complex_dir",        type=str,   default="")
    parser.add_argument(
        "--start_state", type=str, default="",
        help="Optional fixed starting state as 'x,y'. Passed to run_reward_shaping.py.",
    )
    parser.add_argument(
        "--min_goal_distance", type=int, default=0,
        help="Minimum taxi distance from start to goal. Passed to run_reward_shaping.py.",
    )
    args = parser.parse_args()

    if args.allo_dir:
        args.skip_allo = True
    if args.complex_dir:
        args.skip_complex = True

    repo_root = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print(" Pipeline configuration")
    print("=" * 60)
    print(f"  seed          : {args.seed}")
    print(f"  steps         : {args.steps}")
    print(f"  results_dir   : {args.results_dir}")
    print(f"  env           : {args.env_file_name}")
    print(f"  shaping_coef  : {args.shaping_coef}")
    print(f"  num_episodes  : {args.num_episodes}")
    print(f"  num_seeds     : {args.num_seeds}")
    print(f"  skip_allo     : {args.skip_allo}")
    print(f"  skip_complex  : {args.skip_complex}")
    if args.allo_dir:
        print(f"  allo_dir          : {args.allo_dir}")
    if args.complex_dir:
        print(f"  complex_dir       : {args.complex_dir}")
    if args.start_state:
        print(f"  start_state       : {args.start_state}")
    if args.min_goal_distance > 0:
        print(f"  min_goal_distance : {args.min_goal_distance}")
    print("=" * 60)

    py = sys.executable
    results_file_dir = Path(args.results_dir) / "file"

    # ── Step 1: Train ALLO representation ────────────────────────────────────
    print()
    print("=" * 60)
    print(" Step 1: ALLO representation training")
    print("=" * 60)

    if args.skip_allo:
        if args.allo_dir:
            allo_dir = Path(args.allo_dir)
        else:
            allo_dir = latest_dir(results_file_dir, "__allo__")
            if allo_dir is None:
                print(f"ERROR: --skip_allo set but no allo model found under {results_file_dir}/", file=sys.stderr)
                sys.exit(1)
        print(f"  Skipping training.  Using: {allo_dir}")
    else:
        run([
            py, str(repo_root / "train_allo_rep.py"),
            "--env_type",              "file",
            "--env_file_name",         args.env_file_name,
            "--num_gradient_steps",    str(args.steps),
            "--batch_size",            "256",
            "--num_eigenvector_pairs", "8",
            "--learning_rate",         "0.0003",
            "--gamma",                 "0.9",
            "--use_rejection_sampling",
            "--num_envs",              "1000",
            "--num_steps",             "1000",
            "--step_size_duals",       "1.0",
            "--duals_initial_val",     "-2.0",
            "--barrier_initial_val",   "0.5",
            "--max_barrier_coefs",     "0.5",
            "--seed",                  str(args.seed),
            "--results_dir",           args.results_dir,
            "--exp_name",              "allo",
        ])
        allo_dir = latest_dir(results_file_dir, "__allo__")
        if allo_dir is None:
            print(f"ERROR: Could not locate allo training output under {results_file_dir}/", file=sys.stderr)
            sys.exit(1)
        print(f"  ALLO training complete.  Output: {allo_dir}")

    # ── Step 2: Train complex representation ──────────────────────────────────
    print()
    print("=" * 60)
    print(" Step 2: Complex representation training")
    print("=" * 60)

    if args.skip_complex:
        if args.complex_dir:
            complex_dir = Path(args.complex_dir)
        else:
            complex_dir = latest_dir(results_file_dir, "__complex__")
            if complex_dir is None:
                print(f"ERROR: --skip_complex set but no complex model found under {results_file_dir}/", file=sys.stderr)
                sys.exit(1)
        print(f"  Skipping training.  Using: {complex_dir}")
    else:
        run([
            py, str(repo_root / "train_lap_rep.py"),
            "--env_type",              "file",
            "--env_file_name",         args.env_file_name,
            "--num_gradient_steps",    str(args.steps),
            "--batch_size",            "256",
            "--num_eigenvector_pairs", "8",
            "--learning_rate",         "0.00001",
            "--ema_learning_rate",     "0.0003",
            "--lambda_x",              "10.0",
            "--chirality_factor",      "0.1",
            "--gamma",                 "0.9",
            "--no-use_rejection_sampling",
            "--constraint_mode",       "single_batch",
            "--use_residual",
            "--use_layernorm",
            "--num_envs",              "1000",
            "--num_steps",             "1000",
            "--seed",                  str(args.seed),
            "--results_dir",           args.results_dir,
            "--exp_name",              "complex",
        ])
        complex_dir = latest_dir(results_file_dir, "__complex__")
        if complex_dir is None:
            print(f"ERROR: Could not locate complex training output under {results_file_dir}/", file=sys.stderr)
            sys.exit(1)
        print(f"  Complex training complete.  Output: {complex_dir}")

    # ── Step 3: Submit distributed reward-shaping Q-learning ──────────────────
    # Each (method, seed) pair is a separate SLURM array task so that no single
    # job needs to hold all trajectories in memory at once.  A dependent
    # analysis job aggregates the partial results and produces the same plot.
    print()
    print("=" * 60)
    print(" Step 3: Submit distributed reward-shaping Q-learning")
    print(f"   complex model : {complex_dir}")
    print(f"   allo model    : {allo_dir}")
    print("=" * 60)

    output_dir     = complex_dir / "reward_shaping"
    array_script   = repo_root / "scripts" / "run_reward_shaping_array.sh"
    analyze_script = repo_root / "scripts" / "analyze_reward_shaping.sh"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "partial").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    num_methods  = 3 if allo_dir else 2
    num_tasks    = num_methods * args.num_seeds
    max_task_id  = num_tasks - 1

    # Build the environment inherited by the array and analysis jobs.
    env = os.environ.copy()
    env.update({
        "MODEL_DIR":         str(complex_dir),
        "ALLO_MODEL_DIR":    str(allo_dir),
        "OUTPUT_DIR":        str(output_dir),
        "NUM_SEEDS":         str(args.num_seeds),
        "NUM_METHODS":       str(num_methods),
        "NUM_EPISODES":      str(args.num_episodes),
        "MAX_STEPS":         str(args.max_steps),
        "SHAPING_COEF":      str(args.shaping_coef),
        "GAMMA_RL":          "0.99",
        "LR":                "0.1",
        "EPSILON":           "0.1",
        "LOG_INTERVAL":      "500",
        "EVAL_SEED":         "0",
        "NUM_EVAL_EPISODES": "30",
        "MIN_GOAL_DISTANCE": str(args.min_goal_distance),
        "START_STATE":       args.start_state or "",
    })

    if shutil.which("sbatch"):
        # SLURM: submit array job then analysis job with dependency.
        array_job_id = sbatch(
            ["sbatch", f"--array=0-{max_task_id}", "--export=ALL", str(array_script)],
            env=env,
        )
        print(f"  Q-learning array job : {array_job_id}  (tasks 0-{max_task_id})")

        analyze_job_id = sbatch(
            ["sbatch", f"--dependency=afterok:{array_job_id}", "--export=ALL",
             str(analyze_script)],
            env=env,
        )
        print(f"  Analysis job         : {analyze_job_id}  (runs after {array_job_id})")
        print()
        print(f"  Monitor: squeue -j {array_job_id},{analyze_job_id}")
    else:
        # Local fallback: run tasks sequentially then analysis.
        print("  SLURM not available — running locally (sequential)...")
        for task_id in range(num_tasks):
            print(f"\n  --- Task {task_id}/{max_task_id} ---")
            run(["bash", str(array_script), str(task_id)], env=env)
        print("\n  Running analysis...")
        run(["bash", str(analyze_script)], env=env)

    print()
    print("=" * 60)
    print(" Pipeline complete!")
    print("  Q-learning jobs submitted (or finished, if running locally).")
    print(f"  Results will appear in : {output_dir}")
    print(f"  Final plot             : {output_dir / 'learning_curves.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
