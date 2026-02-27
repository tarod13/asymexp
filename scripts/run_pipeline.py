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
import subprocess
import sys
from pathlib import Path


def latest_dir(base: Path, fragment: str) -> Path | None:
    """Return the most recently modified subdirectory whose name contains fragment."""
    matches = [d for d in base.iterdir() if d.is_dir() and fragment in d.name]
    if not matches:
        return None
    return max(matches, key=lambda d: d.stat().st_mtime)


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training + reward-shaping pipeline.")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--steps",         type=int,   default=100_000)
    parser.add_argument("--results_dir",   type=str,   default="./results")
    parser.add_argument("--env_file_name", type=str,   default="GridRoom-4-Doors")
    parser.add_argument("--shaping_coef",  type=float, default=0.1)
    parser.add_argument("--num_episodes",  type=int,   default=3000)
    parser.add_argument("--num_seeds",     type=int,   default=5)
    parser.add_argument("--skip_allo",     action="store_true")
    parser.add_argument("--skip_complex",  action="store_true")
    parser.add_argument("--allo_dir",      type=str,   default="")
    parser.add_argument("--complex_dir",   type=str,   default="")
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
        print(f"  allo_dir      : {args.allo_dir}")
    if args.complex_dir:
        print(f"  complex_dir   : {args.complex_dir}")
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
            "--constraint_mode",       "same_episodes",
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

    # ── Step 3: Reward shaping experiment ─────────────────────────────────────
    print()
    print("=" * 60)
    print(" Step 3: Reward shaping experiment")
    print(f"   model_dir (complex) : {complex_dir}")
    print(f"   allo_model_dir      : {allo_dir}")
    print("=" * 60)

    output_dir = complex_dir / "reward_shaping"

    run([
        py, str(repo_root / "experiments" / "reward_shaping" / "run_reward_shaping.py"),
        "--model_dir",      str(complex_dir),
        "--allo_model_dir", str(allo_dir),
        "--num_episodes",   str(args.num_episodes),
        "--num_seeds",      str(args.num_seeds),
        "--shaping_coef",   str(args.shaping_coef),
        "--max_steps",      "500",
        "--gamma_rl",       "0.99",
        "--lr",             "0.1",
        "--epsilon",        "0.1",
        "--output_dir",     str(output_dir),
    ])

    print()
    print("=" * 60)
    print(" Pipeline complete!")
    print(f"  Results  : {output_dir}")
    print(f"  Plot     : {output_dir / 'learning_curves.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
