"""
Aggregate partial Q-learning results produced by distributed single-seed runs
(via run_reward_shaping.py --method X --seed_idx Y) and generate the combined
learning-curve figure that looks identical to the one produced by the
non-distributed pipeline.

Usage
-----
    python experiments/reward_shaping/analyze_reward_shaping.py \\
        --output_dir ./results/file/<run>/reward_shaping \\
        [--window 100] [--max_steps 500]

The script reads all  output_dir/partial/results_<method>_<seed>.pkl  files,
concatenates the seed axis for each condition, and calls the same plot_results()
function used by run_reward_shaping.py.  The combined results are saved to
output_dir/results.pkl and output_dir/learning_curves.png.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.reward_shaping.run_reward_shaping import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate distributed reward-shaping Q-learning results."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory that contains the 'partial/' sub-folder written by "
             "run_reward_shaping.py --method X --seed_idx Y.",
    )
    parser.add_argument(
        "--window", type=int, default=100,
        help="Smoothing window (episodes) for training curves (default: 100).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Max steps per episode shown on the plot y-axis (default: 500).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    partial_dir = output_dir / "partial"

    if not partial_dir.is_dir():
        print(f"ERROR: partial results directory not found: {partial_dir}", file=sys.stderr)
        sys.exit(1)

    partial_files = sorted(partial_dir.glob("results_*.pkl"))
    if not partial_files:
        print(f"ERROR: No partial result files found in {partial_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(partial_files)} partial result file(s) in {partial_dir}")

    # ── Load and group by condition name ────────────────────────────────────
    # by_cond[cond_name][seed_idx] = {steps, reached, eval_sr, eval_episodes}
    by_cond: dict[str, dict[int, dict]] = {}
    sample_args = None

    for fpath in partial_files:
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        cond_name = d["cond_name"]
        seed_idx  = d["seed_idx"]
        if sample_args is None:
            sample_args = d["args"]

        if cond_name not in by_cond:
            by_cond[cond_name] = {}
        if seed_idx in by_cond[cond_name]:
            print(
                f"WARNING: duplicate seed_idx={seed_idx} for condition '{cond_name}'. "
                f"Keeping the one from {fpath.name}.",
                file=sys.stderr,
            )
        by_cond[cond_name][seed_idx] = d["results"][cond_name]

    # ── Aggregate ────────────────────────────────────────────────────────────
    results: dict = {}
    for cond_name, seed_dict in by_cond.items():
        sorted_idx = sorted(seed_dict.keys())
        print(f"  '{cond_name}': {len(sorted_idx)} seed(s)  [idx {sorted_idx}]")

        # Each entry has shape [1, num_episodes] or [num_chunks, 1].
        steps_list   = [seed_dict[si]["steps"]    for si in sorted_idx]
        reached_list = [seed_dict[si]["reached"]  for si in sorted_idx]
        eval_sr_list = [seed_dict[si]["eval_sr"]  for si in sorted_idx]
        eval_ep      = seed_dict[sorted_idx[0]]["eval_episodes"]

        # Sanity-check that eval_episodes arrays are consistent across seeds.
        for si in sorted_idx[1:]:
            if not np.array_equal(seed_dict[si]["eval_episodes"], eval_ep):
                print(
                    f"WARNING: eval_episodes mismatch at seed_idx={si} for '{cond_name}'. "
                    "Using the first seed's eval_episodes.",
                    file=sys.stderr,
                )

        results[cond_name] = dict(
            steps         = np.concatenate(steps_list,   axis=0),  # [num_seeds, num_ep]
            reached       = np.concatenate(reached_list, axis=0),  # [num_seeds, num_ep]
            eval_sr       = np.concatenate(eval_sr_list, axis=1),  # [num_chunks, num_seeds]
            eval_episodes = eval_ep,                                # [num_chunks]
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    num_episodes = next(iter(results.values()))["steps"].shape[1]
    window = min(args.window, max(1, num_episodes // 10))

    plot_results(
        results,
        output_dir / "learning_curves.png",
        window    = window,
        max_steps = args.max_steps,
    )

    # ── Save combined results ─────────────────────────────────────────────────
    out_pkl = output_dir / "results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(results=results, args=sample_args), f)
    print(f"Combined results saved → {out_pkl}")
    print(f"\nDone.  All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
