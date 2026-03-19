"""
Aggregate partial Q-learning results produced by distributed single-seed runs
(via run_reward_shaping.py --method X --seed_idx Y) and generate the combined
learning-curve figure.

Usage
-----
    python experiments/reward_shaping/analyze_reward_shaping.py \\
        --output_dir ./results/file/<run>/reward_shaping

The script reads all  output_dir/partial/results_<method>_<seed>.pkl  files,
stacks the seed axis for each condition, and calls plot_results().
The combined results are saved to output_dir/results.pkl and
output_dir/learning_curves.png.
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
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
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
    # by_cond[cond_name][seed_idx] = {eval_sr, eval_len_all, eval_len_suc, eval_steps}
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

        # Each partial has shape [num_chunks, 1]; stack along seed axis → [num_chunks, num_seeds]
        results[cond_name] = dict(
            eval_sr      = np.concatenate(
                [seed_dict[si]["eval_sr"]      for si in sorted_idx], axis=1),
            eval_len_all = np.concatenate(
                [seed_dict[si]["eval_len_all"] for si in sorted_idx], axis=1),
            eval_len_suc = np.concatenate(
                [seed_dict[si]["eval_len_suc"] for si in sorted_idx], axis=1),
            eval_steps   = seed_dict[sorted_idx[0]]["eval_steps"],
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_results(results, output_dir / "learning_curves.png")

    # ── Save combined results ─────────────────────────────────────────────────
    out_pkl = output_dir / "results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(results=results, args=sample_args), f)
    print(f"Combined results saved → {out_pkl}")
    print(f"\nDone.  All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
