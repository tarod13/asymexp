#!/bin/bash
# Run the reward-shaping experiment with a pre-trained Laplacian model.
#
# Usage:
#   bash scripts/run_reward_shaping.sh <model_dir> [extra args]
#
# Examples:
#   # Quickstart with defaults
#   bash scripts/run_reward_shaping.sh ./results/file/my_run
#
#   # Choose a specific goal state and shaping coefficient
#   bash scripts/run_reward_shaping.sh ./results/file/my_run \
#       --goal_state 42 --shaping_coef 0.05
#
#   # Compare against ground-truth eigenvectors
#   bash scripts/run_reward_shaping.sh ./results/file/my_run \
#       --use_gt --shaping_coef 0.1
#
#   # Full sweep of shaping coefficients (run sequentially)
#   for beta in 0.01 0.05 0.1 0.5 1.0; do
#       bash scripts/run_reward_shaping.sh ./results/file/my_run \
#           --shaping_coef $beta \
#           --output_dir ./results/file/my_run/reward_shaping/beta_$beta
#   done
#
# Arguments after <model_dir> are passed directly to run_reward_shaping.py.

set -euo pipefail

MODEL_DIR="${1:?Error: provide a model_dir as first argument.
Usage: $0 <model_dir> [--goal_state N] [--shaping_coef 0.1] [--use_gt] ...}"
shift

# Move to repo root so relative imports work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python experiments/reward_shaping/run_reward_shaping.py \
    --model_dir  "$MODEL_DIR" \
    --num_episodes 3000 \
    --num_seeds    5 \
    --shaping_coef 0.1 \
    --max_steps    500 \
    --gamma_rl     0.99 \
    --lr           0.1 \
    --epsilon      0.1 \
    "$@"
