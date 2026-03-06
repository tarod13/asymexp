#!/bin/bash
# Run the reward-shaping experiment for a given environment.
#
# Usage:
#   bash scripts/run_reward_shaping.sh <env_name> [extra args]
#
# Examples:
#   # Baseline only (no model dir needed)
#   bash scripts/run_reward_shaping.sh GridRoom-4-Doors
#
#   # With a complex representation model
#   bash scripts/run_reward_shaping.sh GridRoom-4-Doors \
#       --model_dir ./results/file/my_run
#
#   # With both models and a custom shaping coefficient
#   bash scripts/run_reward_shaping.sh GridRoom-4-Doors \
#       --model_dir     ./results/file/my_run \
#       --allo_model_dir ./results/file/my_allo_run \
#       --shaping_coef 0.05
#
# Arguments after <env_name> are passed directly to run_reward_shaping.py.

set -euo pipefail

ENV_NAME="${1:?Error: provide an environment name as the first argument.
Usage: $0 <env_name> [--model_dir DIR] [--shaping_coef 0.1] ...}"
shift

# Move to repo root so relative imports work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python experiments/reward_shaping/run_reward_shaping.py \
    --env          "$ENV_NAME" \
    --num_episodes 3000 \
    --num_seeds    5 \
    --shaping_coef 0.1 \
    --max_steps    500 \
    --gamma_rl     0.99 \
    --lr           0.1 \
    --epsilon      0.1 \
    "$@"
