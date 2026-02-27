#!/bin/bash
# Train the ALLO representation on GridRoom-4-Doors.
#
# Usage:
#   bash scripts/train_allo.sh [extra args]
#
# Examples:
#   bash scripts/train_allo.sh
#   bash scripts/train_allo.sh --seed 1 --num_gradient_steps 50000
#
# All extra arguments are forwarded directly to train_allo_rep.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python train_allo_rep.py \
    --env_type             file \
    --env_file_name        GridRoom-4-Doors \
    --num_gradient_steps   100000 \
    --batch_size           256 \
    --num_eigenvector_pairs 8 \
    --learning_rate        0.00001 \
    --ema_learning_rate    0.0003 \
    --lambda_x             10.0 \
    --chirality_factor     0.1 \
    --gamma                0.9 \
    --no-use_rejection_sampling \
    --constraint_mode      same_episodes \
    --use_residual \
    --use_layernorm \
    --num_envs             1000 \
    --num_steps            1000 \
    --exp_name             allo \
    "$@"
