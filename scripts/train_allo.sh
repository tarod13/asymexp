#!/bin/bash
# Train the ALLO representation on GridRoom-4-Doors.
#
# ALLO uses an augmented-Lagrangian loss that is unweighted by design, so
# rejection sampling is used to obtain a near-uniform state distribution.
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
    --env_type              file \
    --env_file_name         GridRoom-4-Doors \
    --num_gradient_steps    100000 \
    --batch_size            256 \
    --num_eigenvector_pairs 8 \
    --learning_rate         0.0003 \
    --gamma                 0.9 \
    --use_rejection_sampling \
    --num_envs              1000 \
    --num_steps             1000 \
    --step_size_duals       1.0 \
    --duals_initial_val     -2.0 \
    --barrier_initial_val   0.5 \
    --max_barrier_coefs     0.5 \
    --exp_name              allo \
    "$@"
