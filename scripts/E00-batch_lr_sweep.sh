#!/bin/bash
#SBATCH --job-name=batch_lr_sweep
#SBATCH --account=aip-machado
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-999
#SBATCH --output=logs/E00-batch_lr_sweep_%a.out
#SBATCH --error=logs/E00-batch_lr_sweep_%a.err

# =============================================================================
# Batch size × Learning rate sweep
#
# 10 batch sizes × 10 learning rates × 10 seeds = 1000 jobs
#
# Index encoding:
#   SLURM_ARRAY_TASK_ID = seed_idx + 10 * lr_idx + 100 * bs_idx
#
# Usage:
#   sbatch scripts/E00-batch_lr_sweep.sh                    # on SLURM
#   bash scripts/E00-batch_lr_sweep.sh <task_id>            # local (single run)
# =============================================================================

mkdir -p logs

JOB_ID="${1:-$SLURM_ARRAY_TASK_ID}"

# Sweep grid
BATCH_SIZES=(16 32 64 128 256 512 1024 2048 4096 8192)
LEARNING_RATES=(3e-3 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 1e-7)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Decode job index → (bs_idx, lr_idx, seed_idx)
NUM_SEEDS=${#SEEDS[@]}        # 10
NUM_LRS=${#LEARNING_RATES[@]} # 10

seed_idx=$(( JOB_ID % NUM_SEEDS ))
lr_idx=$(( (JOB_ID / NUM_SEEDS) % NUM_LRS ))
bs_idx=$(( JOB_ID / (NUM_SEEDS * NUM_LRS) ))

BATCH_SIZE=${BATCH_SIZES[$bs_idx]}
LR=${LEARNING_RATES[$lr_idx]}
SEED=${SEEDS[$seed_idx]}

echo "========================================"
echo "Batch size × LR sweep"
echo "Job ID:       $JOB_ID"
echo "Batch size:   $BATCH_SIZE  (index $bs_idx)"
echo "Learning rate: $LR  (index $lr_idx)"
echo "Seed:          $SEED  (index $seed_idx)"
echo "Node:          ${SLURM_NODELIST:-local}"
echo "========================================"

module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 mujoco/3.3.0
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_lap_rep.py \
    --env_type file \
    --env_file_name GridRoom-4-Doors \
    --no-use_doors \
    --num_gradient_steps 200000 \
    --batch_size "$BATCH_SIZE" \
    --num_eigenvector_pairs 8 \
    --learning_rate "$LR" \
    --ema_learning_rate 0.03 \
    --lambda_x 10.0 \
    --chirality_factor 0.1 \
    --gamma 0.9 \
    --no-use_rejection_sampling \
    --constraint_mode single_batch \
    --seed "$SEED" \
    --exp_name E00-batch_lr_sweep \
    --results_dir ./results/sweeps

echo "========================================"
echo "Done: bs=$BATCH_SIZE lr=$LR seed=$SEED"
echo "========================================"
