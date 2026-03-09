#!/bin/bash
# =============================================================================
# Parallel representation training across all environments (no importance sampling)
#
# Launches one job per file-based environment (6 total). Results are saved to:
#   ./results/parallel_eval/{env_file_name}/{env_file_name}__parallel_eval__0__42__{timestamp}/
#
# After each job completes the resolved directory is written to:
#   ./results/eval_manifest/{ENV_NAME}.txt
#
# Usage:
#   sbatch scripts/eval_parallel_train.sh
#
# To submit with the companion plotting job automatically, use:
#   bash scripts/submit_parallel_eval.sh
# =============================================================================

#SBATCH --job-name=par_eval_train
#SBATCH --account=rrg-machado
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=logs/par_eval_train_%a.out
#SBATCH --error=logs/par_eval_train_%a.err

mkdir -p logs
mkdir -p results/eval_manifest

# ---------------------------------------------------------------------------
# Environment table  (index 0-5)
# ---------------------------------------------------------------------------
ENV_FILES=(
    "GridRoom-4-Doors"  "GridRoom-4"  "GridRoom-4-DoorsIm"  "GridRoom-1"
    "GridRoom-1-Portals-1"  "GridRoom-1-Doors"
)

ENV_FILE=${ENV_FILES[$SLURM_ARRAY_TASK_ID]}
ENV_NAME=$ENV_FILE

echo "========================================"
echo "Parallel Eval Training — No Importance Sampling"
echo "Job ID      : $SLURM_JOB_ID"
echo "Array Task  : $SLURM_ARRAY_TASK_ID"
echo "Environment : $ENV_NAME"
echo "Node        : $SLURM_NODELIST"
echo "========================================"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 scipy-stack/2024b
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------------------------------------------------------------------------
# Run training (CLF, no importance sampling)
# ---------------------------------------------------------------------------
python train.py clf \
    --env_file_name        "$ENV_FILE" \
    --num_gradient_steps   100000 \
    --batch_size           256 \
    --num_eigenvector_pairs 8 \
    --learning_rate        0.00001 \
    --ema_learning_rate    0.0003 \
    --lambda_x             10.0 \
    --chirality_factor     0.1 \
    --gamma                0.9 \
    --sampling_mode        none \
    --constraint_mode      single_batch \
    --use_residual \
    --use_layernorm \
    --num_envs             1000 \
    --num_steps            1000 \
    --exp_name             parallel_eval \
    --exp_number           0 \
    --seed                 42 \
    --results_dir          ./results/parallel_eval \
    --no-plot_during_training \
    --save_model

# ---------------------------------------------------------------------------
# Find the results directory and write to manifest
# The directory pattern is:
#   ./results/parallel_eval/{env_file_name}/{env_file_name}__parallel_eval__0__42__*/
# We take the most recent one in case of reruns.
# ---------------------------------------------------------------------------
RESULTS_DIR=$(ls -td "./results/parallel_eval/${ENV_FILE}/${ENV_FILE}__parallel_eval__0__42__"*/ 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: Could not find results directory for $ENV_NAME" >&2
    exit 1
fi

# Strip trailing slash for consistency
RESULTS_DIR="${RESULTS_DIR%/}"
echo "$RESULTS_DIR" > "./results/eval_manifest/${ENV_NAME}.txt"

echo "========================================"
echo "Training complete for: $ENV_NAME"
echo "Results saved to     : $RESULTS_DIR"
echo "Manifest written to  : ./results/eval_manifest/${ENV_NAME}.txt"
echo "========================================"
