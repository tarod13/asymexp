#!/bin/bash
#SBATCH --job-name=wind_sweep
#SBATCH --account=aip-machado
#SBATCH --array=0-30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=logs/wind_sweep_%A_%a.out
#SBATCH --error=logs/wind_sweep_%A_%a.err

# =============================================================================
# Wind-parameter sweep: train ALLO for 31 equally-spaced wind values in
# [-0.99, 0.99].  Each array task handles one value.
#
# Results land in:
#   <results_dir>/task_<id>/file/<run_name>/
#
# Usage
# -----
#   sbatch scripts/sweep_wind_allo.sh [options]        # on SLURM
#   bash   scripts/sweep_wind_allo.sh <task_id>        # local (single run)
#
# Options (all optional)
# -----------------------
#   --env_file_name ENV    Environment text file name   [GridRoom-1]
#   --steps         N      Gradient steps per run       [100000]
#   --seed          S      Random seed                  [42]
#   --results_dir   DIR    Top-level output directory   [./results/wind_sweep]
#   --num_eigvecs   N      Eigenvector pairs to learn   [8]
# =============================================================================

mkdir -p logs

# Support local execution: bash sweep_wind_allo.sh <task_id>
JOB_ID="${1:-$SLURM_ARRAY_TASK_ID}"

# ── Environment setup ────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 mujoco/3.3.0
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-8}"

# ── Defaults ─────────────────────────────────────────────────────────────────
ENV_FILE_NAME="GridRoom-1"
STEPS=100000
SEED=42
RESULTS_DIR="./results/wind_sweep"
NUM_EIGVECS=8

# ── Parse options ────────────────────────────────────────────────────────────
# Shift past the optional positional task_id before parsing flags
[[ "${1:-}" =~ ^[0-9]+$ ]] && shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env_file_name) ENV_FILE_NAME="$2"; shift 2 ;;
        --steps)         STEPS="$2";         shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --results_dir)   RESULTS_DIR="$2";   shift 2 ;;
        --num_eigvecs)   NUM_EIGVECS="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Compute wind value for this task ─────────────────────────────────────────
# 31 values: np.linspace(-0.99, 0.99, 31)
WIND=$(python3 -c "import numpy as np; print(f'{np.linspace(-0.99, 0.99, 31)[$JOB_ID]:.6f}')")

echo "========================================"
echo "Wind sweep"
echo "  Array job:    ${SLURM_ARRAY_JOB_ID:-local}_${JOB_ID}"
echo "  Task ID:      $JOB_ID  (of 0-30)"
echo "  Wind:         $WIND"
echo "  Env:          $ENV_FILE_NAME"
echo "  Steps:        $STEPS"
echo "  Seed:         $SEED"
echo "  Num eigvecs:  $NUM_EIGVECS"
echo "  Results dir:  $RESULTS_DIR/task_${JOB_ID}"
echo "========================================"

python train_allo_rep.py \
    --env_type              file \
    --env_file_name         "$ENV_FILE_NAME" \
    --num_gradient_steps    "$STEPS" \
    --batch_size            256 \
    --num_eigenvector_pairs "$NUM_EIGVECS" \
    --learning_rate         0.0003 \
    --gamma                 0.9 \
    --no-use_rejection_sampling \
    --num_envs              1000 \
    --num_steps             1000 \
    --step_size_duals       1.0 \
    --duals_initial_val     -2.0 \
    --barrier_initial_val   0.5 \
    --max_barrier_coefs     0.5 \
    --exp_name              allo \
    --exp_number            "$JOB_ID" \
    --seed                  "$SEED" \
    --results_dir           "$RESULTS_DIR/task_${JOB_ID}" \
    --windy \
    --wind                  "$WIND"

echo "========================================"
echo "Task $JOB_ID (wind=$WIND) complete."
echo "========================================"
