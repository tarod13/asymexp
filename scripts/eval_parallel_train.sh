#!/bin/bash
# =============================================================================
# Parallel representation training across all environments (no importance sampling)
#
# Usage (standalone):
#   sbatch scripts/eval_parallel_train.sh
#
# Usage (from run_full_pipeline.sh, which overrides account / array size and
# passes the path / experiment arguments below):
#   sbatch --account=<acct> --array=0-N \
#       scripts/eval_parallel_train.sh \
#       [--results_dir PATH] [--manifest_dir PATH] \
#       [--exp_name STR]     [--seed N] \
#       [--envs "E1 E2 ..."]
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

# ── Arguments (all optional; defaults match standalone behaviour) ──────────────
RESULTS_DIR="./results/parallel_clf_eval"
MANIFEST_DIR="./results/eval_manifest"
EXP_NAME="parallel_clf_eval"
EXP_NUMBER=1
SEED=42
ENV_LIST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results_dir)  RESULTS_DIR="$2"; shift 2 ;;
        --manifest_dir) MANIFEST_DIR="$2"; shift 2 ;;
        --exp_name)     EXP_NAME="$2";    shift 2 ;;
        --exp_number)   EXP_NUMBER="$2";  shift 2 ;;
        --seed)         SEED="$2";        shift 2 ;;
        --envs)         ENV_LIST="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$MANIFEST_DIR"

# ── Environment table (index 0-7) ──────────────────────────────────────────────
if [ -n "$ENV_LIST" ]; then
    read -ra ENV_FILES <<< "$ENV_LIST"
else
    ENV_FILES=(
        "GridRoom-4"  "GridRoom-4-Doors"
        "GridRoom-1"  "GridRoom-1-Portals"
        "GridMaze-OGBench"  "GridMaze-OGBench-Portals"
        "GridMaze-OGBench-Hard"  "GridMaze-OGBench-Hard-Portals"
    )
fi

ENV_FILE=${ENV_FILES[$SLURM_ARRAY_TASK_ID]}
ENV_NAME=$ENV_FILE

echo "========================================"
echo "Parallel Eval Training — No Importance Sampling"
echo "Job ID      : $SLURM_JOB_ID"
echo "Array Task  : $SLURM_ARRAY_TASK_ID"
echo "Environment : $ENV_NAME"
echo "Node        : $SLURM_NODELIST"
echo "========================================"

# ── Environment setup ──────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 scipy-stack/2024b
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ── Run training (CLF, no importance sampling) ────────────────────────────────
python train.py clf \
    --env_file_name        "$ENV_FILE" \
    --num_gradient_steps   100000 \
    --batch_size           1024 \
    --num_eigenvector_pairs 8 \
    --learning_rate        0.00001 \
    --ema_learning_rate    0.0003 \
    --exp_name             "$EXP_NAME" \
    --exp_number           "$EXP_NUMBER" \
    --seed                 "$SEED" \
    --results_dir          "$RESULTS_DIR" \
    --no-plot_during_training \
    --save_model

# ── Find the results directory and write to manifest ──────────────────────────
# Directory pattern: {results_dir}/{env}/{env}__{exp_name}__{exp_number}__{seed}__*/
TRAIN_OUT=$(ls -td "${RESULTS_DIR}/${ENV_FILE}/${ENV_FILE}__${EXP_NAME}__${EXP_NUMBER}__${SEED}__"*/ 2>/dev/null | head -1)

if [ -z "$TRAIN_OUT" ]; then
    echo "ERROR: Could not find results directory for $ENV_NAME" >&2
    exit 1
fi

# Strip trailing slash for consistency
TRAIN_OUT="${TRAIN_OUT%/}"
# Resolve to absolute path so the manifest is portable across jobs
TRAIN_OUT=$(realpath "$TRAIN_OUT")
echo "$TRAIN_OUT" > "${MANIFEST_DIR}/${ENV_NAME}.txt"

echo "========================================"
echo "Training complete for: $ENV_NAME"
echo "Results saved to     : $TRAIN_OUT"
echo "Manifest written to  : ${MANIFEST_DIR}/${ENV_NAME}.txt"
echo "========================================"
