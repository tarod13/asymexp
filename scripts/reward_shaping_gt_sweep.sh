#!/bin/bash
#SBATCH --job-name=rs_gt_sweep
#SBATCH --account=aip-machado
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rs_gt_sweep_%A_%a.out
#SBATCH --error=logs/rs_gt_sweep_%A_%a.err

# =============================================================================
# GT eigenvector sweep array job.
# Each task handles one value of num_eigenvectors=k and runs all seeds
# sequentially within that job.
#
# Task-ID encoding:
#   k = K_MIN + SLURM_ARRAY_TASK_ID
#
# Required env vars (pass via --export=ALL or --export=...):
#   OUTPUT_DIR, ENV, K_MIN, K_MAX, NUM_SEEDS, GT_GAMMA, GT_DELTA,
#   NUM_EPISODES, MAX_STEPS, SHAPING_COEF, GAMMA_RL, LR, EPSILON,
#   LOG_INTERVAL, EVAL_SEED, NUM_EVAL_EPISODES
# Optional:
#   MIN_GOAL_DISTANCE, START_STATE
#
# Usage
# -----
#   sbatch --array=0-<K_MAX-K_MIN> --export=ALL \
#       scripts/reward_shaping_gt_sweep.sh
# =============================================================================

mkdir -p logs

# ── Environment setup ─────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 mujoco/3.3.0
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"

# ── Defaults ──────────────────────────────────────────────────────────────────
ENV="${ENV:-GridRoom-4-Doors}"
K_MIN="${K_MIN:-1}"
K_MAX="${K_MAX:-104}"
NUM_SEEDS="${NUM_SEEDS:-5}"
GT_GAMMA="${GT_GAMMA:-0.95}"
GT_DELTA="${GT_DELTA:-0.1}"
NUM_EPISODES="${NUM_EPISODES:-30000}"
MAX_STEPS="${MAX_STEPS:-500}"
SHAPING_COEF="${SHAPING_COEF:-0.1}"
GAMMA_RL="${GAMMA_RL:-0.99}"
LR="${LR:-0.1}"
EPSILON="${EPSILON:-0.1}"
LOG_INTERVAL="${LOG_INTERVAL:-500}"
EVAL_SEED="${EVAL_SEED:-0}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-30}"
MIN_GOAL_DISTANCE="${MIN_GOAL_DISTANCE:-0}"
START_STATE="${START_STATE:-}"

# ── Decode k from task ID ─────────────────────────────────────────────────────
K=$(( K_MIN + SLURM_ARRAY_TASK_ID ))

K_OUTPUT_DIR="${OUTPUT_DIR}/k_${K}"
mkdir -p "${K_OUTPUT_DIR}"

echo "========================================"
echo "GT eigenvector sweep"
echo "  Array job        : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  num_eigenvectors : $K  (range $K_MIN..$K_MAX)"
echo "  Seeds            : 0..$((NUM_SEEDS - 1))"
echo "  Output dir       : $K_OUTPUT_DIR"
echo "========================================"

BASE_CMD=(
    python experiments/reward_shaping/run_reward_shaping.py
    --env                "$ENV"
    --method             "complex"
    --num_seeds          "$NUM_SEEDS"
    --num_episodes       "$NUM_EPISODES"
    --max_steps          "$MAX_STEPS"
    --shaping_coef       "$SHAPING_COEF"
    --gamma_rl           "$GAMMA_RL"
    --lr                 "$LR"
    --epsilon            "$EPSILON"
    --log_interval       "$LOG_INTERVAL"
    --eval_seed          "$EVAL_SEED"
    --num_eval_episodes  "$NUM_EVAL_EPISODES"
    --output_dir         "$K_OUTPUT_DIR"
    --use_gt
    --num_eigenvectors   "$K"
    --gt_gamma           "$GT_GAMMA"
    --gt_delta           "$GT_DELTA"
)

if [ "${MIN_GOAL_DISTANCE:-0}" -gt 0 ]; then
    BASE_CMD+=(--min_goal_distance "$MIN_GOAL_DISTANCE")
fi
if [ -n "${START_STATE:-}" ]; then
    BASE_CMD+=(--start_state "$START_STATE")
fi

for seed_idx in $(seq 0 $(( NUM_SEEDS - 1 ))); do
    echo "--- seed $seed_idx ---"
    "${BASE_CMD[@]}" --seed_idx "$seed_idx"
done

echo "========================================"
echo "Task ${SLURM_ARRAY_TASK_ID} (k=$K, all $NUM_SEEDS seeds) complete."
echo "========================================"
