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
# GT eigenvector sweep: array worker.
# Each task handles one (num_eigenvectors=k, seed) pair.
#
# Task-ID encoding:
#   k_idx    = task_id / NUM_SEEDS          (0-indexed into K_MIN..K_MAX)
#   seed_idx = task_id % NUM_SEEDS
#   k        = K_MIN + k_idx                (actual num_eigenvectors value)
#
# All configuration is passed via environment variables exported by
# scripts/submit_reward_shaping_gt_sweep.sh (or set manually):
#
#   Required : OUTPUT_DIR, ENV
#   Sweep    : K_MIN, K_MAX, NUM_SEEDS
#   GT       : GT_GAMMA, GT_DELTA
#   Q-learn  : NUM_EPISODES, MAX_STEPS, SHAPING_COEF, GAMMA_RL, LR,
#               EPSILON, LOG_INTERVAL, EVAL_SEED, NUM_EVAL_EPISODES
#   Optional : MIN_GOAL_DISTANCE, START_STATE
#
# Usage
# -----
#   sbatch --array=0-<max_id> --export=ALL scripts/run_reward_shaping_gt_sweep_array.sh
#   bash   scripts/run_reward_shaping_gt_sweep_array.sh <task_id>   # local run
# =============================================================================

mkdir -p logs

# Support local execution: bash run_reward_shaping_gt_sweep_array.sh <task_id>
JOB_ID="${1:-$SLURM_ARRAY_TASK_ID}"

# ── Environment setup ─────────────────────────────────────────────────────────
module --force purge 2>/dev/null || true
module load StdEnv/2023 gcc/14.3 python/3.11 mujoco/3.3.0 2>/dev/null || true
source ~/ENV/bin/activate 2>/dev/null || true

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"

# ── Defaults (overridden by exported env vars) ────────────────────────────────
ENV="${ENV:-GridRoom-4-Doors}"
NUM_SEEDS="${NUM_SEEDS:-5}"
K_MIN="${K_MIN:-1}"
K_MAX="${K_MAX:-104}"
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

# ── Decode (k, seed) from task ID ────────────────────────────────────────────
k_idx=$(( JOB_ID / NUM_SEEDS ))
seed_idx=$(( JOB_ID % NUM_SEEDS ))
K=$(( K_MIN + k_idx ))

# Per-k output subdirectory so partial files don't collide across k values.
K_OUTPUT_DIR="${OUTPUT_DIR}/k_${K}"

echo "========================================"
echo "GT eigenvector sweep (distributed)"
echo "  Array job      : ${SLURM_ARRAY_JOB_ID:-local}_${JOB_ID}"
echo "  Task ID        : $JOB_ID"
echo "  num_eigenvectors: $K  (k_idx=$k_idx, range $K_MIN..$K_MAX)"
echo "  Seed idx       : $seed_idx"
echo "  Output dir     : $K_OUTPUT_DIR"
echo "========================================"

mkdir -p "${K_OUTPUT_DIR}/partial"

CMD=(
    python experiments/reward_shaping/run_reward_shaping.py
    --env                "$ENV"
    --method             "complex"
    --seed_idx           "$seed_idx"
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
    CMD+=(--min_goal_distance "$MIN_GOAL_DISTANCE")
fi

if [ -n "${START_STATE:-}" ]; then
    CMD+=(--start_state "$START_STATE")
fi

"${CMD[@]}"

echo "========================================"
echo "Task $JOB_ID (k=$K, seed=$seed_idx) complete."
echo "========================================"
