#!/bin/bash
#SBATCH --job-name=rs_qlearn
#SBATCH --account=rrg-machado
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rs_qlearn_%A_%a.out
#SBATCH --error=logs/rs_qlearn_%A_%a.err

# =============================================================================
# Distributed Q-learning reward-shaping array job.
# Each task handles one (method, seed) pair, keeping per-job memory small.
#
# Task-ID encoding:
#   task_id    = method_idx * NUM_SEEDS + seed_idx
#   method_idx : 0=baseline, 1=complex, 2=gt_truncated, 3=gt_full,
#                4=allo_hitting_time, 5=allo_squared_diff,
#                6=allo_weighted_squared_diff
#   (If ALLO_MODEL_DIR is unset, NUM_METHODS is 4 and allo tasks are omitted.)
#
# All configuration is passed via environment variables exported by
# scripts/run_reward_shaping_array.sh (or set manually before sbatch):
#
#   Required : MODEL_DIR, OUTPUT_DIR, NUM_SEEDS, NUM_METHODS
#   Q-learning: TOTAL_STEPS, MAX_STEPS, SHAPING_COEF, GAMMA_RL, LR,
#               EPSILON, EVAL_INTERVAL, EVAL_SEED, NUM_EVAL_EPISODES
#   Optional  : MIN_GOAL_DISTANCE, START_STATE, NUM_EIGENVECTORS, ALLO_MODEL_DIR
#
# Usage
# -----
#   sbatch --array=0-699 --export=ALL scripts/run_reward_shaping_task.sh
#   bash   scripts/run_reward_shaping_task.sh <task_id>   # local single run
# =============================================================================

mkdir -p logs

# Support local execution: bash run_reward_shaping_task.sh <task_id>
JOB_ID="${1:-$SLURM_ARRAY_TASK_ID}"

# ── Environment setup ─────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 mujoco/3.3.0
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"

# ── Defaults (overridden by exported env vars from run_reward_shaping_array.sh) ──
ENV="${ENV:-GridRoom-4-Doors}"
NUM_SEEDS="${NUM_SEEDS:-100}"
NUM_METHODS="${NUM_METHODS:-4}"
ALLO_MODEL_DIR="${ALLO_MODEL_DIR:-}"
TOTAL_STEPS="${TOTAL_STEPS:-12000000}"
MAX_STEPS="${MAX_STEPS:-200}"
SHAPING_COEF="${SHAPING_COEF:-0.1}"
GAMMA_RL="${GAMMA_RL:-0.99}"
LR="${LR:-0.1}"
EPSILON="${EPSILON:-0.5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500000}"
EVAL_SEED="${EVAL_SEED:-0}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-30}"
MIN_GOAL_DISTANCE="${MIN_GOAL_DISTANCE:-8}"
START_STATE="${START_STATE:-1,1}"
GOAL_STATE="${GOAL_STATE:-}"
NUM_EIGENVECTORS="${NUM_EIGENVECTORS:-8}"
N_STEP_TD="${N_STEP_TD:-1}"
POTENTIAL_MODE="${POTENTIAL_MODE:-negative}"
POTENTIAL_TEMP="${POTENTIAL_TEMP:-1.0}"
POTENTIAL_DELTA="${POTENTIAL_DELTA:-1.0}"

# ── Decode (method, seed) from task ID ───────────────────────────────────────
method_idx=$(( JOB_ID / NUM_SEEDS ))
seed_idx=$(( JOB_ID % NUM_SEEDS ))

METHODS=("baseline" "complex" "gt_truncated" "gt_full" "allo_hitting_time" "allo_squared_diff" "allo_weighted_squared_diff")
METHOD="${METHODS[$method_idx]}"

# Graceful skip: allo methods require ALLO_MODEL_DIR
if [[ "$METHOD" == allo_* ]] && [ -z "${ALLO_MODEL_DIR:-}" ]; then
    echo "SKIP: method=$METHOD requires ALLO_MODEL_DIR (not set). Exiting cleanly."
    exit 0
fi

echo "========================================"
echo "Reward-shaping Q-learning (distributed)"
echo "  Array job : ${SLURM_ARRAY_JOB_ID:-local}_${JOB_ID}"
echo "  Task ID   : $JOB_ID  (of 0-$(( NUM_METHODS * NUM_SEEDS - 1 )))"
echo "  Method    : $METHOD (idx $method_idx)"
echo "  Seed idx  : $seed_idx"
echo "  Model dir : ${MODEL_DIR:-<unset>}"
echo "  Output dir: ${OUTPUT_DIR:-<unset>}"
echo "========================================"

CMD=(
    python experiments/reward_shaping/run_reward_shaping_cluster.py
    --env                "$ENV"
    --method             "$METHOD"
    --seed_idx           "$seed_idx"
    --num_seeds          "$NUM_SEEDS"
    --total_steps        "$TOTAL_STEPS"
    --max_steps          "$MAX_STEPS"
    --shaping_coef       "$SHAPING_COEF"
    --gamma_rl           "$GAMMA_RL"
    --lr                 "$LR"
    --epsilon            "$EPSILON"
    --eval_interval      "$EVAL_INTERVAL"
    --eval_seed          "$EVAL_SEED"
    --num_eval_episodes  "$NUM_EVAL_EPISODES"
    --output_dir         "$OUTPUT_DIR"
    --n_step_td          "$N_STEP_TD"
    --potential_mode     "$POTENTIAL_MODE"
    --potential_temp     "$POTENTIAL_TEMP"
    --potential_delta    "$POTENTIAL_DELTA"
)

# Complex: load from trained model dir.
if [ "$METHOD" = "complex" ] && [ -n "${MODEL_DIR:-}" ]; then
    CMD+=(--model_dir "$MODEL_DIR")
fi

# GT truncated: load gt_* eigenvectors + apply truncation.
if [ "$METHOD" = "gt_truncated" ]; then
    CMD+=(--num_eigenvectors "$NUM_EIGENVECTORS")
    if [ -n "${MODEL_DIR:-}" ]; then
        CMD+=(--model_dir "$MODEL_DIR")
    fi
fi

# GT full: load full_* eigenvectors / precomputed file — no truncation.
if [ "$METHOD" = "gt_full" ]; then
    if [ -n "${MODEL_DIR:-}" ]; then
        CMD+=(--model_dir "$MODEL_DIR")
    fi
fi

# ALLO variants: all require --allo_model_dir.
if [[ "$METHOD" == allo_* ]]; then
    CMD+=(--allo_model_dir "$ALLO_MODEL_DIR")
    CMD+=(--num_eigenvectors "$NUM_EIGENVECTORS")
fi

if [ "${MIN_GOAL_DISTANCE:-0}" -gt 0 ]; then
    CMD+=(--min_goal_distance "$MIN_GOAL_DISTANCE")
fi

if [ -n "${START_STATE:-}" ]; then
    CMD+=(--start_state "$START_STATE")
fi

if [ -n "${GOAL_STATE:-}" ]; then
    CMD+=(--goal_state "$GOAL_STATE")
fi

"${CMD[@]}"

echo "========================================"
echo "Task $JOB_ID (method=$METHOD, seed=$seed_idx) complete."
echo "========================================"
