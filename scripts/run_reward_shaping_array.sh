#!/bin/bash
#SBATCH --job-name=rs_qlearn
#SBATCH --account=aip-machado
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rs_qlearn_%A_%a.out
#SBATCH --error=logs/rs_qlearn_%A_%a.err

# =============================================================================
# Distributed Q-learning reward-shaping array job.
# Each task handles one (method, seed) pair, keeping per-job memory small.
#
# Task-ID encoding:
#   task_id = method_idx * NUM_SEEDS + seed_idx
#   method_idx : 0 = baseline, 1 = complex, 2 = allo
#
# All configuration is passed via environment variables exported by
# scripts/submit_reward_shaping.sh (or set manually before sbatch):
#
#   Required : MODEL_DIR, OUTPUT_DIR, NUM_SEEDS, NUM_METHODS
#   Q-learning: NUM_EPISODES, MAX_STEPS, SHAPING_COEF, GAMMA_RL, LR,
#               EPSILON, LOG_INTERVAL, EVAL_SEED, NUM_EVAL_EPISODES
#   Optional  : ALLO_MODEL_DIR, MIN_GOAL_DISTANCE, START_STATE
#
# Usage
# -----
#   sbatch --array=0-14 --export=ALL scripts/run_reward_shaping_array.sh
#   bash   scripts/run_reward_shaping_array.sh <task_id>   # local single run
# =============================================================================

mkdir -p logs

# Support local execution: bash run_reward_shaping_array.sh <task_id>
JOB_ID="${1:-$SLURM_ARRAY_TASK_ID}"

# ── Environment setup ─────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 mujoco/3.3.0
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"

# ── Defaults (overridden by exported env vars from submit_reward_shaping.sh) ──
NUM_SEEDS="${NUM_SEEDS:-5}"
NUM_METHODS="${NUM_METHODS:-3}"
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

# ── Decode (method, seed) from task ID ───────────────────────────────────────
method_idx=$(( JOB_ID / NUM_SEEDS ))
seed_idx=$(( JOB_ID % NUM_SEEDS ))

METHODS=("baseline" "complex" "allo")
METHOD="${METHODS[$method_idx]}"

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
    python experiments/reward_shaping/run_reward_shaping.py
    --model_dir          "$MODEL_DIR"
    --method             "$METHOD"
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
    --output_dir         "$OUTPUT_DIR"
)

# Only pass --allo_model_dir when we actually need it (method=allo).
if [ "$METHOD" = "allo" ] && [ -n "${ALLO_MODEL_DIR:-}" ]; then
    CMD+=(--allo_model_dir "$ALLO_MODEL_DIR")
fi

if [ "${MIN_GOAL_DISTANCE:-0}" -gt 0 ]; then
    CMD+=(--min_goal_distance "$MIN_GOAL_DISTANCE")
fi

if [ -n "${START_STATE:-}" ]; then
    CMD+=(--start_state "$START_STATE")
fi

"${CMD[@]}"

echo "========================================"
echo "Task $JOB_ID (method=$METHOD, seed=$seed_idx) complete."
echo "========================================"
