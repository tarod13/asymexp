#!/bin/bash
# =============================================================================
# Orchestrate distributed Q-learning reward-shaping jobs.
#
# Submits one job per (method × seed) pair via a SLURM array, then submits
# the analysis job as a dependent job that runs once all Q-learning tasks
# finish.  Falls back to sequential local execution when SLURM is unavailable.
#
# Called from scripts/run_pipeline.py after model training is complete, but
# can also be invoked directly:
#
#   bash scripts/submit_reward_shaping.sh \
#       --model_dir     ./results/file/<complex_run> \
#       --allo_model_dir ./results/file/<allo_run>  \
#       --output_dir    ./results/file/<complex_run>/reward_shaping \
#       [--num_seeds 5] [--num_episodes 30000] [--max_steps 500] \
#       [--shaping_coef 0.1] [--gamma_rl 0.99] [--lr 0.1] \
#       [--epsilon 0.1] [--log_interval 500] [--eval_seed 0] \
#       [--num_eval_episodes 30] [--min_goal_distance 0] \
#       [--start_state "x,y"]
#
# When --allo_model_dir is omitted, only the baseline and complex conditions
# are run (NUM_METHODS=2, task IDs 0..2*num_seeds-1).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_DIR=""
ALLO_MODEL_DIR=""
OUTPUT_DIR=""
NUM_SEEDS=5
NUM_EPISODES=30000
MAX_STEPS=500
SHAPING_COEF=0.1
GAMMA_RL=0.99
LR=0.1
EPSILON=0.1
LOG_INTERVAL=500
EVAL_SEED=0
NUM_EVAL_EPISODES=30
MIN_GOAL_DISTANCE=0
START_STATE=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_dir)          MODEL_DIR="$2";          shift 2 ;;
        --allo_model_dir)     ALLO_MODEL_DIR="$2";     shift 2 ;;
        --output_dir)         OUTPUT_DIR="$2";         shift 2 ;;
        --num_seeds)          NUM_SEEDS="$2";          shift 2 ;;
        --num_episodes)       NUM_EPISODES="$2";       shift 2 ;;
        --max_steps)          MAX_STEPS="$2";          shift 2 ;;
        --shaping_coef)       SHAPING_COEF="$2";       shift 2 ;;
        --gamma_rl)           GAMMA_RL="$2";           shift 2 ;;
        --lr)                 LR="$2";                 shift 2 ;;
        --epsilon)            EPSILON="$2";            shift 2 ;;
        --log_interval)       LOG_INTERVAL="$2";       shift 2 ;;
        --eval_seed)          EVAL_SEED="$2";          shift 2 ;;
        --num_eval_episodes)  NUM_EVAL_EPISODES="$2";  shift 2 ;;
        --min_goal_distance)  MIN_GOAL_DISTANCE="$2";  shift 2 ;;
        --start_state)        START_STATE="$2";        shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate required arguments ───────────────────────────────────────────────
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: --model_dir is required." >&2; exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output_dir is required." >&2; exit 1
fi

# ── Determine number of methods ───────────────────────────────────────────────
if [ -n "$ALLO_MODEL_DIR" ]; then
    NUM_METHODS=3
    METHODS_LABEL="baseline, complex, allo"
else
    NUM_METHODS=2
    METHODS_LABEL="baseline, complex"
fi

NUM_TASKS=$(( NUM_METHODS * NUM_SEEDS ))
MAX_TASK_ID=$(( NUM_TASKS - 1 ))

echo "========================================"
echo "Reward-shaping distributed submission"
echo "  Model dir      : $MODEL_DIR"
echo "  ALLO model dir : ${ALLO_MODEL_DIR:-<none>}"
echo "  Output dir     : $OUTPUT_DIR"
echo "  Methods        : $METHODS_LABEL  (NUM_METHODS=$NUM_METHODS)"
echo "  Num seeds      : $NUM_SEEDS"
echo "  Total tasks    : $NUM_TASKS  (task IDs 0-$MAX_TASK_ID)"
echo "  Num episodes   : $NUM_EPISODES"
echo "  Shaping coef   : $SHAPING_COEF"
echo "========================================"

# ── Export env vars so child scripts inherit them ─────────────────────────────
export MODEL_DIR ALLO_MODEL_DIR OUTPUT_DIR
export NUM_SEEDS NUM_METHODS NUM_EPISODES MAX_STEPS
export SHAPING_COEF GAMMA_RL LR EPSILON LOG_INTERVAL EVAL_SEED NUM_EVAL_EPISODES
export MIN_GOAL_DISTANCE START_STATE

mkdir -p logs
mkdir -p "${OUTPUT_DIR}/partial"

# ── Choose SLURM or local execution ──────────────────────────────────────────
if command -v sbatch &>/dev/null; then
    # ── SLURM: submit array job then analysis job with dependency ─────────────
    echo ""
    echo "Detected SLURM — submitting jobs..."

    ARRAY_JOB_ID=$(
        sbatch \
            --array=0-${MAX_TASK_ID} \
            --export=ALL \
            "${SCRIPT_DIR}/run_reward_shaping_array.sh" \
        | awk '{print $4}'
    )
    echo "  Q-learning array job submitted: $ARRAY_JOB_ID  (tasks 0-${MAX_TASK_ID})"

    ANALYZE_JOB_ID=$(
        sbatch \
            --dependency=afterok:${ARRAY_JOB_ID} \
            --export=ALL \
            "${SCRIPT_DIR}/analyze_reward_shaping.sh" \
        | awk '{print $4}'
    )
    echo "  Analysis job submitted:         $ANALYZE_JOB_ID  (runs after $ARRAY_JOB_ID)"
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -j ${ARRAY_JOB_ID},${ANALYZE_JOB_ID}"
    echo ""
    echo "Results will appear in: $OUTPUT_DIR"

else
    # ── Local fallback: run tasks sequentially ────────────────────────────────
    echo ""
    echo "SLURM not available — running locally (sequential)..."
    echo ""

    for task_id in $(seq 0 $MAX_TASK_ID); do
        echo "--- Task $task_id / $MAX_TASK_ID ---"
        bash "${SCRIPT_DIR}/run_reward_shaping_array.sh" "$task_id"
    done

    echo ""
    echo "All Q-learning tasks complete. Running analysis..."
    bash "${SCRIPT_DIR}/analyze_reward_shaping.sh"

    echo ""
    echo "Done!  Results in: $OUTPUT_DIR"
fi
