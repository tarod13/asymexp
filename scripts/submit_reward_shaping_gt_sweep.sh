#!/bin/bash
# =============================================================================
# Sweep --num_eigenvectors from K_MIN to K_MAX using ground-truth Laplacian
# eigenvectors (--use_gt) and run the reward-shaping Q-learning experiment for
# each value.
#
# Submits one SLURM array job covering all (k, seed) pairs, then a dependent
# analysis job.  Falls back to sequential local execution without SLURM.
#
# Usage
# -----
#   bash scripts/submit_reward_shaping_gt_sweep.sh \
#       --output_dir ./results/gt_sweep \
#       [--env GridRoom-4-Doors] \
#       [--k_min 1] [--k_max 104] \
#       [--num_seeds 5] \
#       [--gt_gamma 0.95] [--gt_delta 0.1] \
#       [--num_episodes 30000] [--max_steps 500] \
#       [--shaping_coef 0.1] [--gamma_rl 0.99] [--lr 0.1] [--epsilon 0.1] \
#       [--log_interval 500] [--eval_seed 0] [--num_eval_episodes 30] \
#       [--min_goal_distance 0] [--start_state "x,y"]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
ENV="GridRoom-4-Doors"
OUTPUT_DIR=""
K_MIN=1
K_MAX=104
NUM_SEEDS=5
GT_GAMMA=0.95
GT_DELTA=0.1
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
        --env)               ENV="$2";               shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --k_min)             K_MIN="$2";             shift 2 ;;
        --k_max)             K_MAX="$2";             shift 2 ;;
        --num_seeds)         NUM_SEEDS="$2";         shift 2 ;;
        --gt_gamma)          GT_GAMMA="$2";          shift 2 ;;
        --gt_delta)          GT_DELTA="$2";          shift 2 ;;
        --num_episodes)      NUM_EPISODES="$2";      shift 2 ;;
        --max_steps)         MAX_STEPS="$2";         shift 2 ;;
        --shaping_coef)      SHAPING_COEF="$2";      shift 2 ;;
        --gamma_rl)          GAMMA_RL="$2";          shift 2 ;;
        --lr)                LR="$2";                shift 2 ;;
        --epsilon)           EPSILON="$2";           shift 2 ;;
        --log_interval)      LOG_INTERVAL="$2";      shift 2 ;;
        --eval_seed)         EVAL_SEED="$2";         shift 2 ;;
        --num_eval_episodes) NUM_EVAL_EPISODES="$2"; shift 2 ;;
        --min_goal_distance) MIN_GOAL_DISTANCE="$2"; shift 2 ;;
        --start_state)       START_STATE="$2";       shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output_dir is required." >&2; exit 1
fi
if [ "$K_MIN" -lt 1 ]; then
    echo "ERROR: --k_min must be >= 1." >&2; exit 1
fi
if [ "$K_MAX" -lt "$K_MIN" ]; then
    echo "ERROR: --k_max must be >= --k_min." >&2; exit 1
fi

NUM_K=$(( K_MAX - K_MIN + 1 ))
NUM_TASKS=$(( NUM_K * NUM_SEEDS ))
MAX_TASK_ID=$(( NUM_TASKS - 1 ))

echo "========================================"
echo "GT eigenvector sweep submission"
echo "  Env            : $ENV"
echo "  Output dir     : $OUTPUT_DIR"
echo "  k range        : $K_MIN .. $K_MAX  ($NUM_K values)"
echo "  Num seeds      : $NUM_SEEDS"
echo "  Total tasks    : $NUM_TASKS  (task IDs 0-$MAX_TASK_ID)"
echo "  gt_gamma       : $GT_GAMMA"
echo "  gt_delta       : $GT_DELTA"
echo "  Num episodes   : $NUM_EPISODES"
echo "  Shaping coef   : $SHAPING_COEF"
echo "========================================"

# ── Export env vars so the array worker inherits them ─────────────────────────
export ENV OUTPUT_DIR K_MIN K_MAX NUM_SEEDS GT_GAMMA GT_DELTA
export NUM_EPISODES MAX_STEPS SHAPING_COEF GAMMA_RL LR EPSILON
export LOG_INTERVAL EVAL_SEED NUM_EVAL_EPISODES MIN_GOAL_DISTANCE START_STATE

mkdir -p logs "$OUTPUT_DIR"

# ── Choose SLURM or local execution ──────────────────────────────────────────
if command -v sbatch &>/dev/null; then
    echo ""
    echo "Detected SLURM — submitting jobs..."

    ARRAY_JOB_ID=$(
        sbatch \
            --array=0-${MAX_TASK_ID} \
            --export=ALL \
            "${SCRIPT_DIR}/run_reward_shaping_gt_sweep_array.sh" \
        | awk '{print $4}'
    )
    echo "  GT sweep array job submitted: $ARRAY_JOB_ID  (tasks 0-${MAX_TASK_ID})"
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -j ${ARRAY_JOB_ID}"
    echo ""
    echo "Results will appear in: $OUTPUT_DIR/k_<K>/"

else
    echo ""
    echo "SLURM not available — running locally (sequential)..."
    echo ""

    for task_id in $(seq 0 $MAX_TASK_ID); do
        echo "--- Task $task_id / $MAX_TASK_ID ---"
        bash "${SCRIPT_DIR}/run_reward_shaping_gt_sweep_array.sh" "$task_id"
    done

    echo ""
    echo "Done!  Results in: $OUTPUT_DIR/k_<K>/"
fi
