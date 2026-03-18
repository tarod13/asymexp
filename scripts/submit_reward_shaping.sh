#!/bin/bash
#SBATCH --job-name=rs_submit
#SBATCH --account=rrg-machado
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=0:05:00
#SBATCH --output=logs/rs_submit_%j.out
#SBATCH --error=logs/rs_submit_%j.err
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
#       --model_dir  ./results/file/<complex_run> \
#       --output_dir ./results/file/<complex_run>/reward_shaping \
#       [--num_seeds 100] [--num_episodes 60000] [--max_steps 200] \
#       [--shaping_coef 0.1] [--gamma_rl 0.99] [--lr 0.1] \
#       [--epsilon 0.5] [--log_interval 500] [--eval_seed 0] \
#       [--num_eval_episodes 30] [--min_goal_distance 8] \
#       [--start_state "1,1"] [--num_eigenvectors 8]
#
# Always runs three conditions: baseline, complex (learned model), gt (ground-truth
# Laplacian eigenvectors via --use_gt).  Total tasks = 3 * num_seeds.
# =============================================================================

set -euo pipefail

# When running as a SLURM job, BASH_SOURCE[0] points to a spool copy.
# Use SLURM_SUBMIT_DIR (set by SLURM to the sbatch invocation directory)
# combined with the known 'scripts/' subdirectory instead.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
ACCOUNT="${ACCOUNT:-rrg-machado}"
ENV=""
MODEL_DIR=""
OUTPUT_DIR=""
NUM_SEEDS=100
NUM_EPISODES=60000
MAX_STEPS=200
SHAPING_COEF=0.1
GAMMA_RL=0.99
LR=0.1
EPSILON=0.5
LOG_INTERVAL=500
EVAL_SEED=0
NUM_EVAL_EPISODES=30
MIN_GOAL_DISTANCE=8
START_STATE="1,1"
NUM_EIGENVECTORS=8

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)            ACCOUNT="$2";            shift 2 ;;
        --env)                ENV="$2";                shift 2 ;;
        --model_dir)          MODEL_DIR="$2";          shift 2 ;;
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
        --num_eigenvectors)   NUM_EIGENVECTORS="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate required arguments ───────────────────────────────────────────────
if [ -z "$ENV" ]; then
    echo "ERROR: --env is required." >&2; exit 1
fi
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: --model_dir is required." >&2; exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output_dir is required." >&2; exit 1
fi

# ── Three fixed methods: baseline, complex (learned), gt (ground-truth) ───────
NUM_METHODS=3
METHODS_LABEL="baseline, complex, gt"

NUM_TASKS=$(( NUM_METHODS * NUM_SEEDS ))
MAX_TASK_ID=$(( NUM_TASKS - 1 ))

echo "========================================"
echo "Reward-shaping distributed submission"
echo "  Model dir      : $MODEL_DIR"
echo "  Output dir     : $OUTPUT_DIR"
echo "  Methods        : $METHODS_LABEL  (NUM_METHODS=$NUM_METHODS)"
echo "  Num seeds      : $NUM_SEEDS"
echo "  Total tasks    : $NUM_TASKS  (task IDs 0-$MAX_TASK_ID)"
echo "  Num episodes   : $NUM_EPISODES"
echo "  Shaping coef   : $SHAPING_COEF"
echo "  Num eigenvectors: $NUM_EIGENVECTORS"
echo "========================================"

# ── Export env vars so child scripts inherit them ─────────────────────────────
export ENV MODEL_DIR OUTPUT_DIR
export NUM_SEEDS NUM_METHODS NUM_EPISODES MAX_STEPS
export SHAPING_COEF GAMMA_RL LR EPSILON LOG_INTERVAL EVAL_SEED NUM_EVAL_EPISODES
export MIN_GOAL_DISTANCE START_STATE NUM_EIGENVECTORS

mkdir -p logs
mkdir -p "${OUTPUT_DIR}/partial"

# ── Choose SLURM or local execution ──────────────────────────────────────────
if command -v sbatch &>/dev/null; then
    # ── SLURM: submit array job then analysis job with dependency ─────────────
    echo ""
    echo "Detected SLURM — submitting jobs..."

    ARRAY_JOB_ID=$(
        sbatch \
            --account="$ACCOUNT" \
            --array=0-${MAX_TASK_ID} \
            --export=ALL \
            "${SCRIPT_DIR}/run_reward_shaping_array.sh" \
        | awk '{print $4}'
    )
    echo "  Q-learning array job submitted: $ARRAY_JOB_ID  (tasks 0-${MAX_TASK_ID})"

    ANALYZE_JOB_ID=$(
        sbatch \
            --account="$ACCOUNT" \
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
