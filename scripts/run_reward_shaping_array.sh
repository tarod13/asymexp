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
# Called from scripts/run_reward_shaping_per_env.sh or directly:
#
#   bash scripts/run_reward_shaping_task.sh \
#       --model_dir  ./results/file/<complex_run> \
#       --output_dir ./results/file/<complex_run>/reward_shaping \
#       [--num_seeds 100] [--total_steps 12000000] [--max_steps 200] \
#       [--shaping_coef 0.1] [--gamma_rl 0.99] [--lr 0.1] \
#       [--epsilon 0.5] [--eval_interval 500000] [--eval_seed 0] \
#       [--num_eval_episodes 30] [--min_goal_distance 8] \
#       [--start_state "1,1"] [--num_eigenvectors 8] [--skip_qlearning]
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
ALLO_MODEL_DIR=""
OUTPUT_DIR=""
NUM_SEEDS=100
TOTAL_STEPS=12000000
MAX_STEPS=200
SHAPING_COEF=0.1
GAMMA_RL=0.99
LR=0.1
EPSILON=0.5
EVAL_INTERVAL=500000
EVAL_SEED=0
NUM_EVAL_EPISODES=30
MIN_GOAL_DISTANCE=8
START_STATE="1,1"
NUM_EIGENVECTORS=8
N_STEP_TD=1
POTENTIAL_MODE=negative
POTENTIAL_TEMP=1.0
POTENTIAL_DELTA=1.0
SKIP_QLEARNING=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)            ACCOUNT="$2";            shift 2 ;;
        --env)                ENV="$2";                shift 2 ;;
        --model_dir)          MODEL_DIR="$2";          shift 2 ;;
        --allo_model_dir)     ALLO_MODEL_DIR="$2";     shift 2 ;;
        --output_dir)         OUTPUT_DIR="$2";         shift 2 ;;
        --num_seeds)          NUM_SEEDS="$2";          shift 2 ;;
        --total_steps)        TOTAL_STEPS="$2";        shift 2 ;;
        --max_steps)          MAX_STEPS="$2";          shift 2 ;;
        --shaping_coef)       SHAPING_COEF="$2";       shift 2 ;;
        --gamma_rl)           GAMMA_RL="$2";           shift 2 ;;
        --lr)                 LR="$2";                 shift 2 ;;
        --epsilon)            EPSILON="$2";            shift 2 ;;
        --eval_interval)      EVAL_INTERVAL="$2";      shift 2 ;;
        --eval_seed)          EVAL_SEED="$2";          shift 2 ;;
        --num_eval_episodes)  NUM_EVAL_EPISODES="$2";  shift 2 ;;
        --min_goal_distance)  MIN_GOAL_DISTANCE="$2";  shift 2 ;;
        --start_state)        START_STATE="$2";        shift 2 ;;
        --num_eigenvectors)   NUM_EIGENVECTORS="$2";   shift 2 ;;
        --n_step_td)          N_STEP_TD="$2";          shift 2 ;;
        --potential_mode)     POTENTIAL_MODE="$2";     shift 2 ;;
        --potential_temp)     POTENTIAL_TEMP="$2";     shift 2 ;;
        --potential_delta)    POTENTIAL_DELTA="$2";    shift 2 ;;
        --skip_qlearning)     SKIP_QLEARNING=true;     shift ;;
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

# ── Method count: 7 (with ALLO) or 4 (without) ───────────────────────────────
if [ -n "$ALLO_MODEL_DIR" ]; then
    NUM_METHODS=7
    METHODS_LABEL="baseline, complex, gt_truncated, gt_full, allo_hitting_time, allo_squared_diff, allo_weighted_squared_diff"
else
    NUM_METHODS=4
    METHODS_LABEL="baseline, complex, gt_truncated, gt_full"
fi

NUM_TASKS=$(( NUM_METHODS * NUM_SEEDS ))
MAX_TASK_ID=$(( NUM_TASKS - 1 ))

echo "========================================"
echo "Reward-shaping distributed submission"
echo "  Model dir      : $MODEL_DIR"
echo "  Allo model dir : ${ALLO_MODEL_DIR:-(not set)}"
echo "  Output dir     : $OUTPUT_DIR"
echo "  Methods        : $METHODS_LABEL  (NUM_METHODS=$NUM_METHODS)"
echo "  Num seeds      : $NUM_SEEDS"
echo "  Total tasks    : $NUM_TASKS  (task IDs 0-$MAX_TASK_ID)"
echo "  Total steps    : $TOTAL_STEPS"
echo "  Shaping coef   : $SHAPING_COEF"
echo "  Num eigenvectors: $NUM_EIGENVECTORS"
echo "  Potential mode : $POTENTIAL_MODE"
echo "  Potential temp : $POTENTIAL_TEMP"
echo "  Potential delta: $POTENTIAL_DELTA"
echo "========================================"

# ── Export env vars so child scripts inherit them ─────────────────────────────
export ENV MODEL_DIR OUTPUT_DIR
export NUM_SEEDS NUM_METHODS TOTAL_STEPS MAX_STEPS
export SHAPING_COEF GAMMA_RL LR EPSILON EVAL_INTERVAL EVAL_SEED NUM_EVAL_EPISODES
export MIN_GOAL_DISTANCE START_STATE NUM_EIGENVECTORS N_STEP_TD POTENTIAL_MODE POTENTIAL_TEMP POTENTIAL_DELTA ALLO_MODEL_DIR

mkdir -p logs
mkdir -p "${OUTPUT_DIR}/partial"

# ── Choose SLURM or local execution ──────────────────────────────────────────
if command -v sbatch &>/dev/null; then
    echo ""
    echo "Detected SLURM — submitting jobs..."

    if [ "$SKIP_QLEARNING" = true ]; then
        # ── Analysis only ─────────────────────────────────────────────────────
        echo "  --skip_qlearning set: submitting analysis job only."
        ANALYZE_JOB_ID=$(
            sbatch \
                --account="$ACCOUNT" \
                --export=ALL \
                "${SCRIPT_DIR}/analyze_reward_shaping.sh" \
            | awk '{print $4}'
        )
        echo "  Analysis job submitted: $ANALYZE_JOB_ID"
        echo "  Monitor: squeue -j ${ANALYZE_JOB_ID}"
    else
        # ── SLURM: submit array job then analysis job with dependency ──────────
        ARRAY_JOB_ID=$(
            sbatch \
                --account="$ACCOUNT" \
                --array=0-${MAX_TASK_ID} \
                --export=ALL \
                "${SCRIPT_DIR}/run_reward_shaping_task.sh" \
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
        echo "  Monitor: squeue -j ${ARRAY_JOB_ID},${ANALYZE_JOB_ID}"
    fi
    echo ""
    echo "Results will appear in: $OUTPUT_DIR"

else
    # ── Local fallback ────────────────────────────────────────────────────────
    echo ""
    if [ "$SKIP_QLEARNING" = true ]; then
        echo "  --skip_qlearning set: running analysis only..."
    else
        echo "SLURM not available — running locally (sequential)..."
        for task_id in $(seq 0 $MAX_TASK_ID); do
            echo "--- Task $task_id / $MAX_TASK_ID ---"
            bash "${SCRIPT_DIR}/run_reward_shaping_task.sh" "$task_id"
        done
        echo ""
        echo "All Q-learning tasks complete. Running analysis..."
    fi
    bash "${SCRIPT_DIR}/analyze_reward_shaping.sh"
    echo ""
    echo "Done!  Results in: $OUTPUT_DIR"
fi
