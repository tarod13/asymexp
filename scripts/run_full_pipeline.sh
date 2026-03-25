#!/bin/bash
# =============================================================================
# Submit the full eigenvector pipeline as three chained SLURM jobs.
#
# Stages (each depends on the previous):
#   1. eval_parallel_train.sh  — train eigenvector models (array job, 1 per env)
#   2. eval_parallel_plot.sh   — visualize results
#   3. run_reward_shaping_per_env.sh — reward-shaping evaluation per env
#
# Usage:
#   sbatch scripts/run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --account STR             SLURM account       (default: rrg-machado)
#   --base_dir PATH           Root output dir     (default: ./results)
#   --exp_name STR            Experiment label    (default: parallel_clf_eval)
#   --seed N                  Training seed       (default: 42)
#   --envs "E1 E2 ..."        Environments        (default: all 6)
#   --num_eigenvectors N                          (default: 8)
#   --num_seeds N             Q-learning seeds    (default: 100)
#   --total_steps N           Q-learning steps    (default: 12000000)
#   --max_steps N             Max steps/episode   (default: 200)
#   --shaping_coef F                              (default: 0.1)
#   --gamma_rl F                                  (default: 0.99)
#   --lr F                                        (default: 0.1)
#   --epsilon F                                   (default: 0.5)
#   --eval_interval N                             (default: 500000)
#   --eval_seed N                                 (default: 0)
#   --num_eval_episodes N                         (default: 30)
#   --min_goal_distance N                         (default: 8)
#   --start_state "R,C"                           (default: "1,1")
#   --skip_train              Skip stage 1 (reuse existing manifests)
#   --skip_plot               Skip stage 2
#   --skip_qlearning          Skip stage 3 array job; run analysis only
# =============================================================================

#SBATCH --job-name=pipeline
#SBATCH --account=rrg-machado
#SBATCH --time=0:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=logs/pipeline.out
#SBATCH --error=logs/pipeline.err

set -euo pipefail

SCRIPTS_DIR="${SLURM_SUBMIT_DIR}/scripts"

# ── Defaults ──────────────────────────────────────────────────────────────────
ACCOUNT="rrg-machado"
BASE_DIR="./results"
EXP_NAME="parallel_reward_shaping_eval"
EXP_NUMBER=1
SEED=42
NUM_EIGENVECTORS=8
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
SKIP_TRAIN=false
SKIP_PLOT=false
SKIP_QLEARNING=false
ENV_LIST="GridRoom-4 GridRoom-4-Doors GridRoom-1 GridRoom-1-Portals GridMaze-OGBench GridMaze-OGBench-Portals"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)            ACCOUNT="$2";            shift 2 ;;
        --base_dir)           BASE_DIR="$2";           shift 2 ;;
        --exp_name)           EXP_NAME="$2";           shift 2 ;;
        --exp_number)         EXP_NUMBER="$2";         shift 2 ;;
        --seed)               SEED="$2";               shift 2 ;;
        --envs)               ENV_LIST="$2";           shift 2 ;;
        --num_eigenvectors)   NUM_EIGENVECTORS="$2";   shift 2 ;;
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
        --skip_train)         SKIP_TRAIN=true;         shift ;;
        --skip_plot)          SKIP_PLOT=true;          shift ;;
        --skip_qlearning)     SKIP_QLEARNING=true;     shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Derived paths ─────────────────────────────────────────────────────────────
RESULTS_DIR="${BASE_DIR}/parallel_reward_shaping_eval"
MANIFEST_DIR="${BASE_DIR}/eval_manifest"
PLOTS_DIR="${BASE_DIR}/parallel_reward_shaping_eval_plots"

read -ra ENVS <<< "$ENV_LIST"
NUM_ENVS=${#ENVS[@]}
MAX_ARRAY_ID=$(( NUM_ENVS - 1 ))

mkdir -p logs
# Wipe manifest on a fresh training run so stale env entries don't accumulate
if [ "$SKIP_TRAIN" = false ]; then
    rm -rf "$MANIFEST_DIR"
fi
mkdir -p "$MANIFEST_DIR"

echo "============================================================"
echo "  Full Eigenvector Pipeline"
echo "  Account      : $ACCOUNT"
echo "  Results dir  : $RESULTS_DIR"
echo "  Manifest dir : $MANIFEST_DIR"
echo "  Plots dir    : $PLOTS_DIR"
echo "  Exp name     : $EXP_NAME  |  seed: $SEED"
echo "  Environments : ${ENVS[*]}"
echo "============================================================"

# ── Stage 1: Training ─────────────────────────────────────────────────────────
TRAIN_JID=""

if [ "$SKIP_TRAIN" = false ]; then
    TRAIN_JID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --array=0-${MAX_ARRAY_ID} \
        "${SCRIPTS_DIR}/eval_parallel_train.sh" \
            --results_dir  "$RESULTS_DIR" \
            --manifest_dir "$MANIFEST_DIR" \
            --exp_name     "$EXP_NAME" \
            --exp_number   "$EXP_NUMBER" \
            --seed         "$SEED" \
            --envs         "$ENV_LIST")
    echo "[1/3] Training array submitted: job $TRAIN_JID  (tasks 0-$MAX_ARRAY_ID)"
else
    echo "[1/3] Training skipped (--skip_train)."
fi

# ── Stage 2: Plotting ─────────────────────────────────────────────────────────
PLOT_JID=""

if [ "$SKIP_PLOT" = false ]; then
    PLOT_ARGS=(--account="$ACCOUNT")
    [ -n "$TRAIN_JID" ] && PLOT_ARGS+=(--dependency=afterok:${TRAIN_JID})

    PLOT_JID=$(sbatch --parsable \
        "${PLOT_ARGS[@]}" \
        "${SCRIPTS_DIR}/eval_parallel_plot.sh" \
            --manifest_dir "$MANIFEST_DIR" \
            --plots_dir    "$PLOTS_DIR")
    echo "[2/3] Plotting job submitted: job $PLOT_JID$([ -n "$TRAIN_JID" ] && echo "  (after $TRAIN_JID)")"
else
    echo "[2/3] Plotting skipped (--skip_plot)."
fi

# ── Stage 3: Reward shaping ───────────────────────────────────────────────────
RS_ARGS=(--account="$ACCOUNT")
[ -n "$PLOT_JID" ] && RS_ARGS+=(--dependency=afterok:${PLOT_JID})

RS_JID=$(sbatch --parsable \
    "${RS_ARGS[@]}" \
    "${SCRIPTS_DIR}/run_reward_shaping_per_env.sh" \
        --account            "$ACCOUNT" \
        --manifest_dir       "$MANIFEST_DIR" \
        --num_seeds          "$NUM_SEEDS" \
        --total_steps        "$TOTAL_STEPS" \
        --max_steps          "$MAX_STEPS" \
        --shaping_coef       "$SHAPING_COEF" \
        --gamma_rl           "$GAMMA_RL" \
        --lr                 "$LR" \
        --epsilon            "$EPSILON" \
        --eval_interval      "$EVAL_INTERVAL" \
        --eval_seed          "$EVAL_SEED" \
        --num_eval_episodes  "$NUM_EVAL_EPISODES" \
        --min_goal_distance  "$MIN_GOAL_DISTANCE" \
        --start_state        "$START_STATE" \
        --num_eigenvectors   "$NUM_EIGENVECTORS" \
        $([ "$SKIP_QLEARNING" = true ] && echo "--skip_qlearning"))
echo "[3/3] Reward-shaping launcher submitted: job $RS_JID$([ -n "$PLOT_JID" ] && echo "  (after $PLOT_JID)")"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Jobs submitted."
[ -n "$TRAIN_JID" ] && echo "  [1] Training   : $TRAIN_JID"
[ -n "$PLOT_JID"  ] && echo "  [2] Plotting   : $PLOT_JID"
echo "  [3] RS launcher: $RS_JID"
echo ""
echo "  squeue -u \$USER"
echo "============================================================"
