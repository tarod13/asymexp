#!/bin/bash
# =============================================================================
# Iterate over manifest files and submit reward-shaping jobs for each env.
#
# This script runs as a short SLURM job (after the plot job finishes) and
# calls submit_reward_shaping.sh once per manifest entry.
#
# Usage (from run_full_pipeline.sh):
#   sbatch --dependency=afterok:<plot_jid> \
#       scripts/run_reward_shaping_per_env.sh \
#       [--manifest_dir PATH] [--num_seeds N] [--num_episodes N] \
#       [--max_steps N] [--shaping_coef F] [--gamma_rl F] [--lr F] \
#       [--epsilon F] [--log_interval N] [--eval_seed N] \
#       [--num_eval_episodes N] [--min_goal_distance N] \
#       [--start_state "R,C"] [--num_eigenvectors N]
# =============================================================================

#SBATCH --job-name=rs_per_env
#SBATCH --account=rrg-machado
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=logs/rs_per_env.out
#SBATCH --error=logs/rs_per_env.err

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
ACCOUNT="rrg-machado"
MANIFEST_DIR="./results/eval_manifest"
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)            ACCOUNT="$2";            shift 2 ;;
        --manifest_dir)       MANIFEST_DIR="$2";       shift 2 ;;
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

SUBMIT_RS="${SLURM_SUBMIT_DIR}/scripts/submit_reward_shaping.sh"

echo "========================================"
echo "Reward-shaping per-env launcher"
echo "Job ID       : $SLURM_JOB_ID"
echo "Manifest dir : $MANIFEST_DIR"
echo "========================================"

if [ ! -d "$MANIFEST_DIR" ] || [ -z "$(ls -A "$MANIFEST_DIR"/*.txt 2>/dev/null)" ]; then
    echo "ERROR: No manifest files found in $MANIFEST_DIR" >&2
    exit 1
fi

for MANIFEST_FILE in "$MANIFEST_DIR"/*.txt; do
    ENV=$(basename "$MANIFEST_FILE" .txt)
    export ENV
    MODEL_DIR=$(cat "$MANIFEST_FILE")
    OUTPUT_DIR="${MODEL_DIR}/reward_shaping"

    echo ""
    echo "--- $ENV ---"
    echo "  Model dir  : $MODEL_DIR"
    echo "  Output dir : $OUTPUT_DIR"

    bash "$SUBMIT_RS" \
        --account           "$ACCOUNT" \
        --model_dir         "$MODEL_DIR" \
        --output_dir        "$OUTPUT_DIR" \
        --num_seeds         "$NUM_SEEDS" \
        --num_episodes      "$NUM_EPISODES" \
        --max_steps         "$MAX_STEPS" \
        --shaping_coef      "$SHAPING_COEF" \
        --gamma_rl          "$GAMMA_RL" \
        --lr                "$LR" \
        --epsilon           "$EPSILON" \
        --log_interval      "$LOG_INTERVAL" \
        --eval_seed         "$EVAL_SEED" \
        --num_eval_episodes "$NUM_EVAL_EPISODES" \
        --min_goal_distance "$MIN_GOAL_DISTANCE" \
        --start_state       "$START_STATE" \
        --num_eigenvectors  "$NUM_EIGENVECTORS"
done

echo ""
echo "All reward-shaping jobs submitted."
