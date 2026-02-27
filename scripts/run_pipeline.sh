#!/bin/bash
# Full training + reward-shaping pipeline.
#
# Steps
# -----
# 1. Train the ALLO representation          (train_allo_rep.py)
# 2. Train the complex representation        (train_lap_rep.py)
# 3. Run reward-shaping Q-learning comparing
#      baseline | shaped-ALLO | shaped-complex
#    and generate a single visualisation.
#
# Usage
# -----
#   bash scripts/run_pipeline.sh [options]
#
# Options (all optional, with defaults shown):
#   --seed           SEED        Random seed for both training runs  [42]
#   --steps          N           Gradient steps for each training run [100000]
#   --results_dir    DIR         Where to save training outputs       [./results]
#   --env_file_name  ENV         Environment name                     [GridRoom-4-Doors]
#   --shaping_coef   BETA        Reward-shaping coefficient           [0.1]
#   --num_episodes   N           Q-learning episodes per seed         [3000]
#   --num_seeds      N           Number of Q-learning seeds           [5]
#   --skip_allo                  Skip allo training (reuse existing)
#   --skip_complex               Skip complex training (reuse existing)
#   --allo_dir       DIR         Pre-existing allo model dir (sets --skip_allo)
#   --complex_dir    DIR         Pre-existing complex model dir (sets --skip_complex)
#
# Examples
# --------
#   # Full pipeline from scratch
#   bash scripts/run_pipeline.sh
#
#   # Quick smoke-test with fewer gradient steps and episodes
#   bash scripts/run_pipeline.sh --steps 5000 --num_episodes 500 --num_seeds 2
#
#   # Reuse pre-trained models, just re-run reward shaping
#   bash scripts/run_pipeline.sh \
#       --allo_dir    ./results/file/my_allo_run \
#       --complex_dir ./results/file/my_complex_run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Default parameters ────────────────────────────────────────────────────────
SEED=42
STEPS=100000
RESULTS_DIR="./results"
ENV_FILE_NAME="GridRoom-4-Doors"
SHAPING_COEF=0.1
NUM_EPISODES=3000
NUM_SEEDS=5
SKIP_ALLO=false
SKIP_COMPLEX=false
ALLO_DIR=""
COMPLEX_DIR=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)           SEED="$2";          shift 2 ;;
        --steps)          STEPS="$2";         shift 2 ;;
        --results_dir)    RESULTS_DIR="$2";   shift 2 ;;
        --env_file_name)  ENV_FILE_NAME="$2"; shift 2 ;;
        --shaping_coef)   SHAPING_COEF="$2";  shift 2 ;;
        --num_episodes)   NUM_EPISODES="$2";  shift 2 ;;
        --num_seeds)      NUM_SEEDS="$2";     shift 2 ;;
        --skip_allo)      SKIP_ALLO=true;     shift ;;
        --skip_complex)   SKIP_COMPLEX=true;  shift ;;
        --allo_dir)       ALLO_DIR="$2";      SKIP_ALLO=true;    shift 2 ;;
        --complex_dir)    COMPLEX_DIR="$2";   SKIP_COMPLEX=true; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "============================================================"
echo " Pipeline configuration"
echo "============================================================"
echo "  seed          : $SEED"
echo "  steps         : $STEPS"
echo "  results_dir   : $RESULTS_DIR"
echo "  env           : $ENV_FILE_NAME"
echo "  shaping_coef  : $SHAPING_COEF"
echo "  num_episodes  : $NUM_EPISODES"
echo "  num_seeds     : $NUM_SEEDS"
echo "  skip_allo     : $SKIP_ALLO"
echo "  skip_complex  : $SKIP_COMPLEX"
[[ -n "$ALLO_DIR"    ]] && echo "  allo_dir      : $ALLO_DIR"
[[ -n "$COMPLEX_DIR" ]] && echo "  complex_dir   : $COMPLEX_DIR"
echo "============================================================"

# ── Helper: find most recently modified subdirectory ─────────────────────────
latest_dir() {
    # Usage: latest_dir <base_dir> <exp_name_fragment>
    local base="$1"
    local fragment="$2"
    # find directories whose name contains the fragment, sort by modification
    # time (newest first), print the first one
    find "$base" -maxdepth 1 -type d -name "*${fragment}*" \
        | xargs -I{} stat --format="%Y %n" {} 2>/dev/null \
        | sort -rn \
        | awk 'NR==1{print $2}'
}

# ── Step 1: Train ALLO representation ────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 1: ALLO representation training"
echo "============================================================"

if $SKIP_ALLO; then
    if [[ -z "$ALLO_DIR" ]]; then
        ALLO_DIR="$(latest_dir "${RESULTS_DIR}/file" "__allo__")"
        if [[ -z "$ALLO_DIR" ]]; then
            echo "ERROR: --skip_allo set but no allo model found under ${RESULTS_DIR}/file/" >&2
            exit 1
        fi
    fi
    echo "  Skipping training.  Using: $ALLO_DIR"
else
    python train_allo_rep.py \
        --env_type              file \
        --env_file_name         "$ENV_FILE_NAME" \
        --num_gradient_steps    "$STEPS" \
        --batch_size            256 \
        --num_eigenvector_pairs 8 \
        --learning_rate         0.00001 \
        --ema_learning_rate     0.0003 \
        --lambda_x              10.0 \
        --chirality_factor      0.1 \
        --gamma                 0.9 \
        --no-use_rejection_sampling \
        --constraint_mode       same_episodes \
        --use_residual \
        --use_layernorm \
        --num_envs              1000 \
        --num_steps             1000 \
        --seed                  "$SEED" \
        --results_dir           "$RESULTS_DIR" \
        --exp_name              allo

    # Locate the directory that was just created
    ALLO_DIR="$(latest_dir "${RESULTS_DIR}/file" "__allo__")"
    if [[ -z "$ALLO_DIR" ]]; then
        echo "ERROR: Could not locate allo training output under ${RESULTS_DIR}/file/" >&2
        exit 1
    fi
    echo "  ALLO training complete.  Output: $ALLO_DIR"
fi

# ── Step 2: Train complex representation ─────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 2: Complex representation training"
echo "============================================================"

if $SKIP_COMPLEX; then
    if [[ -z "$COMPLEX_DIR" ]]; then
        COMPLEX_DIR="$(latest_dir "${RESULTS_DIR}/file" "__complex__")"
        if [[ -z "$COMPLEX_DIR" ]]; then
            echo "ERROR: --skip_complex set but no complex model found under ${RESULTS_DIR}/file/" >&2
            exit 1
        fi
    fi
    echo "  Skipping training.  Using: $COMPLEX_DIR"
else
    python train_lap_rep.py \
        --env_type              file \
        --env_file_name         "$ENV_FILE_NAME" \
        --num_gradient_steps    "$STEPS" \
        --batch_size            256 \
        --num_eigenvector_pairs 8 \
        --learning_rate         0.00001 \
        --ema_learning_rate     0.0003 \
        --lambda_x              10.0 \
        --chirality_factor      0.1 \
        --gamma                 0.9 \
        --no-use_rejection_sampling \
        --constraint_mode       same_episodes \
        --use_residual \
        --use_layernorm \
        --num_envs              1000 \
        --num_steps             1000 \
        --seed                  "$SEED" \
        --results_dir           "$RESULTS_DIR" \
        --exp_name              complex

    COMPLEX_DIR="$(latest_dir "${RESULTS_DIR}/file" "__complex__")"
    if [[ -z "$COMPLEX_DIR" ]]; then
        echo "ERROR: Could not locate complex training output under ${RESULTS_DIR}/file/" >&2
        exit 1
    fi
    echo "  Complex training complete.  Output: $COMPLEX_DIR"
fi

# ── Step 3: Reward shaping experiment ────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 3: Reward shaping experiment"
echo "   model_dir (complex) : $COMPLEX_DIR"
echo "   allo_model_dir      : $ALLO_DIR"
echo "============================================================"

OUTPUT_DIR="${COMPLEX_DIR}/reward_shaping"

python experiments/reward_shaping/run_reward_shaping.py \
    --model_dir      "$COMPLEX_DIR" \
    --allo_model_dir "$ALLO_DIR" \
    --num_episodes   "$NUM_EPISODES" \
    --num_seeds      "$NUM_SEEDS" \
    --shaping_coef   "$SHAPING_COEF" \
    --max_steps      500 \
    --gamma_rl       0.99 \
    --lr             0.1 \
    --epsilon        0.1 \
    --output_dir     "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo " Pipeline complete!"
echo "  Results  : $OUTPUT_DIR"
echo "  Plot     : $OUTPUT_DIR/learning_curves.png"
echo "============================================================"
