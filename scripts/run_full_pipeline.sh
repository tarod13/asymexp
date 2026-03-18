#!/bin/bash
# =============================================================================
# Master pipeline: eigenvector training → visualization → reward shaping
#
# Executes three stages in strict order:
#   1. eval_parallel_train.sh  — train eigenvector models for each environment
#   2. eval_parallel_plot.sh   — visualize training results
#   3. submit_reward_shaping.sh — reward-shaping evaluation per environment
#
# On SLURM clusters: submits jobs with --dependency=afterok chaining.
# Locally:           runs training in parallel (& / wait), then sequentially.
#
# Usage:
#   bash scripts/run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --account STR             SLURM account (default: rrg-machado)
#   --base_dir PATH           Root output directory (default: ./results)
#   --exp_name STR            Experiment label (default: parallel_clf_eval)
#   --seed N                  Training seed (default: 42)
#   --envs "E1 E2 ..."        Space-separated env list (default: all 6)
#   --num_eigenvectors N      Eigenvector pairs — train + RS (default: 8)
#   --num_seeds N             Q-learning seeds per env (default: 100)
#   --num_episodes N          Q-learning episodes (default: 60000)
#   --max_steps N             Max steps per episode (default: 200)
#   --shaping_coef F          Reward shaping coefficient (default: 0.1)
#   --gamma_rl F              RL discount factor (default: 0.99)
#   --lr F                    Q-learning learning rate (default: 0.1)
#   --epsilon F               Exploration epsilon (default: 0.5)
#   --log_interval N          Logging interval (default: 500)
#   --eval_seed N             Evaluation seed (default: 0)
#   --num_eval_episodes N     Evaluation episodes (default: 30)
#   --min_goal_distance N     Minimum goal distance (default: 8)
#   --start_state "R,C"       Start state row,col (default: "1,1")
#   --skip_train              Skip stage 1 (reuse existing manifests)
#   --skip_plot               Skip stage 2
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Defaults ──────────────────────────────────────────────────────────────────
ACCOUNT="rrg-machado"
BASE_DIR="./results"
EXP_NAME="parallel_clf_eval"
SEED=42
NUM_EIGENVECTORS=8
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
SKIP_TRAIN=false
SKIP_PLOT=false
ENV_LIST="GridRoom-4 GridRoom-4-Doors GridRoom-1 GridRoom-1-Portals GridMaze-OGBench GridMaze-OGBench-Portals"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)            ACCOUNT="$2";            shift 2 ;;
        --base_dir)           BASE_DIR="$2";           shift 2 ;;
        --exp_name)           EXP_NAME="$2";           shift 2 ;;
        --seed)               SEED="$2";               shift 2 ;;
        --envs)               ENV_LIST="$2";           shift 2 ;;
        --num_eigenvectors)   NUM_EIGENVECTORS="$2";   shift 2 ;;
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
        --skip_train)         SKIP_TRAIN=true;         shift ;;
        --skip_plot)          SKIP_PLOT=true;          shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Derive paths (single source of truth) ─────────────────────────────────────
RESULTS_DIR="${BASE_DIR}/parallel_clf_eval"
MANIFEST_DIR="${BASE_DIR}/eval_manifest"
PLOTS_DIR="${BASE_DIR}/parallel_eval_plots"

# ── Export for sub-scripts ────────────────────────────────────────────────────
export PIPELINE_RESULTS_DIR="$RESULTS_DIR"
export PIPELINE_MANIFEST_DIR="$MANIFEST_DIR"
export PIPELINE_PLOTS_DIR="$PLOTS_DIR"
export PIPELINE_EXP_NAME="$EXP_NAME"
export PIPELINE_SEED="$SEED"
export PIPELINE_ENV_LIST="$ENV_LIST"

# ── Build environment array ───────────────────────────────────────────────────
read -ra ENVS <<< "$ENV_LIST"
NUM_ENVS=${#ENVS[@]}
MAX_ARRAY_ID=$((NUM_ENVS - 1))

mkdir -p logs "$MANIFEST_DIR"

# ── Detect execution mode ─────────────────────────────────────────────────────
if command -v sbatch &>/dev/null; then
    USE_SLURM=true
else
    USE_SLURM=false
fi

echo "============================================================"
echo "  Full Eigenvector Pipeline"
echo "  Account      : $ACCOUNT"
echo "  Base dir     : $BASE_DIR"
echo "  Exp name     : $EXP_NAME"
echo "  Seed         : $SEED"
echo "  Environments : ${ENVS[*]}"
echo "  Eigenvectors : $NUM_EIGENVECTORS"
echo "  RS seeds     : $NUM_SEEDS  |  episodes: $NUM_EPISODES"
echo "  Execution    : $([ "$USE_SLURM" = true ] && echo "SLURM" || echo "Local")"
echo "============================================================"

# =============================================================================
# SLURM path
# =============================================================================
if [ "$USE_SLURM" = true ]; then

    TRAIN_JID=""
    PLOT_JID=""

    # ── Stage 1: Training ─────────────────────────────────────────────────────
    if [ "$SKIP_TRAIN" = false ]; then
        echo ""
        echo "[1/3] Submitting training array ($NUM_ENVS environments)..."

        TRAIN_JID=$(sbatch --parsable \
            --account="$ACCOUNT" \
            --array=0-${MAX_ARRAY_ID} \
            --export=ALL \
            scripts/eval_parallel_train.sh)

        [ -z "$TRAIN_JID" ] && { echo "ERROR: Training submission failed." >&2; exit 1; }
        echo "      Job $TRAIN_JID  (array 0-$MAX_ARRAY_ID)"
    else
        echo "[1/3] Skipping training (--skip_train set)."
    fi

    # ── Stage 2: Plotting ─────────────────────────────────────────────────────
    if [ "$SKIP_PLOT" = false ]; then
        echo ""
        echo "[2/3] Submitting plotting job..."

        PLOT_SBATCH_ARGS=("--account=$ACCOUNT" "--export=ALL")
        [ -n "$TRAIN_JID" ] && PLOT_SBATCH_ARGS+=("--dependency=afterok:${TRAIN_JID}")

        PLOT_JID=$(sbatch --parsable "${PLOT_SBATCH_ARGS[@]}" scripts/eval_parallel_plot.sh)

        [ -z "$PLOT_JID" ] && { echo "ERROR: Plotting submission failed." >&2; exit 1; }
        PLOT_MSG="Job $PLOT_JID"
        [ -n "$TRAIN_JID" ] && PLOT_MSG+="  (depends on training $TRAIN_JID)"
        echo "      $PLOT_MSG"
    else
        echo "[2/3] Skipping plotting (--skip_plot set)."
    fi

    # ── Stage 3: Reward-shaping launcher ──────────────────────────────────────
    # A small launcher job is generated and submitted with a dependency on the
    # plot job.  It runs after the manifests exist and fires one
    # submit_reward_shaping.sh call per environment.
    echo ""
    echo "[3/3] Generating and submitting reward-shaping launcher..."

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RS_LAUNCHER="logs/rs_launcher_${TIMESTAMP}.sh"

    cat > "$RS_LAUNCHER" << LAUNCHER_EOF
#!/bin/bash
#SBATCH --job-name=rs_launcher
#SBATCH --account=${ACCOUNT}
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=logs/rs_launcher_${TIMESTAMP}.out
#SBATCH --error=logs/rs_launcher_${TIMESTAMP}.err

set -euo pipefail

# When running as a SLURM job, SLURM_SUBMIT_DIR is the sbatch invocation dir.
SCRIPT_DIR="\${SLURM_SUBMIT_DIR}/scripts"
ENV_LIST="${ENV_LIST}"
MANIFEST_DIR="${MANIFEST_DIR}"

read -ra ENVS <<< "\$ENV_LIST"

echo "========================================"
echo "Reward-Shaping Launcher"
echo "Environments : \${ENVS[*]}"
echo "Manifest dir : \$MANIFEST_DIR"
echo "========================================"

for ENV in "\${ENVS[@]}"; do
    MANIFEST_FILE="\${MANIFEST_DIR}/\${ENV}.txt"

    if [ ! -f "\$MANIFEST_FILE" ]; then
        echo "ERROR: Manifest not found for \$ENV at \$MANIFEST_FILE" >&2
        exit 1
    fi

    MODEL_DIR=\$(cat "\$MANIFEST_FILE")
    OUTPUT_DIR="\${MODEL_DIR}/reward_shaping"

    echo ""
    echo "--- \$ENV ---"
    echo "  Model dir  : \$MODEL_DIR"
    echo "  Output dir : \$OUTPUT_DIR"

    bash "\${SCRIPT_DIR}/submit_reward_shaping.sh" \\
        --model_dir         "\$MODEL_DIR" \\
        --output_dir        "\$OUTPUT_DIR" \\
        --num_seeds         "${NUM_SEEDS}" \\
        --num_episodes      "${NUM_EPISODES}" \\
        --max_steps         "${MAX_STEPS}" \\
        --shaping_coef      "${SHAPING_COEF}" \\
        --gamma_rl          "${GAMMA_RL}" \\
        --lr                "${LR}" \\
        --epsilon           "${EPSILON}" \\
        --log_interval      "${LOG_INTERVAL}" \\
        --eval_seed         "${EVAL_SEED}" \\
        --num_eval_episodes "${NUM_EVAL_EPISODES}" \\
        --min_goal_distance "${MIN_GOAL_DISTANCE}" \\
        --start_state       "${START_STATE}" \\
        --num_eigenvectors  "${NUM_EIGENVECTORS}"
done

echo ""
echo "All reward-shaping jobs submitted."
LAUNCHER_EOF

    chmod +x "$RS_LAUNCHER"

    LAUNCH_SBATCH_ARGS=("--account=$ACCOUNT" "--export=ALL")
    [ -n "$PLOT_JID" ] && LAUNCH_SBATCH_ARGS+=("--dependency=afterok:${PLOT_JID}")

    LAUNCH_JID=$(sbatch --parsable "${LAUNCH_SBATCH_ARGS[@]}" "$RS_LAUNCHER")

    [ -z "$LAUNCH_JID" ] && { echo "ERROR: rs_launcher submission failed." >&2; exit 1; }
    LAUNCH_MSG="Job $LAUNCH_JID"
    [ -n "$PLOT_JID" ] && LAUNCH_MSG+="  (depends on plot $PLOT_JID)"
    echo "      $LAUNCH_MSG"
    echo "      Launcher script: $RS_LAUNCHER"

    # ── Summary ───────────────────────────────────────────────────────────────
    echo ""
    echo "============================================================"
    echo "  Pipeline submitted."
    echo ""
    if [ "$SKIP_TRAIN" = false ]; then
        echo "  [1] Training array  : job $TRAIN_JID  ($NUM_ENVS envs, ~6 h each)"
    else
        echo "  [1] Training        : skipped"
    fi
    if [ "$SKIP_PLOT" = false ]; then
        echo "  [2] Plotting        : job $PLOT_JID"
    else
        echo "  [2] Plotting        : skipped"
    fi
    echo "  [3] RS launcher     : job $LAUNCH_JID"
    echo ""
    echo "  Outputs:"
    echo "    Training  → $RESULTS_DIR/"
    echo "    Manifests → $MANIFEST_DIR/"
    echo "    Figures   → $PLOTS_DIR/"
    echo "    RS        → <model_dir>/reward_shaping/  (per env)"
    echo ""
    echo "  Monitor:"
    echo "    squeue -u \$USER"
    echo "============================================================"

# =============================================================================
# Local path
# =============================================================================
else

    # ── Stage 1: Training (parallel background jobs) ──────────────────────────
    if [ "$SKIP_TRAIN" = false ]; then
        echo ""
        echo "[1/3] Running training locally ($NUM_ENVS environments in parallel)..."

        TRAIN_PIDS=()
        for i in $(seq 0 "$MAX_ARRAY_ID"); do
            SLURM_ARRAY_TASK_ID=$i \
                bash scripts/eval_parallel_train.sh \
                > "logs/par_eval_train_${i}.out" \
                2> "logs/par_eval_train_${i}.err" &
            TRAIN_PIDS+=($!)
            echo "  Started: ${ENVS[$i]}  (PID $!,  log: logs/par_eval_train_${i}.out)"
        done

        echo ""
        echo "  Waiting for all training jobs..."
        TRAIN_FAILED=false
        for pid in "${TRAIN_PIDS[@]}"; do
            if ! wait "$pid"; then
                echo "ERROR: Training job failed (PID $pid)." >&2
                TRAIN_FAILED=true
            fi
        done
        [ "$TRAIN_FAILED" = true ] && exit 1

        echo "[1/3] All training complete."
    else
        echo "[1/3] Skipping training (--skip_train set)."
    fi

    # ── Stage 2: Plotting ─────────────────────────────────────────────────────
    if [ "$SKIP_PLOT" = false ]; then
        echo ""
        echo "[2/3] Running plotting..."
        bash scripts/eval_parallel_plot.sh
        echo "[2/3] Plotting complete."
    else
        echo "[2/3] Skipping plotting (--skip_plot set)."
    fi

    # ── Stage 3: Reward shaping (one call per env, sequential) ───────────────
    echo ""
    echo "[3/3] Running reward shaping ($NUM_ENVS environments)..."

    for ENV in "${ENVS[@]}"; do
        MANIFEST_FILE="${MANIFEST_DIR}/${ENV}.txt"

        if [ ! -f "$MANIFEST_FILE" ]; then
            echo "ERROR: Manifest not found for $ENV at $MANIFEST_FILE" >&2
            exit 1
        fi

        MODEL_DIR=$(cat "$MANIFEST_FILE")
        OUTPUT_DIR="${MODEL_DIR}/reward_shaping"

        echo ""
        echo "  --- $ENV ---"
        echo "  Model dir  : $MODEL_DIR"
        echo "  Output dir : $OUTPUT_DIR"

        bash scripts/submit_reward_shaping.sh \
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
    echo "============================================================"
    echo "  Pipeline complete."
    echo "    Training  → $RESULTS_DIR/"
    echo "    Manifests → $MANIFEST_DIR/"
    echo "    Figures   → $PLOTS_DIR/"
    echo "    RS        → <model_dir>/reward_shaping/  (per env)"
    echo "============================================================"

fi
