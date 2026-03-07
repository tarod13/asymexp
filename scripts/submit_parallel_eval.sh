#!/bin/bash
# =============================================================================
# Master submission script for the parallel representation evaluation pipeline.
#
# Submits two jobs:
#   1. eval_parallel_train.sh — SLURM array (one job per environment, no IS)
#   2. eval_parallel_plot.sh  — depends on ALL training jobs succeeding
#
# Usage:
#   bash scripts/submit_parallel_eval.sh
#
# Optional environment variables:
#   ACCOUNT   — SLURM account override (default: from script headers)
#   PARTITION — SLURM partition override
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p logs results/eval_manifest

echo "============================================================"
echo "  Parallel Representation Evaluation — Pipeline Submission"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Submit training array
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Submitting training array job (11 environments)…"

TRAIN_JID=$(sbatch --parsable scripts/eval_parallel_train.sh)

if [ -z "$TRAIN_JID" ]; then
    echo "ERROR: Failed to submit training job." >&2
    exit 1
fi

echo "      Submitted: job $TRAIN_JID  (array 0-10)"

# ---------------------------------------------------------------------------
# 2. Submit plotting job (afterok = all array tasks must succeed)
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Submitting plotting job (depends on training array)…"

PLOT_JID=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JID} \
    scripts/eval_parallel_plot.sh)

if [ -z "$PLOT_JID" ]; then
    echo "ERROR: Failed to submit plotting job." >&2
    exit 1
fi

echo "      Submitted: job $PLOT_JID  (depends on $TRAIN_JID)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline submitted successfully."
echo ""
echo "  Training array job : $TRAIN_JID  (11 envs, ~6 h each)"
echo "  Plotting job       : $PLOT_JID   (starts after all training)"
echo ""
echo "  Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f logs/par_eval_train_0.out"
echo ""
echo "  Outputs:"
echo "    Training  → ./results/parallel_eval/"
echo "    Manifests → ./results/eval_manifest/"
echo "    Figures   → ./results/parallel_eval_plots/"
echo "============================================================"
