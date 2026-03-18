#!/bin/bash
# =============================================================================
# Combined eigenvector visualization across all environments.
#
# Usage (standalone):
#   sbatch scripts/eval_parallel_plot.sh
#
# Usage (from run_full_pipeline.sh):
#   sbatch --dependency=afterok:<train_jid> \
#       scripts/eval_parallel_plot.sh \
#       [--manifest_dir PATH] [--plots_dir PATH]
# =============================================================================

#SBATCH --job-name=par_eval_plot
#SBATCH --account=rrg-machado
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/par_eval_plot.out
#SBATCH --error=logs/par_eval_plot.err

mkdir -p logs

# ── Arguments (all optional; defaults match standalone behaviour) ──────────────
MANIFEST_DIR="./results/eval_manifest"
PLOTS_DIR="./results/parallel_eval_plots"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest_dir) MANIFEST_DIR="$2"; shift 2 ;;
        --plots_dir)    PLOTS_DIR="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "========================================"
echo "Parallel Eval — Combined Eigenvector Plots"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "========================================"

# ── Environment setup ──────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 scipy-stack/2024b
source ~/ENV/bin/activate

# JAX on CPU is fine for plotting (no gradient computation)
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ── Sanity check: at least one manifest file exists ───────────────────────────
if [ ! -d "$MANIFEST_DIR" ] || [ -z "$(ls -A "$MANIFEST_DIR"/*.txt 2>/dev/null)" ]; then
    echo "ERROR: No manifest files found in $MANIFEST_DIR" >&2
    echo "Make sure the training array job (eval_parallel_train.sh) completed successfully." >&2
    exit 1
fi

echo "Manifest files found:"
ls "$MANIFEST_DIR"/*.txt

# ── Run plotting ───────────────────────────────────────────────────────────────
python plot_parallel_eval.py \
    --manifest-dir  "$MANIFEST_DIR" \
    --output-dir    "$PLOTS_DIR" \
    --subplot-size  2.2 \
    --dpi           150

echo "========================================"
echo "Plotting complete."
echo "Figures saved to: $PLOTS_DIR/"
echo "========================================"
