#!/bin/bash
# =============================================================================
# Combined eigenvector visualization across all environments.
#
# This job should be submitted with a dependency on the full training array
# so that it runs only after all environments have been trained:
#
#   TRAIN_JID=$(sbatch --parsable scripts/eval_parallel_train.sh)
#   sbatch --dependency=afterok:${TRAIN_JID} scripts/eval_parallel_plot.sh
#
# The easy way to do both at once:
#   bash scripts/submit_parallel_eval.sh
#
# What this script does:
#   Calls plot_parallel_eval.py which reads the manifest files written by
#   each training job and produces 16 combined figures (8 learned + 8 ground-
#   truth), one per component type:
#     right-real, right-imag, left-real, left-imag,
#     right-magnitude, right-phase, left-magnitude, left-phase
#   Each figure has eigenvectors as rows and environments as columns.
#
# Output:
#   ./results/parallel_eval_plots/*.png
# =============================================================================

#SBATCH --job-name=par_eval_plot
#SBATCH --account=rrg-machado
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/par_eval_plot.out
#SBATCH --error=logs/par_eval_plot.err

mkdir -p logs

echo "========================================"
echo "Parallel Eval — Combined Eigenvector Plots"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "========================================"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 scipy-stack/2024b
source ~/ENV/bin/activate

# JAX on CPU is fine for plotting (no gradient computation)
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------------------------------------------------------------------------
# Path overrides — exported by run_full_pipeline.sh; fall back to defaults.
# ---------------------------------------------------------------------------
MANIFEST_DIR="${PIPELINE_MANIFEST_DIR:-./results/eval_manifest}"
PLOTS_DIR="${PIPELINE_PLOTS_DIR:-./results/parallel_eval_plots}"

# ---------------------------------------------------------------------------
# Sanity check: at least one manifest file exists
# ---------------------------------------------------------------------------
if [ ! -d "$MANIFEST_DIR" ] || [ -z "$(ls -A $MANIFEST_DIR/*.txt 2>/dev/null)" ]; then
    echo "ERROR: No manifest files found in $MANIFEST_DIR" >&2
    echo "Make sure the training array job (eval_parallel_train.sh) completed successfully." >&2
    exit 1
fi

echo "Manifest files found:"
ls "$MANIFEST_DIR"/*.txt

# ---------------------------------------------------------------------------
# Run plotting
# ---------------------------------------------------------------------------
python plot_parallel_eval.py \
    --manifest-dir  "$MANIFEST_DIR" \
    --output-dir    "$PLOTS_DIR" \
    --subplot-size  2.2 \
    --dpi           150

echo "========================================"
echo "Plotting complete."
echo "Figures saved to: $PLOTS_DIR/"
echo "========================================"
