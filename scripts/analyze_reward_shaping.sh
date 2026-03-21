#!/bin/bash
#SBATCH --job-name=rs_analyze
#SBATCH --account=rrg-machado
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rs_analyze_%j.out
#SBATCH --error=logs/rs_analyze_%j.err

# =============================================================================
# Analysis job: aggregate partial Q-learning results from the distributed array
# job and produce the combined learning-curve figure.
#
# This job is typically submitted with --dependency=afterok:<array_job_id> by
# scripts/run_reward_shaping_array.sh so it runs only after every Q-learning task
# has finished successfully.
#
# Configuration via environment variables (exported by run_reward_shaping_array.sh):
#   Required : OUTPUT_DIR
#
# Usage
# -----
#   sbatch --dependency=afterok:<array_job_id> --export=ALL \
#          scripts/analyze_reward_shaping.sh
#   bash   scripts/analyze_reward_shaping.sh   # local (env vars must be set)
# =============================================================================

mkdir -p logs

# ── Environment setup ─────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 mujoco/3.3.0
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "========================================"
echo "Reward-shaping analysis"
echo "  Output dir : ${OUTPUT_DIR:-<unset>}"
echo "========================================"

python experiments/reward_shaping/analyze_reward_shaping.py \
    --output_dir "$OUTPUT_DIR"

echo "========================================"
echo "Analysis complete.  Results in: $OUTPUT_DIR"
echo "========================================"
