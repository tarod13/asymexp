#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --account=aip-machado
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# =============================================================================
# SLURM submission script for asymexp analysis tasks
#
# This script runs analysis for portal, door, or sweep experiments.
#
# Usage:
#   sbatch scripts/submit_analysis.sh portal --num-envs 20 --k 30
#   sbatch scripts/submit_analysis.sh door --num-doors 10 --num-envs 50
#   sbatch scripts/submit_analysis.sh sweep --exp_name batch_lr_sweep
#
# Or set custom resources:
#   sbatch --mem=32G --time=8:00:00 scripts/submit_analysis.sh sweep --exp_name batch_lr_sweep
# =============================================================================

set -e  # Exit on error

mkdir -p logs

echo "========================================"
echo "Running analysis on SLURM"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Time limit:    $SLURM_TIMELIMIT"
echo "Arguments:     $@"
echo "========================================"

# Load required modules
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 scipy-stack/2024b

# Activate virtual environment (has JAX)
source ~/ENV/bin/activate

# Prevent JAX from grabbing all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Set project root for analysis scripts
export ASYMEXP_ROOT="$HOME/asymexp"

# Run the analysis using run_analysis.sh
cd "$ASYMEXP_ROOT"
bash scripts/run_analysis.sh "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "Analysis completed successfully!"
    echo "========================================"
else
    echo "========================================"
    echo "Analysis failed with exit code $EXIT_CODE"
    echo "========================================"
    exit $EXIT_CODE
fi
