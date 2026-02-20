#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --account=aip-machado
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# Usage:
#   sbatch scripts/submit_analysis.sh portal --num-envs 20 --k 30
#   sbatch scripts/submit_analysis.sh door --num-doors 10 --num-envs 50
#   sbatch scripts/submit_analysis.sh sweep --exp_name batch_lr_sweep

mkdir -p logs

ANALYSIS_TYPE=$1
shift

# Load modules and activate environment
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 scipy-stack/2024b
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the appropriate analysis
cd ~/asymexp

case "$ANALYSIS_TYPE" in
    portal)
        python experiments/01/run_analysis.py "$@"
        ;;
    door)
        python experiments/02/run_analysis.py "$@"
        ;;
    sweep)
        python scripts/analyze_sweep.py "$@"
        ;;
    *)
        echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
        echo "Usage: sbatch scripts/submit_analysis.sh [portal|door|sweep] [options...]"
        exit 1
        ;;
esac
