#!/bin/bash
#SBATCH --job-name=sweep_analysis
#SBATCH --account=aip-machado
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/sweep_analysis_%j.out
#SBATCH --error=logs/sweep_analysis_%j.err

mkdir -p logs

# Load modules and activate environment
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 scipy-stack/2024b
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run sweep analysis
cd ~/asymexp
python scripts/analyze_sweep.py "$@"
