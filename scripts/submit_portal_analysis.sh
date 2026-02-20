#!/bin/bash
#SBATCH --job-name=portal_analysis
#SBATCH --account=aip-machado
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/portal_analysis_%j.out
#SBATCH --error=logs/portal_analysis_%j.err

mkdir -p logs

# Load modules and activate environment
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 scipy-stack/2024b
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run portal analysis
cd ~/asymexp
python experiments/01/run_analysis.py "$@"
