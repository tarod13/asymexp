#!/bin/bash
#SBATCH --job-name=door_analysis
#SBATCH --account=aip-machado
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/door_analysis_%j.out
#SBATCH --error=logs/door_analysis_%j.err

mkdir -p logs

# Load modules and activate environment
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 scipy-stack/2024b
source ~/ENV/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run door analysis
cd ~/asymexp
python experiments/02/run_analysis.py "$@"
