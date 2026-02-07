#!/bin/bash
#SBATCH --job-name=lap_rep_training
#SBATCH --account=aip-machado
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-1
#SBATCH --output=logs/lap_rep_seed_%a.out
#SBATCH --error=logs/lap_rep_seed_%a.err

mkdir -p logs

SEED=$((42 + SLURM_ARRAY_TASK_ID))

echo "========================================"
echo "Starting Laplacian representation training"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Node: $SLURM_NODELIST"
echo "========================================"

module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 mujoco/3.3.0
source ~/ENV/bin/activate

# Dynamic allocation of GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run training
python train_lap_rep.py \
    --seed $SEED \
    --num_eigenvector_pairs 10 \
    --learning_rate 0.000001 \
    --ema_learning_rate 0.00001 \
    --batch_size 1024 \
    --num_gradient_steps 10000 \
    --gamma 0.95 \
    --lambda_x 10.0 \
    --chirality_factor 0.1

echo "========================================"
echo "Job completed for seed $SEED"
echo "Results saved to: results/ (see logs above)"
echo "========================================"