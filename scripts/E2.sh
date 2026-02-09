#!/bin/bash
#SBATCH --job-name=lap_rep_training
#SBATCH --account=aip-machado
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-5
#SBATCH --output=logs/lap_rep_seed_%a.out
#SBATCH --error=logs/lap_rep_seed_%a.err

mkdir -p logs

learning_rates=(0.01 0.001 0.0001 0.00001 0.000001 0.0000001)
LR=${learning_rates[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Starting Laplacian representation training"
echo "Experiment: Learning Rate Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Learning Rate: $LR"
echo "Node: $SLURM_NODELIST"
echo "========================================"

module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 mujoco/3.3.0
source ~/ENV/bin/activate

# Dynamic allocation of GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run training
python train_lap_rep.py \
    --seed 42 \
    --num_eigenvector_pairs 8 \
    --learning_rate $LR \
    --ema_learning_rate 0.01 \
    --batch_size 256 \
    --num_gradient_steps 100000 \
    --gamma 0.9 \
    --lambda_x 10.0 \
    --chirality_factor 0.1

echo "========================================"
echo "Job completed for seed $SEED"
echo "Results saved to: results/ (see logs above)"
echo "========================================"