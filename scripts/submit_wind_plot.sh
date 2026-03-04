#!/bin/bash
#SBATCH --job-name=lap_rep_training
#SBATCH --account=aip-machado
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-1
#SBATCH --output=logs/lap_rep_seed_%a.out
#SBATCH --error=logs/lap_rep_seed_%a.err

# =============================================================================
# Run the wind-sweep eigenvector visualisation.
#
# Usage
# -----
#   bash scripts/submit_wind_plot.sh [options forwarded to plot_wind_sweep.py]
#
# Examples
# --------
#   bash scripts/submit_wind_plot.sh
#   bash scripts/submit_wind_plot.sh --results_dir ./results/wind_sweep \
#                                    --output_dir  ./figures \
#                                    --num_eigvecs 4
#
# Produces (all in --output_dir):
#   gt_right_real.png   gt_left_real.png   allo_right_real.png
#   gt_right_imag.png   gt_left_imag.png
#   gt_right_abs.png    gt_left_abs.png
# =============================================================================

mkdir -p logs

module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 mujoco/3.3.0
source ~/ENV/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"

python scripts/plot_wind_sweep.py "$@"
