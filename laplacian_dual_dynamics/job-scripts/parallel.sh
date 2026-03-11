#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

#SBATCH --array=0-109

envs=("GridRoom-64" "GridRoom-16" "GridMaze-19" "GridRoom-1" "GridRoom-4" "GridRoomSym-4" "GridMaze-7" "GridMaze-17" "GridMaze-9" "GridMaze-32")
ws=(20.0 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01)
configs=("coefficient_augmented_martin.yaml")

R_ENV=$((${SLURM_ARRAY_TASK_ID} % 110))
N_ENV=$((${R_ENV} / 11))
R_W=$((${R_ENV} % 11))
N_CONFIG=0
N_W=$((${R_W} / 1))

SEEDS="./rl_lap/hyperparam/seed_list_min.txt"
CONFIG=${configs[$N_CONFIG]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=${envs[$N_ENV]}
W=${ws[$N_W]}

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/off --config_file $CONFIG --env_name {3} --seed {1} --regularization_weight {2} --total_train_steps {4} --max_episode_steps {5} --lr {6} --batch_size {7} --discount {8} --n_samples {9} :::: $SEEDS ::: $W ::: $ENV ::: 8000000 ::: 50 ::: 0.0001 ::: 32 ::: 0.9 ::: 3000000
#apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B /project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/lk_haiku.sif python3 train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/off --config_file coefficient_augmented_martin.yaml --env_name GridRoom-1 --seed 1234 --regularization_weight 5.0 --total_train_steps 400000 --max_episode_steps 1000000 --lr 0.0001 --hidden_dims "256 256" --batch_size 32 --discount 0.9 --n_samples 2000000
#apptainer shell --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B /project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/lk_haiku.sif
