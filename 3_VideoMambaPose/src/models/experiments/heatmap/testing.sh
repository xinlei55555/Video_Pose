#!/bin/bash
#SBATCH --job-name=model_info_and_shape   # Job name
#SBATCH --account=def-btaati              # Replace with your account
#SBATCH --time=00-03:00                   # Time limit (DD-HH:MM)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:1                      # Number of GPUs per node 
#SBATCH --mem-per-gpu=32000M                    # Total memory per node (less means faster allocation) 
#SBATCH --output=test-output/%x-%j.out                # Standard output and error log (%x = job name, %j = job ID)

# module load python/3.10
# module load opencv
# module load gcc
#module load cudacore/.11.8.0
#source /home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/mamba_env2/bin/activate               # Activate your virtual environment
module restore mamba_modules125
source /home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/mamba_env125/bin/activate
wandb offline
python HeatMapTrain.py                    # Command to run your Python script
