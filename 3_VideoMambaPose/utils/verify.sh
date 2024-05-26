#!/bin/bash
#SBATCH --job-name=real_heatmap_train          # Job name
#SBATCH --account=def-btaati              # Replace with your account
#SBATCH --time=00-12:00                   # Time limit (DD-HH:MM)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:1                      # Number of GPUs per node 
#SBATCH --mem=4000M                    # Total memory per node (less means faster allocation) 
#SBATCH --output=outputs/%x-%j.out                # Standard output and error log (%x = job name, %j = job ID)

source /home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/mamba_env118/bin/activate
module restore mamba_modules
python verify_installation.py                    # Command to run your Python script
