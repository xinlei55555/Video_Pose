#!/bin/bash
#SBATCH --job-name=heatmap_train          # Job name
#SBATCH --account=def-btaati              # Replace with your account
#SBATCH --time=00-02:00                   # Time limit (DD-HH:MM)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem-per-cpu=4000                  
#SBATCH --output=%x-%j.out                # Standard output and error log (%x = job name, %j = job ID)

module load python/3.10
module load opencv
module load gcc
module load cuda
source /home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/mamba_env2/bin/activate               # Activate your virtual environment

python DataFormat.py                    # Command to run your Python script
