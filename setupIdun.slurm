#!/bin/bash                    # This is the shell
#SBATCH -J trainNetwork        # Sensible name for the job
#SBATCH -N 1                   # Nodes requested
#SBATCH --ntasks-per-node=1    # Sets one task per node
#SBATCH --gres=gpu:1           # Number of GPU requested per node
#SBATCH -c 20                  # Sets number of cores
#SBATCH --constraint="A100"    # Any job constraints, here the A100 GPU is needed
#SBATCH -t 2-24:00:00          # Upper time-limit for the job
#SBATCH -p GPUQ                # Set as either GPUQ or CPUQ
#SBATCH --mail-user=<olanordmann@stud.ntnu.no> # Where to be notified
#SBATCH --mail-type=ALL        # ALL notifies when job starts and ends

module purge                   # Unloads all modules from users enviroment
module load Python/3.8.6-GCCcore-10.2.0 # Loads a given configuration
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 matplotlib==3.4.2 
python3 program1.py           # Runs the given program(s)
python3 program2.py

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html matplotlib==3.4.2 torchgeometry==0.1.2 albumentations==0.5.2 seaborn==0.11.2 pillow=8.3.2

