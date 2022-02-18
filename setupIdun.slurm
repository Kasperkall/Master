#!/bin/bash
#SBATCH -J trainNetwork             # Sensible name for the job
#SBATCH -N 1                   # Allocate 2 nodes for the job
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --constraint="A100"
#SBATCH -t 2-24:00:00             # Upper time limit for the job
#SBATCH -p GPUQ
#SBATCH --mail-user=<kasperka@stud.ntnu.no>
#SBATCH --mail-type=ALL

module purge
module load Python/3.8.6-GCCcore-10.2.0
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html matplotlib==3.4.2 torchgeometry==0.1.2 albumentations==0.5.2 seaborn==0.11.2
python3 unettrain.py 