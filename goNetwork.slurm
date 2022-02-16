#!/bin/bash
#SBATCH -J trainNetwork             # Sensible name for the job
#SBATCH -N 1                   # Allocate 2 nodes for the job
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100"
#SBATCH -c 20
#SBATCH -t 2-24:00:00             # Upper time limit for the job
#SBATCH -p GPUQ
#SBATCH --mail-user=<kasperka@stud.ntnu.no>
#SBATCH --mail-type=ALL

module purge
module load Python/3.8.6-GCCcore-10.2.0
python3 unettrain.py 