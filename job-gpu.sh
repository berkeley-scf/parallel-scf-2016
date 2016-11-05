#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=co_stat
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# GPU:
#SBATCH --gres=gpu:1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify total number of processors twice of number of GPUs):
#SBATCH --cpus-per-task=2
#
# Wall clock limit (3 hours here):
#SBATCH --time=3:00:00
#
## Command(s) to run:

# usually need cuda loaded
module load cuda
# now run your GPU job
