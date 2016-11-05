#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_paciorek
#
# Partition:
#SBATCH --partition=savio2
#
# Number of processors
#SBATCH --ntasks=48
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run:
module load python/2.7.8 ipython gcc openmpi
ipcontroller --ip='*' &
sleep 5
# srun here should start as many engines as tasks
srun ipengine &   
sleep 15  # wait until all engines have successfully started
ipython parallel-analysis.py
