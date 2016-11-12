#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=co_stat
#
# Partition:
#SBATCH --partition=savio2
#
# Number of processors
#SBATCH --ntasks=48
#
# Wall clock limit:
#SBATCH --time=00:05:00
#
## Command(s) to run:
module load python/2.7.8 pandas scipy ipython gcc openmpi
ipcontroller --ip='*' &
sleep 20
# srun here will start as many engines as SLURM tasks
srun ipengine &   
sleep 50  # wait until all engines have successfully started
export DATADIR=/global/scratch/paciorek
ipython parallel-analysis.py > parallel-analysis.pyout
