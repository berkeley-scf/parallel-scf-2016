#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Number of nodes:
#SBATCH --nodes=1
#
# Processors per node:
#SBATCH --ntasks-per-node=8
#
# Wall clock limit:
#SBATCH --time=00:05:00
#
## Command(s) to run:
ipcluster start -n $SLURM_NTASKS_PER_NODE &
sleep 40
ipython parallel-analysis.py > parallel-analysis.pyout
