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
# Number of nodes
#SBATCH --nodes=8
#
# Wall clock limit (1 day here):
#SBATCH --time=1-00:00:00
#
## Command(s) to run:

module load java spark
source /global/home/groups/allhands/bin/spark_helper.sh

spark-start

spark-submit --master $SPARK_URL analysis.py 

spark-stop
