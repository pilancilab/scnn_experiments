#!/bin/bash
#
#SBATCH --job-name=cpu
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G


ml reset
ml restore teleport
source .venv/bin/activate

eval "$JOB_STR -I $SLURM_ARRAY_TASK_ID"
