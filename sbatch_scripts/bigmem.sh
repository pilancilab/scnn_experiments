#!/bin/bash
#
#SBATCH --job-name=bigmem
#SBATCH --time=24:00:00
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

ml reset
ml restore scnn
source venv/bin/activate

eval $JOB_STR
