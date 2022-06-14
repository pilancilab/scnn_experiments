#!/bin/bash
#
#SBATCH --job-name=pilanci_big_mem
#SBATCH --time=24:00:00
#SBATCH --partition=pilanci
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=48G

ml reset
ml restore convex_nn

eval $JOB_STR
