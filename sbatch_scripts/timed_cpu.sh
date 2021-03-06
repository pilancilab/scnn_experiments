#!/bin/bash
#
#SBATCH --job-name=cpu
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --constraint=CPU_SKU:7502

ml reset
ml restore scnn

eval $JOB_STR
