#!/bin/bash
#
#SBATCH --job-name=gpu
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=64G
#SBATCH --constraint=GPU_SKU:RTX_2080Ti

ml reset
ml restore scnn
source venv/bin/activate

eval $JOB_STR
