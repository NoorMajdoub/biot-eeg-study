#!/bin/bash
#SBATCH --account=def-amiilab_gpu

#SBATCH --gres=gpu:h100:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --time=01:00:00



cd /home/nourmaj/BIOT

/scratch/nourmaj/envs/eeg-gpu/bin/python biot_production_finetune_multiclass.py "$@"

