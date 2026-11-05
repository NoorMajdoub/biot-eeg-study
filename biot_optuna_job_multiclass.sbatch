#!/bin/bash
#SBATCH --account=def-amiilab_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -e
DATASET=""; ARM=""; FRACTION="100"; N_TRIALS="20"
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        --fraction=*) FRACTION="${arg#*=}" ;;
        --n_trials=*) N_TRIALS="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
source ~/envs/eeg-gpu/bin/activate
export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR
cd /home/nourmaj/BIOT
python /home/nourmaj/BIOT/biot_optuna_search_multiclass.py \
    --dataset "$DATASET" --arm "$ARM" --fraction "$FRACTION" --n_trials "$N_TRIALS"