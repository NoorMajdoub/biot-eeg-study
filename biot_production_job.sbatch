#!/bin/bash
#SBATCH --account=def-amiilab_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#
# Runs ONE production BIOT fine-tuning run (one dataset, one arm, one
# fraction, one seed) using the winning hyperparameters from the Optuna
# search, and reports real test_acc/test_bacc/test_pr_auc/test_auroc.
#
# Usage (via the wrapper, not directly):
#   bash submit_biot_production_job.sh --dataset=chbmit --arm=pretrained --fraction=50 --seed=1
#
# --dataset: one of chbmit, clinical, dream, karaone, yang2025, japaneeg
# --arm: pretrained or random
# --fraction: 100, 50, or 25
# --seed: 1, 2, or 42

set -e

DATASET=""
ARM=""
FRACTION=""
SEED=""
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        --fraction=*) FRACTION="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
for v in DATASET ARM FRACTION SEED; do
    if [ -z "${!v}" ]; then
        echo "Missing required --${v,,}=... argument"; exit 1
    fi
done

echo "=== BIOT production job ==="
echo "dataset=$DATASET arm=$ARM fraction=$FRACTION seed=$SEED"
echo "============================"

source $SCRATCH/envs/eeg-gpu/bin/activate
export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR
cd /home/nourmaj/BIOT

# Assumes biot_production_finetune.py is placed at
# /home/nourmaj/BIOT/biot_production_finetune.py -- update this path if
# you put it somewhere else.
python /home/nourmaj/BIOT/biot_production_finetune.py \
    --dataset "$DATASET" --arm "$ARM" --fraction "$FRACTION" --seed "$SEED"