#!/bin/bash
#SBATCH --account=def-amiilab_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Runs ONE (dataset, arm, fraction) Optuna study for BIOT -- 20 trials by
# default, each trial up to 100 epochs (early-stopped on val_auroc,
# patience=5). Resumable via the study's SQLite file.
#
# Usage (via the wrapper, not directly):
#   bash submit_biot_optuna_job.sh --dataset=chbmit --arm=pretrained --fraction=50
#
# --dataset: one of chbmit, clinical, dream, karaone, yang2025, japaneeg
# --arm: pretrained or random
# --fraction: 100, 50, or 25
# --n_trials: number of Optuna trials (default 20)

set -e

DATASET=""
ARM=""
FRACTION="100"
N_TRIALS="20"
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        --fraction=*) FRACTION="${arg#*=}" ;;
        --n_trials=*) N_TRIALS="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
if [ -z "$DATASET" ] || [ -z "$ARM" ]; then
    echo "Missing required --dataset=... and/or --arm=... argument"; exit 1
fi

echo "=== BIOT Optuna search job ==="
echo "dataset=$DATASET arm=$ARM fraction=$FRACTION n_trials=$N_TRIALS"
echo "==============================="

source ~/envs/eeg-gpu/bin/activate

export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME

export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR

cd /home/nourmaj/BIOT

python /home/nourmaj/BIOT/biot_optuna_search.py \
    --dataset "$DATASET" --arm "$ARM" --fraction "$FRACTION" --n_trials "$N_TRIALS"