#!/bin/bash
#SBATCH --account=def-amiilab_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#
# Extracts BIOT embeddings for every task x arm x seed x fraction.
# Pretrained encoder is deterministic -> seed 1 only. Random arm -> 3 random encoders.
# Already-extracted files are skipped, so you can resubmit after a timeout.
#   sbatch biot_lp_extract.sbatch
#   sbatch --export=ALL,TASKS="isruc",FRACTIONS="100" biot_lp_extract.sbatch

set -uo pipefail
TASKS=${TASKS:-"chbmit clinical karaone japaneeg dream yang2025 isruc"}
FRACTIONS=${FRACTIONS:-"100"}
OUT=/scratch/nourmaj/biot_lp_feats

source $SCRATCH/envs/eeg-gpu/bin/activate
export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR
export MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp

SCRIPT=/scratch/nourmaj/biot_extract_feats.py

for T in $TASKS; do
  for F in $FRACTIONS; do
    for ARM_SEED in "pretrained 1" "random 1" "random 2" "random 3"; do
      set -- $ARM_SEED
      FILE=$OUT/${T}_${1}_seed${2}_${F}pct.npz
      if [ -f "$FILE" ]; then echo "[skip] $FILE"; continue; fi
      python "$SCRIPT" --task "$T" --arm "$1" --seed "$2" --fraction "$F" \
        || echo "[FAILED] $T $1 seed $2 ${F}% (see .err)"
    done
  done
done
echo "===== DONE"; ls -lh $OUT
