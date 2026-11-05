"""
Flatten CBraMod's ISRUC sequence format (20 consecutive, non-overlapping
30s epochs per file) into individual single-epoch pkl files BIOT can consume
via a simple train/val/test folder convention (same pattern as CHB-MIT/JapanEEG).

Source: /scratch/nourmaj/scratch_iteration2/isruc_processed_80h/{seq,labels}/ISRUC-group1-{1..100}/*.npy
  each seq file: (20, 6, 6000) float64 -- 20 epochs x 6 channels x 6000 samples (30s @ 200Hz)
  each label file: (20,) int64 -- 5-class AASM sleep stage (0=Wake,1=N1,2=N2,3=N3,4=REM)

Sequences are confirmed NON-OVERLAPPING (verified directly from CBraMod's own
prepare_ISRUC_1.py: raw data is reshaped via consecutive, non-overlapping
chunks, no sliding window) -- so flattening into individual epochs introduces
no duplication or leakage.

Split by subject (matching CBraMod's own split exactly, for comparability):
  train: subjects 1-80
  val:   subjects 81-90
  test:  subjects 91-100
"""
import os
import glob
import pickle
import numpy as np

SRC_ROOT = "/scratch/nourmaj/scratch_iteration2/isruc_processed_80h"
OUT_ROOT = "/scratch/nourmaj/isruc_biot_processed"

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(OUT_ROOT, split), exist_ok=True)


def split_for_subject(n):
    if n <= 80:
        return "train"
    elif n <= 90:
        return "val"
    return "test"


total_counts = {"train": 0, "val": 0, "test": 0}

for subject_num in range(1, 101):
    split = split_for_subject(subject_num)
    seq_dir = os.path.join(SRC_ROOT, "seq", f"ISRUC-group1-{subject_num}")
    label_dir = os.path.join(SRC_ROOT, "labels", f"ISRUC-group1-{subject_num}")

    if not os.path.isdir(seq_dir):
        print(f"WARNING: missing seq dir for subject {subject_num}, skipping")
        continue

    seq_files = sorted(os.listdir(seq_dir))
    n_saved = 0
    for seq_fname in seq_files:
        seq_path = os.path.join(seq_dir, seq_fname)
        label_path = os.path.join(label_dir, seq_fname)
        if not os.path.exists(label_path):
            print(f"WARNING: no matching label for {seq_path}, skipping")
            continue

        seq = np.load(seq_path)      # (20, 6, 6000)
        labels = np.load(label_path)  # (20,)

        if seq.shape[0] != labels.shape[0]:
            print(f"WARNING: shape mismatch {seq_path} ({seq.shape}) vs "
                  f"{label_path} ({labels.shape}), skipping")
            continue

        base = seq_fname.replace(".npy", "")
        for epoch_idx in range(seq.shape[0]):
            X = seq[epoch_idx].astype(np.float32)  # (6, 6000)
            y = int(labels[epoch_idx])
            out = {"X": X, "y": y}
            out_fname = os.path.join(OUT_ROOT, split, f"{base}_e{epoch_idx}.pkl")
            with open(out_fname, "wb") as f:
                pickle.dump(out, f)
            n_saved += 1

    total_counts[split] += n_saved
    print(f"subject {subject_num} ({split}): {n_saved} epochs saved")

print("DONE")
print("total_counts:", total_counts)