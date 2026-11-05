import pickle
import csv
import glob
import torch
import numpy as np
import torch.nn.functional as F
import os
from scipy.signal import resample
from scipy.signal import butter, iirnotch, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
import json
import mne

class SleepEDFLoader(torch.utils.data.Dataset):
    """
    Sleep-EDF 5-class sleep staging (W/N1/N2/N3/REM). seq/<rec>.npy
    (n_epochs, 2, 6000) float32 @ 200Hz (Fpz-Cz, Pz-Oz; 30s epochs),
    labels/<rec>.npy (n_epochs,) int64 (0=W,1=N1,2=N2,3=N3,4=REM),
    epoch_qc/<rec>.npy (n_epochs,) float32 -- a CONTINUOUS fraction of
    artifact/noise per epoch (observed range 0 to ~0.96, not binary).
    qc_threshold excludes any epoch above that fraction; 0.2 is an
    assumed default.
    """
    def __init__(self, root, recordings, sampling_rate=200, qc_threshold=0.2):
        self.root = root
        self.sampling_rate = sampling_rate
        self.default_rate = 200
        self._seq_cache = {}
        self._labels_cache = {}

        self.index = []  # (recording, epoch_idx)
        for rec in recordings:
            labels = np.load(os.path.join(root, "labels", f"{rec}.npy"))
            qc = np.load(os.path.join(root, "epoch_qc", f"{rec}.npy"))
            for i in range(len(labels)):
                if qc[i] > qc_threshold:
                    continue
                self.index.append((rec, i))

    def __len__(self):
        return len(self.index)

    def _get_seq(self, rec):
        if rec not in self._seq_cache:
            self._seq_cache[rec] = np.load(os.path.join(self.root, "seq", f"{rec}.npy"), mmap_mode="r")
        return self._seq_cache[rec]

    def _get_labels(self, rec):
        if rec not in self._labels_cache:
            self._labels_cache[rec] = np.load(os.path.join(self.root, "labels", f"{rec}.npy"))
        return self._labels_cache[rec]

    def __getitem__(self, idx):
        rec, i = self.index[idx]
        X = np.array(self._get_seq(rec)[i])  # (2, 6000)
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(30 * self.sampling_rate), axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8)
        X = torch.FloatTensor(X)
        y = int(self._get_labels(rec)[i])
        return X, y

class ThingsEEG1Loader(torch.utils.data.Dataset):
    """
    THINGS-EEG1 binary istarget (RSVP oddball fixation-colour-change) detection.
    Each "file" is really a subject directory (e.g. 'sub-30') containing
    eeg/<subject>_task-rsvp_eeg.{vhdr,eeg,vmrk} (BrainVision, native 1000Hz,
    63 real channels) and eeg/<subject>_task-rsvp_events.tsv. Every row of
    the events file becomes one window: [time_stimon, time_stimon+window_seconds)
    at the recording's native rate, resampled to sampling_rate (must be 200Hz
    for pretrained-weight compatibility). Label = istarget (0/1); istarget=1
    is extremely rare (~1.2% of events). When balance_ratio is set (e.g. 10),
    non-target events are randomly subsampled per-subject so the loader
    yields at most balance_ratio * (# target events) non-target windows --
    intended for the TRAIN split only; pass balance_ratio=None for val/test
    to keep the natural distribution for honest AUROC/PR-AUC reporting.
    """
    def __init__(self, root, subject_dirs, sampling_rate=200, window_seconds=0.5,

                 balance_ratio=None, seed=42):

        self.root = root

        self.sampling_rate = sampling_rate

        self.window_seconds = window_seconds

        self._raw_cache = {}

        self._sfreq_cache = {}



        rng = np.random.RandomState(seed)

        self.index = []  # (subject, start_sample_native, label)



        for subj in subject_dirs:

            eeg_dir = os.path.join(root, subj, "eeg")

            vhdr_candidates = glob.glob(os.path.join(eeg_dir, "*_eeg.vhdr"))

            events_candidates = glob.glob(os.path.join(eeg_dir, "*_events.tsv"))

            if not vhdr_candidates or not events_candidates:

                print(f"WARNING: missing vhdr/events for {subj}, skipping")

                continue

            vhdr = vhdr_candidates[0]

            events_tsv = events_candidates[0]



            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)

            native_sfreq = raw.info["sfreq"]

            n_times = raw.n_times

            self._sfreq_cache[subj] = native_sfreq



            window_samples = int(round(window_seconds * native_sfreq))



            with open(events_tsv) as f:

                reader = csv.DictReader(f, delimiter="	")

                rows = list(reader)



            if not rows or "time_stimon" not in rows[0]:

                print(f"WARNING: {subj} events file missing time_stimon column, skipping subject")

                continue



            target_rows = []

            nontarget_rows = []

            for row in rows:

                onset_s = float(row["time_stimon"])

                start = int(round(onset_s * native_sfreq))

                if start + window_samples > n_times:

                    continue

                label = int(row["istarget"])

                if label == 1:

                    target_rows.append((subj, start, label))

                else:

                    nontarget_rows.append((subj, start, label))



            if balance_ratio is not None and len(target_rows) > 0:

                keep_n = min(len(nontarget_rows), balance_ratio * len(target_rows))

                keep_idx = rng.choice(len(nontarget_rows), size=keep_n, replace=False)

                nontarget_rows = [nontarget_rows[i] for i in keep_idx]



            self.index.extend(target_rows)

            self.index.extend(nontarget_rows)



    def __len__(self):
        return len(self.index)

    def _get_raw(self, subj):
        if subj not in self._raw_cache:
            eeg_dir = os.path.join(self.root, subj, "eeg")
            vhdr = glob.glob(os.path.join(eeg_dir, "*_eeg.vhdr"))[0]
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
            self._raw_cache[subj] = raw
        return self._raw_cache[subj]

    def __getitem__(self, idx):
        subj, start, label = self.index[idx]
        raw = self._get_raw(subj)
        native_sfreq = self._sfreq_cache[subj]
        window_samples = int(round(self.window_seconds * native_sfreq))
        X = raw.get_data(start=start, stop=start + window_samples)  # (63, window_samples) @ native_sfreq

        if self.sampling_rate != native_sfreq:
            target_samples = int(round(self.window_seconds * self.sampling_rate))
            X = resample(X, target_samples, axis=-1)

        X = X / (np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8)
        X = torch.FloatTensor(X)
        return X, label
class CAUEEGLoader(torch.utils.data.Dataset):
    """
    CAUEEG Dementia (3-class: Normal/MCI/Dementia). Each EDF recording
    (~17 min avg) is chunked into non-overlapping 10s windows at index-build
    time; every window inherits its recording's single class label. Only
    the 19 real EEG channels are kept (EKG, Photic dropped). Labels come
    from dementia-no-overlap.json's {split}_split lists, matched by serial
    number parsed from each EDF filename -- NOT from the train-only capped
    manifest, so this works identically for train/val/test.
    """
    EEG_CHANNELS = [
        'Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'O1-AVG',
        'Fp2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'O2-AVG',
        'F7-AVG', 'T3-AVG', 'T5-AVG', 'F8-AVG', 'T4-AVG', 'T6-AVG',
        'FZ-AVG', 'CZ-AVG', 'PZ-AVG',
    ]
    ANNOTATION_PATH = "/scratch/nourmaj/caueeg/caueeg-dataset/dementia-no-overlap.json"

    def __init__(self, root, files, sampling_rate=200, window_seconds=10):
        self.root = root
        self.files = files
        self.sampling_rate = sampling_rate
        self.window_seconds = window_seconds
        self._raw_cache = {}

        with open(self.ANNOTATION_PATH) as f:
            ann = json.load(f)
        self.serial_to_label = {}
        for key in ("train_split", "validation_split", "test_split"):
            for entry in ann[key]:
                self.serial_to_label[entry["serial"]] = entry["class_label"]

        # Build a flat (filename, window_index) index so __getitem__ can
        # address individual windows directly.
        self.index = []
        for fname in self.files:
            serial = os.path.splitext(fname)[0]
            if serial not in self.serial_to_label:
                print(f"WARNING: {fname} has no label in annotation file, skipping")
                continue
            raw = mne.io.read_raw_edf(os.path.join(root, fname), preload=False, verbose=False)
            sfreq = raw.info["sfreq"]
            n_windows = int(raw.n_times // (sfreq * window_seconds))
            for w in range(n_windows):
                self.index.append((fname, w))

    def __len__(self):
        return len(self.index)

    def _get_raw(self, fname):
        if fname not in self._raw_cache:
            raw = mne.io.read_raw_edf(os.path.join(self.root, fname), preload=False, verbose=False)
            raw.pick_channels(self.EEG_CHANNELS, ordered=True)
            self._raw_cache[fname] = raw
        return self._raw_cache[fname]

    def __getitem__(self, idx):
        fname, w = self.index[idx]
        serial = os.path.splitext(fname)[0]
        y = self.serial_to_label[serial]

        raw = self._get_raw(fname)
        sfreq = raw.info["sfreq"]
        start = int(w * self.window_seconds * sfreq)
        stop = start + int(self.window_seconds * sfreq)
        X = raw.get_data(start=start, stop=stop)  # (19, 2000) at 200Hz/10s

        X = X / (np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8)
        X = torch.FloatTensor(X)
        return X, y

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class PTBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=500):
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self.freq * 5, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


class ISRUCLoader(torch.utils.data.Dataset):
    """
    ISRUC 5-class sleep staging (Wake/N1/N2/N3/REM), flattened from CBraMod's
    non-overlapping 20-epoch sequence format into individual 30s epochs by
    preprocess_isruc_biot.py. Each pkl file has keys 'X' (6, 6000) @ 200Hz
    and 'y' (int, 0-4).
    """
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200  # already resampled to 200Hz during CBraMod's own preprocessing
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]  # (6, 6000), 30s @ 200Hz
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(30 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class Yang2025Loader(torch.utils.data.Dataset):
    """
    Yang2025 (WBCIC-SHU) binary motor imagery: left hand vs right hand.
    Yang, Rong, Xie, et al. (2025), Scientific Data, DOI: 10.1038/s41597-025-04826-y.
    Recorded at 1000Hz, downsampled to 250Hz during preprocessing. Each pkl file
    has keys 'X' (59, 1500) float32 @ 250Hz (6s trials: 1.5s cue window trimmed,
    leaving the labeled MI-period epoch) and 'y' (int, 0=left hand, 1=right hand).
    59 real EEG channels, standard_1005 (10-05) montage, Neuracle NeuSen W cap.
    """
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 250  # native rate after the dataset's own preprocessing
        self.sampling_rate = sampling_rate
        self.trial_seconds = 6  # 1500 samples / 250Hz

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]  # (59, 1500), 6s @ 250Hz
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(self.trial_seconds * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class ClinicalLoader(torch.utils.data.Dataset):
    """
    ds004504 clinical EEG, binary FTD-vs-HC classification (matching the REVE
    build's label convention). Each pkl file has keys 'X' (19, 5000) float32
    @ 500Hz (10s resting-state windows, raw volts) and 'y' (int, 0/1).
    19 real 10-20 system channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
    F7, F8, T3, T4, T5, T6, Fz, Cz, Pz.
    """
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 500  # native rate of the dataset's own preprocessing
        self.sampling_rate = sampling_rate
        self.trial_seconds = 10  # 5000 samples / 500Hz

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]  # (19, 5000), 10s @ 500Hz
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(self.trial_seconds * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class DreamLoader(torch.utils.data.Dataset):
    """
    DREAM (Tononi Serial Awakenings, Siclari et al. 2017) binary
    Experience-vs-no-Experience classification. Each pkl file has keys
    'X' (257, 10000) float32 @ 500Hz (last 20s of each 120s episode,
    raw volts) and 'y' (int, 0/1). 257 real channels on a custom EGI
    HydroCelGSN256v10 net (no standard 10-20 names) -- BIOT does not need
    real positions (unlike REVE), so this is a non-issue here. Note: 257
    channels is far more than the pretrained checkpoint's 18, so the
    channel-mismatch skip logic in run_binary_supervised.py will
    reinitialize most/all channel-dependent layers even for "pretrained" runs.
    """
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 500  # native rate of the dataset's own preprocessing
        self.sampling_rate = sampling_rate
        self.trial_seconds = 20  # 10000 samples / 500Hz

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]  # (257, 10000), 20s @ 500Hz
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(self.trial_seconds * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class ThoughtVizLoader(torch.utils.data.Dataset):
    """
    ThoughtViz 10-class object-category classification. Each pkl file has
    keys 'X' (14, 32) float32 @ 128Hz (0.25s window, real EEGNet stimulus
    data -- known unfixable train/test overlap-window leakage per the
    ThoughtViz paper, same caveat as the REVE build) and 'y' (int, 0-9).
    Unlike REVE (hard patch_size=200, forced zero-padding 50->200), BIOT's
    STFT-based tokenization can process the real 0.25s window directly with
    no padding -- just requires a much smaller --token_size/--hop_length
    than BIOT's usual defaults (200/100), since the real signal is only
    ~50 samples after resampling to 200Hz.
    """
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 128  # native rate (14, 32) was recorded at
        self.sampling_rate = sampling_rate
        self.trial_seconds = 0.25  # 32 samples / 128Hz

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]  # (14, 32), 0.25s @ 128Hz
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(round(self.trial_seconds * self.sampling_rate)), axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class KaraOneLoader(torch.utils.data.Dataset):
    """
    KaraOne imagined-speech binary classification. Each pkl file has keys
    'X' (60, ~4960-4970) float32 @ 1000Hz (trial length varies slightly
    trial-to-trial, ~5s) and 'y' (int, 0/1). 60 real 10-10 system channels.
    Resamples to a fixed 5s target regardless of the exact native length,
    same as the REVE build's convention.
    """
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 1000  # nominal native rate; actual trial length varies slightly
        self.sampling_rate = sampling_rate
        self.trial_seconds = 5

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]  # (60, ~4960-4970), ~5s @ 1000Hz
        X = resample(X, int(self.trial_seconds * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class HARLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs, sampling_rate=50):
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, "rb"))
        X, y = sample["X"], self.label_map.index(sample["y"])
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(
                np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        return torch.FloatTensor(X), y


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root_prest, root_shhs):

        # prest dataset
        self.root_prest = root_prest
        exception_files = ["319431_data.npy"]
        self.prest_list = list(
            filter(
                lambda x: ("data" in x) and (x not in exception_files),
                os.listdir(self.root_prest),
            )
        )

        PREST_LENGTH = 2000
        WINDOW_SIZE = 200

        print("(prest) unlabeled data size:", len(self.prest_list) * 16)
        self.prest_idx_all = np.arange(PREST_LENGTH // WINDOW_SIZE)
        self.prest_mask_idx_N = PREST_LENGTH // WINDOW_SIZE // 3

        SHHS_LENGTH = 6000
        # shhs dataset
        self.root_shhs = root_shhs
        self.shhs_list = os.listdir(self.root_shhs)
        print("(shhs) unlabeled data size:", len(self.shhs_list))
        self.shhs_idx_all = np.arange(SHHS_LENGTH // WINDOW_SIZE)
        self.shhs_mask_idx_N = SHHS_LENGTH // WINDOW_SIZE // 5

    def __len__(self):
        return len(self.prest_list) + len(self.shhs_list)

    def prest_load(self, index):
        sample_path = self.prest_list[index]
        # (16, 16, 2000), 10s
        samples = np.load(os.path.join(self.root_prest, sample_path)).astype("float32")

        # find all zeros or all 500 signals and then remove them
        samples_max = np.max(samples, axis=(1, 2))
        samples_min = np.min(samples, axis=(1, 2))
        valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        samples = samples[valid]

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0

    def shhs_load(self, index):
        sample_path = self.shhs_list[index]
        # (2, 3750) sampled at 125
        sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
        # (2, 6000) resample to 200
        samples = resample(sample, 6000, axis=-1)

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # generate samples and targets and mask_indices
        samples = torch.FloatTensor(samples)

        return samples, 1

    def __getitem__(self, index):
        if index < len(self.prest_list):
            return self.prest_load(index)
        else:
            index = index - len(self.prest_list)
            return self.shhs_load(index)


def collate_fn_unsupervised_pretrain(batch):
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples


class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, tuev_data, chb_mit_data, iiic_data, tuab_data):
        # for TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # for CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        # for IIIC seizure
        iiic_x, iiic_y = iiic_data
        self.iiic_x = iiic_x
        self.iiic_y = iiic_y
        self.iiic_size = len(self.iiic_x)

        # for TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)

    def __len__(self):
        return self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size

    def tuev_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
        )
        X = sample["signal"]
        # 256 * 5 -> 1000
        X = resample(X, 1000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y, 0

    def chb_mit_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
        )
        X = sample["X"]
        # 2560 -> 2000
        X = resample(X, 2000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 1

    def iiic_load(self, index):
        data = self.iiic_x[index]
        samples = torch.FloatTensor(data)
        samples = samples / (
            torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
        )
        y = np.argmax(self.iiic_y[index])
        return samples, y, 2

    def tuab_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
        )
        X = sample["X"]
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 3

    def __getitem__(self, index):
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.iiic_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.iiic_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.iiic_size
            return self.tuab_load(index)
        else:
            raise ValueError("index out of range")


def collate_fn_supervised_pretrain(batch):
    tuev_samples, tuev_labels = [], []
    iiic_samples, iiic_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            iiic_samples.append(sample)
            iiic_labels.append(labels)
        elif idx == 2:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError("idx out of range")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        iiic_labels = torch.LongTensor(iiic_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (iiic_samples, iiic_labels),
        (chb_mit_samples, chb_mit_labels),
        (tuab_samples, tuab_labels),
    )


# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


# define binary cross entropy loss
def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()