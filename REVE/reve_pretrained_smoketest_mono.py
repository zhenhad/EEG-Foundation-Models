from pathlib import Path
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset, DataLoader

from braindecode.models import REVE
from build_dataset import ROOT, CSV_PATH, TARGET_SFREQ, subject_edf_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_ELECTRODES = [
    "FP1", "F7", "T7", "P7", "O1",
    "F3", "C3", "P3",
    "FP2", "F4", "C4", "P4", "O2",
    "F8", "T8", "P8",
    "FZ", "CZ", "PZ"
]


def parse_bipolar_name(ch):
    ch = str(ch).upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "")
    parts = ch.split("-")
    if len(parts) < 2:
        return None
    a, b = parts[0], parts[1]
    # ignore suffixes like T8-P8-0 / T8-P8-1
    if b in {"0", "1"} and len(parts) >= 3:
        b = parts[1]
    return a, b


def build_pair_list(ch_names):
    pairs = []
    for ch in ch_names:
        parsed = parse_bipolar_name(ch)
        if parsed is None:
            continue
        a, b = parsed
        if a in TARGET_ELECTRODES and b in TARGET_ELECTRODES:
            pairs.append((ch, a, b))
    return pairs


def reconstruct_mono_from_bipolar(x_bipolar, raw_ch_names, target_electrodes):
    pairs = build_pair_list(raw_ch_names)
    if len(pairs) < 5:
        raise ValueError(f"Not enough usable bipolar pairs: {len(pairs)}")

    elecs = list(target_electrodes)
    e2i = {e: i for i, e in enumerate(elecs)}

    K = len(pairs)
    E = len(elecs)

    A = np.zeros((K, E), dtype=np.float32)
    y = np.zeros((K, x_bipolar.shape[1]), dtype=np.float32)

    raw_index = {ch.upper(): i for i, ch in enumerate(raw_ch_names)}

    for r, (raw_name, a, b) in enumerate(pairs):
        A[r, e2i[a]] = 1.0
        A[r, e2i[b]] = -1.0
        y[r, :] = x_bipolar[raw_index[raw_name.upper()], :]

    # gauge constraint: mean(V)=0
    A2 = np.vstack([A, np.ones((1, E), dtype=np.float32) / E])
    y2 = np.vstack([y, np.zeros((1, y.shape[1]), dtype=np.float32)])

    V, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    return V.astype(np.float32), elecs


class CHBMITMonoWindowDataset(Dataset):
    def __init__(self, df, target_electrodes, target_sfreq=256.0, zscore=True):
        self.df = df.reset_index(drop=True)
        self.target_electrodes = [e.upper() for e in target_electrodes]
        self.target_sfreq = float(target_sfreq)
        self.zscore = bool(zscore)
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _load_edf(self, edf_path):
        if edf_path in self._cache:
            return self._cache[edf_path]

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        ch_names = [c.upper() for c in raw.ch_names]
        data = raw.get_data().astype(np.float32)

        if sfreq != self.target_sfreq:
            data = mne.filter.resample(
                data,
                up=self.target_sfreq,
                down=sfreq,
                axis=1,
                npad="auto",
                verbose=False,
            )
            sfreq = self.target_sfreq

        self._cache[edf_path] = (data, ch_names, sfreq)
        return self._cache[edf_path]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        edf_path = row["edf_path"]
        win_start_sec = float(row["win_start_sec"])
        win_end_sec = float(row["win_end_sec"])
        y = int(row["label"])

        data, ch_names, sfreq = self._load_edf(edf_path)

        start_samp = int(round(win_start_sec * sfreq))
        stop_samp = int(round(win_end_sec * sfreq))
        x = data[:, start_samp:stop_samp]

        expected_len = int(round((win_end_sec - win_start_sec) * self.target_sfreq))
        if x.shape[1] != expected_len:
            if x.shape[1] < expected_len:
                pad = np.zeros((x.shape[0], expected_len - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, :expected_len]

        V, elecs = reconstruct_mono_from_bipolar(x, ch_names, self.target_electrodes)

        if self.zscore:
            mean = V.mean(axis=1, keepdims=True)
            std = V.std(axis=1, keepdims=True)
            std[std < 1e-6] = 1.0
            V = (V - mean) / std

        return torch.from_numpy(V), torch.tensor(y, dtype=torch.long)


def make_chs_info(target_electrodes):
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]

    name_map = {
        "FP1": "Fp1", "FP2": "Fp2",
        "F7": "F7", "F3": "F3", "FZ": "Fz", "F4": "F4", "F8": "F8",
        "T7": "T7", "C3": "C3", "CZ": "Cz", "C4": "C4", "T8": "T8",
        "P7": "P7", "P3": "P3", "PZ": "Pz", "P4": "P4", "P8": "P8",
        "O1": "O1", "O2": "O2",
    }

    ch_names = [name_map[e] for e in target_electrodes]
    info = mne.create_info(ch_names=ch_names, sfreq=TARGET_SFREQ, ch_types=["eeg"] * len(ch_names))

    for ch in info["chs"]:
        ch["loc"][:3] = pos[ch["ch_name"]]

    return info["chs"]


def main():
    print("Device:", DEVICE)

    df = pd.read_csv(CSV_PATH)
    train_df, val_df, test_df = subject_edf_split(df, test_subject="chb03")

    train_ds = CHBMITMonoWindowDataset(
        train_df,
        TARGET_ELECTRODES,
        target_sfreq=TARGET_SFREQ,
        zscore=True,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)
    xb, yb = next(iter(train_loader))

    print("Batch X shape:", xb.shape)
    print("Batch y shape:", yb.shape)

    chs_info = make_chs_info(TARGET_ELECTRODES)

    model = REVE.from_pretrained(
        "brain-bzh/reve-base",
        n_outputs=2,
        n_chans=xb.shape[1],
        n_times=xb.shape[2],
        sfreq=TARGET_SFREQ,
        chs_info=chs_info,
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():
        logits = model(xb.to(DEVICE))

    print("Forward pass successful")
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()
