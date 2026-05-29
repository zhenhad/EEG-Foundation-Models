from pathlib import Path
import numpy as np
import pandas as pd
import mne
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from braindecode.models import REVE
from tqdm.auto import tqdm

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")
CSV_PATH = ROOT / "windows_chbmit_all_4s_50ol.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SFREQ = 256.0
TEST_SUBJECT = "chb03"
MIN_RECONSTRUCTABLE_PAIRS = 5
CKPT_PATH = ROOT / f"linear_probe_reve_{TEST_SUBJECT}_mono_balanced_best.pt"
OUT_CSV = ROOT / f"test_predictions_{TEST_SUBJECT}_mono_balanced.csv"

TARGET_ELECTRODES = [
    "FP1", "F7", "T7", "P7", "O1",
    "F3", "C3", "P3",
    "FP2", "F4", "C4", "P4", "O2",
    "F8", "T8", "P8",
    "FZ", "CZ", "PZ"
]

def subject_edf_split(df, test_subject="chb03", val_fraction=0.1, seed=42):
    test_df = df[df["subject"] == test_subject].copy()
    train_pool = df[df["subject"] != test_subject].copy()
    rng = np.random.default_rng(seed)
    unique_edfs = train_pool[["subject", "edf_path"]].drop_duplicates()
    val_n = max(1, int(round(len(unique_edfs) * val_fraction)))
    val_idx = rng.choice(len(unique_edfs), size=val_n, replace=False)
    val_edfs = set(unique_edfs.iloc[val_idx]["edf_path"].tolist())
    val_df = train_pool[train_pool["edf_path"].isin(val_edfs)].copy()
    train_df = train_pool[~train_pool["edf_path"].isin(val_edfs)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def parse_bipolar_name(ch):
    ch = str(ch).upper().strip()
    ch = ch.replace("EEG ", "").replace("-REF", "").replace("-LE", "").replace(" ", "")
    parts = ch.split("-")
    if len(parts) < 2:
        return None
    a = parts[0].strip()
    b = parts[1].strip()
    if not a or not b:
        return None
    return a, b

def build_pair_list(ch_names):
    pairs = []
    seen = set()
    for ch in ch_names:
        parsed = parse_bipolar_name(ch)
        if parsed is None:
            continue
        a, b = parsed
        if a in TARGET_ELECTRODES and b in TARGET_ELECTRODES:
            key = (a, b)
            if key not in seen:
                pairs.append((ch, a, b))
                seen.add(key)
    return pairs

def count_usable_pairs_for_edf(edf_path):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        ch_names = [c.upper() for c in raw.ch_names]
        return len(build_pair_list(ch_names))
    except Exception:
        return 0

def filter_df_by_reconstructable_edfs(df, min_pairs=5):
    good_edfs = []
    for edf_path in df["edf_path"].drop_duplicates().tolist():
        if count_usable_pairs_for_edf(edf_path) >= min_pairs:
            good_edfs.append(edf_path)
    return df[df["edf_path"].isin(good_edfs)].copy().reset_index(drop=True)

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
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        ch_names = [c.upper() for c in raw.ch_names]
        self._cache[edf_path] = (raw, ch_names, sfreq)
        return self._cache[edf_path]
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        edf_path = row["edf_path"]
        win_start_sec = float(row["win_start_sec"])
        win_end_sec = float(row["win_end_sec"])
        y = int(row["label"])
        raw, ch_names, sfreq = self._load_edf(edf_path)
        start_samp = int(round(win_start_sec * sfreq))
        stop_samp = int(round(win_end_sec * sfreq))
        x = raw.get_data(start=start_samp, stop=stop_samp).astype(np.float32)
        if sfreq != self.target_sfreq:
            x = mne.filter.resample(x, up=self.target_sfreq, down=sfreq, axis=1, npad="auto", verbose=False).astype(np.float32)
        expected_len = int(round((win_end_sec - win_start_sec) * self.target_sfreq))
        if x.shape[1] != expected_len:
            if x.shape[1] < expected_len:
                pad = np.zeros((x.shape[0], expected_len - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, :expected_len]
        V, _ = reconstruct_mono_from_bipolar(x, ch_names, self.target_electrodes)
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

class REVELinearProbe(nn.Module):
    def __init__(self, backbone, n_outputs=2):
        super().__init__()
        self.backbone = backbone
        self._features = None
        self.head = None
        for p in self.backbone.parameters():
            p.requires_grad = False
        target_layer = None
        for _, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                target_layer = module
        if target_layer is None:
            raise RuntimeError("Could not find a Linear layer in REVE.")
        def hook_fn(module, inputs, output):
            self._features = inputs[0]
        self._hook = target_layer.register_forward_hook(hook_fn)
    def build_head(self, xb, n_outputs=2):
        self.backbone.eval()
        with torch.no_grad():
            _ = self.backbone(xb)
            feats = self._features
            if feats is None:
                raise RuntimeError("Feature hook did not capture anything.")
            feat_dim = feats.shape[-1]
        self.head = nn.Linear(feat_dim, n_outputs).to(xb.device)
    def forward(self, x):
        _ = self.backbone(x)
        feats = self._features
        if feats is None:
            raise RuntimeError("No features captured from backbone.")
        return self.head(feats)

def main():
    print("Device:", DEVICE)
    print("Checkpoint:", CKPT_PATH)
    df = pd.read_csv(CSV_PATH)
    _, _, test_df = subject_edf_split(df, test_subject=TEST_SUBJECT)
    test_df = filter_df_by_reconstructable_edfs(test_df, min_pairs=MIN_RECONSTRUCTABLE_PAIRS)
    test_ds = CHBMITMonoWindowDataset(test_df, TARGET_ELECTRODES, target_sfreq=TARGET_SFREQ, zscore=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    xb0, _ = next(iter(test_loader))
    xb0 = xb0.to(DEVICE)
    chs_info = make_chs_info(TARGET_ELECTRODES)
    backbone = REVE.from_pretrained(
        "brain-bzh/reve-base",
        n_outputs=2,
        n_chans=xb0.shape[1],
        n_times=xb0.shape[2],
        sfreq=TARGET_SFREQ,
        chs_info=chs_info,
    ).to(DEVICE)
    model = REVELinearProbe(backbone=backbone, n_outputs=2).to(DEVICE)
    model.build_head(xb0, n_outputs=2)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    results = []
    row_ptr = 0
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Saving test predictions"):
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            seizure_prob = probs[:, 1].cpu().numpy()
            pred_default = (seizure_prob >= 0.5).astype(int)
            y_true = yb.numpy()
            batch_size = len(y_true)
            batch_df = test_df.iloc[row_ptr:row_ptr + batch_size]
            for i in range(batch_size):
                row = batch_df.iloc[i]
                results.append({
                    "subject": row["subject"],
                    "edf": row["edf"],
                    "edf_path": row["edf_path"],
                    "win_start_sec": row["win_start_sec"],
                    "win_end_sec": row["win_end_sec"],
                    "y_true": int(y_true[i]),
                    "y_pred_default": int(pred_default[i]),
                    "prob_nonseizure": float(probs[i, 0].cpu().item()),
                    "prob_seizure": float(seizure_prob[i]),
                })
            row_ptr += batch_size
    pred_df = pd.DataFrame(results).sort_values(["subject", "edf", "win_start_sec"]).reset_index(drop=True)
    pred_df.to_csv(OUT_CSV, index=False)
    print(f"Saved predictions to: {OUT_CSV}")
    print(pred_df.head())
    print("Done.")

if __name__ == "__main__":
    main()
