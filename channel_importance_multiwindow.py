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



# ============================================================
# Multi-window channel importance analysis
# ============================================================
# This script:
# 1) loads your trained REVE linear-probe model
# 2) reads test_predictions_chb03_mono_balanced.csv
# 3) selects multiple confident seizure windows
# 4) computes gradient-based channel importance per window
# 5) averages importance across windows
# 6) saves CSV + bar plots
# ============================================================

import matplotlib.pyplot as plt


PRED_CSV = ROOT / f"test_predictions_{TEST_SUBJECT}_mono_balanced.csv"
OUT_DIR = ROOT / f"channel_importance_{TEST_SUBJECT}_multiwindow"

# Select windows with true seizure and model confidence above this threshold
MIN_PROB_SEIZURE = 0.80

# Maximum number of seizure windows to analyze
# Use 20 first. You can increase to 50 later.
TOP_N_WINDOWS = 50

# If True, only keep true-positive seizure windows at default threshold 0.5
ONLY_TRUE_POSITIVES = True


def compute_one_window_importance(model, test_ds, test_df, edf_path, start_sec):
    """
    Gradient-based importance for one selected window.

    importance[channel] = mean(abs(d P(seizure) / d input[channel, time]))
    """
    match = test_df[
        (test_df["edf_path"] == edf_path) &
        (test_df["win_start_sec"].astype(float) == float(start_sec))
    ]

    if len(match) == 0:
        raise ValueError(
            f"Could not find selected window in test_df:\n"
            f"edf_path={edf_path}\nstart={start_sec}"
        )

    idx = int(match.index[0])
    row = test_df.loc[idx]

    x, y = test_ds[idx]  # (C, T)
    x = x.unsqueeze(0).to(DEVICE)  # (1, C, T)
    x.requires_grad_(True)

    model.eval()
    model.zero_grad(set_to_none=True)

    logits = model(x)
    prob_seizure = torch.softmax(logits, dim=1)[0, 1]
    prob_seizure.backward()

    grad = x.grad.detach().cpu().numpy()[0]  # (C, T)
    importance_raw = np.mean(np.abs(grad), axis=1)  # (C,)

    # Normalize each window so no single window dominates the average
    importance_norm = importance_raw / (importance_raw.max() + 1e-8)

    return {
        "row": row,
        "true_label": int(y.item()),
        "prob_seizure": float(prob_seizure.item()),
        "importance_raw": importance_raw,
        "importance_norm": importance_norm,
    }


def main():
    print("Device:", DEVICE)
    print("Checkpoint:", CKPT_PATH)
    print("Prediction CSV:", PRED_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load test split and dataset
    # -------------------------
    df = pd.read_csv(CSV_PATH)
    _, _, test_df = subject_edf_split(df, test_subject=TEST_SUBJECT)
    test_df = filter_df_by_reconstructable_edfs(test_df, min_pairs=MIN_RECONSTRUCTABLE_PAIRS)

    test_ds = CHBMITMonoWindowDataset(
        test_df,
        TARGET_ELECTRODES,
        target_sfreq=TARGET_SFREQ,
        zscore=True,
    )

    # -------------------------
    # Load model
    # -------------------------
    x0, _ = test_ds[0]
    xb0 = x0.unsqueeze(0).to(DEVICE)

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

    # -------------------------
    # Select seizure windows from prediction CSV
    # -------------------------
    pred_df = pd.read_csv(PRED_CSV)

    selected = pred_df[
        (pred_df["y_true"] == 1) &
        (pred_df["prob_seizure"] >= MIN_PROB_SEIZURE)
    ].copy()

    if ONLY_TRUE_POSITIVES:
        selected = selected[selected["prob_seizure"] >= 0.5].copy()

    selected = selected.sort_values("prob_seizure", ascending=False).head(TOP_N_WINDOWS)
    selected = selected.reset_index(drop=True)

    if len(selected) == 0:
        raise RuntimeError(
            "No windows selected. Try lowering MIN_PROB_SEIZURE, for example 0.5 or 0.6."
        )

    print(f"\nSelected {len(selected)} seizure windows for channel-importance averaging.")
    print(selected[["edf", "win_start_sec", "win_end_sec", "y_true", "prob_seizure"]].head(20).to_string(index=False))

    # -------------------------
    # Compute importance for each selected window
    # -------------------------
    all_norm = []
    all_raw = []
    per_window_rows = []

    for i, row in selected.iterrows():
        print(
            f"\n[{i+1}/{len(selected)}] Computing importance | "
            f"{row['edf']} | {row['win_start_sec']}-{row['win_end_sec']} sec | "
            f"prob={row['prob_seizure']:.4f}"
        )

        result = compute_one_window_importance(
            model=model,
            test_ds=test_ds,
            test_df=test_df,
            edf_path=row["edf_path"],
            start_sec=row["win_start_sec"],
        )

        imp_raw = result["importance_raw"]
        imp_norm = result["importance_norm"]

        all_raw.append(imp_raw)
        all_norm.append(imp_norm)

        for ch, raw_val, norm_val in zip(TARGET_ELECTRODES, imp_raw, imp_norm):
            per_window_rows.append({
                "window_rank": i + 1,
                "edf": row["edf"],
                "edf_path": row["edf_path"],
                "win_start_sec": row["win_start_sec"],
                "win_end_sec": row["win_end_sec"],
                "prob_seizure_csv": row["prob_seizure"],
                "prob_seizure_recomputed": result["prob_seizure"],
                "channel": ch,
                "importance_raw": raw_val,
                "importance_norm": norm_val,
            })

    all_raw = np.vstack(all_raw)
    all_norm = np.vstack(all_norm)

    # -------------------------
    # Average across windows
    # -------------------------
    mean_norm = all_norm.mean(axis=0)
    std_norm = all_norm.std(axis=0)

    mean_raw = all_raw.mean(axis=0)
    std_raw = all_raw.std(axis=0)

    summary_df = pd.DataFrame({
        "channel": TARGET_ELECTRODES,
        "mean_importance_norm": mean_norm,
        "std_importance_norm": std_norm,
        "mean_importance_raw": mean_raw,
        "std_importance_raw": std_raw,
    }).sort_values("mean_importance_norm", ascending=False).reset_index(drop=True)

    per_window_df = pd.DataFrame(per_window_rows)

    # -------------------------
    # Save CSV outputs
    # -------------------------
    summary_csv = OUT_DIR / f"average_channel_importance_top{len(selected)}_windows.csv"
    per_window_csv = OUT_DIR / f"per_window_channel_importance_top{len(selected)}_windows.csv"
    selected_csv = OUT_DIR / f"selected_windows_top{len(selected)}.csv"

    summary_df.to_csv(summary_csv, index=False)
    per_window_df.to_csv(per_window_csv, index=False)
    selected.to_csv(selected_csv, index=False)

    print("\n===== AVERAGE CHANNEL IMPORTANCE =====")
    print(summary_df.head(15).to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved per-window CSV: {per_window_csv}")
    print(f"Saved selected windows CSV: {selected_csv}")

    # -------------------------
    # Save plots
    # -------------------------
    # Original channel order plot
    plt.figure(figsize=(11, 4))
    plt.bar(TARGET_ELECTRODES, mean_norm, yerr=std_norm, capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean normalized gradient importance")
    plt.title(f"Average Channel Importance | Top {len(selected)} Seizure Windows")
    plt.tight_layout()
    out_plot_ordered = OUT_DIR / f"average_channel_importance_top{len(selected)}_ordered.png"
    plt.savefig(out_plot_ordered, dpi=180)
    plt.close()
    print(f"Saved plot: {out_plot_ordered}")

    # Ranked plot
    ranked = summary_df.sort_values("mean_importance_norm", ascending=False)

    plt.figure(figsize=(11, 4))
    plt.bar(ranked["channel"], ranked["mean_importance_norm"], yerr=ranked["std_importance_norm"], capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean normalized gradient importance")
    plt.title(f"Ranked Average Channel Importance | Top {len(selected)} Seizure Windows")
    plt.tight_layout()
    out_plot_ranked = OUT_DIR / f"average_channel_importance_top{len(selected)}_ranked.png"
    plt.savefig(out_plot_ranked, dpi=180)
    plt.close()
    print(f"Saved plot: {out_plot_ranked}")

    # Heatmap: windows x channels
    plt.figure(figsize=(11, max(4, 0.25 * len(selected))))
    plt.imshow(all_norm, aspect="auto")
    plt.colorbar(label="Normalized importance")
    plt.xticks(np.arange(len(TARGET_ELECTRODES)), TARGET_ELECTRODES, rotation=45, ha="right")
    plt.yticks(np.arange(len(selected)), [f"{r.edf}:{int(r.win_start_sec)}" for _, r in selected.iterrows()])
    plt.xlabel("Channel")
    plt.ylabel("Selected seizure window")
    plt.title("Per-window Channel Importance Heatmap")
    plt.tight_layout()
    out_heatmap = OUT_DIR / f"channel_importance_heatmap_top{len(selected)}.png"
    plt.savefig(out_heatmap, dpi=180)
    plt.close()
    print(f"Saved heatmap: {out_heatmap}")

    print("\nDone.")


if __name__ == "__main__":
    main()
