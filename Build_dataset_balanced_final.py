from pathlib import Path
import pandas as pd
import numpy as np
import mne
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")
CSV_PATH = ROOT / "windows_chbmit_all_4s_50ol.csv"

TARGET_SFREQ = 256.0
Z_SCORE_PER_CHANNEL = True
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 42

# Balancing config: training set only
DOWNSAMPLE_NEG_POS_RATIO = 4          # keep non-seizure ~= 4x seizure
AUGMENT_POSITIVES = True
SEIZURE_SHIFT_SEC = 0.5               # use ±0.5 s shifts
MIN_SHIFTED_SEIZURE_OVERLAP_FRAC = 0.50
WINDOW_LEN_SEC = 4.0
USE_WEIGHTED_SAMPLER = True
MIN_USABLE_BIPOLAR_PAIRS = 5

# Canonical CHB-MIT bipolar montage we want to keep
VALID_BIPOLAR_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
]
VALID_BIPOLAR_SET = set(VALID_BIPOLAR_CHANNELS)


def clean_ch_name(ch: str) -> str:
    ch = str(ch).strip().upper()
    ch = ch.replace("EEG ", "")
    ch = ch.replace("-REF", "")
    ch = ch.replace("-LE", "")
    ch = ch.replace("--", "-")
    ch = ch.replace(" ", "")
    return ch


def is_valid_bipolar_eeg(ch: str) -> bool:
    ch = clean_ch_name(ch)
    return ch in VALID_BIPOLAR_SET


def get_valid_channel_mapping(raw_ch_names):
    cleaned_to_rawidx = {}
    for i, ch in enumerate(raw_ch_names):
        ch_clean = clean_ch_name(ch)
        if is_valid_bipolar_eeg(ch_clean) and ch_clean not in cleaned_to_rawidx:
            cleaned_to_rawidx[ch_clean] = i

    kept_names = [ch for ch in VALID_BIPOLAR_CHANNELS if ch in cleaned_to_rawidx]
    raw_idx = [cleaned_to_rawidx[ch] for ch in kept_names]
    return kept_names, raw_idx


def filter_bad_edfs(df: pd.DataFrame, min_channels: int = 5) -> pd.DataFrame:
    print("[FILTER] Checking EDF channel usability...")

    good_edfs = []
    bad_edfs = []

    for edf_path in df["edf_path"].unique():
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
            kept_names, _ = get_valid_channel_mapping(raw.ch_names)
            n_kept = len(kept_names)

            if n_kept >= min_channels:
                good_edfs.append(edf_path)
            else:
                bad_edfs.append((edf_path, n_kept))
        except Exception:
            bad_edfs.append((edf_path, "error"))

    if bad_edfs:
        print(f"[WARNING] Dropping {len(bad_edfs)} EDF files with < {min_channels} usable bipolar pairs")
        for p, n in bad_edfs:
            print(f"  {Path(p).name}: {n} usable pairs")

    df_filtered = df[df["edf_path"].isin(good_edfs)].copy()
    print(f"[FILTER] Remaining windows after EDF filtering: {len(df_filtered):,}")
    return df_filtered.reset_index(drop=True)


class CHBMITWindowDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, global_ch_names, target_sfreq=256.0, zscore=True):
        self.df = df.reset_index(drop=True)
        self.global_ch_names = list(global_ch_names)
        self.ch_to_idx = {ch: i for i, ch in enumerate(self.global_ch_names)}
        self.target_sfreq = float(target_sfreq)
        self.zscore = bool(zscore)
        self._edf_cache = {}

    def __len__(self):
        return len(self.df)

    def _load_edf_array(self, edf_path: str):
        if edf_path in self._edf_cache:
            return self._edf_cache[edf_path]

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        orig_sfreq = float(raw.info["sfreq"])

        kept_names, raw_idx = get_valid_channel_mapping(raw.ch_names)
        if len(kept_names) == 0:
            raise RuntimeError(f"No valid bipolar EEG channels found in {edf_path}")

        data = raw.get_data(picks=raw_idx).astype(np.float32)

        if orig_sfreq != self.target_sfreq:
            data = mne.filter.resample(
                data,
                up=self.target_sfreq,
                down=orig_sfreq,
                axis=1,
                npad="auto",
                verbose=False,
            )

        sfreq = self.target_sfreq
        self._edf_cache[edf_path] = (data, kept_names, sfreq)
        return self._edf_cache[edf_path]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        edf_path = row["edf_path"]
        win_start_sec = float(row["win_start_sec"])
        win_end_sec = float(row["win_end_sec"])
        y = int(row["label"])

        data, ch_names, sfreq = self._load_edf_array(edf_path)

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

        X = np.zeros((len(self.global_ch_names), x.shape[1]), dtype=np.float32)
        local_index = {ch: i for i, ch in enumerate(ch_names)}

        for ch in ch_names:
            if ch in self.ch_to_idx:
                X[self.ch_to_idx[ch], :] = x[local_index[ch], :]

        if self.zscore:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)
            std[std < 1e-6] = 1.0
            X = (X - mean) / std

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


def build_global_channel_list(df):
    present = set()
    for p in df["edf_path"].unique():
        raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
        kept_names, _ = get_valid_channel_mapping(raw.ch_names)
        present.update(kept_names)

    return [ch for ch in VALID_BIPOLAR_CHANNELS if ch in present]


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


def summarize_labels(df: pd.DataFrame, name: str):
    counts = df["label"].value_counts().sort_index()
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    ratio = (n_neg / n_pos) if n_pos > 0 else float("inf")
    print(f"[{name}] total={len(df):,} | non-seizure={n_neg:,} | seizure={n_pos:,} | neg:pos={ratio:.2f}:1")


def downsample_negatives(train_df: pd.DataFrame, neg_pos_ratio: int = 4, seed: int = 42) -> pd.DataFrame:
    seizure_df = train_df[train_df["label"] == 1].copy()
    non_seizure_df = train_df[train_df["label"] == 0].copy()

    if len(seizure_df) == 0:
        raise ValueError("Training set has zero seizure windows; cannot balance.")

    target_non_seizure = min(len(non_seizure_df), int(neg_pos_ratio * len(seizure_df)))
    non_seizure_down = non_seizure_df.sample(n=target_non_seizure, random_state=seed)

    out = pd.concat([seizure_df, non_seizure_down], axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _find_interval_columns(df: pd.DataFrame):
    candidates = [
        ("seizure_start_sec", "seizure_end_sec"),
        ("sz_start_sec", "sz_end_sec"),
        ("seiz_start_sec", "seiz_end_sec"),
        ("ictal_start_sec", "ictal_end_sec"),
    ]
    cols = set(df.columns)
    for s_col, e_col in candidates:
        if s_col in cols and e_col in cols:
            return s_col, e_col
    return None, None


def augment_seizure_windows(
    train_df: pd.DataFrame,
    shift_sec: float = 0.5,
    min_overlap_frac: float = 0.5,
    window_len_sec: float = 4.0,
) -> pd.DataFrame:
    pos_df = train_df[train_df["label"] == 1].copy()
    if len(pos_df) == 0:
        return train_df.reset_index(drop=True)

    s_col, e_col = _find_interval_columns(pos_df)
    use_exact_intervals = s_col is not None and e_col is not None
    use_fraction = ("seizure_fraction" in pos_df.columns)

    augmented_rows = []
    kept_by_fraction = 0
    kept_by_exact = 0

    for _, row in pos_df.iterrows():
        orig_start = float(row["win_start_sec"])
        orig_end = float(row["win_end_sec"])

        for shift in (-shift_sec, +shift_sec):
            new_start = orig_start + shift
            new_end = orig_end + shift

            if new_start < 0:
                continue

            keep = False
            new_frac = None

            if use_exact_intervals:
                sz_start = float(row[s_col])
                sz_end = float(row[e_col])
                inter = max(0.0, min(new_end, sz_end) - max(new_start, sz_start))
                frac = inter / max(1e-8, (new_end - new_start))
                keep = frac >= min_overlap_frac
                new_frac = frac
                if keep:
                    kept_by_exact += 1
            elif use_fraction:
                orig_frac = float(row["seizure_fraction"])
                est_overlap_sec = max(0.0, orig_frac * window_len_sec - abs(shift))
                est_frac = est_overlap_sec / window_len_sec
                keep = est_frac >= min_overlap_frac
                new_frac = est_frac
                if keep:
                    kept_by_fraction += 1
            else:
                keep = False

            if not keep:
                continue

            new_row = row.copy()
            new_row["win_start_sec"] = new_start
            new_row["win_end_sec"] = new_end
            new_row["aug_shift_sec"] = shift
            if new_frac is not None and "seizure_fraction" in new_row.index:
                new_row["seizure_fraction"] = float(new_frac)
            augmented_rows.append(new_row)

    aug_df = pd.DataFrame(augmented_rows)
    if len(aug_df) == 0:
        print("[AUG] No shifted seizure windows were added.")
        return train_df.reset_index(drop=True)

    if use_exact_intervals:
        print(
            f"[AUG] Used exact seizure intervals ({s_col}, {e_col}) with minimum overlap {min_overlap_frac:.2f}. "
            f"Kept shifted windows: {kept_by_exact:,}."
        )
    elif use_fraction:
        print(
            f"[AUG] Used seizure_fraction with conservative shift rule and minimum overlap {min_overlap_frac:.2f}. "
            f"Kept shifted windows: {kept_by_fraction:,}."
        )
    else:
        print(
            "[AUG][WARNING] No exact seizure intervals or seizure_fraction column found. "
            "Skipping seizure shift augmentation."
        )
        return train_df.reset_index(drop=True)

    out = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
    return out


def make_weighted_sampler_from_df(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df["label"].to_numpy(dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(f"Cannot build sampler because one class is missing. class_counts={class_counts}")

    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_weights_from_df(df: pd.DataFrame) -> torch.Tensor:
    labels = df["label"].to_numpy(dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError(f"Cannot compute class weights because one class is missing. class_counts={class_counts}")

    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def assert_split_integrity(train_df, val_df, test_df):
    train_edfs = set(train_df["edf_path"].unique())
    val_edfs = set(val_df["edf_path"].unique())
    test_edfs = set(test_df["edf_path"].unique())

    assert train_edfs.isdisjoint(val_edfs), "Leakage: train and val share EDF files"
    assert train_edfs.isdisjoint(test_edfs), "Leakage: train and test share EDF files"
    assert val_edfs.isdisjoint(test_edfs), "Leakage: val and test share EDF files"

    print(
        f"[SPLIT] OK | train_edfs={len(train_edfs)} | val_edfs={len(val_edfs)} | test_edfs={len(test_edfs)}"
    )


def inspect_loader_batches(loader, n_batches: int = 5):
    print(f"[BATCHES] Inspecting first {n_batches} training batches...")
    pos_total = 0
    neg_total = 0

    it = iter(loader)
    for i in range(n_batches):
        xb, yb = next(it)
        binc = torch.bincount(yb, minlength=2)
        neg = int(binc[0].item())
        pos = int(binc[1].item())
        neg_total += neg
        pos_total += pos
        print(f"  batch {i+1}: X={tuple(xb.shape)} | non-seizure={neg} | seizure={pos}")

    ratio = (neg_total / pos_total) if pos_total > 0 else float("inf")
    print(f"[BATCHES] aggregate over inspected batches | non-seizure={neg_total} | seizure={pos_total} | neg:pos={ratio:.2f}:1")


def report_channel_coverage(global_ch_names):
    print(f"[CHANNELS] Global channel count: {len(global_ch_names)}")
    print(f"[CHANNELS] {global_ch_names}")


def preview_examples(df: pd.DataFrame, name: str, n: int = 3):
    cols = [c for c in ["subject", "edf_path", "win_start_sec", "win_end_sec", "label", "aug_shift_sec"] if c in df.columns]
    print(f"[{name}] preview:")
    if len(df) == 0:
        print("  <empty>")
        return
    print(df[cols].head(n).to_string(index=False))


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded windows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Remove invalid EDFs first
    df = filter_bad_edfs(df, min_channels=MIN_USABLE_BIPOLAR_PAIRS)
    summarize_labels(df, "ALL")

    train_df, val_df, test_df = subject_edf_split(df, test_subject="chb03", val_fraction=0.1, seed=SEED)
    assert_split_integrity(train_df, val_df, test_df)

    summarize_labels(train_df, "TRAIN_RAW")
    summarize_labels(val_df, "VAL")
    summarize_labels(test_df, "TEST")

    global_ch_names = build_global_channel_list(train_df)
    report_channel_coverage(global_ch_names)

    train_balanced_df = downsample_negatives(
        train_df,
        neg_pos_ratio=DOWNSAMPLE_NEG_POS_RATIO,
        seed=SEED,
    )
    summarize_labels(train_balanced_df, "TRAIN_AFTER_DOWNSAMPLE")

    if AUGMENT_POSITIVES:
        before_aug = len(train_balanced_df)
        train_balanced_df = augment_seizure_windows(
            train_balanced_df,
            shift_sec=SEIZURE_SHIFT_SEC,
            min_overlap_frac=MIN_SHIFTED_SEIZURE_OVERLAP_FRAC,
            window_len_sec=WINDOW_LEN_SEC,
        )
        added = len(train_balanced_df) - before_aug
        print(f"[AUG] Added {added:,} shifted seizure windows")
        summarize_labels(train_balanced_df, "TRAIN_AFTER_AUG")

    preview_examples(train_balanced_df, "TRAIN_BALANCED", n=5)

    train_ds = CHBMITWindowDataset(
        train_balanced_df,
        global_ch_names,
        target_sfreq=TARGET_SFREQ,
        zscore=Z_SCORE_PER_CHANNEL,
    )
    val_ds = CHBMITWindowDataset(
        val_df,
        global_ch_names,
        target_sfreq=TARGET_SFREQ,
        zscore=Z_SCORE_PER_CHANNEL,
    )
    test_ds = CHBMITWindowDataset(
        test_df,
        global_ch_names,
        target_sfreq=TARGET_SFREQ,
        zscore=Z_SCORE_PER_CHANNEL,
    )

    if USE_WEIGHTED_SAMPLER:
        sampler = make_weighted_sampler_from_df(train_balanced_df)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
        )
        print("[LOADER] Using WeightedRandomSampler for train_loader")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        print("[LOADER] Using shuffle=True for train_loader")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_weights = compute_class_weights_from_df(train_balanced_df)
    print(f"[LOSS] Suggested class weights from balanced train set: {class_weights.tolist()}")

    inspect_loader_batches(train_loader, n_batches=5)

    xb, yb = next(iter(train_loader))
    print(f"[ONE_BATCH] X={tuple(xb.shape)} | y={tuple(yb.shape)} | y_counts={torch.bincount(yb, minlength=2).tolist()}")
    print(f"[READY] train_ds={len(train_ds):,} | val_ds={len(val_ds):,} | test_ds={len(test_ds):,}")

    return {
        "df_cleaned": df,
        "train_df_raw": train_df,
        "train_df_balanced": train_balanced_df,
        "val_df": val_df,
        "test_df": test_df,
        "global_ch_names": global_ch_names,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
    }


if __name__ == "__main__":
    main()
