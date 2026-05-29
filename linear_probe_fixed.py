from pathlib import Path
import copy
import numpy as np
import pandas as pd
import mne
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
from tqdm.auto import tqdm

from braindecode.models import REVE

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")
CSV_PATH = ROOT / "windows_chbmit_all_4s_50ol.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SFREQ = 256.0
SEED = 42

TARGET_ELECTRODES = [
    "FP1", "F7", "T7", "P7", "O1",
    "F3", "C3", "P3",
    "FP2", "F4", "C4", "P4", "O2",
    "F8", "T8", "P8",
    "FZ", "CZ", "PZ"
]

TEST_SUBJECT = "chb03"
N_EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

MIN_RECONSTRUCTABLE_PAIRS = 5
DOWNSAMPLE_NEG_POS_RATIO = 4
AUGMENT_POSITIVES = True
SEIZURE_SHIFT_SEC = 0.5
MIN_SHIFTED_SEIZURE_OVERLAP_FRAC = 0.50
WINDOW_LEN_SEC = 4.0
USE_WEIGHTED_SAMPLER = True

USE_SMOKE_SUBSET = False
SMOKE_TRAIN_N = 5000
SMOKE_VAL_N = 1000

RESUME_IF_AVAILABLE = False
BEST_CKPT_PATH = ROOT / f"linear_probe_reve_{TEST_SUBJECT}_mono_balanced_best.pt"
LAST_CKPT_PATH = ROOT / f"linear_probe_reve_{TEST_SUBJECT}_mono_balanced_last.pt"
FINAL_CKPT_PATH = ROOT / f"linear_probe_reve_{TEST_SUBJECT}_mono_balanced_final.pt"


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


def parse_bipolar_name(ch):
    ch = str(ch).upper().strip()
    ch = ch.replace("EEG ", "")
    ch = ch.replace("-REF", "")
    ch = ch.replace("-LE", "")
    ch = ch.replace(" ", "")

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
        pairs = build_pair_list(ch_names)
        return len(pairs)
    except Exception as e:
        print(f"[WARNING] Failed to inspect EDF {edf_path}: {e}")
        return 0


def filter_df_by_reconstructable_edfs(df, min_pairs=5):
    unique_edfs = df["edf_path"].drop_duplicates().tolist()
    good_edfs = []
    bad_edfs = []

    for edf_path in unique_edfs:
        n_pairs = count_usable_pairs_for_edf(edf_path)
        if n_pairs >= min_pairs:
            good_edfs.append(edf_path)
        else:
            bad_edfs.append((edf_path, n_pairs))

    if bad_edfs:
        print(f"[WARNING] Dropping {len(bad_edfs)} EDF files with < {min_pairs} usable bipolar pairs")
        for edf_path, n_pairs in bad_edfs[:10]:
            print(f"  {Path(edf_path).name}: {n_pairs} usable pairs")
        if len(bad_edfs) > 10:
            print(f"  ... and {len(bad_edfs) - 10} more")

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

    out = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
    return out


def make_weighted_sampler(labels):
    labels = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=2)
    class_counts = torch.clamp(class_counts, min=1)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


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
            x = mne.filter.resample(
                x,
                up=self.target_sfreq,
                down=sfreq,
                axis=1,
                npad="auto",
                verbose=False,
            ).astype(np.float32)
            sfreq = self.target_sfreq

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
        self.n_outputs = n_outputs

        for p in self.backbone.parameters():
            p.requires_grad = False

        target_layer = None
        target_name = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                target_layer = module
                target_name = name

        if target_layer is None:
            raise RuntimeError("Could not find final linear layer in REVE.")

        print(f"Hooking penultimate features from layer: {target_name}")

        def hook_fn(module, inputs, output):
            self._features = inputs[0]

        self._hook = target_layer.register_forward_hook(hook_fn)

    def build_head(self, xb):
        self.backbone.eval()
        with torch.no_grad():
            _ = self.backbone(xb)
            feats = self._features
            if feats is None:
                raise RuntimeError("Feature hook did not capture anything.")
            feat_dim = feats.shape[-1]

        self.head = nn.Linear(feat_dim, self.n_outputs).to(xb.device)
        print(f"Linear head created: {feat_dim} -> {self.n_outputs}")

    def forward(self, x):
        _ = self.backbone(x)
        feats = self._features
        if feats is None:
            raise RuntimeError("No features captured from backbone.")
        return self.head(feats)


def evaluate(model, loader, criterion, desc="Evaluating"):
    model.eval()
    all_y, all_prob, all_pred = [], [], []
    total_loss, n_total = 0.0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n_total += bs

            all_y.extend(yb.cpu().numpy().tolist())
            all_prob.extend(probs.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(n_total, 1),
        "acc": accuracy_score(all_y, all_pred),
        "bal_acc": balanced_accuracy_score(all_y, all_pred),
        "f1": f1_score(all_y, all_pred, zero_division=0),
        "auroc": roc_auc_score(all_y, all_prob) if len(set(all_y)) > 1 else float("nan"),
        "auprc": average_precision_score(all_y, all_prob) if len(set(all_y)) > 1 else float("nan"),
    }


def save_checkpoint(path, epoch, model, optimizer, best_val_auprc):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_auprc": best_val_auprc,
            "target_electrodes": TARGET_ELECTRODES,
            "target_sfreq": TARGET_SFREQ,
            "test_subject": TEST_SUBJECT,
        },
        path,
    )


def inspect_loader_batches(loader, n_batches=3):
    print(f"[BATCHES] Inspecting first {n_batches} train batches...")
    it = iter(loader)
    for i in range(n_batches):
        xb, yb = next(it)
        binc = torch.bincount(yb, minlength=2)
        print(
            f"  batch {i+1}: X={tuple(xb.shape)} | "
            f"non-seizure={int(binc[0])} | seizure={int(binc[1])}"
        )


def main():
    print("Device:", DEVICE)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded windows: {len(df):,}")
    summarize_labels(df, "ALL_RAW")

    train_df, val_df, test_df = subject_edf_split(df, test_subject=TEST_SUBJECT, seed=SEED)

    train_df = filter_df_by_reconstructable_edfs(train_df, min_pairs=MIN_RECONSTRUCTABLE_PAIRS)
    val_df = filter_df_by_reconstructable_edfs(val_df, min_pairs=MIN_RECONSTRUCTABLE_PAIRS)
    test_df = filter_df_by_reconstructable_edfs(test_df, min_pairs=MIN_RECONSTRUCTABLE_PAIRS)

    summarize_labels(train_df, "TRAIN_RAW_CLEAN")
    summarize_labels(val_df, "VAL_CLEAN")
    summarize_labels(test_df, "TEST_CLEAN")

    if USE_SMOKE_SUBSET:
        train_df = train_df.sample(min(SMOKE_TRAIN_N, len(train_df)), random_state=SEED)
        val_df = val_df.sample(min(SMOKE_VAL_N, len(val_df)), random_state=SEED)
        print(f"Smoke subset enabled | train={len(train_df)} val={len(val_df)}")

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
        print(f"[AUG] Added {len(train_balanced_df) - before_aug:,} shifted seizure windows")
        summarize_labels(train_balanced_df, "TRAIN_AFTER_AUG")

    train_ds = CHBMITMonoWindowDataset(train_balanced_df, TARGET_ELECTRODES, target_sfreq=TARGET_SFREQ, zscore=True)
    val_ds = CHBMITMonoWindowDataset(val_df, TARGET_ELECTRODES, target_sfreq=TARGET_SFREQ, zscore=True)
    test_ds = CHBMITMonoWindowDataset(test_df, TARGET_ELECTRODES, target_sfreq=TARGET_SFREQ, zscore=True)

    if USE_WEIGHTED_SAMPLER:
        sampler = make_weighted_sampler(train_balanced_df["label"].tolist())
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        print("[LOADER] Using WeightedRandomSampler for train_loader")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        print("[LOADER] Using shuffle=True for train_loader")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    inspect_loader_batches(train_loader, n_batches=3)

    xb0, yb0 = next(iter(train_loader))
    xb0 = xb0.to(DEVICE)
    print("One batch:", xb0.shape, yb0.shape)

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
    model.build_head(xb0)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss()

    best_val_auprc = -1.0
    best_state = None
    start_epoch = 1

    if RESUME_IF_AVAILABLE and LAST_CKPT_PATH.exists():
        print(f"Resuming from checkpoint: {LAST_CKPT_PATH}")
        ckpt = torch.load(LAST_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_auprc = ckpt.get("best_val_auprc", -1.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resume start_epoch={start_epoch}, best_val_auprc={best_val_auprc:.4f}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [train]", leave=True)
        for xb, yb in train_pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n_total += bs
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(n_total, 1)
        val_metrics = evaluate(model, val_loader, criterion, desc=f"Epoch {epoch:02d} [val]")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_bal_acc={val_metrics['bal_acc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f}"
        )

        save_checkpoint(LAST_CKPT_PATH, epoch, model, optimizer, best_val_auprc)
        print(f"Saved last checkpoint: {LAST_CKPT_PATH}")

        if np.isfinite(val_metrics["auprc"]) and val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(BEST_CKPT_PATH, epoch, model, optimizer, best_val_auprc)
            print(f"Saved best checkpoint by val_auprc: {BEST_CKPT_PATH}")

    if best_state is None and BEST_CKPT_PATH.exists():
        print("\nLoading best checkpoint from disk...")
        best_ckpt = torch.load(BEST_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state_dict"])
    elif best_state is not None:
        print("\nLoading best validation model from memory...")
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, desc="Final test")
    print("\nTEST RESULTS")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    save_checkpoint(FINAL_CKPT_PATH, N_EPOCHS, model, optimizer, best_val_auprc)
    print(f"\nSaved final checkpoint to: {FINAL_CKPT_PATH}")


if __name__ == "__main__":
    main()
