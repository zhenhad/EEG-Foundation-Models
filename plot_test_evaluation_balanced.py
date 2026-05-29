from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")
TEST_SUBJECT = "chb03"
PRED_CSV = ROOT / f"test_predictions_{TEST_SUBJECT}_mono_balanced.csv"
OUT_DIR = ROOT / f"plots_{TEST_SUBJECT}_mono_balanced"
THRESHOLD = 0.50

def find_contiguous_regions(starts, ends, labels):
    regions = []
    in_region = False
    region_start = None
    region_end = None
    for s, e, lab in zip(starts, ends, labels):
        if lab == 1 and not in_region:
            region_start = s
            region_end = e
            in_region = True
        elif lab == 1 and in_region:
            region_end = e
        elif lab == 0 and in_region:
            regions.append((region_start, region_end))
            in_region = False
            region_start = None
            region_end = None
    if in_region:
        regions.append((region_start, region_end))
    return regions

def save_confusion_matrix(df):
    y_true = df["y_true"].values
    y_pred = (df["prob_seizure"].values >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-seizure", "Seizure"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix | {TEST_SUBJECT} | threshold={THRESHOLD:.2f}")
    plt.tight_layout()
    out_path = OUT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")

def save_roc_curve(df):
    y_true = df["y_true"].values
    y_prob = df["prob_seizure"].values
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve | {TEST_SUBJECT}")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "roc_curve.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")

def save_pr_curve(df):
    y_true = df["y_true"].values
    y_prob = df["prob_seizure"].values
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve | {TEST_SUBJECT}")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "pr_curve.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")

def save_probability_histogram(df):
    plt.figure(figsize=(7, 5))
    plt.hist(df.loc[df["y_true"] == 0, "prob_seizure"], bins=50, alpha=0.6, label="True non-seizure")
    plt.hist(df.loc[df["y_true"] == 1, "prob_seizure"], bins=50, alpha=0.6, label="True seizure")
    plt.axvline(THRESHOLD, linestyle="--", linewidth=1.5, label=f"Threshold = {THRESHOLD:.2f}")
    plt.xlabel("Predicted seizure probability")
    plt.ylabel("Count")
    plt.title(f"Probability Histogram | {TEST_SUBJECT}")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "probability_histogram.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")

def save_threshold_sweep(df):
    y_true = df["y_true"].values
    y_prob = df["prob_seizure"].values
    thresholds = np.arange(0.01, 0.99, 0.01)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        })
    thr_df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
    plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
    plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
    plt.plot(thr_df["threshold"], thr_df["bal_acc"], label="Balanced Acc")
    plt.axvline(THRESHOLD, linestyle="--", linewidth=1.5, label=f"Chosen threshold = {THRESHOLD:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title(f"Threshold Sweep | {TEST_SUBJECT}")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "threshold_sweep.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")
    best_row = thr_df.sort_values("f1", ascending=False).iloc[0]
    print("\nBest threshold by F1 from prediction CSV:")
    print(best_row.to_string())

def save_probability_timelines(df):
    df = df.sort_values(["edf", "win_start_sec"]).reset_index(drop=True)
    for edf_name, g in df.groupby("edf"):
        g = g.sort_values("win_start_sec").reset_index(drop=True)
        x = g["win_start_sec"].values
        x_end = g["win_end_sec"].values
        y_prob = g["prob_seizure"].values
        y_true = g["y_true"].values
        plt.figure(figsize=(14, 5))
        plt.plot(x, y_prob, linewidth=1.5, label="Predicted seizure probability")
        plt.axhline(THRESHOLD, linestyle="--", linewidth=1, label=f"Threshold = {THRESHOLD:.2f}")
        seizure_regions = find_contiguous_regions(x, x_end, y_true)
        for i, (s, e) in enumerate(seizure_regions):
            plt.axvspan(s, e, alpha=0.2, label="True seizure interval" if i == 0 else None)
        plt.title(f"{TEST_SUBJECT} | {edf_name}")
        plt.xlabel("Window start time (sec)")
        plt.ylabel("Predicted seizure probability")
        plt.ylim(-0.02, 1.02)
        plt.legend()
        plt.tight_layout()
        out_path = OUT_DIR / f"{edf_name}_prob_plot.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED_CSV)
    save_confusion_matrix(df)
    save_roc_curve(df)
    save_pr_curve(df)
    save_probability_histogram(df)
    save_threshold_sweep(df)
    save_probability_timelines(df)
    print("\nDone.")

if __name__ == "__main__":
    main()
