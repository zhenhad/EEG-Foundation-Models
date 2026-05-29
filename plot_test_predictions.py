from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")

TEST_SUBJECT = "chb03"
PRED_CSV = ROOT / f"test_predictions_{TEST_SUBJECT}.csv"
OUT_DIR = ROOT / f"plots_{TEST_SUBJECT}"
THRESHOLD = 0.5


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PRED_CSV)
    df = df.sort_values(["edf", "win_start_sec"]).reset_index(drop=True)

    for edf_name, g in df.groupby("edf"):
        g = g.sort_values("win_start_sec").reset_index(drop=True)

        x = g["win_start_sec"].values
        x_end = g["win_end_sec"].values
        y_prob = g["prob_seizure"].values
        y_true = g["y_true"].values

        plt.figure(figsize=(14, 5))
        plt.plot(x, y_prob, linewidth=1.5, label="Predicted seizure probability")
        plt.axhline(THRESHOLD, linestyle="--", linewidth=1, label=f"Threshold = {THRESHOLD}")

        seizure_regions = find_contiguous_regions(x, x_end, y_true)
        for i, (s, e) in enumerate(seizure_regions):
            plt.axvspan(
                s, e,
                alpha=0.2,
                label="True seizure interval" if i == 0 else None
            )

        plt.title(f"{TEST_SUBJECT} | {edf_name}")
        plt.xlabel("Window start time (sec)")
        plt.ylabel("Predicted seizure probability")
        plt.ylim(-0.02, 1.02)
        plt.legend()
        plt.tight_layout()

        out_path = OUT_DIR / f"{edf_name}_prob_plot_shaded.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()