import re
from pathlib import Path
import pandas as pd
import mne

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")

# Automatically detect subject folders like chb01, chb02, ..., chb24
SUBJECTS = sorted([p.name for p in ROOT.glob("chb*") if p.is_dir()])

WIN_SEC = 4.0
OVERLAP = 0.5
STEP_SEC = WIN_SEC * (1 - OVERLAP)   # 2 seconds


def parse_summary(summary_path: Path):
    """
    Parse chbXX-summary.txt and return:
        dict: {edf_filename: [(start_sec, end_sec), ...]}
    Works with both:
      - "Seizure 1 Start Time: ..."
      - "Seizure Start Time: ..."
    """
    text = summary_path.read_text(errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    seizures = {}
    current_file = None

    for ln in lines:
        m = re.match(r"File Name:\s*(\S+)", ln)
        if m:
            current_file = m.group(1).strip()
            seizures.setdefault(current_file, [])
            continue

        if current_file is None:
            continue

        m_start = re.match(r"Seizure(\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", ln)
        if m_start:
            start = float(m_start.group(2))
            seizures[current_file].append([start, None])
            continue

        m_end = re.match(r"Seizure(\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", ln)
        if m_end and seizures[current_file]:
            end = float(m_end.group(2))
            for k in range(len(seizures[current_file]) - 1, -1, -1):
                if seizures[current_file][k][1] is None:
                    seizures[current_file][k][1] = end
                    break

    cleaned = {}
    for fname, intervals in seizures.items():
        good = []
        for s, e in intervals:
            if e is not None and e > s:
                good.append((float(s), float(e)))
        cleaned[fname] = good

    return cleaned


def overlap_duration(win_start, win_end, intervals):
    """Return total overlap duration between a window and seizure intervals."""
    total = 0.0
    for s, e in intervals:
        ov = max(0.0, min(win_end, e) - max(win_start, s))
        total += ov
    return total


rows = []

print(f"Detected subject folders: {SUBJECTS}")

for subj in SUBJECTS:
    subj_dir = ROOT / subj

    summary_path = subj_dir / f"{subj}-summary.txt"
    if not summary_path.exists():
        print(f"[WARNING] Missing summary file for {subj}, skipping.")
        continue

    seizure_map = parse_summary(summary_path)
    print(f"[{subj}] Seizure EDFs in summary: {sum(len(v) > 0 for v in seizure_map.values())}")

    edfs = sorted(subj_dir.glob("*.edf"))
    if not edfs:
        print(f"[WARNING] No EDF files found in {subj_dir}, skipping.")
        continue

    print(f"[{subj}] Found {len(edfs)} EDF files")

    for edf_path in edfs:
        edf_name = edf_path.name
        intervals = seizure_map.get(edf_name, [])

        try:
            raw = mne.io.read_raw_edf(edf_path.as_posix(), preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            duration_sec = raw.n_times / sfreq
            n_channels = len(raw.ch_names)
        except Exception as e:
            print(f"[WARNING] Could not read {edf_path.name}: {e}")
            continue

        win_start = 0.0
        while (win_start + WIN_SEC) <= duration_sec:
            win_end = win_start + WIN_SEC

            ov_sec = overlap_duration(win_start, win_end, intervals)
            seizure_fraction = ov_sec / WIN_SEC
            y = 1 if ov_sec > 0 else 0

            rows.append({
                "subject": subj,
                "edf": edf_name,
                "edf_path": str(edf_path),
                "sfreq": sfreq,
                "n_channels": n_channels,
                "duration_sec": duration_sec,
                "win_start_sec": win_start,
                "win_end_sec": win_end,
                "overlap_sec": ov_sec,
                "seizure_fraction": seizure_fraction,
                "label": y
            })

            win_start += STEP_SEC

print("\nBuilding dataframe...")
df = pd.DataFrame(rows)

# Sort for clean downstream processing
df = df.sort_values(["subject", "edf", "win_start_sec"]).reset_index(drop=True)

out_csv = ROOT / "windows_chbmit_all_4s_50ol.csv"
df.to_csv(out_csv, index=False)

print(f"\nSaved: {out_csv}")
print("\nOverall label counts:")
print(df["label"].value_counts().rename({0: "non-seizure", 1: "seizure"}))

print("\nPer-subject label counts:")
print(df.groupby("subject")["label"].value_counts())

print("\nDone.")