from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")
TEST_SUBJECT = "chb03"

# Updated for filtered FIF pipeline
PRED_CSV = ROOT / f"test_predictions_{TEST_SUBJECT}_mono_balanced_filtered_fif.csv"

# Best validation threshold
THRESHOLD = 0.98


def get_true_events(df):
    events = []
    in_event = False

    for _, row in df.iterrows():
        if row["y_true"] == 1 and not in_event:
            start = row["win_start_sec"]
            end = row["win_end_sec"]
            in_event = True

        elif row["y_true"] == 1 and in_event:
            end = row["win_end_sec"]

        elif row["y_true"] == 0 and in_event:
            events.append((start, end))
            in_event = False

    if in_event:
        events.append((start, end))

    return events


def get_predicted_events(df):
    events = []
    in_event = False

    for _, row in df.iterrows():
        pred = row["prob_seizure"] >= THRESHOLD

        if pred and not in_event:
            start = row["win_start_sec"]
            end = row["win_end_sec"]
            in_event = True

        elif pred and in_event:
            end = row["win_end_sec"]

        elif (not pred) and in_event:
            events.append((start, end))
            in_event = False

    if in_event:
        events.append((start, end))

    return events


def event_detected(true_event, pred_events):
    s_true, e_true = true_event

    for s_pred, e_pred in pred_events:
        # Any overlap counts as a detection.
        if (s_pred < e_true) and (e_pred > s_true):
            return True

    return False


def compute_latency(true_event, pred_events):
    """
    Detection latency:
    0 sec if seizure was detected before onset and overlaps.
    Positive value if first detection occurs after onset.
    """
    s_true, e_true = true_event

    for s_pred, e_pred in pred_events:
        if (s_pred < e_true) and (e_pred > s_true):
            return max(0.0, s_pred - s_true)

    return None


def compute_false_alarms(pred_events, true_events):
    false_alarms = 0

    for s_pred, e_pred in pred_events:
        overlap = False

        for s_true, e_true in true_events:
            if (s_pred < e_true) and (e_pred > s_true):
                overlap = True
                break

        if not overlap:
            false_alarms += 1

    return false_alarms


def main():
    df = pd.read_csv(PRED_CSV)
    df = df.sort_values(["edf", "win_start_sec"]).reset_index(drop=True)

    total_duration_sec = 0.0
    total_false_alarms = 0
    total_detected = 0
    total_events = 0
    latencies = []

    print(f"Loaded prediction CSV: {len(df):,} windows")

    for edf_name, g in df.groupby("edf"):
        g = g.sort_values("win_start_sec").reset_index(drop=True)

        duration = float(g["win_end_sec"].max())
        total_duration_sec += duration

        true_events = get_true_events(g)
        pred_events = get_predicted_events(g)

        total_events += len(true_events)

        for event in true_events:
            if event_detected(event, pred_events):
                total_detected += 1

                latency = compute_latency(
                    event,
                    pred_events,
                )

                if latency is not None:
                    latencies.append(latency)

        total_false_alarms += compute_false_alarms(
            pred_events,
            true_events,
        )

    sensitivity = (
        total_detected / total_events
        if total_events > 0
        else 0.0
    )

    hours = total_duration_sec / 3600.0

    fp_per_hour = (
        total_false_alarms / hours
        if hours > 0
        else 0.0
    )

    fp_per_24h = fp_per_hour * 24.0

    mean_latency = (
        float(np.mean(latencies))
        if len(latencies) > 0
        else np.nan
    )

    median_latency = (
        float(np.median(latencies))
        if len(latencies) > 0
        else np.nan
    )

    print("\n===== EVENT-LEVEL RESULTS =====")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Total seizure events: {total_events}")
    print(f"Detected events: {total_detected}")
    print(f"Event sensitivity: {sensitivity:.4f}")
    print(f"False alarms: {total_false_alarms}")
    print(f"False alarms per hour: {fp_per_hour:.4f}")
    print(f"False alarms per 24 h: {fp_per_24h:.2f}")

    if len(latencies) > 0:
        print(f"Mean latency (sec): {mean_latency:.2f}")
        print(f"Median latency (sec): {median_latency:.2f}")
    else:
        print("No detection latencies computed.")

    # Save a summary table for the Results section
    summary = pd.DataFrame(
        {
            "Metric": [
                "Threshold",
                "Total seizure events",
                "Detected events",
                "Event sensitivity",
                "False alarms",
                "False alarms per hour",
                "False alarms per 24 h",
                "Mean latency (sec)",
                "Median latency (sec)",
            ],
            "Value": [
                THRESHOLD,
                total_events,
                total_detected,
                sensitivity,
                total_false_alarms,
                fp_per_hour,
                fp_per_24h,
                mean_latency,
                median_latency,
            ],
        }
    )

    out_csv = (
        ROOT
        / f"event_level_metrics_{TEST_SUBJECT}_mono_balanced_filtered_fif.csv"
    )

    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary table:\n{out_csv}")


if __name__ == "__main__":
    main()
