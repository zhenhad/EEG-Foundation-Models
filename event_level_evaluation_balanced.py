from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")
TEST_SUBJECT = "chb03"
PRED_CSV = ROOT / f"test_predictions_{TEST_SUBJECT}_mono_balanced.csv"
THRESHOLD = 0.50

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
        elif not pred and in_event:
            events.append((start, end))
            in_event = False
    if in_event:
        events.append((start, end))
    return events

def event_detected(true_event, pred_events):
    s_true, e_true = true_event
    for s_pred, e_pred in pred_events:
        if (s_pred < e_true) and (e_pred > s_true):
            return True
    return False

def compute_latency(true_event, pred_events):
    s_true, _ = true_event
    for s_pred, _ in pred_events:
        if s_pred >= s_true:
            return s_pred - s_true
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
    df = pd.read_csv(PRED_CSV).sort_values(["edf", "win_start_sec"])
    total_duration_sec = 0
    total_false_alarms = 0
    total_detected = 0
    total_events = 0
    latencies = []
    for _, g in df.groupby("edf"):
        g = g.sort_values("win_start_sec")
        duration = g["win_end_sec"].max()
        total_duration_sec += duration
        true_events = get_true_events(g)
        pred_events = get_predicted_events(g)
        total_events += len(true_events)
        for event in true_events:
            if event_detected(event, pred_events):
                total_detected += 1
                latency = compute_latency(event, pred_events)
                if latency is not None:
                    latencies.append(latency)
        total_false_alarms += compute_false_alarms(pred_events, true_events)
    sensitivity = total_detected / max(total_events, 1)
    hours = total_duration_sec / 3600.0
    fp_per_hour = total_false_alarms / max(hours, 1e-6)
    avg_latency = np.mean(latencies) if latencies else None
    print("\n===== EVENT-LEVEL RESULTS =====")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Total seizure events: {total_events}")
    print(f"Detected events: {total_detected}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"False alarms: {total_false_alarms}")
    print(f"False alarms per hour: {fp_per_hour:.4f}")
    print(f"Average latency (sec): {avg_latency:.2f}" if avg_latency is not None else "No latency computed")

if __name__ == "__main__":
    main()
