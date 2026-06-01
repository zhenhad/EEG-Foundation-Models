# EEG Foundation Models for Clinical EEG Analysis

This repository contains my research and implementation work on EEG foundation models and their application to downstream clinical EEG tasks.

The current repository includes experiments with:

- REVE (Representation for EEG with Versatile Embeddings)
- EEGPT (EEG Foundation Model)

The primary focus of this work is investigating whether pretrained EEG foundation models can learn transferable representations that generalize to unseen clinical datasets and support downstream seizure detection tasks.

---

## Repository Structure

```text
EEG-Foundation-Models/
│
├── REVE/
│   ├── download_chbmit.py
│   ├── make_windows.py
│   ├── build_dataset_balanced.py
│   ├── smoke_test_reve.py
│   ├── train_linear_probe.py
│   ├── train_linear_probe_threshold_tuned.py
│   ├── save_test_predictions.py
│   ├── plot_test_predictions.py
│   ├── event_level_evaluation.py
│   ├── channel_importance.py
│   └── README.md
│
├── EEGPT/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
