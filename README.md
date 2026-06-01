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
```
## Project Overview


EEG recordings are highly heterogeneous across subjects, devices, sampling rates, and electrode montages. This makes generalization difficult for traditional task-specific EEG models.

This project explores whether pretrained EEG foundation models can transfer to a clinically relevant downstream task: seizure detection.

For the REVE experiment, I used the CHB-MIT scalp EEG dataset, converted continuous EDF recordings into labeled EEG windows, and trained a linear classifier on top of a frozen pretrained REVE backbone.

## Current Status

- REVE pipeline implemented
- CHB-MIT preprocessing completed
- Linear probing experiments completed
- Class imbalance handling added
- Test-time prediction export added
- Event-level evaluation added
- Gradient-based channel importance analysis added

## Author

Zhenous Hadi Jafari
PhD Student, Biomedical Engineering
University of Texas at Arlington
- GitHub: [zhenhad](https://github.com/zhenhad)
- LinkedIn: [Zhenous (Zee) Hadi Jafari](https://www.linkedin.com/in/zhenous-hadi-jafari-b086b36a/)
