# REVE for EEG Seizure Detection

This folder contains my implementation of a REVE-based seizure detection pipeline using the CHB-MIT scalp EEG dataset.

REVE, Representation for EEG with Versatile Embeddings, is an EEG foundation model designed to generalize across different EEG montages by using 4D spatio-temporal positional encoding based on electrode coordinates and time.

## Goal

The goal of this experiment was to test whether frozen pretrained REVE embeddings contain seizure-related information that can be used for downstream seizure vs. non-seizure classification.

Instead of fully fine-tuning REVE, I used a linear probing setup:

```text
EEG window → frozen REVE backbone → feature vector → linear classifier → seizure prediction
