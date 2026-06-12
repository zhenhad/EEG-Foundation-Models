"""
download_chbmit_whole.py

Utility script for downloading the complete CHB-MIT Scalp EEG Database
from PhysioNet and organizing recordings by subject.

Dataset:
    CHB-MIT Scalp EEG Database
    https://physionet.org/content/chbmit/1.0.0/

Author:
    Zhenous Hadi Jafari
    zhenous.hadijafari@uta.edu
    PhD Student, Bioengineering
    University of Texas at Arlington

Project:
    EEG Foundation Models for Seizure Detection
    REVE-based seizure classification on CHB-MIT EEG recordings

Description:
    This script automatically:
    1. Retrieves all subject folders from PhysioNet.
    2. Downloads EDF recordings and annotation files.
    3. Organizes files into subject-specific directories.
    4. Skips files already downloaded.

Usage:
    python download_chbmit_whole.py
"""

import os
import urllib.request
import re

base_url = "https://physionet.org/files/chbmit/1.0.0/"

def fetch_html(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")

def get_subject_list():
    """Find all subject folders like chb01, chb02, ..."""
    print("Fetching subject list...")
    html = fetch_html(base_url)

    subjects = sorted(set(re.findall(r'href="(chb\d{2})/"', html)))
    print(f"Found subjects: {subjects}")
    return subjects

def get_file_list(subj_url, subj_id):
    """Get all files inside a subject folder"""
    try:
        html = fetch_html(subj_url)
        return set(re.findall(r'href="([^"]+)"', html))
    except Exception as e:
        print(f"Error fetching {subj_id}: {e}")
        return set()

subjects = get_subject_list()

for subj in subjects:
    print(f"\n--- DOWNLOADING {subj} ---")
    subj_url = f"{base_url}{subj}/"

    os.makedirs(subj, exist_ok=True)

    files = get_file_list(subj_url, subj)

    for file_name in sorted(files):
        # skip directories or parent links
        if file_name.endswith("/") or ".." in file_name:
            continue

        file_url = subj_url + file_name
        destination = os.path.join(subj, file_name)

        if os.path.exists(destination):
            print(f"Skipping {file_name}")
            continue

        print(f"Downloading {file_name}...")
        try:
            urllib.request.urlretrieve(file_url, destination)
        except Exception as e:
            print(f"Failed: {file_name} -> {e}")

print("\n--- ALL SUBJECTS DOWNLOADED ---")
