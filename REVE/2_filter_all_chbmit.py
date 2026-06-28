from pathlib import Path
import mne
# PATHS-------------------------------------------------------
INPUT_ROOT = Path(
    r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit"
)
OUTPUT_ROOT = Path(
    r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit_filtered_0p5_40"
)
OUTPUT_ROOT.mkdir(exist_ok=True)
# FIND ALL EDF FILES------------------------------------------------
edf_files = sorted(INPUT_ROOT.rglob("*.edf"))
print(f"Found {len(edf_files)} EDF files")

# FILTER + SAVE---------------------------------------------------
for i, edf_path in enumerate(edf_files, start=1):
    rel_path = edf_path.relative_to(INPUT_ROOT)
    out_dir = OUTPUT_ROOT / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{edf_path.stem}_bp_0p5_40_raw.fif"
    if out_file.exists():
        print(f"[SKIP] {out_file.name}")
        continue
    try:
        if i % 25 == 0 or i == 1:
            print(f"[{i}/{len(edf_files)}] {edf_path.name}")
        raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
        )
        raw.set_meas_date(None)
        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            fir_design="firwin",
            verbose=False
        )
        raw.save(
            out_file,
            overwrite=True,
            verbose=False
        )
    except Exception as e:
        print(f"[ERROR] {edf_path}")
        print(e)
print("\nDONE")
#from pathlib import Path
#root = Path(r"A:\UTA\Dr. Papadelis\Dr.P\Foundation Models\REVE\chbmit")
#print("EDF:", len(list(root.rglob("*.edf"))))
#print("EDF.GZ:", len(list(root.rglob("*.edf.gz"))))