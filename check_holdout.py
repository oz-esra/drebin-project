"""Holdout integrity check for the 2024 out-of-time set.

Trains nothing. Answers four questions:
  1. Do the metadata and feature file have matching SHA-256 keys?
  2. Which version is the feature file (does it contain 2024 samples by gp_date)?
  3. How large is the 2024 holdout (by gp_date), and how many samples have features?
  4. Did any 2024 sample (by gp_date) enter the development set that
     run_advanced_experiments.py built using dex_date?

The only file it writes is results/holdout_leaked_shas.txt, and only if leaks are found.

Run from the project folder:  python check_holdout.py
"""
import json
from pathlib import Path

import pandas as pd

from run_advanced_experiments import FEATURES_PATH, METADATA_PATH

DEV_START, DEV_END = "2021-01", "2023-12"
HOLD_START, HOLD_END = "2024-01", "2024-12"


def to_month(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m")


# --- Load metadata exactly as run_advanced_experiments.py does ----------------
with open(METADATA_PATH) as f:
    raw = json.load(f)
meta = pd.DataFrame.from_dict(raw, orient="index") if isinstance(raw, dict) else pd.DataFrame(raw)
if "sha256" not in meta.columns:
    meta["sha256"] = meta.index
del raw

used_col = "dex_date" if "dex_date" in meta.columns else ("added" if "added" in meta.columns else "gp_date")
meta["used_month"] = to_month(meta[used_col])
meta["gp_month"] = to_month(meta["gp_date"])

# --- Feature file keys only ----------------------------------------------------
with open(FEATURES_PATH) as f:
    feat_shas = set(json.load(f).keys())

meta_shas = set(meta["sha256"])
print("1. SHA-256 key match")
print(f"   metadata samples:        {len(meta_shas):,}")
print(f"   feature-file samples:    {len(feat_shas):,}")
print(f"   in both:                 {len(meta_shas & feat_shas):,}")
print(f"   metadata only:           {len(meta_shas - feat_shas):,}")
print(f"   feature file only:       {len(feat_shas - meta_shas):,}")

meta["has_features"] = meta["sha256"].isin(feat_shas)

print("\n2. Feature-file content by gp_date year")
years = meta.loc[meta["has_features"], "gp_month"].str[:4].fillna("missing")
print(years.value_counts().sort_index().to_string())

holdout = meta["gp_month"].between(HOLD_START, HOLD_END)
print(f"\n3. 2024 holdout by gp_date: {holdout.sum():,} samples in metadata, "
      f"{(holdout & meta['has_features']).sum():,} with features")
if (holdout & meta["has_features"]).sum() == 0:
    print("   -> The feature file has no 2024 samples: the extended_2024 features are needed.")

# --- Rebuild the development set the experiments used --------------------------
dev = meta[meta["used_month"].between(DEV_START, DEV_END) & meta["has_features"]]
print(f"\n4. Development set as built by the experiments ({used_col}, {DEV_START}..{DEV_END}): "
      f"{len(dev):,} samples")
print("   gp_date year of those samples:")
print("   " + dev["gp_month"].str[:4].fillna("missing").value_counts().sort_index()
      .to_string().replace("\n", "\n   "))

leaked = dev[dev["gp_month"].between(HOLD_START, HOLD_END)]
if len(leaked):
    out = Path("results/holdout_leaked_shas.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(leaked["sha256"])) + "\n")
    print(f"\nVERDICT: {len(leaked):,} samples with a 2024 gp_date were in the development set "
          f"({len(leaked) / max(holdout.sum(), 1):.1%} of the 2024 holdout). List saved to {out}.")
else:
    print("\nVERDICT: no sample with a 2024 gp_date entered the development set. Holdout is clean.")