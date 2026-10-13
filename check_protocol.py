"""Read-only protocol check for run_advanced_experiments.py.

Trains nothing, writes nothing. Answers three questions:
  1. Which date column did the experiments use, and how much does it differ from gp_date?
  2. How are labels distributed by VirusTotal detection count?
  3. Which feature prefixes exist, and which S-group does map_feature_to_group put them in?

Run from the project folder:  python check_protocol.py
"""
import collections
import json

import pandas as pd

from run_advanced_experiments import FEATURES_PATH, METADATA_PATH, map_feature_to_group

# --- 1. Dates ------------------------------------------------------------------
with open(METADATA_PATH) as f:
    raw = json.load(f)
meta = pd.DataFrame.from_dict(raw, orient="index") if isinstance(raw, dict) else pd.DataFrame(raw)

present = [c for c in ("dex_date", "added", "gp_date") if c in meta.columns]
used = "dex_date" if "dex_date" in meta.columns else ("added" if "added" in meta.columns else "gp_date")
print("Date columns present:", present)
print("Column the experiments used:", used)

months = {c: pd.to_datetime(meta[c], errors="coerce").dt.strftime("%Y-%m") for c in present}
if used != "gp_date" and "gp_date" in months:
    both = months[used].notna() & months["gp_date"].notna()
    diff = (months[used] != months["gp_date"]) & both
    print(f"Samples where {used} month != gp_date month: {diff.sum():,} of {both.sum():,} "
          f"({diff.sum() / max(both.sum(), 1):.1%})")
    dev_used = months[used].between("2021-01", "2023-12")
    dev_gp = months["gp_date"].between("2021-01", "2023-12")
    print(f"In 2021-2023 by {used}: {dev_used.sum():,} | by gp_date: {dev_gp.sum():,}")

# --- 2. Labels -----------------------------------------------------------------
vt = pd.to_numeric(meta["vt_detection"], errors="coerce")
buckets = pd.cut(vt, bins=[-1, 0, 3, 1e9], labels=["0", "1-3", ">=4"])
print("\nvt_detection buckets (code labels everything > 0 as malware):")
print(buckets.value_counts(dropna=False).to_string())

# --- 3. Feature prefixes -------------------------------------------------------
with open(FEATURES_PATH) as f:
    feats = json.load(f)

occurrences = collections.Counter()
distinct = collections.defaultdict(set)
example = {}
for f_list in feats.values():
    for feat in f_list:
        prefix = feat.split("::")[0] if "::" in feat else feat.split("_")[0]
        occurrences[prefix] += 1
        distinct[prefix].add(feat)
        example.setdefault(prefix, feat)

rows = [{
    "prefix": p,
    "mapped_to": map_feature_to_group(example[p]),
    "distinct_features": len(distinct[p]),
    "occurrences": n,
    "example": example[p][:60],
} for p, n in occurrences.most_common()]
table = pd.DataFrame(rows)
print("\nFeature prefixes and their S-group:")
print(table.to_string(index=False))

per_group = table.groupby("mapped_to")["distinct_features"].sum().sort_index()
print("\nDistinct features per group:")
print(per_group.to_string())