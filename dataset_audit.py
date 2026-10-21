"""Dataset audit for the corrected protocol (v2). Trains nothing.

Reports, as requested by Prof. Lanzi on 14 Oct 2026:
  * exact metadata-feature intersection (both directions)
  * counts by year / month / class (VTT=2: malware if vt_detection >= 2)
  * duplicate hashes: repeated SHA-256 in the metadata and repeated keys in
    each feature file (a plain json.load would silently keep only the last one)
  * empty feature vectors (and any feature value other than 1)
  * final sample counts for development (2021-2023) and holdout (2024)
  * whether the extended feature file actually contains the 2024 samples
  * full two-way metadata-feature intersection per file and year
    (intersection_two_way.csv)
  * 2024 samples without features (excluded by the missing-data rule) against
    the retained ones, by month and class (holdout_missing_vs_retained.csv)

Also reports, for information:
  * whether the 2021-2023 samples have identical feature sets in the original
    and extended files (training uses one file, the 2024 evaluation the other)
  * groups of distinct SHA-256 with identical feature sets (possible
    train/test near-duplicates)

Feature files are loaded one at a time to limit memory. Outputs go to
results/audit/. Run from the project folder:

  python dataset_audit.py --extended "C:\\path\\to\\extended_2024\\features.json"
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METADATA = r"C:\Users\esrao\Downloads\metadata.json"
DEFAULT_ORIGINAL = r"C:\Users\esrao\Downloads\hypercube_drebin.json"

DATE_COL = "gp_date"
VTT = 2
DEV_YEARS = (2021, 2022, 2023)
HOLDOUT_YEAR = 2024


# ------------------------------------------------------------------ helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_top_level_with_duplicates(path):
    """Load a JSON object, recording keys that repeat at the top level.

    Returns (object, Counter of duplicated top-level keys -> extra occurrences).
    Only the top-level object is checked: object_pairs_hook is called innermost
    first, so the last call is always the top-level object.
    """
    last = [None]  # keep only the most recent object's pairs to save memory

    def hook(pairs):
        last[0] = pairs
        return dict(pairs)

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f, object_pairs_hook=hook)

    dups = Counter()
    if isinstance(obj, dict) and last[0] is not None:
        counts = Counter(k for k, _ in last[0])
        dups = Counter({k: c - 1 for k, c in counts.items() if c > 1})
    last[0] = None
    return obj, dups


def feature_digest(features):
    return hashlib.sha1("\n".join(sorted(set(features))).encode("utf-8")).hexdigest()


def summarise_feature_file(path, label):
    """Load one feature file and keep only what the audit needs."""
    print(f"Loading {label} feature file: {path}")
    obj, dup_keys = load_top_level_with_duplicates(path)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected a {{sha256: [features]}} object, got {type(obj).__name__}")

    # Hypercube stores each sample as {feature: 1}; lists are also accepted.
    digests, n_features = {}, {}
    bad_type = 0
    value_counts = Counter()
    for sha, feats in obj.items():
        if isinstance(feats, dict):
            value_counts.update(feats.values())
            feats = [k for k, v in feats.items() if v]
        elif not isinstance(feats, list):
            bad_type += 1
            feats = []
        n_features[sha] = len(feats)
        digests[sha] = feature_digest(feats)
    del obj

    non_unit = {str(k): n for k, n in value_counts.items() if k != 1}
    print(f"  {len(digests):,} samples | feature values other than 1: {non_unit or 'none'}"
          f" | unsupported value types: {bad_type:,}")

    return {
        "path": str(path),
        "keys": set(digests),
        "duplicate_keys": dup_keys,
        "n_features": n_features,
        "digests": digests,
        "non_list_values": bad_type,
        "non_unit_feature_values": non_unit,
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Dataset audit (no models).")
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    ap.add_argument("--original", default=DEFAULT_ORIGINAL, help="original (2021-2023) feature file")
    ap.add_argument("--extended", required=True, help="extended_2024 feature file")
    ap.add_argument("--out", default="results/audit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- metadata
    print(f"Loading metadata: {args.metadata}")
    raw, meta_dup_keys = load_top_level_with_duplicates(args.metadata)
    if isinstance(raw, dict):
        meta = pd.DataFrame.from_dict(raw, orient="index")
        if "sha256" not in meta.columns:
            meta["sha256"] = meta.index
    else:
        meta = pd.DataFrame(raw)
    del raw
    meta = meta.reset_index(drop=True)

    meta_dup_rows = meta[meta["sha256"].duplicated(keep=False)]
    meta["date"] = pd.to_datetime(meta[DATE_COL], errors="coerce")
    meta["year"] = meta["date"].dt.year
    meta["month"] = meta["date"].dt.strftime("%Y-%m")
    vt = pd.to_numeric(meta["vt_detection"], errors="coerce")
    meta["label"] = (vt >= VTT).astype("Int64").where(vt.notna())

    # ---- feature files, one at a time
    orig = summarise_feature_file(args.original, "original")
    ext = summarise_feature_file(args.extended, "extended_2024")

    meta["in_original"] = meta["sha256"].isin(orig["keys"])
    meta["in_extended"] = meta["sha256"].isin(ext["keys"])
    meta["empty_original"] = meta["sha256"].map(lambda s: orig["n_features"].get(s, -1) == 0)
    meta["empty_extended"] = meta["sha256"].map(lambda s: ext["n_features"].get(s, -1) == 0)

    meta_shas = set(meta["sha256"])
    is_dev = meta["year"].isin(DEV_YEARS)
    is_hold = meta["year"] == HOLDOUT_YEAR

    # ---- intersection
    intersection = {}
    for name, f in (("original", orig), ("extended_2024", ext)):
        intersection[name] = {
            "feature_file_keys": len(f["keys"]),
            "metadata_and_features": len(meta_shas & f["keys"]),
            "metadata_only": len(meta_shas - f["keys"]),
            "features_only": len(f["keys"] - meta_shas),
            "duplicate_keys": int(sum(f["duplicate_keys"].values())),
            "unsupported_value_types": f["non_list_values"],
            "feature_values_other_than_1": f["non_unit_feature_values"],
            "empty_vectors": int(sum(1 for n in f["n_features"].values() if n == 0)),
        }
        pd.Series(sorted(f["keys"] - meta_shas), name="sha256").to_csv(
            out / f"features_only_{name}.csv", index=False)

    # ---- full two-way metadata-feature intersection, per file and per year
    # (feature-only hashes have no metadata, hence no year: reported as "no metadata")
    two_way = []
    for name, f, col in (("original", orig, "in_original"), ("extended_2024", ext, "in_extended")):
        for year, g in meta.groupby("year", dropna=False):
            two_way.append({
                "feature_file": name, "year": str(int(year)) if pd.notna(year) else "unknown",
                "metadata_and_features": int(g[col].sum()),
                "metadata_only": int((~g[col]).sum()),
                "features_only": 0,
            })
        two_way.append({"feature_file": name, "year": "no metadata",
                        "metadata_and_features": 0, "metadata_only": 0,
                        "features_only": len(f["keys"] - meta_shas)})
        two_way.append({"feature_file": name, "year": "total",
                        "metadata_and_features": len(meta_shas & f["keys"]),
                        "metadata_only": len(meta_shas - f["keys"]),
                        "features_only": len(f["keys"] - meta_shas)})
    pd.DataFrame(two_way).to_csv(out / "intersection_two_way.csv", index=False)

    # ---- counts by month (dev uses original features, holdout uses extended)
    meta["split"] = "other"
    meta.loc[is_dev, "split"] = "development"
    meta.loc[is_hold, "split"] = "holdout_2024"
    meta["has_features"] = (is_dev & meta["in_original"]) | (is_hold & meta["in_extended"])
    meta["empty"] = (is_dev & meta["empty_original"]) | (is_hold & meta["empty_extended"])
    meta["final"] = (meta["has_features"] & ~meta["empty"] & meta["label"].notna()).astype(bool)

    print("\nIntermediate counts (before the detailed analysis):")
    print(meta.groupby("split")[["has_features", "empty", "final"]].sum().astype(int).to_string())
    if not meta["final"].any():
        raise SystemExit("No samples passed the final filter; please send this output.")

    def counts(group_cols):
        g = meta.groupby(group_cols, dropna=False)
        df = pd.DataFrame({
            "metadata": g.size(),
            "with_features": g["has_features"].sum(),
            "empty_vectors": g["empty"].sum(),
            "final": g["final"].sum(),
            "final_malware": g.apply(lambda x: int((x["final"] & (x["label"] == 1)).sum()), include_groups=False),
            "final_goodware": g.apply(lambda x: int((x["final"] & (x["label"] == 0)).sum()), include_groups=False),
        })
        df["final_malware_ratio"] = (df["final_malware"] / df["final"]).round(4)
        return df.reset_index()

    by_month = counts(["split", "month"])
    by_year = counts(["split", "year"])
    by_month.to_csv(out / "counts_by_month.csv", index=False)
    by_year.to_csv(out / "counts_by_year.csv", index=False)

    # ---- 2024 samples without features (excluded by the missing-data rule)
    # compared with the retained 2024 samples, by month and by class
    hold = meta[is_hold].copy()
    hold["status"] = np.where(hold["has_features"], "retained", "missing_features")
    by_m = hold.groupby(["month", "status", "label"]).size().unstack(["status", "label"], fill_value=0)
    by_m.columns = [f"{s}_{'malware' if l == 1 else 'goodware'}" for s, l in by_m.columns]
    for s in ("retained", "missing_features"):
        for c in ("malware", "goodware"):
            by_m[f"{s}_{c}"] = by_m.get(f"{s}_{c}", 0)
        by_m[f"{s}_total"] = by_m[f"{s}_malware"] + by_m[f"{s}_goodware"]
    by_m["missing_share"] = (by_m["missing_features_total"] /
                             (by_m["missing_features_total"] + by_m["retained_total"])).round(4)
    by_m["retained_malware_ratio"] = (by_m["retained_malware"] / by_m["retained_total"]).round(4)
    by_m["missing_malware_ratio"] = (by_m["missing_features_malware"] /
                                     by_m["missing_features_total"].replace(0, np.nan)).round(4)
    cols = ["retained_total", "retained_malware", "retained_goodware", "retained_malware_ratio",
            "missing_features_total", "missing_features_malware", "missing_features_goodware",
            "missing_malware_ratio", "missing_share"]
    by_m = by_m[cols].reset_index()
    total = by_m[[c for c in cols if not c.endswith(("ratio", "share"))]].sum()
    total_row = {"month": "total", **total.to_dict(),
                 "retained_malware_ratio": round(total["retained_malware"] / total["retained_total"], 4),
                 "missing_malware_ratio": round(total["missing_features_malware"] /
                                                max(total["missing_features_total"], 1), 4),
                 "missing_share": round(total["missing_features_total"] /
                                        (total["missing_features_total"] + total["retained_total"]), 4)}
    by_m = pd.concat([by_m, pd.DataFrame([total_row])], ignore_index=True)
    by_m.to_csv(out / "holdout_missing_vs_retained.csv", index=False)

    # ---- duplicates
    meta_dup_rows.to_csv(out / "duplicate_sha256_metadata.csv", index=False)
    for name, f in (("original", orig), ("extended_2024", ext)):
        pd.DataFrame(sorted(f["duplicate_keys"].items()), columns=["sha256", "extra_occurrences"]).to_csv(
            out / f"duplicate_keys_{name}.csv", index=False)

    # ---- empty vectors
    meta.loc[meta["empty"], ["sha256", "split", "month", "label"]].to_csv(
        out / "empty_feature_vectors.csv", index=False)

    # ---- consistency of 2021-2023 features between the two files
    dev_shas = meta.loc[is_dev & meta["in_original"] & meta["in_extended"], "sha256"]
    differing = [s for s in dev_shas if orig["digests"][s] != ext["digests"][s]]
    pd.Series(differing, name="sha256").to_csv(out / "dev_features_differ_between_files.csv", index=False)

    # ---- identical feature sets across distinct hashes (final samples only)
    final = meta[meta["final"]].copy()
    final["digest"] = [
        (orig if split == "development" else ext)["digests"][sha]
        for sha, split in zip(final["sha256"], final["split"])
    ]
    grp = final.groupby("digest")
    multi = final[final["digest"].map(grp.size()) > 1].copy()
    multi["group_size"] = multi["digest"].map(grp.size())
    multi["label_conflict"] = multi["digest"].map(grp["label"].nunique()) > 1
    multi["spans_splits_or_months"] = multi["digest"].map(grp["month"].nunique()) > 1
    multi.sort_values(["group_size", "digest"], ascending=[False, True])[
        ["digest", "sha256", "split", "month", "label", "group_size", "label_conflict", "spans_splits_or_months"]
    ].to_csv(out / "identical_feature_sets.csv", index=False)

    # ---- dates
    unparsed = int(meta["date"].isna().sum())
    outside = int((~meta["year"].isin(DEV_YEARS + (HOLDOUT_YEAR,)) & meta["date"].notna()).sum())

    # ---- summary
    def split_total(split, col):
        return int(meta.loc[meta["split"] == split, col].sum())

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_rule": f"malware if vt_detection >= {VTT} (Hypercube VTT={VTT})",
        "date_column": DATE_COL,
        "inputs": {
            "metadata": {"path": args.metadata, "sha256": sha256_file(args.metadata)},
            "original_features": {"path": args.original, "sha256": sha256_file(args.original)},
            "extended_features": {"path": args.extended, "sha256": sha256_file(args.extended)},
        },
        "metadata": {
            "rows": len(meta),
            "unique_sha256": int(meta["sha256"].nunique()),
            "duplicate_sha256_rows": len(meta_dup_rows),
            "duplicate_top_level_keys": int(sum(meta_dup_keys.values())),
            "unparsed_dates": unparsed,
            "dates_outside_2021_2024": outside,
            "missing_vt_detection": int(vt.isna().sum()),
            "vt_detection_eq_1": int((vt == 1).sum()),
        },
        "intersection": intersection,
        "development_2021_2023": {
            "metadata": int(is_dev.sum()),
            "with_original_features": split_total("development", "has_features"),
            "empty_vectors": split_total("development", "empty"),
            "final": split_total("development", "final"),
            "final_malware": int((meta["final"] & is_dev & (meta["label"] == 1)).sum()),
        },
        "holdout_2024": {
            "metadata": int(is_hold.sum()),
            "with_extended_features": split_total("holdout_2024", "has_features"),
            "empty_vectors": split_total("holdout_2024", "empty"),
            "final": split_total("holdout_2024", "final"),
            "final_malware": int((meta["final"] & is_hold & (meta["label"] == 1)).sum()),
            "extended_file_contains_2024": bool(split_total("holdout_2024", "has_features") > 0),
            "missing_features": total_row["missing_features_total"],
            "missing_features_malware": total_row["missing_features_malware"],
            "missing_malware_ratio": total_row["missing_malware_ratio"],
            "retained_malware_ratio": total_row["retained_malware_ratio"],
        },
        "dev_samples_in_both_files": int(len(dev_shas)),
        "dev_samples_with_different_features_between_files": len(differing),
        "identical_feature_sets": {
            "groups": int((grp.size() > 1).sum()),
            "samples_in_groups": len(multi),
            "groups_with_label_conflict": int(multi.loc[multi["label_conflict"], "digest"].nunique()),
            "groups_spanning_months": int(multi.loc[multi["spans_splits_or_months"], "digest"].nunique()),
        },
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- console report
    d, h = summary["development_2021_2023"], summary["holdout_2024"]
    print("\n" + "=" * 70)
    print("DATASET AUDIT (no models trained)")
    print("=" * 70)
    print(f"Metadata rows: {len(meta):,} | unique SHA-256: {summary['metadata']['unique_sha256']:,} | "
          f"duplicate rows: {len(meta_dup_rows):,} | unparsed dates: {unparsed:,}")
    for name, s in intersection.items():
        print(f"{name:>14}: keys {s['feature_file_keys']:,} | in metadata {s['metadata_and_features']:,} | "
              f"metadata-only {s['metadata_only']:,} | features-only {s['features_only']:,} | "
              f"duplicate keys {s['duplicate_keys']:,} | empty {s['empty_vectors']:,}")
    print(f"Development 2021-2023: metadata {d['metadata']:,} -> with features {d['with_original_features']:,} "
          f"-> final {d['final']:,} (malware {d['final_malware']:,})")
    print(f"Holdout 2024:          metadata {h['metadata']:,} -> with features {h['with_extended_features']:,} "
          f"-> final {h['final']:,} (malware {h['final_malware']:,})")
    print(f"Extended file contains 2024 samples: {h['extended_file_contains_2024']}")
    print(f"2024 without features: {h['missing_features']:,} (malware ratio {h['missing_malware_ratio']}) "
          f"vs retained malware ratio {h['retained_malware_ratio']}; by month: holdout_missing_vs_retained.csv")
    print(f"2021-2023 samples whose features differ between files: {len(differing):,} of {len(dev_shas):,}")
    ifs = summary["identical_feature_sets"]
    print(f"Identical feature sets: {ifs['groups']:,} groups, {ifs['samples_in_groups']:,} samples, "
          f"{ifs['groups_with_label_conflict']:,} with conflicting labels, "
          f"{ifs['groups_spanning_months']:,} spanning several months")
    print(f"\nOutputs written to {out}/")


if __name__ == "__main__":
    main()
