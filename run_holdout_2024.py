"""2024 out-of-time evaluation (PROTOCOL v2, Section 8).

This script is to be executed exactly once, after the development results have
been reviewed with Prof. Lanzi. As a safeguard, it terminates if the flag
--confirm-single-run is not given, if the output directory already exists, or
if the repository contains uncommitted changes.

Procedure:
  1. Data. Rows for 2021-2023 are taken from the development cache (original
     feature file); rows for 2024 are taken from the extended feature file.
     2024 samples with a missing or empty feature vector are excluded, as
     specified by the missing-data rule.
  2. Retraining schedule. Each policy continues its development schedule
     without being restarted. Since 2024-01 is the 25th test month, all
     retraining policies are retrained at 2024-01 on data up to 2023-12, while
     the static model remains the one trained on 2021. Within 2024, a month
     enters the training data only after it has been evaluated.
  3. Vocabulary. Each model uses only the features observed in its own
     training data; features that first appear in 2024 are therefore available
     only to models trained on 2024 months.
  4. Evaluation. The metrics of Section 6 are reported for all test samples
     and for the overlap-excluded set, together with the H1 test over the 12
     paired months (Cum-1M vs Cum-3M, margin +/-0.03, 95% CI) and the secondary
     paired comparisons. Both pre-registered variants (primary and
     cost-sensitive) are evaluated in this single run, each with the value of C
     selected for it on 2021 data.

No parameter is selected or tuned in this script: all settings are taken from
v2_common.py and from the C-selection results committed before the run.

Usage:  python run_holdout_2024.py --confirm-single-run
"""
import argparse
import hashlib
import json
from array import array

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, vstack

import v2_common as v2
from paired_comparisons import PAIRS, paired_summary
from run_retraining_experiments import POLICIES, training_range

EXTENDED_FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_extended_2024\hypercube_drebin.json"
EXTENDED_FEATURES_SHA256 = "da0a71e5362720d275cffa771bc219fe979395d01c8f2c9afa11cae78564ccf3"  # PROTOCOL.md, Section 1
EXPECTED_HOLDOUT_SAMPLES = 28_518                                                             # dataset audit
HOLDOUT_START, HOLDOUT_END = "2024-01", "2024-12"
OUT = v2.RESULTS_DIR / "holdout_2024"

MARGIN, ALPHA, N_BOOT = 0.03, 0.05, 10_000      # identical to equivalence_test.py


def load_holdout_rows(feature_names):
    """2024 rows from the extended file, on the development column index (extended with new features)."""
    print("Loading metadata and the extended feature file for 2024...")
    with open(v2.METADATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = pd.DataFrame.from_dict(raw, orient="index") if isinstance(raw, dict) else pd.DataFrame(raw)
    del raw
    if "sha256" not in meta.columns:
        meta["sha256"] = meta.index
    meta = meta.reset_index(drop=True)
    meta["month"] = pd.to_datetime(meta[v2.DATE_COL], errors="coerce").dt.strftime("%Y-%m")
    meta["label"] = (pd.to_numeric(meta["vt_detection"], errors="raise") >= v2.VTT).astype(np.int8)
    meta = meta[meta["month"].between(HOLDOUT_START, HOLDOUT_END)]

    with open(EXTENDED_FEATURES_PATH, "r", encoding="utf-8") as f:
        feats = json.load(f)
    n_meta = len(meta)
    meta = meta[meta["sha256"].map(lambda s: bool(feats.get(s)))]          # missing-data rule
    meta = meta.sort_values(["month", "sha256"]).reset_index(drop=True)
    if len(meta) != EXPECTED_HOLDOUT_SAMPLES:
        raise SystemExit(f"Expected {EXPECTED_HOLDOUT_SAMPLES:,} holdout samples (audit), found {len(meta):,}.")
    print(f"  2024: {n_meta:,} in metadata -> {len(meta):,} with features")

    vocab = {name: j for j, name in enumerate(feature_names)}
    n_dev_cols = len(vocab)
    indptr, indices, digests = array("q", [0]), array("i"), []
    for sha in meta["sha256"]:
        keys = sorted(k for k, v in feats[sha].items() if v)
        for k in keys:
            j = vocab.get(k)
            if j is None:
                j = vocab[k] = len(vocab)
            indices.append(j)
        indptr.append(len(indices))
        digests.append(hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest())
    del feats

    X = csr_matrix((np.ones(len(indices), dtype=np.float32),
                    np.frombuffer(indices, dtype=np.int32), np.frombuffer(indptr, dtype=np.int64)),
                   shape=(len(meta), len(vocab)))
    index = meta[["sha256", "month", "label"]].copy()
    index["digest"] = digests
    print(f"  2024 rows: {X.shape[0]:,}; features first seen in 2024: {len(vocab) - n_dev_cols:,}")
    return X, index, n_dev_cols


def h1_test(d):
    """Same computation as equivalence_test.py, on the 2024 months."""
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    se, dof = sd / np.sqrt(n), n - 1
    t95, t90 = stats.t.ppf(1 - ALPHA / 2, dof), stats.t.ppf(1 - ALPHA, dof)
    ci95, ci90 = [mean - t95 * se, mean + t95 * se], [mean - t90 * se, mean + t90 * se]
    p_two = float(2 * stats.t.sf(abs(mean / se), dof)) if se > 0 else None
    p_tost = float(max(stats.t.sf((mean + MARGIN) / se, dof), stats.t.cdf((mean - MARGIN) / se, dof))) if se > 0 else None
    rng = np.random.default_rng(v2.SEED)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    ci95_boot = np.percentile(boot, [2.5, 97.5]).tolist()
    inside = lambda ci: bool(-MARGIN < ci[0] and ci[1] < MARGIN)
    return {"n_months": n, "mean_diff": mean, "sd_diff": sd, "ci95_t": ci95, "ci95_bootstrap_months": ci95_boot,
            "ci90_t": ci90, "p_two_sided_vs_zero": p_two, "p_tost": p_tost, "margin": MARGIN, "alpha": ALPHA,
            "equivalent_primary": inside(ci95), "equivalent_bootstrap_check": inside(ci95_boot),
            "difference_significant": p_two is not None and p_two < ALPHA}


def run_variant(variant, X, y, months, digests, out):
    C = json.loads((v2.results_dir(variant) / "c_selection.json").read_text())["selected_C"]
    all_test = v2.month_range(v2.TEST_START, HOLDOUT_END)          # 2022-01 .. 2024-12, as one schedule
    holdout_idx = [j for j, m in enumerate(all_test) if m >= HOLDOUT_START]
    print(f"\nVariant {variant}: C = {C}")

    models, rows = {}, []
    for name, (every, window, family) in POLICIES.items():
        for j in holdout_idx:
            month = all_test[j]
            _, first, last = training_range(all_test, j, every, window)
            key = (first, last)
            if key not in models:
                train = (months >= first) & (months <= last)
                clf, cols, info = v2.fit_svm(X[train], y[train], C, variant=variant)
                models[key] = (clf, cols, info, set(digests[train]))
                print(f"  fitted {first}..{last}: n={info['n_train']:,}, converged={info['converged']} "
                      f"({info['fit_seconds']}s)")
            clf, cols, info, train_digests = models[key]
            test = months == month
            scores = v2.decision_scores(clf, cols, X[test])
            seen = np.array([d in train_digests for d in digests[test]])
            rows.append({"policy": name, "family": family, "month": month,
                         "train_first": first, "train_last": last, "n_train": info["n_train"],
                         "model_converged": info["converged"],
                         **v2.evaluate(y[test], scores),
                         "train_test_vector_overlap": round(float(seen.mean()), 4),
                         **{f"excl_overlap_{k}": v for k, v in v2.evaluate(y[test][~seen], scores[~seen]).items()}})

    out.mkdir(parents=True)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame([{"train_first": k[0], "train_last": k[1], **v[2]} for k, v in models.items()]).to_csv(
        out / "models.csv", index=False)

    summary = monthly.groupby("policy", sort=False).agg(
        mean_f1=("f1", "mean"), mean_average_precision=("average_precision", "mean"),
        mean_precision=("precision", "mean"), mean_recall=("recall", "mean"), mean_fpr=("fpr", "mean"),
        mean_train_test_vector_overlap=("train_test_vector_overlap", "mean"),
        excl_overlap_mean_f1=("excl_overlap_f1", "mean")).round(4)
    summary.to_csv(out / "summary.csv")

    results = {}
    for test_set, prefix in (("all", ""), ("excl_overlap", "excl_overlap_")):
        wide = monthly.pivot(index="month", columns="policy", values=prefix + "f1")
        wide.to_csv(out / f"monthly_f1_wide{'' if test_set == 'all' else '_excl_overlap'}.csv")
        h1 = h1_test((wide["Cum-1M"] - wide["Cum-3M"]).to_numpy())
        rng = np.random.default_rng(v2.SEED)
        pairs = []
        for family, policy, comparator in PAIRS:
            for metric in ("f1", "average_precision"):
                w = monthly.pivot(index="month", columns="policy", values=prefix + metric)
                p = w[[policy, comparator]].dropna()
                pairs.append({"family": family, "policy": policy, "comparator": comparator, "metric": metric,
                              **paired_summary((p[policy] - p[comparator]).to_numpy(), rng)})
        pd.DataFrame(pairs).round(6).to_csv(
            out / f"paired_comparisons{'' if test_set == 'all' else '_excl_overlap'}.csv", index=False)
        results[test_set] = h1
        print(f"  H1 ({test_set}): mean d = {h1['mean_diff']:.4f}, 95% CI [{h1['ci95_t'][0]:.4f}, "
              f"{h1['ci95_t'][1]:.4f}], equivalent = {h1['equivalent_primary']}")
    v2.write_json(out / "h1_cum3m_vs_cum1m.json", {"difference": "F1(Cum-1M) - F1(Cum-3M) per 2024 month", **results})
    return {"C": C, "unique_models": len(models),
            "all_converged": bool(all(v[2]["converged"] for v in models.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-single-run", action="store_true",
                    help="required: confirms this is the one and only 2024 evaluation")
    args = ap.parse_args()

    if not args.confirm_single_run:
        raise SystemExit("Refusing to run: the 2024 holdout may be evaluated only once. "
                         "Add --confirm-single-run after the development review.")
    if OUT.exists():
        raise SystemExit(f"Refusing to run: {OUT} already exists, so the holdout has already been evaluated.")
    run = v2.run_info()
    if run["git_dirty"]:
        raise SystemExit("Refusing to run: commit all changes first, so the exact code of this run is recorded.")
    for variant in v2.VARIANTS:
        if not (v2.results_dir(variant) / "c_selection.json").exists():
            raise SystemExit(f"Missing C selection for variant '{variant}'.")

    print("Checking the extended feature file hash...")
    if v2.sha256_file(EXTENDED_FEATURES_PATH) != EXTENDED_FEATURES_SHA256:
        raise SystemExit("The extended feature file does not match the SHA-256 frozen in PROTOCOL.md.")

    X_dev, index_dev, feature_names, provenance = v2.load_development_data()
    X_hold, index_hold, _ = load_holdout_rows(feature_names)
    n_cols = X_hold.shape[1]
    X_dev = csr_matrix((X_dev.data, X_dev.indices, X_dev.indptr), shape=(X_dev.shape[0], n_cols))
    X = vstack([X_dev, X_hold]).tocsr()
    index = pd.concat([index_dev, index_hold], ignore_index=True)
    y, months, digests = index["label"].to_numpy(), index["month"].to_numpy(), index["digest"].to_numpy()

    variants = {v: run_variant(v, X, y, months, digests, OUT / v) for v in v2.VARIANTS}

    v2.write_json(OUT / "run_config.json", {
        "description": "single 2024 out-of-time evaluation, PROTOCOL v2 Section 8",
        "holdout_months": [HOLDOUT_START, HOLDOUT_END], "n_holdout_samples": int(X_hold.shape[0]),
        "variants": variants, "policies": {k: {"every": p[0], "window": p[1], "family": p[2]} for k, p in POLICIES.items()},
        "svm_params": {v: v2.svm_params(v) for v in v2.VARIANTS}, "decision_threshold": v2.DECISION_THRESHOLD,
        "inputs": {**provenance, "extended_features": {"path": EXTENDED_FEATURES_PATH, "sha256": EXTENDED_FEATURES_SHA256}},
        "run": run,
    })
    print(f"\nDone. Results in {OUT}/. Commit them immediately; do not run this script again.")


if __name__ == "__main__":
    main()
