"""2024 out-of-time evaluation (PROTOCOL v2, Section 8), with a development fixture test.

The 2024 evaluation is to be executed exactly once, after the development
results have been reviewed with Prof. Lanzi. Two modes are provided:

  python run_holdout_2024.py --fixture
      Runs the complete procedure on the development period only, treating
      2023 as a pseudo-holdout (schedule from 2022-01, evaluation of 2023-01 ..
      2023-12). No 2024 data or feature file is read. The outputs must match the
      committed development results for 2023 exactly; the check is written to
      results/v2/holdout_fixture_check.json, together with the SHA-256 of the
      source files that were tested.

  python run_holdout_2024.py --confirm-single-run
      The single 2024 run. It refuses to start if the flag is missing, if the
      holdout has already been consumed, if the repository has uncommitted
      changes, or if the fixture test has not passed with exactly the same
      source files.

Procedure (identical in both modes, except for the months involved):
  1. Data. Rows for 2021-2023 come from the development cache (original
     feature file); rows for 2024 come from the extended feature file. 2024
     samples with a missing or empty feature vector are excluded, as specified
     by the missing-data rule.
  2. Retraining schedule. Each policy continues its development schedule
     without being restarted. Since 2024-01 is the 25th test month, all
     retraining policies are retrained at 2024-01 on data up to 2023-12, while
     the static model remains the one trained on 2021. Within 2024, a month
     enters the training data only after it has been evaluated.
  3. Vocabulary. Each model uses only the features observed in its own
     training data.
  4. Evaluation. The metrics of Section 6 for all test samples and for the
     overlap-excluded set, the H1 test over the 12 paired months (Cum-1M vs
     Cum-3M, margin +/-0.03, 95% CI) and the secondary paired comparisons, for
     both pre-registered variants (primary: class_weight=None, confirmatory;
     cost-sensitive: secondary), each with its C selected on 2021 data.

Safeguards for the single run (requested by Prof. Lanzi, 27 Oct):
  * Outputs are first written to a staging folder under cache/ and validated
    (all policies and months present, sample counts as expected, no missing
    metrics, complete H1 files). Only then are they moved to
    results/v2/holdout_2024/ and the holdout is marked as consumed
    (CONSUMED.json, with the SHA-256 of every output file).
  * No result is printed before validation succeeds.
  * If an attempt fails (e.g. an infrastructure error), it is logged in
    cache/holdout_2024_staging/attempts.json and kept for inspection; the log
    is copied into the final results and must be reported.
  * The run records the code commit, the protocol-v2 tag and whether the code
    descends from it, the environment and all input hashes.

No parameter is selected or tuned in this script: all settings come from
v2_common.py and from the C-selection results committed before the run.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import traceback
from array import array
from datetime import datetime, timezone
from pathlib import Path

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
PROTOCOL_TAG = "protocol-v2"

HOLDOUT = ("2024-01", "2024-12")
FIXTURE = ("2023-01", "2023-12")
OUT = v2.RESULTS_DIR / "holdout_2024"
STAGING = v2.CACHE_DIR / "holdout_2024_staging"
ATTEMPTS_LOG = STAGING / "attempts.json"
FIXTURE_CHECK = v2.RESULTS_DIR / "holdout_fixture_check.json"
SOURCE_FILES = ("v2_common.py", "run_holdout_2024.py", "run_retraining_experiments.py", "paired_comparisons.py")

MARGIN, ALPHA, N_BOOT = 0.03, 0.05, 10_000      # identical to equivalence_test.py
METRIC_COLS = ["n", "n_malware", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr",
               "average_precision", "f1_ci_low", "f1_ci_high", "train_test_vector_overlap",
               "excl_overlap_n", "excl_overlap_f1", "excl_overlap_average_precision"]


# ------------------------------------------------------------------------------
# Provenance
# ------------------------------------------------------------------------------
def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def protocol_info():
    tag_commit = git("rev-list", "-n", "1", PROTOCOL_TAG)
    descends = subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL_TAG, "HEAD"],
                              capture_output=True).returncode == 0 if tag_commit else False
    return {"protocol_tag": PROTOCOL_TAG, "protocol_tag_commit": tag_commit, "code_descends_from_tag": descends}


def source_hashes():
    return {f: v2.sha256_file(f) for f in SOURCE_FILES}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
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
    meta = meta[meta["month"].between(*HOLDOUT)]

    with open(EXTENDED_FEATURES_PATH, "r", encoding="utf-8") as f:
        feats = json.load(f)
    n_meta = len(meta)
    meta = meta[meta["sha256"].map(lambda s: bool(feats.get(s)))]          # missing-data rule
    meta = meta.sort_values(["month", "sha256"]).reset_index(drop=True)
    if len(meta) != EXPECTED_HOLDOUT_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_HOLDOUT_SAMPLES:,} holdout samples (audit), found {len(meta):,}.")
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
    return X, index, len(vocab) - n_dev_cols


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
def h1_test(d):
    """Same computation as equivalence_test.py."""
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


def run_variant(variant, X, y, months, digests, period, out):
    """Evaluate every policy on the months of `period`, continuing the schedule from 2022-01."""
    C = json.loads((v2.results_dir(variant) / "c_selection.json").read_text())["selected_C"]
    schedule = v2.month_range(v2.TEST_START, period[1])
    eval_idx = [j for j, m in enumerate(schedule) if m >= period[0]]
    print(f"\nVariant {variant}: C = {C}")

    models, rows = {}, []
    for name, (every, window, family) in POLICIES.items():
        for j in eval_idx:
            month = schedule[j]
            _, first, last = training_range(schedule, j, every, window)
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
    monthly.groupby("policy", sort=False).agg(
        mean_f1=("f1", "mean"), mean_average_precision=("average_precision", "mean"),
        mean_precision=("precision", "mean"), mean_recall=("recall", "mean"), mean_fpr=("fpr", "mean"),
        mean_train_test_vector_overlap=("train_test_vector_overlap", "mean"),
        excl_overlap_mean_f1=("excl_overlap_f1", "mean")).round(4).to_csv(out / "summary.csv")

    results = {}
    for test_set, prefix in (("all", ""), ("excl_overlap", "excl_overlap_")):
        suffix = "" if test_set == "all" else "_excl_overlap"
        wide = monthly.pivot(index="month", columns="policy", values=prefix + "f1")
        wide.to_csv(out / f"monthly_f1_wide{suffix}.csv")
        results[test_set] = h1_test((wide["Cum-1M"] - wide["Cum-3M"]).to_numpy())
        rng = np.random.default_rng(v2.SEED)
        pairs = []
        for family, policy, comparator in PAIRS:
            for metric in ("f1", "average_precision"):
                w = monthly.pivot(index="month", columns="policy", values=prefix + metric)
                p = w[[policy, comparator]].dropna()
                pairs.append({"family": family, "policy": policy, "comparator": comparator, "metric": metric,
                              **paired_summary((p[policy] - p[comparator]).to_numpy(), rng)})
        pd.DataFrame(pairs).round(6).to_csv(out / f"paired_comparisons{suffix}.csv", index=False)
    v2.write_json(out / "h1_cum3m_vs_cum1m.json",
                  {"difference": f"F1(Cum-1M) - F1(Cum-3M) per month, {period[0]}..{period[1]}", **results})
    return {"C": C, "unique_models": len(models),
            "all_converged": bool(all(v[2]["converged"] for v in models.values()))}


# ------------------------------------------------------------------------------
# Validation (no result values are printed here)
# ------------------------------------------------------------------------------
def validate(out, period, expected_n):
    """Raise if any output is missing or incomplete. expected_n: samples per month."""
    errors = []
    months = v2.month_range(*period)
    for variant in v2.VARIANTS:
        d = out / variant
        for f in ("monthly_metrics.csv", "models.csv", "summary.csv", "monthly_f1_wide.csv",
                  "monthly_f1_wide_excl_overlap.csv", "paired_comparisons.csv",
                  "paired_comparisons_excl_overlap.csv", "h1_cum3m_vs_cum1m.json"):
            if not (d / f).exists():
                errors.append(f"{variant}: missing {f}")
        if errors:
            continue
        m = pd.read_csv(d / "monthly_metrics.csv", dtype={"month": str})
        if len(m) != len(POLICIES) * len(months):
            errors.append(f"{variant}: {len(m)} rows instead of {len(POLICIES) * len(months)}")
        if set(zip(m["policy"], m["month"])) != {(p, mo) for p in POLICIES for mo in months}:
            errors.append(f"{variant}: policy/month grid incomplete")
        for _, r in m.iterrows():
            if r["n"] != expected_n.get(r["month"]):
                errors.append(f"{variant}: {r['policy']} {r['month']} has n={r['n']}, expected {expected_n.get(r['month'])}")
        for col in ("precision", "recall", "f1", "fpr"):
            if m[col].isna().any():
                errors.append(f"{variant}: missing values in {col}")
        both_classes = (m["n_malware"] > 0) & (m["n_malware"] < m["n"])
        if m.loc[both_classes, "average_precision"].isna().any():
            errors.append(f"{variant}: missing Average Precision")
        if len(pd.read_csv(d / "summary.csv")) != len(POLICIES):
            errors.append(f"{variant}: summary incomplete")
        for f in ("paired_comparisons.csv", "paired_comparisons_excl_overlap.csv"):
            if len(pd.read_csv(d / f)) != 2 * len(PAIRS):
                errors.append(f"{variant}: {f} incomplete")
        h1 = json.loads((d / "h1_cum3m_vs_cum1m.json").read_text())
        for s in ("all", "excl_overlap"):
            vals = [h1.get(s, {}).get("mean_diff"), *h1.get(s, {}).get("ci95_t", [None, None])]
            if any(v is None or not np.isfinite(v) for v in vals):
                errors.append(f"{variant}: H1 ({s}) incomplete")
    if errors:
        raise RuntimeError("Validation failed:\n  " + "\n  ".join(errors))


# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------
def run_fixture():
    """Full procedure on 2023 as a pseudo-holdout; must reproduce the development results."""
    run = v2.run_info()
    if run["git_dirty"]:
        raise SystemExit("Commit all changes first, so the tested code is recorded.")
    X, index, _, provenance = v2.load_development_data()
    y, months, digests = index["label"].to_numpy(), index["month"].to_numpy(), index["digest"].to_numpy()

    out = v2.CACHE_DIR / "holdout_fixture" / datetime.now().strftime("%Y%m%d_%H%M%S")
    variants = {v: run_variant(v, X, y, months, digests, FIXTURE, out / v) for v in v2.VARIANTS}
    expected_n = index[index["month"].between(*FIXTURE)].groupby("month").size().to_dict()
    validate(out, FIXTURE, expected_n)

    # Compare with the committed development results for the same months.
    comparison, ok = {}, True
    for variant in v2.VARIANTS:
        dev = pd.read_csv(v2.results_dir(variant) / "retraining" / "monthly_metrics.csv", dtype={"month": str})
        dev = dev[dev["month"].between(*FIXTURE)].set_index(["policy", "month"]).sort_index()
        fix = pd.read_csv(out / variant / "monthly_metrics.csv", dtype={"month": str}).set_index(
            ["policy", "month"]).sort_index()
        same_windows = bool((dev[["train_first", "train_last", "n_train"]] ==
                             fix[["train_first", "train_last", "n_train"]]).all().all())
        same_index = dev.index.equals(fix.index)
        same_missing = bool((dev[METRIC_COLS].isna() == fix[METRIC_COLS].isna()).all().all())
        diff = (dev[METRIC_COLS] - fix[METRIC_COLS]).abs()
        max_diff = float(np.nanmax(diff.to_numpy()))
        passed = same_index and same_windows and same_missing and max_diff < 1e-9
        ok &= passed
        comparison[variant] = {"rows_compared": len(dev), "same_training_windows": same_windows,
                               "max_abs_metric_difference": max_diff, "passed": passed}
        print(f"Fixture vs development ({variant}): windows identical = {same_windows}, "
              f"max |difference| = {max_diff:.2e} -> {'PASS' if passed else 'FAIL'}")

    v2.write_json(FIXTURE_CHECK, {
        "description": "complete holdout procedure run on the development period (2023 as pseudo-holdout); "
                       "no 2024 data read",
        "passed": ok, "period": FIXTURE, "comparison": comparison, "variants": variants,
        "fixture_outputs": str(out), "source_sha256": source_hashes(),
        "inputs": provenance, **protocol_info(), "run": run, "finished_utc": now(),
    })
    print(f"\nFixture check {'PASSED' if ok else 'FAILED'} -> {FIXTURE_CHECK}")
    if ok:
        print("Commit this file, then run the holdout with --confirm-single-run.")


def log_attempt(entry):
    STAGING.mkdir(parents=True, exist_ok=True)
    attempts = json.loads(ATTEMPTS_LOG.read_text()) if ATTEMPTS_LOG.exists() else []
    attempts.append(entry)
    ATTEMPTS_LOG.write_text(json.dumps(attempts, indent=2, default=str))
    return attempts


def run_holdout():
    if OUT.exists():
        raise SystemExit(f"Refusing to run: {OUT} exists, so the holdout has already been consumed.")
    run = v2.run_info()
    if run["git_dirty"]:
        raise SystemExit("Refusing to run: commit all changes first, so the exact code of this run is recorded.")
    proto = protocol_info()
    if not proto["code_descends_from_tag"]:
        raise SystemExit(f"Refusing to run: the code does not descend from the {PROTOCOL_TAG} tag.")
    if not FIXTURE_CHECK.exists():
        raise SystemExit("Refusing to run: run the fixture test first (--fixture) and commit its result.")
    fixture = json.loads(FIXTURE_CHECK.read_text())
    if not fixture.get("passed"):
        raise SystemExit("Refusing to run: the fixture test did not pass.")
    if fixture.get("source_sha256") != source_hashes():
        raise SystemExit("Refusing to run: the source files changed after the fixture test; run it again.")
    for variant in v2.VARIANTS:
        if not (v2.results_dir(variant) / "c_selection.json").exists():
            raise SystemExit(f"Missing C selection for variant '{variant}'.")
    print("Checking the extended feature file hash...")
    if v2.sha256_file(EXTENDED_FEATURES_PATH) != EXTENDED_FEATURES_SHA256:
        raise SystemExit("The extended feature file does not match the SHA-256 frozen in PROTOCOL.md.")

    started = now()
    staging = STAGING / f"attempt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    previous = json.loads(ATTEMPTS_LOG.read_text()) if ATTEMPTS_LOG.exists() else []
    try:
        X_dev, index_dev, feature_names, provenance = v2.load_development_data()
        X_hold, index_hold, n_new = load_holdout_rows(feature_names)
        X_dev = csr_matrix((X_dev.data, X_dev.indices, X_dev.indptr), shape=(X_dev.shape[0], X_hold.shape[1]))
        X = vstack([X_dev, X_hold]).tocsr()
        index = pd.concat([index_dev, index_hold], ignore_index=True)
        y, months, digests = index["label"].to_numpy(), index["month"].to_numpy(), index["digest"].to_numpy()

        variants = {v: run_variant(v, X, y, months, digests, HOLDOUT, staging / v) for v in v2.VARIANTS}
        validate(staging, HOLDOUT, index_hold.groupby("month").size().to_dict())

        v2.write_json(staging / "run_config.json", {
            "description": "single 2024 out-of-time evaluation, PROTOCOL v2 Section 8",
            "holdout_months": HOLDOUT, "n_holdout_samples": int(X_hold.shape[0]),
            "features_first_seen_in_2024": int(n_new),
            "variants": variants, "primary_variant": "primary (class_weight=None), confirmatory",
            "secondary_variant": "balanced (cost-sensitive), pre-registered secondary",
            "policies": {k: {"every": p[0], "window": p[1], "family": p[2]} for k, p in POLICIES.items()},
            "svm_params": {v: v2.svm_params(v) for v in v2.VARIANTS}, "decision_threshold": v2.DECISION_THRESHOLD,
            "inputs": {**provenance, "extended_features": {"path": EXTENDED_FEATURES_PATH,
                                                           "sha256": EXTENDED_FEATURES_SHA256}},
            **proto, "source_sha256": source_hashes(),
            "fixture_check": {"file": str(FIXTURE_CHECK), "sha256": v2.sha256_file(FIXTURE_CHECK)},
            "previous_failed_attempts": previous, "run": run, "started_utc": started,
        })
    except BaseException as exc:
        log_attempt({"attempt": staging.name, "started_utc": started, "failed_utc": now(),
                     "commit": run["git_commit"], "error": repr(exc), "traceback": traceback.format_exc()})
        print(f"\nThe attempt FAILED and was logged in {ATTEMPTS_LOG}; no result was printed. "
              "Report this attempt; the partial outputs are kept in the staging folder.")
        raise

    # Validated: consume the holdout.
    shutil.move(str(staging), str(OUT))
    if previous:
        (OUT / "previous_failed_attempts.json").write_text(json.dumps(previous, indent=2, default=str))
    files = {str(p.relative_to(OUT)): v2.sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    v2.write_json(OUT / "CONSUMED.json", {"consumed_utc": now(), "commit": run["git_commit"],
                                          **proto, "output_sha256": files})

    print("\n" + "=" * 70)
    print("2024 holdout consumed. Results (reported as obtained):")
    for variant in v2.VARIANTS:
        h1 = json.loads((OUT / variant / "h1_cum3m_vs_cum1m.json").read_text())
        for s in ("all", "excl_overlap"):
            r = h1[s]
            print(f"  {variant:<8} H1 ({s}): mean d = {r['mean_diff']:.4f}, 95% CI [{r['ci95_t'][0]:.4f}, "
                  f"{r['ci95_t'][1]:.4f}], equivalent = {r['equivalent_primary']}")
    print(f"Outputs: {OUT}/  Commit them now; do not run this script again.")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true", help="test the full procedure on 2023 (no 2024 data)")
    mode.add_argument("--confirm-single-run", action="store_true", help="the one and only 2024 evaluation")
    args = ap.parse_args()
    run_fixture() if args.fixture else run_holdout()


if __name__ == "__main__":
    main()
