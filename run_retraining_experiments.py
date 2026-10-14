"""Step 2 of PROTOCOL v2: retraining policies on the 2021-2023 development set.

Initial training period: all of 2021. Monthly evaluation: 2022-01 .. 2023-12.
C comes from results/v2/c_selection.json (select_c.py) and is the same for
every policy and window.

Two comparison families, as agreed with Prof. Lanzi:
  Family A, retraining frequency with full (cumulative) history:
      Cum-1M vs Cum-3M vs Cum-6M, with the Static model as reference
  Family B, training-history length at a fixed quarterly frequency:
      Cum-3M vs SW12-3M vs SW6-3M

A policy retrains at the first month of every block of `every` test months,
counted from 2022-01, using only months strictly before that point (within its
window). Initial models: Jan-Dec 2021 for Static, cumulative and SW12;
Jul-Dec 2021 for SW6. Identical training ranges share one fitted model.

Outputs go to results/v2/retraining/. This script never reads 2024 data.

Usage:  python run_retraining_experiments.py
"""
import json

import numpy as np
import pandas as pd

import v2_common as v2

OUT = v2.RESULTS_DIR / "retraining"

# name: (retrain every k months or None for static, window in months or None for cumulative, families)
POLICIES = {
    "Static":  (None, None, "A"),
    "Cum-1M":  (1, None, "A"),
    "Cum-3M":  (3, None, "A,B"),
    "Cum-6M":  (6, None, "A"),
    "SW12-3M": (3, 12, "B"),
    "SW6-3M":  (3, 6, "B"),
}


def training_range(test_months, j, every, window):
    """First and last training month for the j-th test month."""
    block_start = 0 if every is None else (j // every) * every
    retrain_month = test_months[block_start]
    last = v2.shift_month(retrain_month, -1)
    first = v2.DEV_START if window is None else v2.shift_month(retrain_month, -window)
    return block_start, first, last


def main():
    selection_file = v2.RESULTS_DIR / "c_selection.json"
    if not selection_file.exists():
        raise SystemExit("results/v2/c_selection.json not found: run select_c.py first.")
    C = json.loads(selection_file.read_text())["selected_C"]

    run = v2.run_info()
    if run["git_dirty"]:
        print("WARNING: uncommitted changes in the repository; the run will be marked as dirty.")

    X, index, _, provenance = v2.load_development_data()
    y = index["label"].to_numpy()
    months = index["month"].to_numpy()
    digests = index["digest"].to_numpy()
    test_months = v2.month_range(v2.TEST_START, v2.TEST_END)
    print(f"C = {C} | policies: {', '.join(POLICIES)} | test months: {test_months[0]}..{test_months[-1]}")

    models = {}        # (first, last) -> (clf, cols, info, set of training digests)
    rows = []
    for name, (every, window, family) in POLICIES.items():
        blocks = set()
        for j, month in enumerate(test_months):
            block, first, last = training_range(test_months, j, every, window)
            blocks.add(block)
            key = (first, last)
            if key not in models:
                train = (months >= first) & (months <= last)
                clf, cols, info = v2.fit_svm(X[train], y[train], C)
                models[key] = (clf, cols, info, set(digests[train]))
                print(f"  fitted {first}..{last}: n={info['n_train']:,}, features={info['n_features']:,}, "
                      f"converged={info['converged']} ({info['fit_seconds']}s)")
            clf, cols, info, train_digests = models[key]

            test = months == month
            metrics = v2.evaluate(y[test], v2.decision_scores(clf, cols, X[test]))
            overlap = float(np.mean([d in train_digests for d in digests[test]]))
            rows.append({
                "policy": name, "family": family, "month": month,
                "train_first": first, "train_last": last, "n_train": info["n_train"],
                "model_converged": info["converged"],
                **metrics,
                "train_test_vector_overlap": round(overlap, 4),
            })
        print(f"{name}: done ({len(blocks) - 1} retraining operations over {len(test_months)} months)")

    OUT.mkdir(parents=True, exist_ok=True)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUT / "monthly_metrics.csv", index=False)
    monthly.pivot(index="month", columns="policy", values="f1")[list(POLICIES)].to_csv(OUT / "monthly_f1_wide.csv")
    monthly.pivot(index="month", columns="policy", values="average_precision")[list(POLICIES)].to_csv(
        OUT / "monthly_ap_wide.csv")

    pd.DataFrame([{"train_first": k[0], "train_last": k[1], **v[2]} for k, v in models.items()]).to_csv(
        OUT / "models.csv", index=False)

    # Aggregate view per policy. AUT = normalised area under the monthly curve.
    def aut(series):
        values = series.to_numpy()
        trapezoid = getattr(np, "trapezoid", None) or np.trapz   # NumPy 1.x and 2.x
        return float(trapezoid(values) / (len(values) - 1))

    summary = []
    for name, (every, window, family) in POLICIES.items():
        p = monthly[monthly["policy"] == name].sort_values("month")
        n_blocks = len({training_range(test_months, j, every, window)[0] for j in range(len(test_months))})
        summary.append({
            "policy": name, "family": family,
            "retrain_every_months": every, "window_months": window,
            "retraining_operations": n_blocks - 1,
            "mean_f1": p["f1"].mean(), "aut_f1": aut(p["f1"]),
            "mean_average_precision": p["average_precision"].mean(), "aut_average_precision": aut(p["average_precision"]),
            "mean_precision": p["precision"].mean(), "mean_recall": p["recall"].mean(), "mean_fpr": p["fpr"].mean(),
            "mean_train_test_vector_overlap": p["train_test_vector_overlap"].mean(),
            "months_with_nonconverged_model": int((~p["model_converged"]).sum()),
        })
    pd.DataFrame(summary).round(4).to_csv(OUT / "summary.csv", index=False)

    v2.write_json(OUT / "run_config.json", {
        "C": C, "c_selection_file": str(selection_file),
        "c_selection_sha256": v2.sha256_file(selection_file),
        "policies": {k: {"every": v[0], "window": v[1], "family": v[2]} for k, v in POLICIES.items()},
        "initial_training": "2021-01..2021-12 (SW6: 2021-07..2021-12)",
        "test_months": [test_months[0], test_months[-1]],
        "svm_params": v2.SVM_PARAMS, "decision_threshold": v2.DECISION_THRESHOLD,
        "label": f"vt_detection >= {v2.VTT}", "date_column": v2.DATE_COL,
        "n_bootstrap": v2.N_BOOTSTRAP, "seed": v2.SEED,
        "unique_models_fitted": len(models),
        "inputs": provenance, "run": run,
    })
    print(f"\nResults written to {OUT}/")


if __name__ == "__main__":
    main()
