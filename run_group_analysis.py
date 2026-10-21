"""Step 4 of PROTOCOL v2: feature-group (S2-S8) degradation and ablation.

All models here are static: trained once on Jan-Dec 2021 with the selected C,
evaluated monthly on 2022-01 .. 2023-12. S1 (hardware) has no features in
Hypercube and is skipped.

Three kinds of model, all fixed in advance:
  * Full:          all feature groups (reference; same as the Static policy)
  * Single-group:  only the features of one group
  * Leave-one-out: all groups except one (controlled ablation)

For every group and month I also measure feature turnover: the share of the
group's feature occurrences in that month whose feature never appeared in the
2021 training data, and the share of samples with at least one such feature.

How to read the outputs: correlation between a group's turnover and the full
model's F1 is only an association. The leave-one-out comparison (does removing
a group change the decline of F1 over time?) is the controlled ablation that
can support a statement about a group contributing to degradation.

This analysis is part of the primary protocol only (class_weight=None).

Usage:  python run_group_analysis.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import v2_common as v2

OUT = v2.results_dir("primary") / "groups"
TRAIN = ("2021-01", "2021-12")


def f1_slope_per_year(months, f1):
    """OLS slope of monthly F1 against time, expressed per 12 months."""
    t = np.arange(len(months))
    return float(stats.linregress(t, f1).slope * 12)


def main():
    selection_file = v2.results_dir("primary") / "c_selection.json"
    if not selection_file.exists():
        raise SystemExit("results/v2/c_selection.json not found: run select_c.py first.")
    C = json.loads(selection_file.read_text())["selected_C"]

    X, index, feature_names, provenance = v2.load_development_data()
    y = index["label"].to_numpy()
    months = index["month"].to_numpy()
    groups = v2.feature_groups(feature_names)
    present = [g for g in sorted(v2.GROUP_NAMES) if (groups == g).any()]
    print(f"C = {C} | groups present: {', '.join(present)} (absent: "
          f"{', '.join(g for g in sorted(v2.GROUP_NAMES) if g not in present) or 'none'})")

    train = (months >= TRAIN[0]) & (months <= TRAIN[1])
    test_months = v2.month_range(v2.TEST_START, v2.TEST_END)
    all_cols = np.arange(X.shape[1])

    # Model definitions, fixed before any fitting.
    specs = [("Full", "all", all_cols)]
    for g in present:
        specs.append((f"Only-{g}", g, np.flatnonzero(groups == g)))
    for g in present:
        specs.append((f"Without-{g}", g, np.flatnonzero(groups != g)))

    rows, model_rows = [], []
    for name, group, allowed in specs:
        clf, cols, info = v2.fit_svm(X[train], y[train], C, allowed_cols=allowed)
        model_rows.append({"model": name, "group": group, **info})
        print(f"  {name:<12} features={info['n_features']:>9,}  converged={info['converged']} ({info['fit_seconds']}s)")
        for month in test_months:
            test = months == month
            m = v2.evaluate(y[test], v2.decision_scores(clf, cols, X[test]))
            rows.append({"model": name, "group": group, "month": month, **m})

    OUT.mkdir(parents=True, exist_ok=True)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUT / "monthly_metrics.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT / "models.csv", index=False)

    # Feature turnover per group and month, relative to the 2021 vocabulary.
    seen_2021 = np.zeros(X.shape[1], dtype=bool)
    seen_2021[np.unique(X[train].indices)] = True
    turnover = []
    for month in test_months:
        Xm = X[months == month].tocsc()
        for g in present:
            gcols = np.flatnonzero(groups == g)
            Xg = Xm[:, gcols]
            occ_per_col = np.diff(Xg.indptr)
            unseen = ~seen_2021[gcols]
            unseen_rows = Xg[:, np.flatnonzero(unseen)].tocsr()
            turnover.append({
                "month": month, "group": g,
                "occurrences": int(occ_per_col.sum()),
                "unseen_occurrence_share": float(occ_per_col[unseen].sum() / max(occ_per_col.sum(), 1)),
                "samples_with_unseen_feature_share": float(np.mean(np.diff(unseen_rows.indptr) > 0)),
            })
    turnover = pd.DataFrame(turnover)
    turnover.to_csv(OUT / "feature_turnover.csv", index=False)

    # Summary: level, decline and (for leave-one-out) change in decline vs Full.
    full = monthly[monthly["model"] == "Full"].sort_values("month")
    full_slope = f1_slope_per_year(full["month"], full["f1"].to_numpy())
    summary = []
    for name, group, _ in specs:
        p = monthly[monthly["model"] == name].sort_values("month")
        slope = f1_slope_per_year(p["month"], p["f1"].to_numpy())
        row = {
            "model": name, "group": group,
            "mean_f1": p["f1"].mean(), "mean_average_precision": p["average_precision"].mean(),
            "f1_first_quarter": p["f1"].head(3).mean(), "f1_last_quarter": p["f1"].tail(3).mean(),
            "f1_slope_per_year": slope,
        }
        if name.startswith("Without-"):
            row["slope_change_vs_full"] = slope - full_slope   # > 0: decline is flatter without the group
        if group in present and name.startswith("Only-"):
            t = turnover[turnover["group"] == group].sort_values("month")
            share, f1 = t["unseen_occurrence_share"].to_numpy(), full["f1"].to_numpy()
            if np.ptp(share) == 0 or np.ptp(f1) == 0:
                rho, pval = np.nan, np.nan            # correlation undefined for a constant series
            else:
                rho, pval = stats.spearmanr(share, f1)
            row["turnover_vs_full_f1_spearman"] = rho
            row["turnover_vs_full_f1_p"] = pval
            row["mean_unseen_occurrence_share"] = t["unseen_occurrence_share"].mean()
        summary.append(row)
    pd.DataFrame(summary).round(4).to_csv(OUT / "summary.csv", index=False)

    v2.write_json(OUT / "run_config.json", {
        "C": C, "c_selection_sha256": v2.sha256_file(selection_file),
        "train_months": TRAIN, "test_months": [test_months[0], test_months[-1]],
        "groups_present": present, "models": [s[0] for s in specs],
        "svm_params": v2.svm_params("primary"), "decision_threshold": v2.DECISION_THRESHOLD,
        "inputs": provenance, "run": v2.run_info(),
    })
    print(f"\nResults written to {OUT}/")


if __name__ == "__main__":
    main()
