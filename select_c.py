"""Step 1 of PROTOCOL v2: select C for LinearSVC, once, inside 2021.

Both the vocabulary and the model are fitted on January-September 2021 only.
Selection metric: the unweighted mean of the monthly F1 scores (threshold 0)
for October, November and December 2021. If several C values give the same
mean F1 at three decimals, the smallest one wins (stronger regularisation).
Pooled F1 over the three months is also stored, for information only.

2022-2023 are never touched here. The selected C is written to
<results>/c_selection.json and then reused, unchanged, by every policy and
window of the same variant. The "balanced" sensitivity variant selects its own
C with exactly the same rule.

Usage:  python select_c.py                     (primary, class_weight=None)
        python select_c.py --variant balanced  (cost-sensitive sensitivity analysis)
"""
import argparse

import numpy as np
import pandas as pd

import v2_common as v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(v2.VARIANTS), default="primary")
    args = ap.parse_args()
    out_dir = v2.results_dir(args.variant)

    X, index, _, provenance = v2.load_development_data()
    y = index["label"].to_numpy()
    months = index["month"].to_numpy()

    train = (months >= v2.C_TRAIN[0]) & (months <= v2.C_TRAIN[1])
    valid_months = list(v2.C_VALID_MONTHS)
    valid = np.isin(months, valid_months)
    print(f"Variant: {args.variant} (class_weight={v2.svm_params(args.variant)['class_weight']})")
    print(f"C selection: train {v2.C_TRAIN[0]}..{v2.C_TRAIN[1]} ({train.sum():,} samples), "
          f"validate {', '.join(valid_months)} ({valid.sum():,} samples)")

    rows = []
    for C in v2.C_GRID:
        clf, cols, info = v2.fit_svm(X[train], y[train], C, variant=args.variant)
        row = {"C": C, **info}
        monthly_f1 = []
        for m in valid_months:
            sel = months == m
            metrics = v2.evaluate(y[sel], v2.decision_scores(clf, cols, X[sel]))
            row[f"f1_{m}"] = metrics["f1"]
            row[f"ap_{m}"] = metrics["average_precision"]
            monthly_f1.append(metrics["f1"])
        row["mean_monthly_f1"] = float(np.mean(monthly_f1))       # selection metric
        pooled = v2.evaluate(y[valid], v2.decision_scores(clf, cols, X[valid]))
        row["pooled_f1_info_only"] = pooled["f1"]
        rows.append(row)
        print(f"  C={C:<5} mean monthly F1={row['mean_monthly_f1']:.4f}  "
              f"(pooled {pooled['f1']:.4f})  converged={info['converged']}  ({info['fit_seconds']}s)")

    table = pd.DataFrame(rows)
    # Selection rule: best mean monthly F1 rounded to three decimals, ties -> smallest C.
    table["mean_monthly_f1_rounded"] = table["mean_monthly_f1"].round(v2.C_TIE_DECIMALS)
    best = table["mean_monthly_f1_rounded"].max()
    selected_C = float(table.loc[table["mean_monthly_f1_rounded"] == best, "C"].min())
    table["selected"] = table["C"] == selected_C

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "c_selection.csv", index=False)
    v2.write_json(out_dir / "c_selection.json", {
        "variant": args.variant,
        "selected_C": selected_C,
        "rule": (f"max of the unweighted mean of monthly F1 (threshold {v2.DECISION_THRESHOLD}) "
                 f"over {', '.join(valid_months)}, rounded to {v2.C_TIE_DECIMALS} decimals; ties -> smallest C"),
        "grid": list(v2.C_GRID),
        "train_months": v2.C_TRAIN, "valid_months": valid_months,
        "vocabulary": "fitted on the training months only",
        "svm_params": v2.svm_params(args.variant),
        "all_converged": bool(table["converged"].all()),
        "inputs": provenance, "run": v2.run_info(),
    })
    print(f"\nSelected C = {selected_C}  ->  {out_dir / 'c_selection.json'}")
    if not table["converged"].all():
        print("NOTE: at least one fit did not converge; this is recorded and settings stay unchanged.")


if __name__ == "__main__":
    main()
