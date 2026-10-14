"""Step 1 of PROTOCOL v2: select C for LinearSVC, once, inside 2021.

Train on January-September 2021, validate on October-December 2021 (pooled),
selection metric = F1 at the frozen threshold of 0. The vocabulary comes from
January-September only. If several C values give the same F1 at three
decimals, the smallest one wins (stronger regularisation).

2022-2023 are never touched here. The selected C is written to
results/v2/c_selection.json and is then reused, unchanged, by every policy
and window in the later scripts.

Usage:  python select_c.py
"""
import numpy as np
import pandas as pd

import v2_common as v2


def main():
    X, index, _, provenance = v2.load_development_data()
    y = index["label"].to_numpy()

    train = index["month"].between(*v2.C_TRAIN).to_numpy()
    valid = index["month"].between(*v2.C_VALID).to_numpy()
    print(f"C selection: train {v2.C_TRAIN[0]}..{v2.C_TRAIN[1]} ({train.sum():,} samples), "
          f"validate {v2.C_VALID[0]}..{v2.C_VALID[1]} ({valid.sum():,} samples)")

    rows = []
    for C in v2.C_GRID:
        clf, cols, info = v2.fit_svm(X[train], y[train], C)
        metrics = v2.evaluate(y[valid], v2.decision_scores(clf, cols, X[valid]))
        rows.append({"C": C, **info, **metrics})
        print(f"  C={C:<5} F1={metrics['f1']:.4f}  AP={metrics['average_precision']:.4f}  "
              f"converged={info['converged']}  ({info['fit_seconds']}s)")

    table = pd.DataFrame(rows)
    # Selection rule: best F1 rounded to three decimals, ties -> smallest C.
    table["f1_rounded"] = table["f1"].round(v2.C_TIE_DECIMALS)
    best = table["f1_rounded"].max()
    selected_C = float(table.loc[table["f1_rounded"] == best, "C"].min())
    table["selected"] = table["C"] == selected_C

    v2.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(v2.RESULTS_DIR / "c_selection.csv", index=False)
    v2.write_json(v2.RESULTS_DIR / "c_selection.json", {
        "selected_C": selected_C,
        "rule": f"max F1 at threshold {v2.DECISION_THRESHOLD} on pooled {v2.C_VALID}, "
                f"rounded to {v2.C_TIE_DECIMALS} decimals; ties -> smallest C",
        "grid": list(v2.C_GRID),
        "train_months": v2.C_TRAIN, "valid_months": v2.C_VALID,
        "svm_params": v2.SVM_PARAMS,
        "all_converged": bool(table["converged"].all()),
        "inputs": provenance, "run": v2.run_info(),
    })
    print(f"\nSelected C = {selected_C}  ->  {v2.RESULTS_DIR / 'c_selection.json'}")
    if not table["converged"].all():
        print("NOTE: at least one fit did not converge; this is recorded and settings stay unchanged.")


if __name__ == "__main__":
    main()
