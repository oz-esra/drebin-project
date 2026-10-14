"""Step 3 of PROTOCOL v2: primary hypothesis, quarterly vs monthly retraining.

Paired difference per test month: d = F1(Cum-1M) - F1(Cum-3M), 24 months
(2022-01 .. 2023-12). Equivalence margin fixed in advance: +/-0.03.

Primary criterion (as requested by Prof. Lanzi): the 95% CI of the mean
paired difference must lie entirely inside (-0.03, +0.03). The t-based CI over
months is primary; a percentile bootstrap over months is reported as a
check. I also report the TOST p-value (equivalent to the 90% CI criterion)
and the ordinary two-sided test against zero, so "statistically different"
and "practically equivalent" can both be stated.

Usage:  python equivalence_test.py
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import v2_common as v2

MARGIN = 0.03
ALPHA = 0.05
N_BOOT = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(v2.RESULTS_DIR / "retraining" / "monthly_f1_wide.csv"))
    ap.add_argument("--reference", default="Cum-1M")
    ap.add_argument("--candidate", default="Cum-3M")
    ap.add_argument("--output", default=str(v2.RESULTS_DIR / "equivalence_cum3m_vs_cum1m.json"))
    args = ap.parse_args()

    wide = pd.read_csv(args.input, dtype={"month": str}).set_index("month").sort_index()
    pair = wide[[args.reference, args.candidate]]
    if pair.isna().any().any():
        raise SystemExit(f"Missing F1 values:\n{pair[pair.isna().any(axis=1)]}")
    d = (pair[args.reference] - pair[args.candidate]).to_numpy()
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    se, dof = sd / np.sqrt(n), n - 1

    t95, t90 = stats.t.ppf(1 - ALPHA / 2, dof), stats.t.ppf(1 - ALPHA, dof)
    ci95 = [mean - t95 * se, mean + t95 * se]
    ci90 = [mean - t90 * se, mean + t90 * se]
    p_two_sided = float(2 * stats.t.sf(abs(mean / se), dof)) if se > 0 else None
    p_tost = float(max(stats.t.sf((mean + MARGIN) / se, dof),
                       stats.t.cdf((mean - MARGIN) / se, dof))) if se > 0 else None

    # Percentile bootstrap over months (resampling whole months).
    rng = np.random.default_rng(v2.SEED)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    ci95_boot = np.percentile(boot, [2.5, 97.5]).tolist()

    inside = lambda ci: bool(-MARGIN < ci[0] and ci[1] < MARGIN)
    result = {
        "hypothesis": f"{args.candidate} is equivalent to {args.reference} within +/-{MARGIN} F1",
        "difference": f"F1({args.reference}) - F1({args.candidate}) per month",
        "months": [pair.index[0], pair.index[-1]], "n_months": n,
        "mean_diff": mean, "sd_diff": sd,
        "ci95_t": ci95, "ci95_bootstrap_months": ci95_boot, "ci90_t": ci90,
        "p_two_sided_vs_zero": p_two_sided, "p_tost": p_tost,
        "margin": MARGIN, "alpha": ALPHA,
        "primary_criterion": "95% t-based CI entirely inside the margin",
        "equivalent_primary": inside(ci95),
        "equivalent_bootstrap_check": inside(ci95_boot),
        "difference_significant": p_two_sided is not None and p_two_sided < ALPHA,
        "diffs": dict(zip(pair.index, d.round(6).tolist())),
        "input_file": args.input, "input_sha256": v2.sha256_file(args.input),
        "run": v2.run_info(),
    }
    v2.write_json(args.output, result)

    print(f"Mean F1 difference ({args.reference} - {args.candidate}) over {n} months: {mean:.4f}")
    print(f"95% CI (t): [{ci95[0]:.4f}, {ci95[1]:.4f}] | 95% CI (bootstrap): "
          f"[{ci95_boot[0]:.4f}, {ci95_boot[1]:.4f}] | 90% CI (t): [{ci90[0]:.4f}, {ci90[1]:.4f}]")
    print(f"Two-sided p vs 0: {p_two_sided} | TOST p: {p_tost}")
    print(f"Equivalent (primary, 95% CI inside +/-{MARGIN}): {result['equivalent_primary']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
