"""Secondary analyses of PROTOCOL v2 (Section 7): paired monthly comparisons.

This script only reports the secondary comparisons that the frozen protocol
already defines; it does not change any analysis. It was added after the tag
because equivalence_test.py (the tool for the primary hypothesis H1) writes a
fixed output file, and running it on other pairs would overwrite the H1 result.
equivalence_test.py itself is left unchanged.

Pairs, fixed by the protocol:
  Family A (retraining frequency, cumulative history):
      Cum-3M vs Cum-1M, Cum-6M vs Cum-1M, Cum-6M vs Cum-3M,
      and Cum-1M, Cum-3M, Cum-6M vs the Static reference
  Family B (history length at quarterly frequency):
      SW12-3M vs Cum-3M, SW6-3M vs Cum-3M, SW6-3M vs SW12-3M

For each pair and metric (F1, Average Precision), over the 24 test months:
d = metric(policy) - metric(comparator); mean, SD, 95% t-based CI, 95%
percentile bootstrap CI over months, and the two-sided paired t-test against
zero. These are descriptive results: no equivalence claim is made outside H1
and no multiple-comparison correction is applied (stated in the report).

Usage:  python paired_comparisons.py                        (primary, all test samples)
        python paired_comparisons.py --set excl_overlap     (overlap-excluded sensitivity)
        python paired_comparisons.py --variant balanced     (cost-sensitive sensitivity)
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import v2_common as v2

PAIRS = [
    ("A", "Cum-3M", "Cum-1M"),
    ("A", "Cum-6M", "Cum-1M"),
    ("A", "Cum-6M", "Cum-3M"),
    ("A", "Cum-1M", "Static"),
    ("A", "Cum-3M", "Static"),
    ("A", "Cum-6M", "Static"),
    ("B", "SW12-3M", "Cum-3M"),
    ("B", "SW6-3M", "Cum-3M"),
    ("B", "SW6-3M", "SW12-3M"),
]
METRICS = ("f1", "average_precision")
ALPHA = 0.05
N_BOOT = 10_000


def paired_summary(d, rng):
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = stats.t.ppf(1 - ALPHA / 2, n - 1)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(2 * stats.t.sf(abs(mean / se), n - 1)) if se > 0 else np.nan
    return {"n_months": n, "mean_diff": mean, "sd_diff": sd,
            "ci95_t_low": mean - t * se, "ci95_t_high": mean + t * se,
            "ci95_boot_low": float(lo), "ci95_boot_high": float(hi),
            "p_two_sided_vs_zero": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(v2.VARIANTS), default="primary")
    ap.add_argument("--set", choices=["all", "excl_overlap"], default="all")
    args = ap.parse_args()
    base = v2.results_dir(args.variant)
    source = base / "retraining" / "monthly_metrics.csv"
    if not source.exists():
        raise SystemExit(f"{source} not found: run run_retraining_experiments.py first.")

    monthly = pd.read_csv(source, dtype={"month": str})
    prefix = "" if args.set == "all" else "excl_overlap_"
    rng = np.random.default_rng(v2.SEED)

    rows = []
    for family, policy, comparator in PAIRS:
        for metric in METRICS:
            wide = monthly.pivot(index="month", columns="policy", values=prefix + metric)
            pair = wide[[policy, comparator]].dropna()
            d = (pair[policy] - pair[comparator]).to_numpy()
            rows.append({
                "family": family, "policy": policy, "comparator": comparator, "metric": metric,
                "difference": f"{metric}({policy}) - {metric}({comparator})",
                "mean_policy": float(pair[policy].mean()), "mean_comparator": float(pair[comparator].mean()),
                **paired_summary(d, rng),
            })

    table = pd.DataFrame(rows)
    suffix = "" if args.set == "all" else "_excl_overlap"
    out_csv = base / f"paired_comparisons{suffix}.csv"
    table.round(6).to_csv(out_csv, index=False)
    v2.write_json(base / f"paired_comparisons{suffix}.json", {
        "variant": args.variant, "test_set": args.set,
        "note": ("Secondary, descriptive comparisons defined in PROTOCOL v2 Section 7. "
                 "No equivalence claim outside H1; no multiple-comparison correction."),
        "alpha": ALPHA, "n_bootstrap": N_BOOT, "seed": v2.SEED,
        "source_file": str(source), "source_sha256": v2.sha256_file(source),
        "run": v2.run_info(),
    })

    show = table[table["metric"] == "f1"]
    print(f"Variant: {args.variant} | test set: {args.set} | F1 differences (policy - comparator):")
    for _, r in show.iterrows():
        print(f"  [{r['family']}] {r['policy']:>8} - {r['comparator']:<8} "
              f"{r['mean_diff']:+.4f}  95% CI [{r['ci95_t_low']:+.4f}, {r['ci95_t_high']:+.4f}]  "
              f"p={r['p_two_sided_vs_zero']:.2g}")
    print(f"Saved: {out_csv} (F1 and Average Precision)")


if __name__ == "__main__":
    main()
