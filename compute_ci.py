import json
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiment2_retraining_frequencies.csv")
    parser.add_argument("--output", default="results/2021_2023_equivalence.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    # 2021-2023 (35 ay) sütun isimleri
    f1_monthly = df["1M_Monthly_F1"].values
    f1_quarterly = df["3M_Quarterly_F1"].values

    diffs = f1_monthly - f1_quarterly
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed=42)
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(10000)]

    ci_95_lower, ci_95_upper = map(float, np.percentile(boot_means, [2.5, 97.5]))
    ci_90_lower, ci_90_upper = map(float, np.percentile(boot_means, [5.0, 95.0]))

    margin = 0.03
    is_equivalent = bool((ci_95_lower >= -margin) and (ci_95_upper <= margin))

    results = {
        "dataset_period": "2021-02 to 2023-12",
        "sample_size_months": len(diffs),
        "mean_f1_difference": round(mean_diff, 4),
        "ci_95": [round(ci_95_lower, 4), round(ci_95_upper, 4)],
        "ci_90": [round(ci_90_lower, 4), round(ci_90_upper, 4)],
        "equivalence_margin": margin,
        "is_equivalent_within_margin": is_equivalent
    }

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()