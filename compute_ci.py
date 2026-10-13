import numpy as np

# 2024 OOT evaluation F1 scores (replace with your actual array/file loading)
f1_monthly = np.array([0.92, 0.91, 0.93, 0.90, 0.92, 0.91, 0.93, 0.92, 0.91, 0.90, 0.92, 0.91])
f1_quarterly = np.array([0.91, 0.90, 0.92, 0.89, 0.91, 0.90, 0.91, 0.91, 0.90, 0.89, 0.91, 0.90])

# Paired differences
diffs = f1_monthly - f1_quarterly
mean_diff = np.mean(diffs)

# Paired bootstrap procedure for 95% confidence interval
rng = np.random.default_rng(seed=42)
boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(10000)]
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

# Verification against fixed equivalence margin (+/-0.03)
margin = 0.03
is_equivalent = (ci_lower >= -margin) and (ci_upper <= margin)

print(f"Mean F1 Difference: {mean_diff:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Within Equivalence Margin (+/-{margin}): {is_equivalent}")