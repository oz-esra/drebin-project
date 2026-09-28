"""
Step 1: Check and Map Drebin Feature Prefixes (S1 - S8)
"""

import json
from collections import Counter

FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"

def analyze_feature_prefixes():
    print("Loading features JSON to inspect prefixes...")
    with open(FEATURES_PATH, 'r') as f:
        all_features = json.load(f)

    prefix_counts = Counter()
    total_samples = len(all_features)
    
    # Extract unique feature prefixes across samples
    for sha, feats in all_features.items():
        for feat in feats:
            # Drebin features typically use 'prefix::feature_name'
            prefix = feat.split("::")[0] if "::" in feat else "UNKNOWN"
            prefix_counts[prefix] += 1

    print(f"\nAnalyzed {total_samples} samples.")
    print("\nFeature Prefix Distribution in Dataset:")
    print("-" * 45)
    for prefix, count in prefix_counts.most_common():
        print(f"{prefix:<25} : {count:>10} occurrences")

if __name__ == "__main__":
    analyze_feature_prefixes()