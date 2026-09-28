"""
Step 1: Feature Mapping Verification (Drebin S1 - S8 Sets)
"""
import json
import pandas as pd

FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"

PREFIX_TO_GROUP = {
    'hardware': 'S1_Hardware',
    'app_permissions': 'S2_RequestedPermissions',
    'permission': 'S2_RequestedPermissions',
    'activities': 'S3_AppComponents',
    'services': 'S3_AppComponents',
    'providers': 'S3_AppComponents',
    'receivers': 'S3_AppComponents',
    'intents': 'S4_FilteredIntents',
    'api_permissions': 'S5_RestrictedAPICalls',
    'api_calls': 'S6_UsedPermissionAPICalls',
    'interesting_calls': 'S7_SuspiciousAPICalls',
    'urls': 'S8_NetworkAddresses'
}

def map_feature_to_group(feature_str):
    prefix = feature_str.split('::')[0] if '::' in feature_str else feature_str.split('_')[0]
    return PREFIX_TO_GROUP.get(prefix, 'UNMAPPED')

def verify_mapping():
    print("Loading hypercube_drebin.json to verify S1-S8 mapping...")
    with open(FEATURES_PATH, 'r') as f:
        features_data = json.load(f)

    all_unique_features = set()
    for feat_list in features_data.values():
        all_unique_features.update(feat_list)

    print(f"Total Unique Features: {len(all_unique_features):,}")

    group_counts = {}
    unmapped = []

    for feat in all_unique_features:
        group = map_feature_to_group(feat)
        group_counts[group] = group_counts.get(group, 0) + 1
        if group == 'UNMAPPED':
            unmapped.append(feat)

    print("\n" + "="*50)
    print("  DREBIN S1-S8 FEATURE GROUP BREAKDOWN")
    print("="*50)
    for group in sorted(group_counts.keys()):
        print(f"{group:<28}: {group_counts[group]:>8,} features")

    if unmapped:
        print(f"\nWARNING: Found {len(unmapped)} unmapped features. First 5: {unmapped[:5]}")
    else:
        print("\nSUCCESS: 100% of features successfully mapped to S1-S8!")

if __name__ == "__main__":
    verify_mapping()