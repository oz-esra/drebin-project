import json
import pandas as pd

# File paths to local dataset files
FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"

# 1. Load dataset metadata containing temporal information and malware labels
print("1. Loading metadata...")
df_meta = pd.read_json(METADATA_PATH)

# 2. Load DREBIN extracted feature dictionary {sha256: [feature_list]}
print("2. Loading DREBIN features JSON...")
with open(FEATURES_PATH, 'r') as f:
    features = json.load(f)

# 3. Validate SHA-256 key alignment between metadata and features
print("3. Validating SHA-256 key alignment...")
meta_shas = set(df_meta['sha256'])
feature_shas = set(features.keys())

# Compute symmetric difference to detect non-matching SHA-256 hashes
diff = meta_shas.symmetric_difference(feature_shas)

if not diff:
    print("\n✓ [SUCCESS] All SHA-256 keys match perfectly!")
    print(f"Total Sample Count: {len(meta_shas)}")
else:
    print(f"\n⚠ [WARNING] Found {len(diff)} non-matching keys.")

# 4. Perform time-aware temporal ordering and split based on gp_date (TESSERACT constraint)
print("\n4. Performing time-aware sorting by gp_date...")
df_meta['gp_date'] = pd.to_datetime(df_meta['gp_date'])
df_sorted = df_meta.sort_values('gp_date').reset_index(drop=True)

# Define time-aware 80/20 train-test split index
split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

print(f"Train Window: {train_df['gp_date'].min().date()} -> {train_df['gp_date'].max().date()} ({len(train_df)} samples)")
print(f"Test Window:  {test_df['gp_date'].min().date()} -> {test_df['gp_date'].max().date()} ({len(test_df)} samples)")