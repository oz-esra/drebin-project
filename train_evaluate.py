import json
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from drebin.vectorize import fit, transform

# Local dataset file paths
FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"

def main():
    print("1. Loading metadata and performing 2021-2023 VTT=2 filtering...")
    df_meta = pd.read_json(METADATA_PATH)
    df_meta['gp_date'] = pd.to_datetime(df_meta['gp_date'])
    
    # Target timeframe filtering (2021-2023)
    df_meta = df_meta[df_meta['gp_date'] <= "2023-12-31"].copy()
    df_meta['label'] = (df_meta['vt_detection'] >= 2).astype(int)

    print("2. Loading DREBIN features JSON...")
    with open(FEATURES_PATH, 'r') as f:
        all_features = json.load(f)

    # Align SHA-256 keys
    common_shas = set(df_meta['sha256']).intersection(set(all_features.keys()))
    df_meta = df_meta[df_meta['sha256'].isin(common_shas)].copy()
    
    # Strictly sort chronologically by gp_date for TESSERACT compliance
    df_sorted = df_meta.sort_values('gp_date').reset_index(drop=True)

    # 3. Time-aware 80/20 train-test split
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    print(f"\nTraining set : {len(train_df)} samples ({train_df['gp_date'].min().date()} to {train_df['gp_date'].max().date()})")
    print(f"Testing set  : {len(test_df)} samples ({test_df['gp_date'].min().date()} to {test_df['gp_date'].max().date()})")

    # Extract raw feature lists corresponding to SHA256 hashes
    train_records = [{"sha256": row['sha256'], "features": all_features[row['sha256']]} for _, row in train_df.iterrows()]
    test_records = [{"sha256": row['sha256'], "features": all_features[row['sha256']]} for _, row in test_df.iterrows()]

    # 4. Decoupled Vectorization (Fit on Train ONLY, Transform on both)
    print("\n4. Fitting vocabulary on training split...")
    vocabulary = fit(train_records)
    print(f"Learned feature space size |S|: {len(vocabulary)}")

    print("Transforming training and testing sets to sparse CSR matrices...")
    X_train = transform(train_records, vocabulary)
    y_train = train_df['label'].values

    X_test = transform(test_records, vocabulary)
    y_test = test_df['label'].values

    # 5. Model Training (Linear SVM via SGD - Fast & Fully Convergent)
    print("\n5. Training Linear SVM (SGD) classifier...")
    clf = SGDClassifier(
        loss='hinge',             # Standard Linear SVM loss
        penalty='l2', 
        alpha=1e-4, 
        class_weight='balanced', 
        max_iter=2000, 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 6. Evaluation
    print("\n6. Evaluating model performance on test split...")
    y_pred = clf.predict(X_test)

    print("\n--- TESSERACT Evaluation Results ---")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malware"], digits=4))

if __name__ == "__main__":
    main()