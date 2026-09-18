import json
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from drebin.vectorize import fit, transform

FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"

def main():
    print("1. Loading metadata and processing extraction records...")
    df_meta = pd.read_json(METADATA_PATH)
    df_meta['gp_date'] = pd.to_datetime(df_meta['gp_date'])
    df_meta = df_meta[df_meta['gp_date'] <= "2023-12-31"].copy()
    df_meta['label'] = (df_meta['vt_detection'] >= 2).astype(int)

    with open(FEATURES_PATH, 'r') as f:
        all_features = json.load(f)

    common_shas = set(df_meta['sha256']).intersection(set(all_features.keys()))
    df_meta = df_meta[df_meta['sha256'].isin(common_shas)].copy()
    df_sorted = df_meta.sort_values('gp_date').reset_index(drop=True)

    # Time-aware chronological 80/20 split (TESSERACT protocol compliance)
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:].copy()

    # Decoupled sparse vectorization
    train_records = [{"sha256": row['sha256'], "features": all_features[row['sha256']]} for _, row in train_df.iterrows()]
    vocabulary = fit(train_records)
    X_train = transform(train_records, vocabulary)
    y_train = train_df['label'].values

    # Model training (Linear SVM via SGD)
    print("2. Training Linear SVM (SGD) classifier...")
    clf = SGDClassifier(
        loss='hinge', 
        penalty='l2', 
        alpha=1e-4, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Monthly grouping and temporal degradation analysis
    print("\n3. Computing monthly performance degradation metrics...\n")
    test_df['year_month'] = test_df['gp_date'].dt.to_period('M')
    
    monthly_results = []
    for month, group in test_df.groupby('year_month'):
        test_records = [{"sha256": row['sha256'], "features": all_features[row['sha256']]} for _, row in group.iterrows()]
        X_month = transform(test_records, vocabulary)
        y_month = group['label'].values
        
        y_pred = clf.predict(X_month)
        
        prec = precision_score(y_month, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_month, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_month, y_pred, pos_label=1, zero_division=0)
        
        monthly_results.append({
            "Month": str(month),
            "Samples": len(group),
            "Malware_Count": y_month.sum(),
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1-Score": f"{f1:.4f}"
        })

    # Export structured evaluation table
    df_results = pd.DataFrame(monthly_results)
    print(df_results.to_string(index=False))
    df_results.to_csv("monthly_degradation.csv", index=False)
    print("\nResults successfully exported to 'monthly_degradation.csv'.")

if __name__ == "__main__":
    main()