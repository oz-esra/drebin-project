"""
Cumulative Monthly Retraining Algorithm Planning
------------------------------------------------
Goal: 
    Test if retraining the model every month using all past data helps 
    prevent performance decay (concept drift) and keeps malware detection rates high.

Steps:
1. Load Metadata & Features
2. Alignment & Chronological Sorting
3. Monthly Grouping
4. Monthly Retraining Loop (May 2023 -> Dec 2023)
   a. Split Data
   b. Vectorize
   c. Model Training
   d. Evaluation
5. Save Results
"""

import json
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from drebin.vectorize import fit, transform

FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"

def main():
    
    # 1. Load Metadata & Features:
    #    - Read metadata.json and hypercube_drebin.json.
    #    - Filter data up to end of 2023.
    #    - Set binary label: VT detection >= 2 -> Malware (1), else Benign (0).
    
    print("1. Loading metadata and extraction records...")
    df_meta = pd.read_json(METADATA_PATH)
    df_meta['gp_date'] = pd.to_datetime(df_meta['gp_date'])
    df_meta = df_meta[df_meta['gp_date'] <= "2023-12-31"].copy()
    df_meta['label'] = (df_meta['vt_detection'] >= 2).astype(int)

    with open(FEATURES_PATH, 'r') as f:
        all_features = json.load(f)

    
    # 2. Alignment & Chronological Sorting:
    #    - Keep only SHA256 keys present in both metadata and feature files.
    #    - Sort all samples chronologically by date (`gp_date`) to respect time order.
    
    common_shas = set(df_meta['sha256']).intersection(set(all_features.keys()))
    df_meta = df_meta[df_meta['sha256'].isin(common_shas)].copy()
    df_sorted = df_meta.sort_values('gp_date').reset_index(drop=True)

    
    # 3. Monthly Grouping:
    #    - Group dataset by month (YYYY-MM format).
    #    - Set May 2023 to December 2023 as the testing timeline.
    
    df_sorted['year_month'] = df_sorted['gp_date'].dt.to_period('M')
    test_months = df_sorted[df_sorted['gp_date'] >= "2023-05-01"]['year_month'].unique()

    print("\n2. Executing Monthly Cumulative Retraining Evaluation...\n")
    retraining_results = []

    
    # 4. Monthly Retraining Loop (May 2023 -> Dec 2023):
    
    for month in test_months:
        
        # --- a. Split Data ---
        # Training set = All samples BEFORE the current test month.
        # Testing set  = Samples from the CURRENT test month only.
        train_sub = df_sorted[df_sorted['year_month'] < month]
        test_sub = df_sorted[df_sorted['year_month'] == month]

        train_records = [{"sha256": row['sha256'], "features": all_features[row['sha256']]} for _, row in train_sub.iterrows()]
        test_records = [{"sha256": row['sha256'], "features": all_features[row['sha256']]} for _, row in test_sub.iterrows()]

        # --- b. Vectorize ---
        # Fit vocabulary on training set ONLY (to avoid data leakage).
        # Transform train and test data into sparse CSR matrices.
        vocabulary = fit(train_records)
        X_train = transform(train_records, vocabulary)
        y_train = train_sub['label'].values

        X_test = transform(test_records, vocabulary)
        y_test = test_sub['label'].values

        # --- c. Model Training ---
        # Train Linear SVM (`SGDClassifier` with balanced class weights).
        clf = SGDClassifier(
            loss='hinge', 
            penalty='l2', 
            alpha=1e-4, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # --- d. Evaluation ---
        # Predict test month labels and calculate Precision, Recall, and F1-score.
        y_pred = clf.predict(X_test)

        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        retraining_results.append({
            "Month": str(month),
            "Train_Samples": len(train_sub),
            "Test_Samples": len(test_sub),
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1-Score": f"{f1:.4f}"
        })

    
    # 5. Save Results:
    #    - Store monthly metrics in `monthly_retraining_results.csv` for comparison.
    
    df_res = pd.DataFrame(retraining_results)
    print(df_res.to_string(index=False))
    df_res.to_csv("monthly_retraining_results.csv", index=False)
    print("\nRetraining baseline results saved to 'monthly_retraining_results.csv'.")

if __name__ == "__main__":
    main()