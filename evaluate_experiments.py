"""
================================================================================
TESSERACT Monthly Retraining & Evaluation Pipeline (2021 - 2023 Development Set)
================================================================================
Features:
- Reserve 2024 set strictly for out-of-time final validation.
- Evaluate on 2021-01 to 2023-12 development window.
- Models Compared:
    1. Static Model (Trained on 2021-01, tested monthly through 2023-12)
    2. Cumulative Retraining (Retrained every month using ALL past data)
    3. Sliding-Window Retraining (Retrained using last 3 months of data)
- Metrics Computed:
    - Sample Count & Class Balance (% Malware)
    - Precision, Recall, F1 Score
    - PR-AUC (Precision-Recall Area Under Curve)
    - 95% Confidence Intervals (CI) via Bootstrapping for F1 Score
- Export: Saves full results to 'experiments_2021_2023_results.csv'
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    auc
)

FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"
OUTPUT_CSV = "experiments_2021_2023_results.csv"

SLIDING_WINDOW_MONTHS = 3  # Kayan pencere boyutu (son 3 ay)


def compute_metrics_with_ci(y_true, y_pred, y_scores, n_bootstraps=300):
    """
    Calculates Precision, Recall, F1, PR-AUC, and 95% CI for F1 via bootstrapping.
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    prec_array, rec_array, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec_array, prec_array)

    # 95% Confidence Interval calculation for F1 via Bootstrapping
    rng = np.random.RandomState(42)
    bootstrapped_f1s = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_samples = len(y_true_arr)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        _, _, b_f1, _ = precision_recall_fscore_support(
            y_true_arr[idx], y_pred_arr[idx], average='binary', zero_division=0
        )
        bootstrapped_f1s.append(b_f1)

    if len(bootstrapped_f1s) > 0:
        ci_lower = np.percentile(bootstrapped_f1s, 2.5)
        ci_upper = np.percentile(bootstrapped_f1s, 97.5)
    else:
        ci_lower, ci_upper = f1, f1

    return p, r, f1, pr_auc, ci_lower, ci_upper


def build_sparse_matrix(sample_features_list, vocab):
    """
    Converts a list of feature lists into a SciPy CSR Sparse Matrix using fixed vocabulary.
    """
    rows, cols = [], []
    for row_idx, feature_list in enumerate(sample_features_list):
        for feat in feature_list:
            if feat in vocab:
                rows.append(row_idx)
                cols.append(vocab[feat])

    data = np.ones(len(rows), dtype=np.float32)
    shape = (len(sample_features_list), len(vocab))
    return csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)


def main():
    print("=" * 70)
    print("  TESSERACT EXPERIMENT PIPELINE (2021-2023 DEVELOPMENT SET)")
    print("=" * 70)

    # 1. Load Metadata
    print("\n[1/5] Loading metadata.json...")
    with open(METADATA_PATH, 'r') as f:
        raw_meta = json.load(f)

    if isinstance(raw_meta, dict):
        df_meta = pd.DataFrame.from_dict(raw_meta, orient='index')
        if 'sha256' not in df_meta.columns:
            df_meta['sha256'] = df_meta.index
    else:
        df_meta = pd.DataFrame(raw_meta)

    # Parse Month and Label
    date_col = 'dex_date' if 'dex_date' in df_meta.columns else ('added' if 'added' in df_meta.columns else 'gp_date')
    df_meta['Month'] = pd.to_datetime(df_meta[date_col], errors='coerce').dt.strftime('%Y-%m')
    df_meta['Label'] = (pd.to_numeric(df_meta['vt_detection'], errors='coerce').fillna(0) > 0).astype(int)

    # Filter strictly for 2021-01 to 2023-12 (2024 is strictly reserved)
    df_dev = df_meta[(df_meta['Month'] >= '2021-01') & (df_meta['Month'] <= '2023-12')].dropna(subset=['Month']).copy()
    df_dev = df_dev.sort_values('Month')

    dev_shas = set(df_dev['sha256'])
    print(f"Loaded {len(df_dev)} valid metadata records for development period (2021-01 to 2023-12).")

    # 2. Load Features JSON
    print("\n[2/5] Loading hypercube_drebin.json...")
    with open(FEATURES_PATH, 'r') as f:
        raw_features = json.load(f)

    # Align Features with Metadata SHAs
    features_dict = {sha: raw_features[sha] for sha in dev_shas if sha in raw_features}
    df_dev = df_dev[df_dev['sha256'].isin(features_dict)].copy()
    print(f"Successfully matched {len(df_dev)} samples between metadata and features.")

    # 3. Build Global Feature Vocabulary (from 2021-2023 development set)
    print("\n[3/5] Building Feature Vocabulary...")
    vocab = {}
    feat_index = 0
    for sha, feat_list in features_dict.items():
        for feat in feat_list:
            if feat not in vocab:
                vocab[feat] = feat_index
                feat_index += 1
    print(f"Total Unique Features in Development Set: {len(vocab):,}")

    # Group samples by month
    months = sorted(df_dev['Month'].unique())
    month_data = {}
    for m in months:
        m_df = df_dev[df_dev['Month'] == m]
        m_shas = m_df['sha256'].tolist()
        m_feats = [features_dict[sha] for sha in m_shas]
        X_m = build_sparse_matrix(m_feats, vocab)
        y_m = m_df['Label'].values
        month_data[m] = {'X': X_m, 'y': y_m, 'df': m_df}

    # Initial Training Month (e.g. 2021-01)
    init_month = months[0]
    eval_months = months[1:]

    print(f"\nInitial Training Month: {init_month} ({len(month_data[init_month]['y'])} samples)")
    print(f"Evaluation Window: {eval_months[0]} to {eval_months[-1]} ({len(eval_months)} months)")

    # 4. Train Initial Models
    print("\n[4/5] Training Baseline Static Model on Initial Month...")
    X_init = month_data[init_month]['X']
    y_init = month_data[init_month]['y']

    static_model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000)
    static_model.fit(X_init, y_init)

    # 5. Run Retraining Evaluation Loop
    print("\n[5/5] Running Monthly Retraining & Evaluation Loop...")
    results = []

    for idx, current_m in enumerate(eval_months):
        X_test = month_data[current_m]['X']
        y_test = month_data[current_m]['y']

        total_samples = len(y_test)
        malware_count = int(np.sum(y_test == 1))
        benign_count = int(np.sum(y_test == 0))
        malware_pct = (malware_count / total_samples) * 100 if total_samples > 0 else 0

        # --- A. Static Model Evaluation ---
        pred_stat = static_model.predict(X_test)
        prob_stat = static_model.predict_proba(X_test)[:, 1]
        p_s, r_s, f1_s, pr_auc_s, ci_low_s, ci_up_s = compute_metrics_with_ci(y_test, pred_stat, prob_stat)

        # --- B. Cumulative Retraining Model ---
        past_months = months[:months.index(current_m)]
        X_cum = csr_matrix(np.vstack([month_data[m]['X'].toarray() for m in past_months]))
        y_cum = np.concatenate([month_data[m]['y'] for m in past_months])

        cum_model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000)
        cum_model.fit(X_cum, y_cum)

        pred_cum = cum_model.predict(X_test)
        prob_cum = cum_model.predict_proba(X_test)[:, 1]
        p_c, r_c, f1_c, pr_auc_c, ci_low_c, ci_up_c = compute_metrics_with_ci(y_test, pred_cum, prob_cum)

        # --- C. Sliding-Window Retraining Model (Last N Months) ---
        window_months = past_months[-SLIDING_WINDOW_MONTHS:]
        X_sw = csr_matrix(np.vstack([month_data[m]['X'].toarray() for m in window_months]))
        y_sw = np.concatenate([month_data[m]['y'] for m in window_months])

        sw_model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000)
        sw_model.fit(X_sw, y_sw)

        pred_sw = sw_model.predict(X_test)
        prob_sw = sw_model.predict_proba(X_test)[:, 1]
        p_sw, r_sw, f1_sw, pr_auc_sw, ci_low_sw, ci_up_sw = compute_metrics_with_ci(y_test, pred_sw, prob_sw)

        # Record Monthly Results
        results.append({
            'Month': current_m,
            'Total_Samples': total_samples,
            'Malware_Count': malware_count,
            'Benign_Count': benign_count,
            'Malware_Ratio_%': round(malware_pct, 2),
            # Static Metrics
            'Static_Precision': round(p_s, 4),
            'Static_Recall': round(r_s, 4),
            'Static_F1': round(f1_s, 4),
            'Static_PR_AUC': round(pr_auc_s, 4),
            'Static_F1_95CI': f"[{ci_low_s:.3f}-{ci_up_s:.3f}]",
            # Cumulative Metrics
            'Cum_Precision': round(p_c, 4),
            'Cum_Recall': round(r_c, 4),
            'Cum_F1': round(f1_c, 4),
            'Cum_PR_AUC': round(pr_auc_c, 4),
            'Cum_F1_95CI': f"[{ci_low_c:.3f}-{ci_up_c:.3f}]",
            # Sliding Window Metrics
            'SW_Precision': round(p_sw, 4),
            'SW_Recall': round(r_sw, 4),
            'SW_F1': round(f1_sw, 4),
            'SW_PR_AUC': round(pr_auc_sw, 4),
            'SW_F1_95CI': f"[{ci_low_sw:.3f}-{ci_up_sw:.3f}]"
        })

        print(f"Processed {current_m} | Total: {total_samples:>5} | Mal%: {malware_pct:>5.1f}% | "
              f"Static F1: {f1_s:.3f} | Cum F1: {f1_c:.3f} | Sliding F1: {f1_sw:.3f}")

    # Export to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 70)
    print(f"SUCCESS: Pipeline complete. Results exported to '{OUTPUT_CSV}'.")
    print("=" * 70)


if __name__ == "__main__":
    main()