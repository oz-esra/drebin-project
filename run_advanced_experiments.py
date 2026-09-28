"""
================================================================================
TESSERACT Advanced Experiments: Feature Group Drift & Retraining Frequencies
================================================================================
Answers Professor's Two Questions:
(1) Which of the eight Drebin feature groups (S1-S8) are responsible for degradation?
(2) How much periodic retraining (1M, 2M, 3M, 6M) is needed to recover performance?
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc

FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"

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
    return PREFIX_TO_GROUP.get(prefix, 'S8_NetworkAddresses')

def compute_metrics_with_ci(y_true, y_pred, y_scores, n_bootstraps=200):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    prec_array, rec_array, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec_array, prec_array)

    rng = np.random.RandomState(42)
    bootstrapped_f1s = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_samples = len(y_true_arr)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        _, _, b_f1, _ = precision_recall_fscore_support(y_true_arr[idx], y_pred_arr[idx], average='binary', zero_division=0)
        bootstrapped_f1s.append(b_f1)

    ci_low = np.percentile(bootstrapped_f1s, 2.5) if bootstrapped_f1s else f1
    ci_up = np.percentile(bootstrapped_f1s, 97.5) if bootstrapped_f1s else f1

    return p, r, f1, pr_auc, ci_low, ci_up

def build_group_sparse_matrices(sample_features_list, group_vocabs):
    group_matrices = {}
    for g_name, vocab in group_vocabs.items():
        rows, cols = [], []
        for r_idx, f_list in enumerate(sample_features_list):
            for feat in f_list:
                if feat in vocab:
                    rows.append(r_idx)
                    cols.append(vocab[feat])
        data = np.ones(len(rows), dtype=np.float32)
        shape = (len(sample_features_list), len(vocab))
        group_matrices[g_name] = csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)
    return group_matrices

def main():
    print("=" * 70)
    print("  TESSERACT ADVANCED EXPERIMENTS (2021-2023 DEVELOPMENT SET)")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/4] Loading metadata & features...")
    with open(METADATA_PATH, 'r') as f:
        raw_meta = json.load(f)

    df_meta = pd.DataFrame.from_dict(raw_meta, orient='index') if isinstance(raw_meta, dict) else pd.DataFrame(raw_meta)
    if 'sha256' not in df_meta.columns:
        df_meta['sha256'] = df_meta.index

    date_col = 'dex_date' if 'dex_date' in df_meta.columns else ('added' if 'added' in df_meta.columns else 'gp_date')
    df_meta['Month'] = pd.to_datetime(df_meta[date_col], errors='coerce').dt.strftime('%Y-%m')
    df_meta['Label'] = (pd.to_numeric(df_meta['vt_detection'], errors='coerce').fillna(0) > 0).astype(int)

    df_dev = df_meta[(df_meta['Month'] >= '2021-01') & (df_meta['Month'] <= '2023-12')].dropna(subset=['Month']).copy()
    
    with open(FEATURES_PATH, 'r') as f:
        raw_features = json.load(f)

    df_dev = df_dev[df_dev['sha256'].isin(raw_features)].sort_values('Month').copy()
    dev_shas = df_dev['sha256'].tolist()
    features_dict = {sha: raw_features[sha] for sha in dev_shas}

    # 2. Build S1-S8 Feature Vocabularies
    print("\n[2/4] Building S1-S8 Vocabularies...")
    group_vocabs = {f'S{i}': {} for i in range(1, 9)}
    group_names = {
        'S1': 'S1_Hardware', 'S2': 'S2_RequestedPermissions', 'S3': 'S3_AppComponents',
        'S4': 'S4_FilteredIntents', 'S5': 'S5_RestrictedAPICalls', 'S6': 'S6_UsedPermissionAPICalls',
        'S7': 'S7_SuspiciousAPICalls', 'S8': 'S8_NetworkAddresses'
    }

    for sha, f_list in features_dict.items():
        for feat in f_list:
            g_key = map_feature_to_group(feat)[:2]
            if g_key in group_vocabs and feat not in group_vocabs[g_key]:
                group_vocabs[g_key][feat] = len(group_vocabs[g_key])

    for g, v in group_vocabs.items():
        print(f"  {g} ({group_names[g]}): {len(v):,} features")

    months = sorted(df_dev['Month'].unique())
    init_month = months[0]
    eval_months = months[1:]

    month_data = {}
    for m in months:
        m_df = df_dev[df_dev['Month'] == m]
        m_shas = m_df['sha256'].tolist()
        m_feats = [features_dict[sha] for sha in m_shas]
        matrices = build_group_sparse_matrices(m_feats, group_vocabs)
        month_data[m] = {'mats': matrices, 'y': m_df['Label'].values, 'df': m_df}

    # --- EXPERIMENT 1: Feature Group Degradation Analysis (Static Models per S1-S8) ---
    print("\n[3/4] Running Experiment 1: S1-S8 Feature Group Degradation Analysis...")
    group_models = {}
    for g in group_vocabs:
        X_init = month_data[init_month]['mats'][g]
        y_init = month_data[init_month]['y']
        if X_init.shape[1] > 0 and len(np.unique(y_init)) > 1:
            clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42)
            clf.fit(X_init, y_init)
            group_models[g] = clf

    exp1_results = []
    for current_m in eval_months:
        y_test = month_data[current_m]['y']
        row = {'Month': current_m, 'Total_Samples': len(y_test), 'Malware_Ratio_%': round((np.sum(y_test==1)/len(y_test))*100, 2)}
        for g, clf in group_models.items():
            X_test = month_data[current_m]['mats'][g]
            if X_test.shape[1] > 0:
                pred = clf.predict(X_test)
                prob = clf.predict_proba(X_test)[:, 1]
                p, r, f1, pr_auc, ci_l, ci_u = compute_metrics_with_ci(y_test, pred, prob, n_bootstraps=50)
                row[f'{g}_F1'] = round(f1, 4)
                row[f'{g}_PR_AUC'] = round(pr_auc, 4)
            else:
                row[f'{g}_F1'] = 0.0
                row[f'{g}_PR_AUC'] = 0.0
        exp1_results.append(row)

    df_exp1 = pd.DataFrame(exp1_results)
    df_exp1.to_csv("experiment1_s1_s8_degradation.csv", index=False)
    print("  -> Saved Experiment 1 results to 'experiment1_s1_s8_degradation.csv'")

    # --- EXPERIMENT 2: Periodic Retraining Frequencies (1M, 2M, 3M, 6M) ---
    print("\n[4/4] Running Experiment 2: Retraining Frequency Analysis (1M, 2M, 3M, 6M)...")
    
    # Combined Vocabulary Matrix for Experiment 2
    full_vocab = {}
    for sha, f_list in features_dict.items():
        for feat in f_list:
            if feat not in full_vocab:
                full_vocab[feat] = len(full_vocab)

    for m in months:
        m_shas = month_data[m]['df']['sha256'].tolist()
        m_feats = [features_dict[sha] for sha in m_shas]
        rows, cols = [], []
        for r_idx, f_list in enumerate(m_feats):
            for feat in f_list:
                if feat in full_vocab:
                    rows.append(r_idx)
                    cols.append(full_vocab[feat])
        shape = (len(m_feats), len(full_vocab))
        month_data[m]['X_full'] = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)

    frequencies = {'1M_Monthly': 1, '2M_Bimonthly': 2, '3M_Quarterly': 3, '6M_Semiannually': 6}
    exp2_results = []

    for current_m in eval_months:
        y_test = month_data[current_m]['y']
        X_test = month_data[current_m]['X_full']
        m_idx = months.index(current_m)

        row = {'Month': current_m, 'Total_Samples': len(y_test), 'Malware_Ratio_%': round((np.sum(y_test==1)/len(y_test))*100, 2)}

        for freq_label, K in frequencies.items():
            # Find the most recent retraining point
            last_train_idx = ((m_idx - 1) // K) * K
            training_months = months[:last_train_idx + 1]

            X_tr = csr_matrix(np.vstack([month_data[m]['X_full'].toarray() for m in training_months]))
            y_tr = np.concatenate([month_data[m]['y'] for m in training_months])

            clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42)
            clf.fit(X_tr, y_tr)

            pred = clf.predict(X_test)
            prob = clf.predict_proba(X_test)[:, 1]
            p, r, f1, pr_auc, ci_l, ci_u = compute_metrics_with_ci(y_test, pred, prob, n_bootstraps=100)

            row[f'{freq_label}_Precision'] = round(p, 4)
            row[f'{freq_label}_Recall'] = round(r, 4)
            row[f'{freq_label}_F1'] = round(f1, 4)
            row[f'{freq_label}_PR_AUC'] = round(pr_auc, 4)
            row[f'{freq_label}_F1_95CI'] = f"[{ci_l:.3f}-{ci_u:.3f}]"

        exp2_results.append(row)
        print(f"Processed {current_m} | 1M F1: {row['1M_Monthly_F1']} | 3M F1: {row['3M_Quarterly_F1']} | 6M F1: {row['6M_Semiannually_F1']}")

    df_exp2 = pd.DataFrame(exp2_results)
    df_exp2.to_csv("experiment2_retraining_frequencies.csv", index=False)
    print("\n" + "=" * 70)
    print("SUCCESS: All advanced experiments completed!")
    print("Results saved to:")
    print("  1. 'experiment1_s1_s8_degradation.csv'")
    print("  2. 'experiment2_retraining_frequencies.csv'")
    print("=" * 70)

if __name__ == "__main__":
    main()