"""
================================================================================
TESSERACT Advanced Experiments: Feature Group Drift & Retraining Frequencies
(Refactored for Full gp_date Dataset, Sparse Memory Efficiency, & FPR/CI Metrics)
================================================================================
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

# ------------------------------------------------------------------------------
# Configuration Parameters
# ------------------------------------------------------------------------------
FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"
OUTPUT_DIR = "results"

# Label Rule Guardrail: must be set explicitly before execution
#   "vt_gt0"        : malware if vt_detection >= 1
#   "vt_ge2"        : malware if vt_detection >= 2 (Hypercube VTT=2, see 14 Sep note)
#   "vt_0_vs_4plus" : goodware if 0, malware if >= 4, samples with 1-3 detections excluded
VALID_LABEL_RULES = ("vt_gt0", "vt_ge2", "vt_0_vs_4plus")
LABEL_RULE = None

# Fixed date column strictly aligned with research protocol
DATE_COL = "gp_date"
DEV_START, DEV_END = "2021-01", "2023-12"

# Expected development-set size when no samples are excluded (vt_gt0, vt_ge2) (all samples with features, see check_holdout.py)
EXPECTED_DEV_SAMPLES_VT_GT0 = 196_447

# Model and evaluation settings (recorded in the config JSON)
SEED = 42
# Linear SVM (hinge loss), as in the Drebin paper (Arp et al., NDSS 2014, Sec. II-C)
# class_weight="balanced" as in the 18 Sep baseline (train_evaluate.py, evaluate_decay.py, retrain_monthly.py)
SGD_PARAMS = dict(loss="hinge", penalty="l2", alpha=1e-4, class_weight="balanced", random_state=SEED)
N_BOOT_EXP1 = 50
N_BOOT_EXP2 = 100

PREFIX_TO_GROUP = {
    'hardware': 'S1_Hardware',
    'app_permissions': 'S2_RequestedPermissions',
    'permission': 'S2_RequestedPermissions',
    'activities': 'S3_AppComponents',
    'services': 'S3_AppComponents',
    'providers': 'S3_AppComponents',
    'receivers': 'S3_AppComponents',
    'intents': 'S4_FilteredIntents',
    'api_calls': 'S5_RestrictedAPICalls',         # Corrected Drebin S5
    'api_permissions': 'S6_UsedPermissions',      # Corrected Drebin S6
    'interesting_calls': 'S7_SuspiciousAPICalls',
    'urls': 'S8_NetworkAddresses'
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def map_feature_to_group(feature_str):
    prefix = feature_str.split('::')[0]
    if prefix not in PREFIX_TO_GROUP:
        raise ValueError(f"Unmapped feature prefix encountered: '{prefix}' in feature '{feature_str}'")
    return PREFIX_TO_GROUP[prefix]


def compute_metrics_with_ci(y_true, y_pred, y_scores, n_bootstraps=200):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    prec_array, rec_array, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec_array, prec_array)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    rng = np.random.RandomState(SEED)
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

    ci_low = np.percentile(bootstrapped_f1s, 2.5) if bootstrapped_f1s else f1
    ci_up = np.percentile(bootstrapped_f1s, 97.5) if bootstrapped_f1s else f1

    return p, r, f1, pr_auc, fpr, ci_low, ci_up


def build_group_sparse_matrices(sample_features_list, group_vocabs):
    group_matrices = {}
    for g_name, vocab in group_vocabs.items():
        rows, cols = [], []
        if len(vocab) > 0:
            for r_idx, f_list in enumerate(sample_features_list):
                for feat in f_list:
                    if feat in vocab:
                        rows.append(r_idx)
                        cols.append(vocab[feat])
            data = np.ones(len(rows), dtype=np.float32)
            shape = (len(sample_features_list), len(vocab))
            group_matrices[g_name] = csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)
        else:
            shape = (len(sample_features_list), 0)
            group_matrices[g_name] = csr_matrix(shape, dtype=np.float32)
    return group_matrices


def main():
    if LABEL_RULE not in VALID_LABEL_RULES:
        raise ValueError(
            f"LABEL_RULE must be explicitly set to one of {VALID_LABEL_RULES} "
            f"before running experiments. Currently set to {LABEL_RULE!r}."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  TESSERACT ADVANCED EXPERIMENTS (2021-2023 DEVELOPMENT SET - REFACTORED)")
    print(f"  Date Column: {DATE_COL} | Label Rule: {LABEL_RULE}")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/4] Loading metadata & features...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        raw_meta = json.load(f)

    df_meta = pd.DataFrame.from_dict(raw_meta, orient='index') if isinstance(raw_meta, dict) else pd.DataFrame(raw_meta)
    del raw_meta
    if 'sha256' not in df_meta.columns:
        df_meta['sha256'] = df_meta.index

    if DATE_COL not in df_meta.columns:
        raise KeyError(f"Specified DATE_COL '{DATE_COL}' not found in metadata columns.")

    df_meta['Month'] = pd.to_datetime(df_meta[DATE_COL], errors='coerce').dt.strftime('%Y-%m')

    # Apply Labeling Rule (missing detection counts are an error, never silently goodware)
    vt_counts = pd.to_numeric(df_meta['vt_detection'], errors='coerce')
    if vt_counts.isna().any():
        raise ValueError(f"{vt_counts.isna().sum():,} samples have a missing vt_detection value.")

    n_excluded_low_confidence = 0
    if LABEL_RULE == "vt_gt0":
        df_meta['Label'] = (vt_counts > 0).astype(int)
    elif LABEL_RULE == "vt_ge2":
        df_meta['Label'] = (vt_counts >= 2).astype(int)
    elif LABEL_RULE == "vt_0_vs_4plus":
        # Exclude low-confidence samples (1-3 detections)
        keep = (vt_counts == 0) | (vt_counts >= 4)
        n_excluded_low_confidence = int((~keep).sum())
        df_meta = df_meta[keep].copy()
        df_meta['Label'] = (vt_counts[keep] >= 4).astype(int)
        print(f"  Excluded low-confidence samples (1-3 detections): {n_excluded_low_confidence:,}")

    df_dev = df_meta[df_meta['Month'].between(DEV_START, DEV_END)].copy()
    assert len(df_dev) > 0, f"Development set is empty after filtering by '{DATE_COL}'."

    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        raw_features = json.load(f)

    df_dev = df_dev[df_dev['sha256'].isin(raw_features)].sort_values('Month').copy()
    dev_shas = df_dev['sha256'].tolist()
    features_dict = {sha: raw_features[sha] for sha in dev_shas}
    del raw_features

    print(f"  Development samples with features ({DATE_COL}, {DEV_START}..{DEV_END}): {len(df_dev):,}")
    if LABEL_RULE in ("vt_gt0", "vt_ge2"):
        assert len(df_dev) == EXPECTED_DEV_SAMPLES_VT_GT0, (
            f"Expected {EXPECTED_DEV_SAMPLES_VT_GT0:,} development samples, got {len(df_dev):,}"
        )

    # 2. Build S1-S8 Feature Vocabularies
    print("\n[2/4] Building S1-S8 Vocabularies...")
    group_vocabs = {f'S{i}': {} for i in range(1, 9)}
    group_names = {
        'S1': 'S1_Hardware',
        'S2': 'S2_RequestedPermissions',
        'S3': 'S3_AppComponents',
        'S4': 'S4_FilteredIntents',
        'S5': 'S5_RestrictedAPICalls',
        'S6': 'S6_UsedPermissions',
        'S7': 'S7_SuspiciousAPICalls',
        'S8': 'S8_NetworkAddresses'
    }

    for sha, f_list in features_dict.items():
        for feat in f_list:
            g_key = map_feature_to_group(feat).split('_')[0]
            if feat not in group_vocabs[g_key]:
                group_vocabs[g_key][feat] = len(group_vocabs[g_key])

    for g, v in group_vocabs.items():
        if len(v) == 0:
            print(f"  {g} ({group_names[g]}): 0 features (not present in Hypercube)")
        else:
            print(f"  {g} ({group_names[g]}): {len(v):,} features")

    months = sorted(df_dev['Month'].unique())
    init_month = months[0]
    eval_months = months[1:]

    month_data = {}
    for m in months:
        m_df = df_dev[df_dev['Month'] == m]
        m_feats = [features_dict[sha] for sha in m_df['sha256']]
        matrices = build_group_sparse_matrices(m_feats, group_vocabs)
        month_data[m] = {'mats': matrices, 'y': m_df['Label'].values, 'df': m_df}

    # --- EXPERIMENT 1: Feature Group Degradation Analysis ---
    print("\n[3/4] Running Experiment 1: S1-S8 Feature Group Degradation Analysis...")
    group_models = {}
    for g in group_vocabs:
        X_init = month_data[init_month]['mats'][g]
        y_init = month_data[init_month]['y']
        if X_init.shape[1] > 0 and len(np.unique(y_init)) > 1:
            clf = SGDClassifier(**SGD_PARAMS)
            clf.fit(X_init, y_init)
            group_models[g] = clf

    exp1_results = []
    for current_m in eval_months:
        y_test = month_data[current_m]['y']
        row = {
            'Month': current_m,
            'Total_Samples': len(y_test),
            'Malware_Ratio_%': round((np.sum(y_test == 1) / len(y_test)) * 100, 2)
        }
        for g in group_vocabs:
            if g in group_models:
                clf = group_models[g]
                X_test = month_data[current_m]['mats'][g]
                pred = clf.predict(X_test)
                scores = clf.decision_function(X_test)  # SVM margin, used for PR-AUC
                p, r, f1, pr_auc, fpr, ci_l, ci_u = compute_metrics_with_ci(
                    y_test, pred, scores, n_bootstraps=N_BOOT_EXP1
                )
                row[f'{g}_F1'] = round(f1, 4)
                row[f'{g}_PR_AUC'] = round(pr_auc, 4)
                row[f'{g}_FPR'] = round(fpr, 4)
            else:
                # Group absent (e.g. S1 in Hypercube): no model, so no score
                row[f'{g}_F1'] = np.nan
                row[f'{g}_PR_AUC'] = np.nan
                row[f'{g}_FPR'] = np.nan
        exp1_results.append(row)

    df_exp1 = pd.DataFrame(exp1_results)
    exp1_file = os.path.join(OUTPUT_DIR, f"experiment1_s1_s8_degradation_{LABEL_RULE}.csv")
    df_exp1.to_csv(exp1_file, index=False)
    print(f"  -> Saved Experiment 1 results to '{exp1_file}'")

    # --- EXPERIMENT 2: Periodic Retraining Frequencies ---
    print("\n[4/4] Running Experiment 2: Retraining Frequency Analysis...")

    full_vocab = {}
    for sha, f_list in features_dict.items():
        for feat in f_list:
            if feat not in full_vocab:
                full_vocab[feat] = len(full_vocab)

    for m in months:
        m_feats = [features_dict[sha] for sha in month_data[m]['df']['sha256']]
        rows, cols = [], []
        for r_idx, f_list in enumerate(m_feats):
            for feat in f_list:
                if feat in full_vocab:
                    rows.append(r_idx)
                    cols.append(full_vocab[feat])
        shape = (len(m_feats), len(full_vocab))
        month_data[m]['X_full'] = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
        )

    frequencies = {
        'Static': None,
        '1M_Monthly': 1,
        '2M_Bimonthly': 2,
        '3M_Quarterly': 3,
        '6M_Semiannually': 6
    }

    # Model caching: with cumulative training, the training set depends only on
    # last_train_idx, so strategies sharing an index share the model.
    # NOTE: when sliding-window strategies are added, the key must include the window size.
    model_cache = {}
    exp2_results = []

    for current_m in eval_months:
        y_test = month_data[current_m]['y']
        X_test = month_data[current_m]['X_full']
        m_idx = months.index(current_m)

        row = {
            'Month': current_m,
            'Total_Samples': len(y_test),
            'Malware_Ratio_%': round((np.sum(y_test == 1) / len(y_test)) * 100, 2)
        }

        for freq_label, K in frequencies.items():
            if K is None:
                # Static baseline: trained on initial month (index 0)
                last_train_idx = 0
            else:
                last_train_idx = ((m_idx - 1) // K) * K

            if last_train_idx not in model_cache:
                training_months = months[:last_train_idx + 1]
                X_tr = vstack([month_data[m]['X_full'] for m in training_months]).tocsr()
                y_tr = np.concatenate([month_data[m]['y'] for m in training_months])

                clf = SGDClassifier(**SGD_PARAMS)
                clf.fit(X_tr, y_tr)
                model_cache[last_train_idx] = clf
            clf = model_cache[last_train_idx]

            pred = clf.predict(X_test)
            scores = clf.decision_function(X_test)  # SVM margin, used for PR-AUC
            p, r, f1, pr_auc, fpr, ci_l, ci_u = compute_metrics_with_ci(
                y_test, pred, scores, n_bootstraps=N_BOOT_EXP2
            )

            row[f'{freq_label}_Precision'] = round(p, 4)
            row[f'{freq_label}_Recall'] = round(r, 4)
            row[f'{freq_label}_F1'] = round(f1, 4)
            row[f'{freq_label}_PR_AUC'] = round(pr_auc, 4)
            row[f'{freq_label}_FPR'] = round(fpr, 4)
            row[f'{freq_label}_F1_95CI'] = f"[{ci_l:.3f}-{ci_u:.3f}]"

        exp2_results.append(row)
        print(f"Processed {current_m} | Static F1: {row['Static_F1']} | "
              f"1M F1: {row['1M_Monthly_F1']} | 3M F1: {row['3M_Quarterly_F1']}")

    df_exp2 = pd.DataFrame(exp2_results)
    exp2_file = os.path.join(OUTPUT_DIR, f"experiment2_retraining_frequencies_{LABEL_RULE}.csv")
    df_exp2.to_csv(exp2_file, index=False)

    # Export Metadata/Configuration Log
    print("\nHashing input files for the configuration log...")
    config_data = {
        "date_column": DATE_COL,
        "label_rule": LABEL_RULE,
        "excluded_low_confidence_samples": n_excluded_low_confidence,
        "dev_period": f"{DEV_START} to {DEV_END}",
        "sample_count_dev": len(df_dev),
        "init_month": init_month,
        "n_eval_months": len(eval_months),
        "training_scheme": "cumulative (all months up to the last retraining point)",
        "vocabulary": "built from the full 2021-2023 development set; features unseen in a model's training data keep zero weight under SGD with L2, so predictions equal a train-only vocabulary",
        "retraining_schedules": {k: v for k, v in frequencies.items()},
        "classifier": "SGDClassifier (linear SVM, hinge loss)",
        "pr_auc_score": "decision_function margin",
        "classifier_params": SGD_PARAMS,
        "seed": SEED,
        "n_bootstraps": {"exp1": N_BOOT_EXP1, "exp2": N_BOOT_EXP2},
        "group_vocab_sizes": {g: len(v) for g, v in group_vocabs.items()},
        "vocab_size_full": len(full_vocab),
        "metadata_path": METADATA_PATH,
        "metadata_sha256": sha256_file(METADATA_PATH),
        "features_path": FEATURES_PATH,
        "features_sha256": sha256_file(FEATURES_PATH),
        "exp1_output": exp1_file,
        "exp2_output": exp2_file
    }
    config_file = os.path.join(OUTPUT_DIR, f"experiment_config_{LABEL_RULE}.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print("\n" + "=" * 70)
    print("SUCCESS: All advanced experiments completed!")
    print(f"  1. {exp1_file}")
    print(f"  2. {exp2_file}")
    print(f"  3. {config_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()