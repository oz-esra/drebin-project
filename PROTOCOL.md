# TESSERACT Out-of-Time (2024) Evaluation Protocol (FROZEN)

## 1. Frozen Repository & Environment
- **Model Parameters**: SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42)
- **Decision Threshold**: 0.5
- **Feature Space**: 2,116,260 mapped Drebin features (S1-S8)
- **Random Seeds**: 42 (Model training & Bootstrap sampling)
- **Bootstrap Samples**: B = 200 (for 95% Confidence Intervals)

## 2. Primary Hypothesis & Equivalence Margin
- **Hypothesis**: Quarterly retraining (3M) is functionally comparable to monthly retraining (1M) on out-of-time (2024) data.
- **Equivalence Margin**: $\Delta F1 \le 0.03$ (The average F1 difference between 1M and 3M across 2024 must be <= 3 percentage points, or their 95% CIs must overlap).

## 3. Evaluated Baselines & Schedules
1. **Static**: Trained on 2021-01, evaluated across 2024 without retraining.
2. **1M (Monthly)**: Retrained every month on rolling historical window.
3. **3M (Quarterly)**: Retrained every 3 months (Preselected Policy).
4. **6M (Semiannual)**: Retrained every 6 months.

## 4. Evaluated Metrics
- **Performance**: Precision, Recall, F1-Score, PR-AUC, False Positive Rate (FPR)
- **Time-Aware Aggregate**: Area Under Time (A-AUT) for F1 and PR-AUC
- **Uncertainty**: 95% Confidence Intervals via non-parametric percentile bootstrap
- **Cost Metric**: Retraining Operations Count (1M=12, 3M=4, 6M=2, Static=0)
