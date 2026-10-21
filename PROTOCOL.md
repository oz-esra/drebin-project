# Evaluation Protocol, version 2 (corrective)

**Status:** FROZEN on 2026-10-21 with the git tag `protocol-v2`, before any development run. Any later change requires a new version with its own changelog entry.
**Supersedes:** version 1, kept unchanged in `protocol/PROTOCOL_v1_superseded.md`.
**Scope:** Drebin-style Android malware detection on the Hypercube dataset; temporal degradation, retraining policies and feature-group analysis.

Version 1 was reopened because the development experiments had used `dex_date` instead of `gp_date` (about 7% of the data), and the reported "2024" result was in fact a re-aggregation of 2021–2023 results. No model has been evaluated on 2024 data. The full list of corrections is in the changelog at the end of this document.

---

## 1. Data

| Item | Value |
|---|---|
| Dataset | Hypercube (SaTML 2026), VTT=2 release |
| Metadata | `metadata.json`, SHA-256 `E2310BDD9B3A9A79A09B253FE8435E3BFA8E76C630DD8A63A5F8C31080E6D9F1` (identical in the `original` and `extended_2024` folders) |
| Development features | `original/hypercube_drebin.json`, SHA-256 `2214c2241d23f10235d11e64b9bc096db9ba8555be920aa9aded8cc13a7dad34` |
| Holdout features | `extended_2024/hypercube_drebin.json`, SHA-256 `da0a71e5362720d275cffa771bc219fe979395d01c8f2c9afa11cae78564ccf3` |
| Timestamp | `gp_date`, grouped by calendar month |
| Label | malware if `vt_detection >= 2`, goodware otherwise (Hypercube VTT=2). Samples with 2–3 detections are kept. |
| Development period | 2021-01 .. 2023-12 |
| Holdout period | 2024-01 .. 2024-12, used once (Section 8) |

Dataset audit (`dataset_audit.py`, outputs in `results/audit/`): no duplicate hashes in the metadata or in either feature file; all feature values equal 1; the 2021–2023 feature sets are identical in both feature files. The full two-way metadata–feature intersection per file and year is in `intersection_two_way.csv`.

**Missing-data rule, fixed in advance.** A sample is excluded if it has no feature vector in the feature file used for its period (3 in 2021–2023, 582 in 2024) or an empty feature vector (3 in 2021–2023). The distribution of the 582 excluded 2024 samples by month and class, against the retained 2024 samples, is reported in `results/audit/holdout_missing_vs_retained.csv`. The excluded 2024 samples are spread over all months (29–84 per month, about 1–3% of each month), but almost all are goodware (580 goodware, 2 malware): excluding them changes the 2024 malware share from 10.0% to 10.2%. This is reported as a limitation of the holdout (slightly fewer goodware samples, hence slightly fewer chances for false positives).

**Identical feature vectors.** Distinct hashes with identical feature vectors are kept in the primary analysis. For every model and test month, the share of test samples whose complete feature vector also appears in that model's training data is reported, and a sensitivity analysis excludes those samples (Section 7).

**Final sample counts.** Development: 196,444 (19,644 malware, 10.0%). Holdout: 28,518 with features out of 29,100 in the metadata (2,908 malware, 10.2%).

## 2. Representation

Binary bag of Drebin features (one column per feature string). For every model, the vocabulary consists only of the features observed in that model's training rows; features unseen in training are ignored at test time. In the code this is implemented with one global column index for 2021–2023 restricted, per model, to the columns present in its training rows, which is equivalent to fitting the vocabulary on the training split alone.

Feature groups follow Drebin: S2 requested permissions (`app_permissions`), S3 app components (`activities`, `services`, `providers`, `receivers`), S4 filtered intents (`intents`), S5 restricted API calls (`api_calls`), S6 used permissions (`api_permissions`), S7 suspicious API calls (`interesting_calls`), S8 network addresses (`urls`). S1 (hardware) is not present in Hypercube. Any unknown prefix stops the run.

## 3. Classifier (frozen)

`sklearn.svm.LinearSVC` (LIBLINEAR), the primary Drebin baseline:

| Parameter | Value |
|---|---|
| penalty / loss | L2 / squared hinge (LIBLINEAR default L2-loss SVM) |
| dual | True (far more features than samples) |
| class_weight | `None` (no error-cost reweighting) |
| fit_intercept / intercept_scaling | True / 1.0 |
| tol | 1e-4 |
| max_iter | 10,000 |
| random_state | 42 |
| decision threshold | 0 (standard SVM threshold, not tuned) |
| C | selected once, Section 4 |

A fit that does not converge is logged and reported (`models.csv`); settings are not changed afterwards. `class_weight="balanced"` changes the error costs and is used only in a separate, pre-registered cost-sensitive sensitivity analysis (Section 7). The SGD-based scripts used before this version are not part of the primary results; if SGD with hinge loss is used later for scalability, it will be reported separately as an approximation.

## 4. Selection of C (once, within 2021)

- Training months 2021-01 .. 2021-09: both the vocabulary and the model are fitted on these months only.
- Validation months 2021-10, 2021-11, 2021-12.
- Candidates C ∈ {0.01, 0.1, 1, 10}.
- Selection metric: the unweighted mean of the three monthly F1 scores (malware class, threshold 0). Pooled F1 is stored for information only.
- Tie rule: mean F1 values equal when rounded to three decimals → the smallest C.
- The selected C is reused for every policy, window and feature-group model. After selection, every operational model is refitted on all data allowed by its policy (Jan–Dec 2021 for cumulative and 12-month policies, Jul–Dec 2021 for the 6-month policy), with its vocabulary rebuilt from that training data. 2022–2023 are never used for tuning.
- The cost-sensitive sensitivity variant selects its own C with exactly the same rule, since class weights can change the best C.

Script: `select_c.py` → `results/v2/c_selection.csv`, `results/v2/c_selection.json` (`--variant balanced` → `results/v2/sensitivity_balanced/`).

## 5. Retraining policies (development, 2022-01 .. 2023-12)

Initial training period: all of 2021. Each policy retrains at the first month of every block of *k* test months counted from 2022-01, using only months strictly before that point and within its window. Monthly evaluation over the 24 months 2022-01 .. 2023-12.

| Policy | Retrain every | Training history | Initial model | Family |
|---|---|---|---|---|
| Static | never | — | 2021-01 .. 2021-12 | A (reference) |
| Cum-1M | 1 month | cumulative from 2021-01 | 2021-01 .. 2021-12 | A |
| Cum-3M | 3 months | cumulative from 2021-01 | 2021-01 .. 2021-12 | A, B |
| Cum-6M | 6 months | cumulative from 2021-01 | 2021-01 .. 2021-12 | A |
| SW12-3M | 3 months | last 12 months | 2021-01 .. 2021-12 | B |
| SW6-3M | 3 months | last 6 months | 2021-07 .. 2021-12 | B |

Family A compares retraining frequency with full history; family B compares history length at a fixed quarterly frequency. The 2-month schedule from version 1 is dropped because it belongs to neither family. Retraining cost is reported as the number of retraining operations in 2022–2023 (Static 0, Cum-1M 23, Cum-3M 7, Cum-6M 3, SW12-3M 7, SW6-3M 7).

Script: `run_retraining_experiments.py` → `results/v2/retraining/`.

## 6. Metrics and uncertainty

For every policy and test month: number of samples and of malware; TP, FP, FN, TN; Precision, Recall, F1 and FPR at threshold 0; **Average Precision** computed from the decision-function scores (labelled as Average Precision, not PR-AUC); 95% CI for F1 by percentile bootstrap over the samples of the month (B = 1,000, seed 42); train–test vector overlap; training size and convergence.

Aggregates per policy: mean and normalised area under the monthly curve (AUT) for F1 and Average Precision; mean Precision, Recall and FPR; retraining operations.

## 7. Hypotheses and analyses

**Primary hypothesis (H1).** Quarterly cumulative retraining is practically equivalent to monthly cumulative retraining. For each of the 24 test months, d = F1(Cum-1M) − F1(Cum-3M). H1 holds if the 95% t-based CI of the mean of d lies entirely inside (−0.03, +0.03). The margin was fixed in version 1 and is not changed. Also reported: 95% percentile bootstrap CI over months (check), TOST p-value (equivalent to the 90% CI criterion) and the two-sided test against zero, so that a statistically detectable but practically negligible difference is stated as such. Script: `equivalence_test.py`.

**Secondary analyses (descriptive).** Family A: how F1 and Average Precision change with retraining frequency, against the Static reference. Family B: Cum-3M vs SW12-3M vs SW6-3M. Paired monthly differences are reported with 95% CIs; no equivalence claim is made outside H1.

**Pre-registered sensitivity analyses.** Each reports the same metrics and the same H1 test, labelled as sensitivity results:
- *Overlap excluded:* for every policy and month, test samples whose complete feature vector is identical to a sample in that model's training data are removed (`excl_overlap_*` columns; `equivalence_test.py --set excl_overlap`).
- *Cost-sensitive SVM:* `class_weight="balanced"`, with its own C selected by the rule of Section 4 (`--variant balanced`).
- *VTT=4 labels:* not run now; may be added later only as a clearly labelled sensitivity analysis.

The feature-group analysis is part of the primary protocol only (`class_weight=None`).

**Feature-group analysis.** Static models trained on 2021 with the selected C, evaluated monthly on 2022–2023: the full model, one model per group (single-group), and one model without each group (leave-one-group-out). Feature turnover per group and month = share of the group's feature occurrences whose feature is absent from the 2021 training data. Script: `run_group_analysis.py` → `results/v2/groups/`.

Interpretation rule, fixed in advance: the correlation between a group's turnover and the full model's F1 is reported as an *association* only. A statement that a group *contributes* to degradation requires the controlled ablation: removing the group must make the decline of F1 over 2022–2023 clearly flatter (change in the yearly F1 slope versus the full model, reported with the monthly series). Without that, groups such as S8 are described as "strongly associated with and plausibly contributing to" degradation.

## 8. 2024 holdout

- Run exactly once, after the development results have been reviewed with Prof. Lanzi, with no change to any setting in this document.
- Features from `extended_2024/hypercube_drebin.json`; months 2024-01 .. 2024-12; samples without features (582) excluded.
- All six policies continue their schedules from Section 5 without restarting: 2024-01 is test month 25, so every retraining policy retrains at 2024-01 on data up to 2023-12 (Static keeps its 2021 model). Within 2024, a policy may train on a 2024 month only after that month has been evaluated, exactly as in development (e.g. Cum-3M retrains at 2024-04 on 2021-01 .. 2024-03). Every vocabulary is built from training rows only.
- The same metrics as Section 6 and the same H1 test (12 paired months) are reported. No model selection, threshold change or tuning based on 2024 results.

## 9. Order of execution

1. Freeze: commit this document and tag `protocol-v2`.
2. Primary: `python select_c.py`, `python run_retraining_experiments.py`, `python equivalence_test.py`, `python run_group_analysis.py`.
3. Sensitivity: `python equivalence_test.py --set excl_overlap`; `python select_c.py --variant balanced`, `python run_retraining_experiments.py --variant balanced`, `python equivalence_test.py --variant balanced`.
4. Review with Prof. Lanzi; then the single 2024 run (script to be added and committed before that run).

Each result folder contains a `run_config.json` with the git commit, the input hashes and library versions.

## 10. Not part of the primary analysis

Threshold tuning; SGD approximation; the 2-month schedule; cost-sensitive weighting (sensitivity only); deduplication of identical feature vectors (sensitivity only); VTT=4 labels (possible later sensitivity analysis).

---

## Changelog: version 1 → version 2

| # | Change | Reason | Commits |
|---|---|---|---|
| 1 | Timestamp `dex_date` → `gp_date` | `dex_date` disagreed with `gp_date` at month level for 93.8% of samples (often 1980-01-01) and kept only 14,231 of 196,447 samples in 2021–2023. Prof. Lanzi had specified `gp_date` on 14 Sep. | `57d1894` |
| 2 | Invalid results moved to `results/invalid_dexdate/`, invalid scripts to `deprecated/` | Kept for traceability; all v1 development numbers (including the S3/S8 analysis and the choice of quarterly retraining) are invalid. | `57d1894`, `fd0ef71`, `683789e` |
| 3 | "2024 out-of-time result" withdrawn | `run_2024_eval.py` only re-aggregated 2021–2023 results; no 2024 evaluation had been run. The holdout is clean: the development feature file contains no 2024 samples. | `683789e` |
| 4 | Equivalence criterion | v1 accepted "ΔF1 ≤ 0.03 **or** overlapping 95% CIs", which is not an equivalence test. v2 requires the 95% CI of the paired monthly difference to lie inside ±0.03 (margin unchanged). | this document |
| 5 | Label rule stated explicitly: VTT=2 | v1 did not state it; the code used `vt_detection > 0`, identical in this dataset because Hypercube has no samples with exactly one detection. Decided by Prof. Lanzi on 14 Oct; VTT=4 only as a later sensitivity analysis. | this document |
| 6 | Classifier: SGD (logistic loss, then hinge) → LinearSVC (LIBLINEAR) | Drebin uses a linear SVM; Prof. Lanzi asked for a proper implementation as the primary baseline. SGD stays outside the primary results. | `b15e545` |
| 7 | `class_weight=None` in the primary analysis; `"balanced"` moved to a separate cost-sensitive sensitivity analysis with its own C (same selection rule) | Reweighting changes the error costs and is not part of the Drebin replication (Prof. Lanzi, 15 Oct). The earlier SGD scripts had used `"balanced"` (commit `10eb453`). | this document |
| 8 | C selected once within 2021: vocabulary and model on Jan–Sep, selection by the unweighted mean of the Oct, Nov and Dec monthly F1 at threshold 0, ties → smaller C | Avoids tuning on 2022–2023 (agreed 14 Oct); mean of monthly F1 instead of pooled F1 (Prof. Lanzi, 15 Oct). | this document |
| 9 | Initial training period: 2021-01 → all of 2021; evaluation 2022–2023 | A single month (~5.6k samples) was too little history; decided by Prof. Lanzi. | this document |
| 10 | Policies reorganised into family A (frequency) and family B (history length); 6- and 12-month sliding windows added; 2-month schedule dropped | Separates retraining frequency from training-history length; decided by Prof. Lanzi. Windows also apply to the initial model (SW6: Jul–Dec 2021). | this document |
| 11 | Vocabulary from training rows only | v1 built one vocabulary from all of 2021–2023; required for a clean 2024 evaluation. | `b15e545` |
| 12 | PR-AUC (trapezoidal) → Average Precision from decision scores; FPR added; monthly train–test vector overlap added | Accurate labelling of the metric (Prof. Lanzi); overlap reported instead of deduplication, following the dataset audit. | `b15e545` |
| 13 | Feature groups: S5/S6 names corrected; S1 reported as absent; unknown prefixes stop the run | v1 swapped S5 (restricted API calls) and S6 (used permissions) and claimed all features mapped to S1–S8 although S1 is empty. | `57d1894` |
| 14 | Group analysis with leave-one-group-out ablation and predeclared interpretation rule | v1 stated that S8 *causes* degradation without an intervention; v2 separates association from the ablation evidence. | this document |
| 15 | Dataset audit added before any run | Requested by Prof. Lanzi; confirms that the extended feature file contains the 2024 samples. | `ca36ce8` |
| 16 | Audit extended with the full two-way metadata–feature intersection and the distribution of the 582 excluded 2024 samples by month and class; missing-data rule stated explicitly | Requested by Prof. Lanzi, 15 Oct. | this document |
| 17 | Sensitivity analysis excluding test samples whose complete feature vector appears in the training data | Pre-registered at Prof. Lanzi's request (15 Oct); identical vectors stay in the primary analysis. | this document |
