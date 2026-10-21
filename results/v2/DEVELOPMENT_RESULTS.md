# Development results under PROTOCOL v2 (2021–2023)

**Protocol:** `PROTOCOL.md`, tag `protocol-v2` (commit `a55900e`); post-tag tooling addition recorded in `protocol/ERRATA_v2.md` (E1).
**Data:** 196,444 samples (2021-01 .. 2023-12), VTT=2, 10.0% malware; initial training on 2021, monthly evaluation over 24 months (2022-01 .. 2023-12).
**Classifier:** LinearSVC (LIBLINEAR), L2, squared hinge, threshold 0; primary `class_weight=None`.
**Status:** development analyses complete; the 2024 holdout has not been touched. All numbers below are copied from the result files listed in the evidence table (Section 8).

---

## 1. Summary

1. **H1 is not supported.** Quarterly cumulative retraining is not equivalent to monthly cumulative retraining within ±0.03 F1: monthly retraining is better by 0.046 F1 on average (95% CI 0.027–0.064). The same holds when test samples whose vector appears in the training data are excluded (0.039, 95% CI 0.022–0.055) and for the cost-sensitive SVM (0.032, 95% CI 0.018–0.047).
2. **Retraining frequency matters strongly.** Without retraining, F1 falls from 0.80 in the first quarter of 2022 to 0.16 in the last quarter of 2023. Every retraining policy beats the static model (+0.14 to +0.28 F1), and every step down in frequency (monthly → quarterly → 6-monthly) costs F1 significantly. Average Precision shows the same ordering.
3. **Training-history length matters little.** At quarterly frequency, 12- and 6-month sliding windows are at most 0.016 F1 better than cumulative training, and this advantage is not robust (it shrinks or disappears in both sensitivity analyses).
4. **No single feature group explains the degradation.** S8 (network addresses) and S5 (restricted API calls) carry the strongest but fastest-ageing signal on their own, but removing any one group, S8 included, leaves the decline almost unchanged (S8: about 7% flatter). Under the predeclared rule, the data do not support a claim that S8 contributes to degradation more than other groups.

Both version-1 conclusions (quarterly ≈ monthly; S8 as a direct cause of degradation) are therefore not supported by the corrected development data.

## 2. Selection of C (within 2021)

Selection metric: unweighted mean of the monthly F1 for Oct, Nov, Dec 2021; model and vocabulary fitted on Jan–Sep 2021 (50,886 samples), validation on 17,066 samples.

| C | Primary: mean monthly F1 | Converged | Balanced: mean monthly F1 | Converged |
|---|---|---|---|---|
| 0.01 | 0.6700 | yes | **0.7561** (selected) | yes |
| 0.1 | **0.7022** (selected) | yes | 0.7210 | yes |
| 1 | 0.6981 | no | 0.6460 | no |
| 10 | 0.6455 | no | 0.5967 | no |

No tie occurred at three decimals. Non-converged fits are recorded as required; the selected values converged. All 40 retraining models and all 15 feature-group models converged.

## 3. Retraining policies (primary)

Means over the 24 test months. AUT = normalised area under the monthly curve. Overlap = share of test samples whose complete feature vector also appears in the model's training data.

| Policy | Retrain ops | F1 | AUT F1 | Average Precision | Precision | Recall | FPR | Overlap |
|---|---|---|---|---|---|---|---|---|
| Static | 0 | 0.370 | 0.364 | 0.451 | 0.736 | 0.267 | 0.008 | 0.018 |
| Cum-1M | 23 | 0.647 | 0.646 | 0.686 | 0.830 | 0.540 | 0.011 | 0.068 |
| Cum-3M | 7 | 0.601 | 0.601 | 0.640 | 0.806 | 0.490 | 0.011 | 0.051 |
| Cum-6M | 3 | 0.513 | 0.510 | 0.564 | 0.758 | 0.399 | 0.013 | 0.040 |
| SW12-3M | 7 | 0.617 | 0.618 | 0.646 | 0.816 | 0.507 | 0.011 | 0.050 |
| SW6-3M | 7 | 0.611 | 0.612 | 0.648 | 0.793 | 0.507 | 0.013 | 0.047 |

Precision stays high and FPR around 1% for all policies; the differences come mainly from recall, which retraining roughly doubles relative to the static model. Monthly overlap ranges from 0.1% to 9.3%.

## 4. Primary hypothesis H1 and paired comparisons

### 4.1 H1: Cum-3M vs Cum-1M (d = F1(Cum-1M) − F1(Cum-3M), 24 months)

| Analysis | Mean d | 95% CI (t) | 95% CI (bootstrap, months) | p vs 0 | TOST p | Equivalent? |
|---|---|---|---|---|---|---|
| Primary | 0.0455 | [0.0273, 0.0638] | [0.0291, 0.0630] | 3.1e-05 | 0.95 | **No** |
| Overlap excluded | 0.0387 | [0.0223, 0.0551] | [0.0241, 0.0545] | 6.3e-05 | 0.86 | No |
| Balanced (cost-sensitive) | 0.0321 | [0.0175, 0.0467] | [0.0189, 0.0463] | 1.4e-04 | 0.62 | No |

In every analysis the difference is clearly different from zero and the CI crosses the +0.03 margin. Equivalence is not shown; non-equivalence (a difference beyond the margin) is not established either, since the lower bounds stay below 0.03.

### 4.2 Secondary paired comparisons (descriptive, no multiplicity correction)

Mean monthly difference, policy − comparator, with 95% t-based CI.

| Family | Comparison | F1, primary | F1, overlap excluded | F1, balanced | Average Precision, primary |
|---|---|---|---|---|---|
| A | Cum-3M − Cum-1M | −0.046 [−0.064, −0.027] | −0.039 [−0.055, −0.022] | −0.032 [−0.047, −0.018] | −0.046 [−0.066, −0.026] |
| A | Cum-6M − Cum-1M | −0.134 [−0.180, −0.087] | −0.125 [−0.172, −0.078] | −0.110 [−0.158, −0.062] | −0.122 [−0.166, −0.078] |
| A | Cum-6M − Cum-3M | −0.088 [−0.137, −0.039] | −0.086 [−0.136, −0.037] | −0.078 [−0.124, −0.033] | −0.076 [−0.118, −0.033] |
| A | Cum-1M − Static | +0.277 [+0.228, +0.326] | +0.263 [+0.215, +0.310] | +0.229 [+0.186, +0.272] | +0.234 [+0.188, +0.281] |
| A | Cum-3M − Static | +0.231 [+0.179, +0.284] | +0.224 [+0.173, +0.275] | +0.197 [+0.150, +0.243] | +0.188 [+0.139, +0.238] |
| A | Cum-6M − Static | +0.143 [+0.092, +0.194] | +0.138 [+0.087, +0.188] | +0.119 [+0.069, +0.168] | +0.113 [+0.062, +0.164] |
| B | SW12-3M − Cum-3M | +0.016 [+0.006, +0.027] | +0.013 [+0.003, +0.022] | +0.006 [−0.000, +0.013] | +0.007 [+0.001, +0.012] |
| B | SW6-3M − Cum-3M | +0.010 [+0.000, +0.020] | +0.006 [−0.003, +0.015] | −0.000 [−0.008, +0.008] | +0.008 [+0.000, +0.015] |
| B | SW6-3M − SW12-3M | −0.006 [−0.011, −0.001] | −0.007 [−0.012, −0.002] | −0.006 [−0.013, +0.000] | +0.001 [−0.006, +0.009] |

Family A results are consistent across all analyses. Family B differences are small; SW6-3M vs Cum-3M loses significance when overlapping samples are excluded, and both sliding-window advantages disappear with the cost-sensitive SVM. A possible explanation, not tested, is that sliding windows partly adapt to the recent class balance, which class weighting already neutralises.

Cost-sensitive variant, means over 24 months: F1 0.415 / 0.644 / 0.612 / 0.534 / 0.618 / 0.612 and FPR 0.024–0.043 (Static, Cum-1M, Cum-3M, Cum-6M, SW12-3M, SW6-3M). Class weighting raises recall (0.34–0.66) at the cost of precision (0.57–0.64) and roughly triples FPR.

## 5. Feature-group analysis (primary)

Static models trained on 2021 with C = 0.1, evaluated monthly on 2022–2023. Of the 858,701 features seen in 2021, 430,070 belong to S8 and 386,314 to S3; S5, S6 and S7 have 318, 55 and 17. S1 has no features in Hypercube.

| Model | Mean F1 | F1, Q1 2022 | F1, Q4 2023 | F1 slope / year | Slope change vs Full |
|---|---|---|---|---|---|
| Full | 0.370 | 0.801 | 0.164 | −0.308 | — |
| Only-S2 | 0.228 | 0.289 | 0.158 | −0.061 | |
| Only-S3 | 0.292 | 0.482 | 0.152 | −0.195 | |
| Only-S4 | 0.119 | 0.238 | 0.070 | −0.083 | |
| Only-S5 | 0.274 | 0.714 | 0.134 | −0.253 | |
| Only-S6 | 0.207 | 0.302 | 0.169 | −0.076 | |
| Only-S7 | 0.150 | 0.155 | 0.112 | −0.044 | |
| Only-S8 | 0.220 | 0.696 | 0.046 | −0.291 | |
| Without-S2 | 0.366 | 0.810 | 0.160 | −0.316 | −0.008 |
| Without-S3 | 0.354 | 0.810 | 0.152 | −0.322 | −0.014 |
| Without-S4 | 0.368 | 0.800 | 0.164 | −0.306 | +0.002 |
| Without-S5 | 0.359 | 0.796 | 0.166 | −0.312 | −0.004 |
| Without-S6 | 0.370 | 0.801 | 0.163 | −0.307 | +0.001 |
| Without-S7 | 0.377 | 0.796 | 0.179 | −0.294 | +0.014 |
| Without-S8 | 0.361 | 0.772 | 0.171 | −0.288 | +0.021 |

Positive slope change = flatter decline without the group.

| Group | Mean share of feature occurrences unseen in 2021 | Spearman ρ with Full-model F1 |
|---|---|---|
| S2 | 7.6% | −0.94 |
| S3 | 16.3% | −0.88 |
| S4 | 0.9% | −0.88 |
| S5 | 0% | −0.11 (p = 0.60) |
| S6 | 0% | −0.14 (p = 0.52) |
| S7 | 0% | undefined (constant) |
| S8 | 11.1% | −0.87 |

**Interpretation (predeclared rule, Section 7 of the protocol).**
- *Association:* turnover in S8 is strongly associated with the decline, but so is turnover in S2, S3 and even S4 (under 1% turnover). Because turnover and F1 both trend over time, these correlations cannot single out a group.
- *Ablation:* removing S8 flattens the yearly decline by 0.021 out of 0.308 (about 7%); the model without S8 still falls from 0.77 to 0.17. No group's removal makes the decline clearly flatter.
- *Additional observation:* S5 has no unseen features at all, yet the S5-only model falls from 0.71 to 0.13. Degradation is therefore not only a matter of new features appearing; the relationship between known features and the label also changes.
- *Wording for the thesis:* S8 carries a strong but rapidly ageing signal and its turnover is strongly associated with degradation, but the controlled ablation does not show that S8 contributes to degradation more than other groups.

## 6. Limitations

- **Temporal dependence:** the 24 monthly differences are treated as independent in the t-based and bootstrap CIs; autocorrelation between months would make the CIs too narrow.
- **C selection:** in the primary variant, C = 1 did not converge and scored 0.004 below the selected C = 0.1; a converged fit might have ranked differently. In the cost-sensitive variant the selected C = 0.01 is at the lower edge of the predeclared grid, so its optimum may lie below the grid. Neither was changed after the fact.
- **Secondary comparisons:** no multiplicity correction; two family B differences have p-values between 0.01 and 0.05 (SW6-3M vs Cum-3M, p = 0.046; SW6-3M vs SW12-3M, p = 0.022).
- **Ablation:** the slope change is a point estimate without a CI (none was predeclared) and summarises each series by a single linear trend.
- **Threshold:** fixed at 0 by protocol. Recall is low (0.27–0.54) while precision is high (0.74–0.83), so the operating point matters; this is a property of the fixed threshold, not a tuned result.
- **Identical vectors:** kept in the primary analysis; monthly overlap 0.1–9.3%. Conclusions are unchanged when they are excluded.
- **Holdout missing data:** 582 2024 samples lack features, 580 of them goodware (Section 1 of the protocol).
- **Run provenance:** the `run_config.json` of the cost-sensitive retraining run is flagged `git_dirty` only because the freshly written C-selection output was not yet committed; the code was at commit `0ea580e`.
- **Tooling after the tag:** `paired_comparisons.py` was added after the H1 result had been seen; it reports every pair within each family to avoid selection (errata E1).

## 7. Points for review before the 2024 run

1. **2024 procedure:** every policy continues its schedule from development; a 2024 month is used for retraining only after it has been evaluated. Please confirm or amend before the holdout is run.
2. **H1 on 2024:** the protocol tests the same hypothesis (Cum-3M vs Cum-1M, ±0.03, 12 paired months). I will keep it unchanged even though H1 was not supported in development.
3. **Optional post-hoc analyses**, clearly labelled as such if you think they are useful: CIs for the ablation slope changes (e.g. bootstrap over months), and the cost-sensitive C selection on an extended grid below 0.01.
4. **Next writing step:** once you have reviewed these results, I will prepare the paper-oriented outline and the evidence table you asked for on 12 October.

## 8. Evidence table

| Claim | Analysis | Result file(s) | Commit |
|---|---|---|---|
| Final development sample of 196,444 | Dataset audit | `results/audit/audit_summary.json` | `a55900e` |
| C = 0.1 (primary), C = 0.01 (balanced) | C selection within 2021 | `results/v2/c_selection.csv`, `results/v2/sensitivity_balanced/c_selection.csv` | `0bc02c9`, `a159391` |
| Policy means (Section 3) | Retraining policies, primary | `results/v2/retraining/summary.csv`, `monthly_metrics.csv` | `abc715b` |
| H1 not supported | Equivalence test, primary | `results/v2/equivalence_cum3m_vs_cum1m.json` | `abc715b` |
| H1 not supported without overlapping samples | Equivalence test, overlap excluded | `results/v2/equivalence_cum3m_vs_cum1m_excl_overlap.json` | `8583e32` |
| H1 not supported with cost-sensitive SVM | Equivalence test, balanced | `results/v2/sensitivity_balanced/equivalence_cum3m_vs_cum1m.json` | `a159391` |
| Frequency effects (family A) and history-length effects (family B) | Paired comparisons | `results/v2/paired_comparisons*.csv`, `results/v2/sensitivity_balanced/paired_comparisons.csv` | `8583e32`, `a159391` |
| Static model degrades from 0.80 to 0.16 F1 | Feature-group analysis, Full model | `results/v2/groups/summary.csv`, `monthly_metrics.csv` | `0ea580e` |
| No single group explains the degradation | Leave-one-group-out ablation | `results/v2/groups/summary.csv` | `0ea580e` |
| Turnover correlations are not group-specific | Feature turnover | `results/v2/groups/feature_turnover.csv`, `summary.csv` | `0ea580e` |