# 2024 holdout results under PROTOCOL v2

**Protocol:** `PROTOCOL.md`, tag `protocol-v2` (commit `a55900e`), Section 8. Post-tag tooling changes are recorded in `protocol/ERRATA_v2.md` (E1, E2).
**Run:** single run of `run_holdout_2024.py --confirm-single-run` on 2026-10-29, code at commit `e2dbb2c` (descends from the tag), outputs committed in `1e2c042`. The holdout was marked as consumed at 2026-10-29 10:32:38 UTC, after output validation. There were no failed attempts.
**Data:** 28,518 samples from 2024-01 to 2024-12 (VTT=2, 10.2% malware), after the predeclared missing-data rule; 343,213 features occur for the first time in 2024.
**Models:** both variants were run in the same execution: the primary model (`class_weight=None`, C = 0.1), which is confirmatory, and the cost-sensitive model (`class_weight="balanced"`, C = 0.01), which is the pre-registered secondary variant. Each variant required 21 distinct models, and all 42 converged.
**Status:** the results below are reported as obtained. No model, setting or analysis was changed after the run, and the script will not be run again.

---

## 1. Summary

1. **H1 is not supported in 2024, which replicates the development result.** Monthly cumulative retraining is better than quarterly retraining by 0.044 F1 on average (95% CI 0.018–0.070), almost the same as in development (0.046). The upper bound of the CI lies outside the +0.03 margin. The difference also remains significant and non-equivalent when overlapping test samples are excluded (0.023, 95% CI 0.006–0.040) and for the cost-sensitive SVM (0.022, 95% CI 0.004–0.039).
2. **There is one exception, in the cost-sensitive variant with overlapping samples excluded.** In this case the difference is 0.007 F1 (95% CI −0.006 to 0.020, TOST p = 0.001), which lies within the equivalence margin. This is a sensitivity analysis and does not change the conclusion on H1, but it shows that the advantage of monthly retraining is smaller once class weighting is used and test vectors already present in training are removed.
3. **Degradation continues and retraining does not fully compensate for it.** The static 2021 model reaches a mean F1 of 0.083 in 2024 (recall 0.047). All retraining policies are 0.13–0.16 F1 below their development level in the primary variant (0.10–0.15 in the cost-sensitive variant). The ordering of the policies is unchanged: monthly > quarterly > 6-monthly, and every step is significant.
4. **The history-length results are consistent with development.** At quarterly frequency, sliding windows give no advantage over cumulative training in 2024 (both CIs include zero), and the 6-month window is slightly worse than the 12-month window.

## 2. Retraining policies

Means over the 12 months of 2024. Development means (24 months, 2022–2023) are given for comparison. "Excl. overlap" is the F1 on the test samples whose complete feature vector does not appear in the model's training data.

### 2.1 Primary (`class_weight=None`, C = 0.1)

| Policy | F1 2024 | F1 dev | Average Precision | Precision | Recall | FPR | Overlap | F1 excl. overlap |
|---|---|---|---|---|---|---|---|---|
| Static | 0.083 | 0.370 | 0.166 | 0.395 | 0.047 | 0.008 | 0.001 | 0.083 |
| Cum-1M | 0.511 | 0.647 | 0.534 | 0.705 | 0.404 | 0.019 | 0.027 | 0.473 |
| Cum-3M | 0.467 | 0.601 | 0.495 | 0.687 | 0.358 | 0.018 | 0.016 | 0.450 |
| Cum-6M | 0.382 | 0.513 | 0.426 | 0.638 | 0.278 | 0.017 | 0.007 | 0.377 |
| SW12-3M | 0.472 | 0.617 | 0.501 | 0.691 | 0.364 | 0.019 | 0.016 | 0.452 |
| SW6-3M | 0.456 | 0.611 | 0.485 | 0.658 | 0.355 | 0.021 | 0.015 | 0.435 |

Compared with development, precision is lower (about 0.64–0.70 compared with 0.76–0.83), FPR is about 1.5–2 times higher (0.017–0.021 compared with 0.011–0.013), and recall is lower for every policy. The static model's monthly F1 falls from 0.14 in January to 0.02 in December 2024. Train–test overlap is lower than in development (at most 2.7% compared with 6.8%).

### 2.2 Cost-sensitive (`class_weight="balanced"`, C = 0.01)

| Policy | F1 2024 | F1 dev | Average Precision | Precision | Recall | FPR | F1 excl. overlap |
|---|---|---|---|---|---|---|---|
| Static | 0.091 | 0.415 | 0.172 | 0.351 | 0.053 | 0.011 | 0.091 |
| Cum-1M | 0.494 | 0.644 | 0.491 | 0.434 | 0.597 | 0.091 | 0.464 |
| Cum-3M | 0.472 | 0.612 | 0.459 | 0.461 | 0.524 | 0.076 | 0.457 |
| Cum-6M | 0.431 | 0.534 | 0.416 | 0.478 | 0.431 | 0.061 | 0.425 |
| SW12-3M | 0.470 | 0.618 | 0.470 | 0.430 | 0.543 | 0.084 | 0.454 |
| SW6-3M | 0.468 | 0.612 | 0.464 | 0.427 | 0.535 | 0.083 | 0.453 |

Class weighting again raises recall at the cost of precision. In 2024, however, the false-positive rate of the retrained models rises to 6–9% (compared with 4% in development).

## 3. Primary hypothesis H1

d = F1(Cum-1M) − F1(Cum-3M) per month. Equivalence requires the 95% CI to lie inside ±0.03.

| Analysis | Mean d | 95% CI (t) | 95% CI (bootstrap, months) | p vs 0 | TOST p | Equivalent? | Development |
|---|---|---|---|---|---|---|---|
| Primary | 0.0443 | [0.0181, 0.0704] | [0.0226, 0.0666] | 0.003 | 0.87 | **No** | 0.0455 [0.0273, 0.0638], no |
| Primary, overlap excluded | 0.0231 | [0.0063, 0.0400] | [0.0094, 0.0381] | 0.012 | 0.19 | No | 0.0387 [0.0223, 0.0551], no |
| Balanced | 0.0217 | [0.0041, 0.0392] | [0.0073, 0.0368] | 0.020 | 0.16 | No | 0.0321 [0.0175, 0.0467], no |
| Balanced, overlap excluded | 0.0071 | [−0.0057, 0.0199] | [−0.0041, 0.0174] | 0.25 | 0.001 | Yes | not computed in development |

The confirmatory result (first row) is the result for H1: **H1 is not supported in 2024**. Non-equivalence in the strict sense (a difference beyond the margin) is not established either, since the lower bound of the CI stays below 0.03. This is the same pattern as in development.

The fourth row is the only analysis whose CI lies within the margin. Two points limit its interpretation. First, it is one of three sensitivity analyses and is not confirmatory. Second, this combination of variant and test set was not reported in development, so it cannot be compared with a development value. One possible explanation, not tested here, is that part of the advantage of monthly retraining comes from recently added training samples whose feature vectors reappear in the following month: in 2024 the overlap is 2.7% for Cum-1M and 1.6% for Cum-3M.

## 4. Secondary paired comparisons (descriptive, no multiplicity correction)

Mean monthly difference over 12 months, policy − comparator, with 95% t-based CI.

| Family | Comparison | F1, primary | F1, primary excl. overlap | F1, balanced | F1, balanced excl. overlap | Average Precision, primary |
|---|---|---|---|---|---|---|
| A | Cum-3M − Cum-1M | −0.044 [−0.070, −0.018] | −0.023 [−0.040, −0.006] | −0.022 [−0.039, −0.004] | −0.007 [−0.020, +0.006] | −0.039 [−0.061, −0.017] |
| A | Cum-6M − Cum-1M | −0.128 [−0.189, −0.068] | −0.096 [−0.146, −0.046] | −0.063 [−0.100, −0.026] | −0.039 [−0.071, −0.007] | −0.108 [−0.162, −0.055] |
| A | Cum-6M − Cum-3M | −0.084 [−0.148, −0.020] | −0.073 [−0.129, −0.017] | −0.041 [−0.069, −0.014] | −0.032 [−0.056, −0.009] | −0.069 [−0.119, −0.020] |
| A | Cum-1M − Static | +0.428 [+0.392, +0.464] | +0.390 [+0.359, +0.421] | +0.402 [+0.374, +0.430] | +0.373 [+0.346, +0.399] | +0.369 [+0.346, +0.391] |
| A | Cum-3M − Static | +0.384 [+0.342, +0.425] | +0.367 [+0.332, +0.402] | +0.381 [+0.355, +0.406] | +0.365 [+0.341, +0.390] | +0.330 [+0.302, +0.357] |
| A | Cum-6M − Static | +0.299 [+0.237, +0.362] | +0.294 [+0.236, +0.353] | +0.339 [+0.300, +0.378] | +0.333 [+0.299, +0.368] | +0.260 [+0.211, +0.309] |
| B | SW12-3M − Cum-3M | +0.005 [−0.010, +0.021] | +0.002 [−0.012, +0.015] | −0.003 [−0.012, +0.007] | −0.003 [−0.013, +0.007] | +0.006 [−0.005, +0.017] |
| B | SW6-3M − Cum-3M | −0.011 [−0.033, +0.012] | −0.015 [−0.034, +0.004] | −0.004 [−0.016, +0.009] | −0.004 [−0.017, +0.009] | −0.011 [−0.027, +0.006] |
| B | SW6-3M − SW12-3M | −0.016 [−0.032, −0.001] | −0.017 [−0.034, −0.000] | −0.001 [−0.016, +0.014] | −0.001 [−0.016, +0.014] | −0.017 [−0.035, +0.001] |

**Family A.** The ordering found in development holds in 2024: every retraining policy beats the static model by a much larger margin than in development (+0.30 to +0.43 F1, compared with +0.14 to +0.28 F1), because the static model has almost stopped detecting malware. Retraining less often costs F1 in every analysis, with the single exception of Cum-3M − Cum-1M in the cost-sensitive, overlap-excluded analysis.

**Family B.** The small advantage of sliding windows seen in development (+0.010 to +0.016 F1, not robust) is not present in 2024: no sliding window differs significantly from cumulative training, and the 6-month window is marginally worse than the 12-month window in the primary variant. Together with the development results, this supports the conclusion that training-history length matters little compared with retraining frequency.

## 5. Interpretation limits

- The holdout has 12 months, so its CIs are wider than in development. Monthly differences are treated as independent; autocorrelation would make the CIs too narrow.
- All results describe one operating point (threshold 0). Average Precision gives the threshold-independent picture; in the primary variant it agrees in direction with F1 in every comparison above.
- The general drop of all policies in 2024 is not explained by this analysis. It coincides with a high number of features not seen before and with lower train–test overlap, but these are observations, not tested causes.
- The feature-group analysis (RQ4) was not part of the holdout run and remains a development-only result.
- The selected C values are those fixed in development. The cost-sensitive C lies at the edge of the grid (0.01); as agreed, the grid was not extended before the holdout, and any extension will be run on development data only and labelled as post hoc.

## 6. Provenance

| Item | Value |
|---|---|
| Protocol tag | `protocol-v2` → `a55900ed2dffcb19ae8189cadad50ef48a2dc846` |
| Code commit of the run | `e2dbb2ccaa984da72ae73bbdb3960dba7b64e90c` (working tree clean) |
| Results commit | `1e2c042` |
| Fixture test | `results/v2/holdout_fixture_check.json` (2023 pseudo-holdout, exact reproduction of development results, both variants) |
| Extended feature file (SHA-256) | `da0a71e5362720d275cffa771bc219fe979395d01c8f2c9afa11cae78564ccf3` |
| Original feature file (SHA-256) | `2214c2241d23f10235d11e64b9bc096db9ba8555be920aa9aded8cc13a7dad34` |
| Metadata (SHA-256) | `e2310bdd9b3a9a79a09b253fe8435e3bfa8e76c630dd8a63a5f8c31080e6d9f1` |
| Environment | Python 3.13.15, scikit-learn 1.9.0, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Windows 11 |
| Consumed marker | `results/v2/holdout_2024/CONSUMED.json` (includes the SHA-256 of every output file) |
| Failed attempts | none |

## 7. Evidence table

| Claim | Result file | Commit |
|---|---|---|
| H1 not supported in 2024 (primary) | `results/v2/holdout_2024/primary/h1_cum3m_vs_cum1m.json` | `1e2c042` |
| H1 not supported with overlap excluded and with the cost-sensitive SVM | `results/v2/holdout_2024/primary/h1_cum3m_vs_cum1m.json`, `results/v2/holdout_2024/balanced/h1_cum3m_vs_cum1m.json` | `1e2c042` |
| Equivalence within the margin in the cost-sensitive, overlap-excluded analysis only | `results/v2/holdout_2024/balanced/h1_cum3m_vs_cum1m.json` | `1e2c042` |
| Static F1 0.083 in 2024; retraining policies 0.13–0.16 below development (primary) | `results/v2/holdout_2024/*/summary.csv`, `results/v2/retraining/summary.csv` | `1e2c042`, `abc715b` |
| Family A ordering and family B results | `results/v2/holdout_2024/*/paired_comparisons*.csv` | `1e2c042` |
| 21 models per variant, all converged | `results/v2/holdout_2024/*/models.csv`, `run_config.json` | `1e2c042` |
