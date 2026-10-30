# Paper-oriented outline (draft)

**Working title:** Retraining frequency, training history and feature-group drift in Drebin-style Android malware detection: a pre-registered temporal evaluation on Hypercube

**Status:** draft for review. Development results (2021–2023) and the single 2024 holdout run are final under `protocol-v2` (holdout consumed on 2026-10-29, results in `results/v2/HOLDOUT_2024_RESULTS.md`). Novelty statements below still have to be checked against the related work (TESSERACT, the Hypercube paper and work on retraining strategies).

---

## 1. Research questions

- **RQ1 (degradation).** How quickly does a Drebin linear SVM trained on one year of data degrade when evaluated month by month on later data?
- **RQ2 (retraining frequency).** How much performance does periodic retraining recover, and is quarterly retraining practically equivalent to monthly retraining (H1, margin ±0.03 F1)?
- **RQ3 (training history).** At a fixed quarterly frequency, does limiting the training history to the last 6 or 12 months help compared with cumulative training?
- **RQ4 (feature groups).** Which Drebin feature groups carry the signal that ages, and does removing any single group reduce the degradation?

## 2. Claimed contributions

1. A corrected, pre-registered time-aware evaluation of Drebin on Hypercube: frozen protocol, dataset audit, single-use holdout, and a public record of the corrections made to an earlier version.
2. A comparison that separates retraining frequency from training-history length (two policy families instead of a single ranking of schedules).
3. An equivalence-testing formulation of the retraining-cost question, with a negative result that replicates on the untouched 2024 holdout: quarterly retraining is not equivalent to monthly retraining.
4. A feature-group analysis combining single-group models, leave-one-group-out ablation and vocabulary turnover, showing that degradation is distributed across groups and is not explained by new features alone.
5. Reproducible artefacts: scripts, frozen settings, input hashes and result files linked to commits.

## 3. Experimental protocol (summary of PROTOCOL v2)

- **Data:** Hypercube (VTT=2 release), timestamps from `gp_date`; 196,444 development samples (2021–2023, 10.0% malware) and 28,518 holdout samples (2024, 10.2% malware) after the predeclared missing-data rule.
- **Representation:** binary Drebin features (S2–S8; S1 absent in Hypercube), vocabulary built from each model's training rows only.
- **Classifier:** LinearSVC (LIBLINEAR), L2, squared hinge, `class_weight=None`, threshold 0; C selected once inside 2021 (Jan–Sep train, mean monthly F1 on Oct–Dec), C = 0.1.
- **Policies:** initial training on 2021, monthly evaluation over 2022–2023. Family A: Static, Cum-1M, Cum-3M, Cum-6M. Family B: Cum-3M, SW12-3M, SW6-3M.
- **Metrics:** F1, Precision, Recall, FPR at threshold 0; Average Precision from decision scores; within-month bootstrap CIs; paired monthly differences with 95% CIs; retraining operations as cost.
- **Primary hypothesis H1:** 95% CI of the mean monthly F1 difference Cum-1M − Cum-3M inside ±0.03.
- **Sensitivity analyses (pre-registered):** test samples whose vector appears in training excluded; cost-sensitive SVM (`class_weight="balanced"`, own C).
- **Holdout:** 2024, evaluated once after review (both variants in the same run, after a fixture test on a 2023 pseudo-holdout), policies continuing their schedules; outputs validated before the holdout was marked as consumed.

## 4. Principal results

| RQ | Development (2021–2023) | Holdout (2024) |
|---|---|---|
| RQ1 | Static model: F1 0.80 (Q1 2022) → 0.16 (Q4 2023); mean F1 0.37, mean recall 0.27 | Static model: mean F1 0.08, recall 0.05; monthly F1 0.14 (Jan) → 0.02 (Dec). Retraining policies 0.13–0.16 F1 below their development level |
| RQ2 | All retraining policies beat Static (+0.14 to +0.28 F1). Monthly > quarterly > 6-monthly, every step significant. **H1 not supported:** Cum-1M − Cum-3M = 0.046 F1 (95% CI 0.027–0.064); same conclusion in both sensitivity analyses | Same ordering, every step significant; Static gap +0.30 to +0.43 F1. **H1 not supported:** 0.044 F1 (95% CI 0.018–0.070); also not equivalent with overlap excluded (0.023) and cost-sensitive (0.022). Only the cost-sensitive, overlap-excluded analysis lies within the margin (0.007, CI −0.006 to 0.020) |
| RQ3 | Sliding windows at most +0.016 F1 over cumulative training; the advantage shrinks or disappears in the sensitivity analyses | No sliding-window advantage (all CIs vs Cum-3M include zero); SW6-3M slightly worse than SW12-3M in the primary variant |
| RQ4 | S8 and S5 carry the strongest, fastest-ageing signal on their own; removing any single group, S8 included, flattens the decline by at most 7%. S5 has no unseen features yet degrades sharply, so degradation is not only vocabulary turnover | (development only) |

Wording to keep: "H1 is not supported" (not "quarterly is worse beyond the margin", since the CI's lower bound is inside the margin, in development and in 2024); S8 "strongly associated with degradation, without ablation evidence of a group-specific contribution"; the cost-sensitive, overlap-excluded equivalence in 2024 is "a sensitivity result that does not change the conclusion on H1", with no development counterpart.

## 5. Threats to validity

- **Internal.** Monthly differences treated as independent (autocorrelation would narrow CIs); C = 1 and C = 10 did not converge during selection; the cost-sensitive C lies at the edge of the grid; threshold fixed at 0, so results describe one operating point; a reporting script was added after the tag (errata E1).
- **Construct.** F1 at a fixed threshold and Average Precision capture different aspects; retraining cost is measured only as the number of retraining operations, not compute time or labelling effort; the ablation summarises each series by one linear slope without a CI.
- **External.** One dataset, one feature set (Drebin) and one classifier family; Android only; 2021–2024; VTT=2 labels include samples with 2–3 detections, so label noise may differ from other thresholds.
- **Conclusion.** 24 development months and 12 holdout months; secondary comparisons without multiplicity correction; identical feature vectors across hashes (0.1–9.3% overlap per month in development, at most 2.7% on average per policy in 2024); without them H1 is still not supported, except in the cost-sensitive variant in 2024, where the difference shrinks to within the margin; 582 holdout samples without features, almost all goodware.
- **Holdout.** The general drop of all policies in 2024 coincides with 343,213 features unseen in development and lower overlap, but its cause is not tested; the feature-group analysis was not part of the holdout run.
- **History of the project.** Version 1 used the wrong timestamp and reported a 2024 result that had not been computed; both are documented, and the invalid results are kept for traceability.

## 6. Planned tables and figures

**Tables**
- T1 Dataset audit: samples per year and class, exclusions, two-way intersection.
- T2 Frozen protocol settings (classifier, C selection rule, policies, metrics).
- T3 C selection for both variants (mean monthly F1, convergence).
- T4 Policy summary: F1, AUT, Average Precision, Precision, Recall, FPR, overlap, retraining operations.
- T5 H1 across the analyses (primary, overlap excluded, cost-sensitive; plus cost-sensitive with overlap excluded in 2024), development and 2024 side by side.
- T6 Paired comparisons for families A and B (F1 and Average Precision).
- T7 Feature-group results: single-group and leave-one-group-out models, slope changes, turnover.
- T8 Holdout 2024 policy summary for both variants, with development means for comparison.

**Figures**
- F1 Monthly F1 of the Static model, with within-month CIs (RQ1).
- F2 Monthly F1 of family A policies (RQ2).
- F3 Mean F1 against number of retraining operations (cost–performance trade-off).
- F4 Forest plot of paired differences with the ±0.03 margin marked (H1 and secondary comparisons).
- F5 Monthly F1 of family B policies (RQ3).
- F6 Monthly F1 of single-group models (RQ4).
- F7 Yearly F1 slope of the Full and leave-one-group-out models (ablation).
- F8 Monthly feature turnover per group.
- F9 Monthly F1 of the main policies over 2022–2024, with the holdout year marked.

## 7. Thesis chapter mapping (proposal)

1. Introduction (motivation, RQs, contributions)
2. Background and related work (Drebin, concept drift, TESSERACT, Hypercube, retraining strategies)
3. Dataset and protocol (Sections 3 and 5 of this outline, dataset audit, protocol corrections)
4. Degradation and retraining policies (RQ1–RQ3)
5. Feature-group analysis (RQ4)
6. Holdout evaluation (2024)
7. Discussion, threats to validity, conclusion

---

## Evidence table

| # | Claim | Experiment | Result file | Commit |
|---|---|---|---|---|
| 1 | 196,444 development and 28,518 holdout samples after the missing-data rule | Dataset audit | `results/audit/audit_summary.json`, `holdout_missing_vs_retained.csv` | `a55900e` |
| 2 | No duplicate hashes; 2021–2023 features identical in both files | Dataset audit | `results/audit/audit_summary.json` | `a55900e` |
| 3 | C = 0.1 (primary), C = 0.01 (cost-sensitive) | C selection within 2021 | `results/v2/c_selection.csv`, `results/v2/sensitivity_balanced/c_selection.csv` | `0bc02c9`, `a159391` |
| 4 | Static F1 falls from 0.80 to 0.16 | Feature-group analysis, Full model | `results/v2/groups/summary.csv` | `0ea580e` |
| 5 | Every retraining policy beats Static; monthly > quarterly > 6-monthly | Retraining policies, paired comparisons | `results/v2/retraining/summary.csv`, `results/v2/paired_comparisons.csv` | `abc715b`, `8583e32` |
| 6 | H1 not supported (primary) | Equivalence test | `results/v2/equivalence_cum3m_vs_cum1m.json` | `abc715b` |
| 7 | H1 not supported without overlapping samples | Equivalence test, overlap excluded | `results/v2/equivalence_cum3m_vs_cum1m_excl_overlap.json` | `8583e32` |
| 8 | H1 not supported with the cost-sensitive SVM | Equivalence test, balanced | `results/v2/sensitivity_balanced/equivalence_cum3m_vs_cum1m.json` | `a159391` |
| 9 | Sliding-window advantage small and not robust | Paired comparisons, family B | `results/v2/paired_comparisons*.csv`, `results/v2/sensitivity_balanced/paired_comparisons.csv` | `8583e32`, `a159391` |
| 10 | No single group's removal reduces degradation clearly | Leave-one-group-out ablation | `results/v2/groups/summary.csv` | `0ea580e` |
| 11 | Turnover correlations are not group-specific; S5 degrades without unseen features | Feature turnover, single-group models | `results/v2/groups/feature_turnover.csv`, `summary.csv` | `0ea580e` |
| 12 | H1 not supported in 2024 (primary, overlap excluded, cost-sensitive) | Holdout run | `results/v2/holdout_2024/*/h1_cum3m_vs_cum1m.json` | `1e2c042` |
| 13 | Cost-sensitive, overlap-excluded difference within the margin in 2024 | Holdout run | `results/v2/holdout_2024/balanced/h1_cum3m_vs_cum1m.json` | `1e2c042` |
| 14 | Static F1 0.08 in 2024; policy ordering unchanged; no sliding-window advantage | Holdout run | `results/v2/holdout_2024/*/summary.csv`, `paired_comparisons*.csv` | `1e2c042` |
| 15 | Single run, validated before consumption, no failed attempts; fixture test passed | Holdout run, fixture test | `results/v2/holdout_2024/CONSUMED.json`, `run_config.json`, `results/v2/holdout_fixture_check.json` | `1e2c042`, `e2dbb2c` |
