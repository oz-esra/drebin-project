# Errata and additions after the `protocol-v2` tag

`PROTOCOL.md` (tag `protocol-v2`, commit `a55900e`) is not modified. This file records every change to the tooling made after the tag, with the reason. None of them changes the data, the models, the metrics, the hypotheses or the comparisons defined in the protocol.

| # | Date | Change | Reason | Effect on the analysis |
|---|---|---|---|---|
| E1 | 2026-10-21 | Added `paired_comparisons.py` for the secondary paired comparisons of Section 7 (families A and B; F1 and Average Precision; all test samples and the overlap-excluded set; both variants). Output: `results/v2/paired_comparisons*.csv`. | `equivalence_test.py`, the tool for H1, writes a fixed output file (`equivalence_cum3m_vs_cum1m.json`); running it on other pairs would overwrite the primary result. `equivalence_test.py` is left unchanged. | None. Section 7 names the families but not individual pairs. The script was written after the H1 result and the policy means had been seen, so to avoid any selection it reports **every** pair within each family (A: Cum-1M, Cum-3M, Cum-6M pairwise; each of them against the Static reference; B: Cum-3M, SW12-3M, SW6-3M pairwise), with the same statistics as for H1 (mean paired monthly difference, 95% t-based CI, 95% bootstrap CI over months, two-sided test against zero). |
