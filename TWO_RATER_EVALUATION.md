# Two-rater audit: adjudicated recorded-label reference

Both blinded response files validate against the frozen 400-case task set. No submitted label was changed.

| Measure | Result |
| --- | ---: |
| Cases | 400 |
| Exact three-way agreement | 398/400 (99.50%) |
| Unweighted Cohen's kappa | 0.9924 |
| Sampling-weighted agreement | 99.980% |
| Sampling-weighted Cohen's kappa | 0.9977 |
| Disagreements adjudicated | 2 |

The two disagreements are `V0144` (`contradicted` / `unresolved`) and `V0281` (`unresolved` / `supported`). The returned adjudication labels are respectively `contradicted` and `unresolved`, yielding the final recorded-label reference. The blind packet contained only original code materials and a blank decision CSV; it did not reveal model labels or either rater's labels.

Rater-to-model agreement remains a descriptive comparison, not ground truth. Rater A1 self-reports no author, family, collaborator, academic, employment, or financial relationship with the project. Rater A2 self-reports no author/family/collaborator/academic/financial conflict, but discloses being a social acquaintance of an author. Adjudicator ADJ self-reports ten years' relevant experience, no prior exposure to the manuscript, model labels, A1/A2 labels/evidence, or results, no external context, and no author/project relationship. These are self-reports, not independently verified. Therefore this audit must not be described as establishing semantic truth or recall.

The machine-readable analysis, immutable input hashes, per-rater matrices, adjudication decisions, and final reference are in `artifact/validation/two_rater_analysis/`.
