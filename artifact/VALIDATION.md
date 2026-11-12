# Single-rater human audit (400 submitted labels analyzed)

The original plan proposed two independent raters. The user subsequently selected a single-rater exploratory audit and supplied 400 completed rows in `validation/annotator_1.csv`, with self-reported provenance in `annotator_notes.md`. All rows pass ID, schema, completeness and allowed-value checks. `annotator_2.csv` remains an unused blank template. No inter-rater agreement or adjudication is available. See [HUMAN_EVALUATION_RESULTS.md](../HUMAN_EVALUATION_RESULTS.md) and `validation/human_analysis/summary.json`. Agent inspections and model-agreement checks remain separate automated analyses. Public review availability: https://claim-survival-artifact.dimpledshrimp2.chatgpt.site. The landing page and initial release were verified without login before publication of this v2 update.

## Distribution and blinding

The single rater was instructed to use only `validation/blinded/` (JSONL plus readable `cases/V0001.md` etc.), their own `annotator_1.csv`, and the labeling instructions. Do **not** share `validation/private/`, worked-example verdicts or model outputs before labeling. For any future second rater, use an independent blank sheet and conceal the first rater's labels. A1 self-reports ten years' Python research/reading experience, no prior exposure to the manuscript or model verdicts, and reliance on the supplied packet. These disclosures have not been independently verified.

Each JSONL record contains an opaque task ID, the original claim, docstring-masked anchor/previous/current code, and a diff. Task IDs do not encode model verdict, source arm, function ID or stage. Function names remain visible as necessary code context. Natural text can suggest its source; blinding is therefore limited rather than guaranteed. Source-only context may be insufficient; mark that uncertainty instead of guessing. Additional repository context, if consulted, must be documented consistently.

The annotation packet normalizes surrounding indentation in decorated-function slices and replaces a docstring-only body with a clearly commented `pass`. This display transformation differs from the original model prompt's `common.code_without_docstring`. Human judgments assess the displayed source; do not describe the packet as byte-identical to the original judge input. New model experiments must record the transformation they use and must not silently change historical prompt/cache keys.

## Labels

For each task, assess the claim against **current_code**, using anchor/previous code only to understand the change. Fill:

- `status`: `supported`, `contradicted`, or `unresolved`.
- `evidence`: code locations and a brief reason. Distinguish missing context from explicit contradictory evidence.
- `decomposition_ok`: `yes`, `no`, or `uncertain`; note compound or distorted claims.
- `claim_type`: `structural`, `dataflow`, `behavioural`, `intent`, or `uncertain`.
- `needs_context`: `yes` or `no`, with details in evidence.

Supported means the supplied evidence supports the claim; contradicted means evidence conflicts with it; unresolved means neither conclusion is justified. A missing identifier alone is not necessarily a behavioral contradiction. Do not infer runtime instance types solely from annotations. Do not treat a claim's fluency as evidence. Do not infer that absence of a contradiction establishes semantic truth outside the provided context.

The submitted single-rater sheet is preserved without relabeling. Model/human disagreements are listed separately in `validation/human_analysis/disagreements.csv`; they are not adjudicated truth labels. If a future second-rater study is conducted, freeze both original sheets before discussion, and save adjudicated labels, adjudicator identity, reasons and any extra context separately. Do not replace raw disagreement with consensus agreement.

## Frame and sampling

Seed: `20261110`; 400 draws without replacement within six stage/verdict strata. Equal allocation in sorted stratum order gives four strata 67 cases and two 66. Both explanation/docstring arms and both cohorts are included. Eligible later observations are effective code-changing touches before an overload-stub stop; format-only placebos and retests are excluded.

The frame contains **recorded judgments**, including unparsed outputs represented as unresolved. It does not contain all possible claim/version observations: later judgments stop after the first contradiction and exclude claims not supported at anchor. This audit can assess recorded-label reliability, not semantic failure at unobserved versions, resurrection, or all decomposition omissions. A separate whole-explanation completeness audit is needed for coverage.

`private/sampling_manifest.json` records stratum populations, sample counts and input hashes. `private/sampling_key.csv` records each case's stratum, model verdict, arm, function, selection probability `n_h/N_h` and inverse weight `N_h/n_h`. These are coordinator-only materials during blinding.

## Completed analysis and interpretation

`src/analyze_human_annotation.py` checks IDs, missing fields, allowed values, frozen source hashes and sampling weights. It reports raw/weighted three-way confusion matrices, agreement and kappa, plus breakdowns by stage, cohort and arm. Binary contradiction metrics exclude human-unresolved judgments from the reference denominator; model-unresolved is counted as no detected contradiction in those binary metrics. The three-way analysis retains all unresolved judgments.

The weighted estimates target the finite frame of 14,511 recorded judgments using inverse inclusion probabilities. Uncertainty includes a stratified SRS-without-replacement variance for agreement with labels treated as fixed, and a separate approximate function-cluster exponential-multiplier bootstrap calibrated to the six stratum totals (2,000 draws, seed 20261111). The latter has no finite-population correction and is not an exact survey-design interval. Normal design intervals may be poor for sparse discrepancies. Neither method measures single-rater error or substitutes for inter-rater reliability.

All 400 rows have submitted labels and evidence; 103 are unresolved and 28 are marked compound. No second-rater or adjudication result exists. The sampling manifest retains its original preparation-time status rather than being rewritten; the current completion record is the hash-bound analysis summary. Do not treat agreement with one rater as adjudicated semantic accuracy, infer unseen failure times, or apply a validated hazard correction from this audit. Historical primary estimates remain unchanged.

## Worked trajectory and supplemental analyses

`validation/worked_trajectory.md` documents a source inspection by the AI assistant. It is illustrative evidence, **not independent human annotation**. `unresolved_sensitivity.json` and `h2_sensitivity.json` are exploratory robustness checks and do not validate the endpoint.
