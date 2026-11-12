# Measurement protocol and limits

This note expands the manuscript's method and audit description. Every count below is recomputed offline by `src/review_followup_evidence.py` from saved records; canonical data are in `validation/review_followup_evidence.json`. No new model calls or labels were generated.

## Decomposition

`src/claims.py` contains the exact SYSTEM/PROMPT and parser. A claim must be one self-contained fact checkable against source, name the function explicitly, avoid pronouns referring to other claims, preserve wording where possible, add no facts and merge no independent facts. One of structural/dataflow/behavioural/intent is assigned, with at most 12 items. The parser accepts recognized JSON/type aliases and keeps only valid items within the cap; successful parsing does not establish completeness or atomicity.

Synthetic illustration, not a labeled dataset case: “The function returns parsed JSON and raises ValueError on invalid input” contains two separately checkable propositions. They should be represented separately, rather than letting support for one imply support for both. Intent/purpose is especially context-sensitive.

Archived checks cover 30 different-family re-extractions and 42 retyped function claim lists. The re-extractions have equal claim counts on 17 functions; equal counts do not mean equivalent facts. Retyping agrees on 335/460 same-claim labels (72.83%). These are cross-family diagnostics, not same-model repeated-run stability or aligned extraction precision/recall. The 400-case audit flags 28 compound claims but does not supply the complete original explanation needed to validate all omissions or distortions. No full-explanation coverage estimate is claimed.

## Human audit

The two-rater plan was changed after planning to a single-rater exploratory audit at the user's request. The rater received the blinded packet and written instructions for status, evidence, atomicity, type and missing-context flags. Judge verdict, arm and private sampling keys were withheld. Function names remain visible; natural language can suggest provenance, so blinding is limited. The rater self-reports experience and no prior exposure; no separate training/calibration session is documented. Experience does not establish accuracy.

Judgment is against current code, with anchor/previous code and diff as context. The packet masks docstrings, normalizes decorated-function indentation and marks a docstring-only body with commented pass; it is not byte-identical to the original model prompt. Supported requires supplied evidence; contradicted requires explicit conflict; unresolved retains insufficient context. A disappearing identifier or a type annotation alone does not establish runtime behavior. All 400 IDs, allowed values and nonempty evidence pass schema checks, which do not independently validate the reasons.

`VALIDATION.md` describes selection and analysis: 400 recorded judgments across six anchor/later-verdict strata, inverse inclusion weights, design-based sampling uncertainty and a separate approximate function-cluster multiplier sensitivity. The later frame is alive-only, not all versions. Human-unresolved cases stay in the three-way matrix and are excluded from the binary reference denominator. There is no second rater, adjudication or estimate of rater bias; no hazard correction or validated semantic failure time is inferred.

## Tracking and endpoint

`src/mine.py` follows Git -M file renames along first-parent history. If a qualified function disappears, fallback requires one newly appearing same-name function in the same file with raw-text SequenceMatcher similarity at least 0.7. Failure to accept a candidate stops tracking rather than creating a contradiction event. There is no independent ground-truth matching audit.

Raw stored tracks contain 1 non-head stop among 129 main records, and 7 among 61 original random records (three function removals/renames, four file removals). These are recorded censor reasons before endpoint-specific overload/exposure handling, not a validated matching-error rate or counts for the new redraw. Matching errors could occur without a stop; a true removal can also correctly stop tracking.

The estimand is freedom from the first detected contradiction. Later support cannot undo a first event, without implying permanent semantic falsity. A reversible-state estimand would need comparable observations after all transitions; the alive-only archive and changed follow-up prompt grouping do not identify those rates. UNR is an uncertain verdict, not proof that a latent failure lies in a known time interval. Primary analysis keeps it at risk; separate censor-before-UNR and UNR-as-event scenarios expose assumptions rather than provide semantic bounds.

## Shared evidence, comparators and scope

The judge, identifier comparator and change flags share parser evidence. Change-type associations therefore describe the detector and cannot be interpreted as independent causal evidence. Parser-free rejudging or a frozen matched-input ablation is not completed. Published CARL-CCI, dependency-aware ReFEree, ETF-style tracing and READU-style internal/external facts are complementary candidates, not implemented baselines here. A new implementation must freeze its inputs/thresholds and account for domain/label differences; an ad hoc substitute must not be named as the published system.

No multi-family majority vote was used to rebuild the main survival cohort. The hosted and fresh cross-family subsets have differing frames/context/runtime identity and do not provide a complete aligned ensemble. The docstring-free regeneration changes generation context; it is not an isolated toggle of docstrings in the judge input and does not identify a causal docstring effect. Non-local callee/configuration/dependency evolution remains outside local-token follow-up.

## Recorded request times

| Fresh condition | Requests with timing | Sum elapsed seconds | Median per request (seconds) |
| --- | ---: | ---: | ---: |
| cross_family | 57 | 120.47 | 1.55 |
| random_redraw_inference | 308 | 1631.70 | 4.29 |
| resurrection | 649 | 1314.29 | 1.38 |
| docstring_free | 786 | 28153.01 | 2.56 |

These are recorded client request durations, including different generation/control/exposure mixes. They exclude environment setup, mining, developer effort and unrecorded historical/billing information. Sums are not a full end-to-end wall-clock or GPU utilization measure and do not establish a per-touch refresh-policy cost or benefit. No dollar cost is reconstructed from missing records.

## Primary sources for complementary methods

- CARL-CCI: https://arxiv.org/abs/2512.19883 ; implementation/model card: https://huggingface.co/CARLCCI-repo/CARL-CCI
- ReFEree: https://arxiv.org/abs/2604.10520
- READU: https://arxiv.org/abs/2607.15780

The manuscript remains an exploratory instrument study. Adding raters, validated matching, stronger comparators, parser-free rejudging or state-transition models would be new work, not results supplied by these clarifications.
