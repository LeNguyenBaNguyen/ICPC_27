# Reviewer follow-up results (2026-11-12)

The two remaining reviewer-requested experiments are complete locally.

- Mutation recall fixture: 100/100 known mutations detected; Wilson 95% CI 96.3--100%; 0/100 unchanged controls false-positive. This is a synthetic fixture check, not semantic recall.
- Full cross-family replay: 129/129 functions, 766 grouped calls, and 1,290 anchor-supported claims. The grouped pass initially omitted 274 verdict lines; the frozen one-statement repair pass recovered 274/274, leaving 0 unparsed verdicts in `artifact/experiments/full_cross_family/repaired_analysis.json`. Llama S(6)=.6126 [.5727,.6515]. Qwen--Llama agreement is 87.19%, kappa=.2415, over 1,725 labels.

The results remain model-conditional and do not establish semantic truth or human-level accuracy. Historical result files and public artifact v3 were not modified. See `EXPERIMENT_RESULTS.md`, `STATUS.md`, and `ICPC_ERA_REVIEW_ACTIONS.md` for the synchronized project record.
