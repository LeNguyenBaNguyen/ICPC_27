# PPV sensitivity audit validation (2026-11-12)

The supplied PPV sensitivity audit originally had 220 rows across 218 unique event keys `(fid, k, idx)` due to arm collisions (`doc:1` vs `llm:1`). Both duplicate keys have been fully investigated and reconciled.

| Event key | Rows | Status & Resolution |
| --- | --- | --- |
| `F142`, touch 1, claim 1 | `PPV_CC_067`, `PPV_CC_068` | Duplicate key: `PPV_CC_067` (`llm:1`) and `PPV_CC_068` (`doc:1`), both labeled `supported`. `PPV_CC_068` removed as duplicate; `PPV_CC_067` retained as authoritative for the LLM cohort. |
| `F031`, touch 6, claim 1 | `PPV_MISS_018`, `PPV_MISS_019` | Conflicting labels: `PPV_MISS_018` (`doc:1`, heuristic `supported`) vs `PPV_MISS_019` (`llm:1`, `contradicted`). Verified from code: at touch 6, `convert_one_fmt_off_pair` signature changed from 3 to 5 parameters and return type changed from `bool` to `int`. Verdict is definitively **`contradicted`**. `PPV_MISS_018` removed; `PPV_MISS_019` retained. |

### Reconciliation & Execution Summary

- **Audit event rows**: 218
- **Unique audited event keys**: 218 (1-to-1 bijection)
- **Primary S6 reproduced**: 0.84518 (84.52%)
- **Human-label S6 (unresolved censored)**: 0.92466 (92.47%)
- **Human-label S6 (unresolved as event)**: 0.92325 (92.32%)
- **Verdict breakdown**: `supported`: 114, `contradicted`: 102, `unresolved`: 2
- **Status**: `artifact/src/analyze_ppv_sensitivity.py` executed successfully; results saved in `artifact/experiments/ppv_sensitivity/summary.json`.
