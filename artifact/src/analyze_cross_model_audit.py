"""Summarize the blinded cross-model disagreement and subtle-mutation audit."""
import csv, json, os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, "validation")
OUT = os.path.join(VAL, "cross_model_audit_summary.json")

with open(os.path.join(VAL, "cross_model_disagreements_audit.csv"), encoding="utf-8", newline="") as f:
    disagreements = list(csv.DictReader(f))
expanded_path = os.path.join(VAL, "cross_model_disagreements_audit_expanded.csv")
if os.path.exists(expanded_path):
    with open(expanded_path, encoding="utf-8", newline="") as f:
        expanded = list(csv.DictReader(f))
    existing = {row["case_id"] for row in disagreements}
    if len({row["case_id"] for row in expanded}) != len(expanded) or existing.intersection(row["case_id"] for row in expanded):
        raise ValueError("Expanded audit has duplicate or overlapping case IDs")
    if any((row["qwen_status"], row["llama_status"]) != ("contradicted", "supported") for row in expanded):
        raise ValueError("Expanded audit must contain only Qwen-CON/Llama-SUP cases")
    disagreements.extend(expanded)
with open(os.path.join(VAL, "subtle_mutations_audit.csv"), encoding="utf-8", newline="") as f:
    mutations = list(csv.DictReader(f))

by_source = defaultdict(Counter)
for row in disagreements:
    by_source[row["disagreement_source"]][row["verdict"]] += 1

qwen_supported = [r for r in disagreements if r["qwen_status"] == "supported"]
qwen_contradicted = [r for r in disagreements if r["qwen_status"] == "contradicted"]
summary = {
    "disagreement_cases": len(disagreements),
    "expanded_qwen_contra_llama_sup_cases": len(expanded) if os.path.exists(expanded_path) else 0,
    "disagreement_verdicts": dict(Counter(r["verdict"] for r in disagreements)),
    "disagreement_sources": {k: dict(v) for k, v in sorted(by_source.items())},
    "qwen_supported_cases": len(qwen_supported),
    "qwen_supported_human_supported": sum(r["verdict"] == "supported" for r in qwen_supported),
    "qwen_supported_human_supported_rate": sum(r["verdict"] == "supported" for r in qwen_supported) / len(qwen_supported),
    "qwen_contradicted_cases": len(qwen_contradicted),
    "qwen_contradicted_human_contradicted": sum(r["verdict"] == "contradicted" for r in qwen_contradicted),
    "qwen_contradicted_human_contradicted_rate": sum(r["verdict"] == "contradicted" for r in qwen_contradicted) / len(qwen_contradicted),
    "subtle_mutation_cases": len(mutations),
    "subtle_mutation_types": dict(Counter(r["mutation_type"] for r in mutations)),
    "subtle_mutation_expected_affected": dict(Counter(r["expected_affected"] for r in mutations)),
    "subtle_mutation_detected": dict(Counter(r["detected_by_rater"] for r in mutations)),
    "interpretation": "Blinded single-rater adjudication of model-disagreement cases and subtle synthetic mutations; not a gold-standard semantic validation.",
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
