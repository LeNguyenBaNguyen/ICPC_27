"""Tool for auditing independent human annotations, extracting disagreements, and computing agreement metrics.

Protocol reference: artifact/VALIDATION.md and HUMAN_ANNOTATION_TASKS.md.
Follows the frozen design: 400 stratified cases, inverse-probability weighting,
cluster-robust uncertainty.
"""

import argparse
import collections
import csv
import json
import math
import random
from pathlib import Path

ALLOWED_STATUS = frozenset({"supported", "contradicted", "unresolved"})
ALLOWED_DECOMPOSITION = frozenset({"yes", "no", "uncertain"})
ALLOWED_CLAIM_TYPE = frozenset({"structural", "dataflow", "behavioural", "intent", "uncertain"})
ALLOWED_NEEDS_CONTEXT = frozenset({"yes", "no"})


def validate_sheet(path):
    path = Path(path)
    if not path.is_file():
        return False, [f"File not found: {path}"], []

    errors = []
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = ["task_id", "status", "evidence", "decomposition_ok", "claim_type", "needs_context"]
        if reader.fieldnames != expected_fields:
            errors.append(f"Invalid header. Expected {expected_fields}, got {reader.fieldnames}")
            return False, errors, []

        for i, row in enumerate(reader, 1):
            rows.append(row)
            tid = row.get("task_id", "")
            expected_tid = f"V{i:04d}"
            if tid != expected_tid:
                errors.append(f"Row {i}: Expected task_id '{expected_tid}', got '{tid}'")

            status = row.get("status", "").strip().lower()
            if status not in ALLOWED_STATUS:
                errors.append(f"Row {i} ({tid}): Invalid status '{row.get('status')}'. Must be one of {sorted(ALLOWED_STATUS)}")

            decomp = row.get("decomposition_ok", "").strip().lower()
            if decomp not in ALLOWED_DECOMPOSITION:
                errors.append(f"Row {i} ({tid}): Invalid decomposition_ok '{row.get('decomposition_ok')}'. Must be one of {sorted(ALLOWED_DECOMPOSITION)}")

            ctype = row.get("claim_type", "").strip().lower()
            if ctype not in ALLOWED_CLAIM_TYPE:
                errors.append(f"Row {i} ({tid}): Invalid claim_type '{row.get('claim_type')}'. Must be one of {sorted(ALLOWED_CLAIM_TYPE)}")

            nctx = row.get("needs_context", "").strip().lower()
            if nctx not in ALLOWED_NEEDS_CONTEXT:
                errors.append(f"Row {i} ({tid}): Invalid needs_context '{row.get('needs_context')}'. Must be one of {sorted(ALLOWED_NEEDS_CONTEXT)}")

            if not row.get("evidence", "").strip():
                errors.append(f"Row {i} ({tid}): Missing evidence explanation")

    if len(rows) != 400:
        errors.append(f"Expected 400 rows, found {len(rows)}")

    return len(errors) == 0, errors, rows


def cohens_kappa(a, b, categories=None):
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    cats = sorted(categories if categories is not None else (set(a) | set(b)))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = {c: sum(1 for x in a if x == c) / n for c in cats}
    pb = {c: sum(1 for y in b if y == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def compute_inter_annotator_agreement(a1_rows, a2_rows):
    assert len(a1_rows) == len(a2_rows) == 400
    n = len(a1_rows)
    s1 = [r["status"].strip().lower() for r in a1_rows]
    s2 = [r["status"].strip().lower() for r in a2_rows]

    raw_agree_count = sum(1 for x, y in zip(s1, s2) if x == y)
    raw_agreement = raw_agree_count / n

    categories = ["supported", "contradicted", "unresolved"]
    confusion = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    for x, y in zip(s1, s2):
        if x in confusion and y in confusion[x]:
            confusion[x][y] += 1

    three_way_kappa = cohens_kappa(s1, s2, categories)

    b1 = ["contradicted" if s == "contradicted" else "non-contradicted" for s in s1]
    b2 = ["contradicted" if s == "contradicted" else "non-contradicted" for s in s2]
    binary_agree_count = sum(1 for x, y in zip(b1, b2) if x == y)
    binary_agreement = binary_agree_count / n
    binary_kappa = cohens_kappa(b1, b2, ["contradicted", "non-contradicted"])

    decomp_agree = sum(1 for r1, r2 in zip(a1_rows, a2_rows) if r1["decomposition_ok"].strip().lower() == r2["decomposition_ok"].strip().lower()) / n
    type_agree = sum(1 for r1, r2 in zip(a1_rows, a2_rows) if r1["claim_type"].strip().lower() == r2["claim_type"].strip().lower()) / n
    context_agree = sum(1 for r1, r2 in zip(a1_rows, a2_rows) if r1["needs_context"].strip().lower() == r2["needs_context"].strip().lower()) / n

    return {
        "sample_size": n,
        "status_raw_agreement": raw_agreement,
        "status_raw_agree_count": raw_agree_count,
        "status_three_way_kappa": three_way_kappa,
        "binary_contradiction_agreement": binary_agreement,
        "binary_contradiction_agree_count": binary_agree_count,
        "binary_contradiction_kappa": binary_kappa,
        "confusion_matrix_annotator1_vs_2": confusion,
        "decomposition_agreement": decomp_agree,
        "claim_type_agreement": type_agree,
        "needs_context_agreement": context_agree,
    }


def export_disagreements(a1_rows, a2_rows, tasks_path, out_csv_path):
    tasks_path = Path(tasks_path)
    tasks = {}
    if tasks_path.is_file():
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            tasks[item["task_id"]] = item

    disagreements = []
    for r1, r2 in zip(a1_rows, a2_rows):
        tid = r1["task_id"]
        diff_fields = []
        for field in ("status", "decomposition_ok", "claim_type", "needs_context"):
            if r1[field].strip().lower() != r2[field].strip().lower():
                diff_fields.append(field)
        if diff_fields:
            claim_text = tasks.get(tid, {}).get("claim", "")
            disagreements.append({
                "task_id": tid,
                "disagreeing_fields": ";".join(diff_fields),
                "claim": claim_text,
                "a1_status": r1["status"],
                "a2_status": r2["status"],
                "a1_evidence": r1["evidence"],
                "a2_evidence": r2["evidence"],
                "a1_decomposition_ok": r1["decomposition_ok"],
                "a2_decomposition_ok": r2["decomposition_ok"],
                "a1_claim_type": r1["claim_type"],
                "a2_claim_type": r2["claim_type"],
                "a1_needs_context": r1["needs_context"],
                "a2_needs_context": r2["needs_context"],
                "adjudicated_status": "",
                "adjudicated_decomposition_ok": "",
                "adjudicated_claim_type": "",
                "adjudicated_needs_context": "",
                "adjudicated_evidence": "",
                "adjudicator_id": "",
                "additional_context_consulted": "",
            })

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if disagreements:
        with out_csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(disagreements[0].keys()))
            writer.writeheader()
            writer.writerows(disagreements)
    return disagreements


def analyze_adjudicated_against_model(adjudicated_rows, sampling_key_path, seed=20261110, bootstraps=1000):
    sampling_key_path = Path(sampling_key_path)
    with sampling_key_path.open(encoding="utf-8", newline="") as stream:
        keys = {r["task_id"]: r for r in csv.DictReader(stream)}

    merged = []
    for r in adjudicated_rows:
        tid = r["task_id"]
        k = keys[tid]
        merged.append({
            "task_id": tid,
            "fid": k["fid"],
            "stage": k["stage"],
            "cohort": k["cohort"],
            "arm": k["arm"],
            "bulk_status": k["bulk_status"].strip().lower(),
            "human_status": r["status"].strip().lower(),
            "weight": float(k["weight"]),
        })

    categories = ["supported", "contradicted", "unresolved"]
    total_weight = sum(r["weight"] for r in merged)

    weighted_confusion = {c1: {c2: 0.0 for c2 in categories} for c1 in categories}
    for r in merged:
        m = r["bulk_status"]
        h = r["human_status"]
        if m in weighted_confusion and h in weighted_confusion[m]:
            weighted_confusion[m][h] += r["weight"]

    weighted_agree = sum(weighted_confusion[c][c] for c in categories) / total_weight

    pop_model_con = sum(r["weight"] for r in merged if r["bulk_status"] == "contradicted")
    pop_human_con = sum(r["weight"] for r in merged if r["human_status"] == "contradicted")
    pop_both_con = sum(r["weight"] for r in merged if r["bulk_status"] == "contradicted" and r["human_status"] == "contradicted")
    pop_both_noncon = sum(r["weight"] for r in merged if r["bulk_status"] != "contradicted" and r["human_status"] != "contradicted")
    weighted_binary_agree = (pop_both_con + pop_both_noncon) / total_weight

    p_model = pop_model_con / total_weight
    p_human = pop_human_con / total_weight
    pe = p_model * p_human + (1 - p_model) * (1 - p_human)
    binary_kappa = (weighted_binary_agree - pe) / (1 - pe) if pe < 1.0 else 1.0

    sensitivity = pop_both_con / pop_human_con if pop_human_con > 0 else 0.0
    pop_human_noncon = total_weight - pop_human_con
    specificity = pop_both_noncon / pop_human_noncon if pop_human_noncon > 0 else 0.0

    groups = collections.defaultdict(list)
    for r in merged:
        groups[r["fid"]].append(r)
    fids = sorted(groups.keys())

    rng = random.Random(seed)
    boot_agree = []
    boot_kappa = []
    for _ in range(bootstraps):
        sample_fids = [rng.choice(fids) for _ in range(len(fids))]
        sample_rows = [r for fid in sample_fids for r in groups[fid]]
        sw = sum(r["weight"] for r in sample_rows)
        if sw == 0:
            continue
        sagreed = sum(r["weight"] for r in sample_rows if r["bulk_status"] == r["human_status"])
        boot_agree.append(sagreed / sw)

        sm_con = sum(r["weight"] for r in sample_rows if r["bulk_status"] == "contradicted") / sw
        sh_con = sum(r["weight"] for r in sample_rows if r["human_status"] == "contradicted") / sw
        sb_agree = sum(r["weight"] for r in sample_rows if (r["bulk_status"] == "contradicted") == (r["human_status"] == "contradicted")) / sw
        spe = sm_con * sh_con + (1 - sm_con) * (1 - sh_con)
        boot_kappa.append((sb_agree - spe) / (1 - spe) if spe < 1.0 else 1.0)

    def quantile(vals, q):
        if not vals:
            return None
        v = sorted(vals)
        pos = (len(v) - 1) * q
        low = int(pos)
        up = min(low + 1, len(v) - 1)
        return v[low] + (pos - low) * (v[up] - v[low])

    return {
        "sample_size": len(merged),
        "total_estimated_population": total_weight,
        "weighted_raw_agreement": weighted_agree,
        "weighted_raw_agreement_95ci": {
            "low": quantile(boot_agree, 0.025),
            "high": quantile(boot_agree, 0.975),
        },
        "binary_contradiction_weighted_agreement": weighted_binary_agree,
        "binary_contradiction_weighted_kappa": binary_kappa,
        "binary_contradiction_weighted_kappa_95ci": {
            "low": quantile(boot_kappa, 0.025),
            "high": quantile(boot_kappa, 0.975),
        },
        "sensitivity_given_human_contradiction": sensitivity,
        "specificity_given_human_non_contradiction": specificity,
        "weighted_confusion_matrix_model_rows_human_cols": weighted_confusion,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit and adjudicate independent human validation labels.")
    parser.add_argument("--sheet1", help="Path to annotator_1.csv")
    parser.add_argument("--sheet2", help="Path to annotator_2.csv")
    parser.add_argument("--tasks", default="validation/blinded/tasks.jsonl")
    parser.add_argument("--disagreements-out", default="validation/disagreements.csv")
    parser.add_argument("--adjudicated", help="Path to finalized adjudicated.csv")
    parser.add_argument("--sampling-key", default="validation/private/sampling_key.csv")

    args = parser.parse_args()

    if args.sheet1 and args.sheet2:
        ok1, err1, rows1 = validate_sheet(args.sheet1)
        ok2, err2, rows2 = validate_sheet(args.sheet2)
        if not ok1:
            print("Errors in sheet 1:")
            for e in err1[:10]:
                print("  -", e)
            if len(err1) > 10:
                print(f"  ... and {len(err1) - 10} more errors")
        if not ok2:
            print("Errors in sheet 2:")
            for e in err2[:10]:
                print("  -", e)
            if len(err2) > 10:
                print(f"  ... and {len(err2) - 10} more errors")
        if ok1 and ok2:
            metrics = compute_inter_annotator_agreement(rows1, rows2)
            print(json.dumps(metrics, indent=2))
            disagreements = export_disagreements(rows1, rows2, args.tasks, args.disagreements_out)
            print(f"Exported {len(disagreements)} disagreements to {args.disagreements_out}")
            return

    if args.adjudicated:
        ok, err, rows = validate_sheet(args.adjudicated)
        if not ok:
            print("Errors in adjudicated sheet:")
            for e in err[:10]:
                print("  -", e)
            return
        results = analyze_adjudicated_against_model(rows, args.sampling_key)
        print(json.dumps(results, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()

