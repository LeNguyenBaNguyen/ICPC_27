"""Offline, deterministic validation packet and unresolved-observation sensitivity.

Reads the archived records; never calls an LLM or changes primary results.
The sampling frame is RECORDED judgments, not all possible claim/version pairs.
"""
import argparse
import ast
import csv
import difflib
import hashlib
import json
import random
import re
import textwrap
from collections import defaultdict
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
RESULTS = ART / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def without_docstrings(code):
    """Keep code layout; mask actual Python docstring expression lines only."""
    source = textwrap.dedent(code)
    # Tree-sitter slices may start at a decorator's '@' while subsequent
    # lines still carry the surrounding class indentation.
    match = re.search(r"(?m)^( *)(?:async )?def ", source)
    if source.startswith("@") and match and match[1]:
        margin = len(match[1])
        source = "\n".join(line[margin:] if line.startswith(" " * margin) else line
                           for line in source.splitlines())
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raise ValueError("Cannot mask docstring: source did not parse")
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            expr = body[0]
            if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                # Typical archived functions put their docstring on separate lines.
                # Reject inline code rather than accidentally dropping executable code.
                if any(getattr(n, "lineno", 0) == expr.end_lineno for n in body[1:]):
                    raise ValueError("Inline docstring requires explicit source handling")
                for i in range(expr.lineno - 1, expr.end_lineno):
                    lines[i] = ""
                if len(body) == 1 and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    original = source.splitlines()[expr.lineno - 1]
                    indent = re.match(r"\s*", original)[0]
                    lines[expr.lineno - 1] = indent + "pass  # docstring removed; no other body statements"
    return "\n".join(lines)


def effective(f, commits):
    valid = []
    for commit in commits:
        k = commit["k"]
        touch = f["touches"][k - 1]
        if touch.get("overload_stub"):
            break
        if not (touch.get("format_only") or touch.get("paren_only")):
            valid.append(k)
    return valid


def km(rows, horizon=6):
    survival = 1.0
    curve = [1.0]
    for k in range(1, horizon + 1):
        risk = sum(t >= k for t, event in rows)
        deaths = sum(t == k and event for t, event in rows)
        survival *= 1 - deaths / risk if risk else 1
        curve.append(survival)
    return curve


def sensitivity(funcs, verdicts):
    """Detection endpoint vs two deliberately different unresolved scenarios.

    censor: stop immediately BEFORE the first unresolved effective touch.
    event: treat first unresolved effective touch as an event.
    These are scenario analyses, not identified semantic-survival bounds.
    """
    primary, censor, event = [], [], []
    affected = 0
    for fid, v in sorted(verdicts.items()):
        f = funcs[fid]
        if f["cohort"] != "main":
            continue
        valid = effective(f, v["commits"])
        eff = {k: e for e, k in enumerate(valid, 1)}
        for a in v["anchor"]:
            if a["arm"] != "llm" or a["status"] != "supported":
                continue
            key = [a["arm"], a["idx"]]
            trajectory = [(c["k"], x["status"]) for c in v["commits"]
                          for pair, x in zip(c["judged"], c["verdicts"]) if pair == key]
            dead = next((k for k, status in trajectory if status == "contradicted"), None)
            baseline = ((eff[dead], 1) if dead in eff else
                        (sum(k < dead for k in valid), 0) if dead is not None else (len(valid), 0))
            primary.append(baseline)
            unknown = next((eff[k] for k, status in trajectory
                            if k in eff and status == "unresolved" and eff[k] <= baseline[0]), None)
            affected += unknown is not None
            censor.append((unknown - 1, 0) if unknown is not None else baseline)
            event.append((unknown, 1) if unknown is not None else baseline)
    archived = load("summary.json")["main_S_K"]
    reproduced = km(primary)[6]
    if abs(archived - reproduced) > 1e-10:
        raise ValueError(f"Primary endpoint mismatch: {reproduced} != {archived}")
    return {
        "estimand": "time to first recorded judge-detected contradiction, conditional on anchor support",
        "horizon": 6, "n_claims": len(primary), "claims_with_unresolved_before_endpoint": affected,
        "primary_S6": reproduced, "censor_before_first_unresolved_S6": km(censor)[6],
        "first_unresolved_as_event_S6": km(event)[6],
        "interpretation": "Exploratory observation scenarios; not semantic-survival bounds. Censoring may be informative.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ART / "validation")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--sensitivity-only", action="store_true",
                        help="Refresh observation scenarios without changing any annotation packet")
    args = parser.parse_args()
    if args.size < 6:
        parser.error("--size must be at least 6")
    funcs = {f["id"]: f for name in ("functions.json", "random_functions.json") for f in load(name)}
    verdicts, claims = load("verdicts.json"), load("claims.json")
    if args.sensitivity_only:
        args.out.mkdir(parents=True, exist_ok=True)
        report = sensitivity(funcs, verdicts)
        write_json(args.out / "unresolved_sensitivity.json", report)
        print(json.dumps(report, indent=2))
        return
    strata = defaultdict(list)
    for fid, v in sorted(verdicts.items()):
        for a in v["anchor"]:
            strata[("anchor", a["status"])].append((fid, 0, a))
        valid = set(effective(funcs[fid], v["commits"]))
        for c in v["commits"]:
            if c["k"] in valid:
                for a in c["verdicts"]:
                    strata[("later", a["status"])].append((fid, c["k"], a))
    keys = sorted(strata)
    if args.size > sum(map(len, strata.values())):
        parser.error("sample exceeds available recorded judgments")
    allocation = {key: 0 for key in keys}
    while sum(allocation.values()) < args.size:
        for key in keys:
            if allocation[key] < len(strata[key]) and sum(allocation.values()) < args.size:
                allocation[key] += 1
    rng = random.Random(args.seed)
    selected = []
    for key in keys:
        frame = sorted(strata[key], key=lambda x: (x[0], x[1], x[2]["arm"], x[2]["idx"]))
        selected.extend((key, entry) for entry in rng.sample(frame, allocation[key]))
    rng.shuffle(selected)
    out = args.out.resolve()
    if (out / "annotator_1.csv").exists() or (out / "annotator_2.csv").exists():
        raise SystemExit("Output already contains annotation sheets; use a new --out to preserve labels.")
    (out / "blinded").mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(exist_ok=True)
    packets, private, sheets = [], [], []
    for number, (stratum, (fid, k, a)) in enumerate(selected, 1):
        f = funcs[fid]
        current = f["anchor"] if k == 0 else f["touches"][k - 1]["new"]
        previous = f["anchor"] if k <= 1 else f["touches"][k - 2]["new"]
        anchor = without_docstrings(f["anchor"]["text"])
        code = without_docstrings(current["text"])
        before = without_docstrings(previous["text"])
        task_id = f"V{number:04}"
        packets.append({"task_id": task_id, "claim": claims[fid][a["arm"]][a["idx"]]["text"],
                        "anchor_code": anchor, "previous_code": before, "current_code": code,
                        "diff": "\n".join(difflib.unified_diff(before.splitlines(), code.splitlines(),
                                                                  fromfile="previous", tofile="current", lineterm=""))})
        private.append({"task_id": task_id, "fid": fid, "cohort": f["cohort"], "repo": f["repo"],
                        "arm": a["arm"], "idx": a["idx"], "mining_k": k,
                        "stage": stratum[0], "bulk_status": a["status"], "bulk_parsed": a.get("parsed", True),
                        "population_n": len(strata[stratum]), "sample_n": allocation[stratum],
                        "selection_probability": allocation[stratum] / len(strata[stratum]),
                        "weight": len(strata[stratum]) / allocation[stratum]})
        sheets.append({"task_id": task_id, "status": "", "evidence": "", "decomposition_ok": "",
                       "claim_type": "", "needs_context": ""})
    (out / "blinded" / "tasks.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in packets), encoding="utf-8")
    write_csv(out / "private" / "sampling_key.csv", private)
    write_csv(out / "annotator_1.csv", sheets)
    write_csv(out / "annotator_2.csv", sheets)
    report = {"seed": args.seed, "sample_size": len(packets),
              "frame": "Recorded anchor judgments and effective later-touch judgments, both arms and cohorts; placebo/retest excluded. Later judgments are conditional on alive-only follow-up.",
              "strata": [{"stage": k[0], "bulk_status": k[1], "population_n": len(strata[k]),
                           "sample_n": allocation[k]} for k in keys],
              "source_sha256": {n: hashlib.sha256((RESULTS / n).read_bytes()).hexdigest()
                                for n in ("functions.json", "random_functions.json", "claims.json", "verdicts.json")},
              "independent_annotation_completed": False}
    write_json(out / "private" / "sampling_manifest.json", report)
    write_json(out / "unresolved_sensitivity.json", sensitivity(funcs, verdicts))
    print(json.dumps({"sample_size": len(packets), "out": str(out),
                      "sensitivity": sensitivity(funcs, verdicts)}, indent=2))


if __name__ == "__main__":
    main()
