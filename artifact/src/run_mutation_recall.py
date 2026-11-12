"""Frozen synthetic mutation-recall check for the primary Qwen judge.

The suite uses behavior-changing mutations with an explicitly affected claim,
plus matched unchanged controls. It measures judge detection on these synthetic
fixtures only; it does not validate semantic survival in repository histories.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from common import code_without_docstring, facts_summary, iter_functions
from judge import JUDGE, SYSTEM, judge_prompt, parse_statuses, udiff

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifact/experiments/mutation_recall"
MODEL = "qwen2.5-coder:14b"
OPTIONS = {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 80}
ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11435/api/generate")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fixtures():
    """Four deterministic behavioral mutation families x 25 values = 100 positives."""
    out = []
    for n in range(1, 26):
        name = f"plus_{n}"
        old = f"def {name}(value: int) -> int:\n    return value + {n}\n"
        new = f"def {name}(value: int) -> int:\n    return value - {n}\n"
        out.append((f"arithmetic-{n:02d}", name, old, new, f"{name} returns value plus {n}."))
        threshold = n * 3
        name = f"band_{n}"
        old = f"def {name}(value: int) -> str:\n    if value < {threshold}:\n        return 'low'\n    return 'high'\n"
        new = f"def {name}(value: int) -> str:\n    if value >= {threshold}:\n        return 'low'\n    return 'high'\n"
        out.append((f"condition-{n:02d}", name, old, new, f"{name} returns 'low' when value is below {threshold}."))
        name = f"nonnegative_{n}"
        old = f"def {name}(value: int) -> int:\n    if value < 0:\n        raise ValueError('negative-{n}')\n    return value\n"
        new = f"def {name}(value: int) -> int:\n    if value < 0:\n        return 0\n    return value\n"
        out.append((f"exception-{n:02d}", name, old, new, f"{name} raises ValueError for negative values."))
        name = f"trim_{n}"
        old = f"def {name}(text: str) -> str:\n    return text.strip()\n"
        new = f"def {name}(text: str) -> str:\n    return text.lower()\n"
        out.append((f"call-{n:02d}", name, old, new, f"{name} returns text after calling strip."))
    return out


def plan():
    tests = []
    for fixture_id, qualname, old, new, claim in fixtures():
        for arm, current, expected in (("mutated", new, "contradicted"), ("control", old, "supported")):
            tests.append({"task_id": f"{fixture_id}:{arm}", "fixture_id": fixture_id, "arm": arm,
                          "qualname": qualname, "old": old, "current": current, "claim": claim,
                          "expected": expected})
    return {
        "experiment": "mutation_recall", "planned_before_fresh_calls": True,
        "model": MODEL, "options": OPTIONS, "endpoint": ENDPOINT,
        "population": "100 deterministic synthetic behavior-changing Python mutations and 100 matched unchanged controls.",
        "endpoint": "Judge verdict for the known affected claim; contradicted is a hit on mutations and a false positive on controls.",
        "limits": "Synthetic fixture recall does not measure recall on historical repository changes, semantic validity of every claim, or survival-curve calibration.",
        "tasks": tests,
    }


def call(prompt):
    body = {"model": MODEL, "prompt": prompt, "system": SYSTEM, "stream": False, "options": OPTIONS}
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read())
    return payload, time.perf_counter() - started


def wilson(x, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = x / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    rad = z * ((p*(1-p)/n + z*z/(4*n*n)) ** .5) / den
    return [max(0, ctr-rad), min(1, ctr+rad)]


def summarize(records, complete):
    by_arm = {arm: [r for r in records.values() if r["arm"] == arm] for arm in ("mutated", "control")}
    def count(arm, status): return sum(r["verdict"]["status"] == status for r in by_arm[arm])
    hits, positives = count("mutated", "contradicted"), len(by_arm["mutated"])
    fps, negatives = count("control", "contradicted"), len(by_arm["control"])
    return {
        "complete": complete, "completed_tasks": len(records), "planned_tasks": 200,
        "mutation_hits": hits, "mutation_total": positives,
        "synthetic_recall": hits / positives if positives else None,
        "synthetic_recall_wilson_95": wilson(hits, positives),
        "control_false_positives": fps, "control_total": negatives,
        "control_false_positive_rate": fps / negatives if negatives else None,
        "control_false_positive_wilson_95": wilson(fps, negatives),
        "unparsed": sum(not r["verdict"]["parsed"] for r in records.values()),
        "by_family": {family: {
            "mutated_total": sum(r["arm"] == "mutated" and r["fixture_id"].split("-")[0] == family for r in records.values()),
            "mutated_hits": sum(r["arm"] == "mutated" and r["fixture_id"].split("-")[0] == family and r["verdict"]["status"] == "contradicted" for r in records.values()),
        } for family in ("arithmetic", "condition", "exception", "call")},
        "interpretation": "Synthetic mutation detection, not human semantic validation or a correction of the historical survival curve.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    spec = plan()
    spec["plan_sha256"] = digest(spec)
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8"))["plan_sha256"] != spec["plan_sha256"]:
        raise RuntimeError("Frozen mutation plan differs; use a new experiment directory")
    atomic_json(plan_path, spec)
    if not args.run:
        print(json.dumps({"planned_tasks": len(spec["tasks"]), "plan": str(plan_path)}, indent=2)); return
    if args.max_calls <= 0:
        raise SystemExit("Fresh inference requires a positive --max-calls budget")
    record_path = OUT / "observations.json"
    records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    calls = 0
    for task in spec["tasks"]:
        if task["task_id"] in records:
            continue
        if calls >= args.max_calls:
            break
        f = {"qualname": task["qualname"], "path": "synthetic_mutations.py"}
        item = ("llm", 0, {"type": "behavioural", "text": task["claim"]})
        current = iter_functions(task["current"].encode("utf-8"))[0]
        prompt = judge_prompt(f, [item], task["current"], facts_summary(current),
                              udiff(code_without_docstring(task["old"]), code_without_docstring(task["current"])),
                              "an earlier version of the function", "Current version of the function")
        try:
            response, elapsed = call(prompt)
            verdict = parse_statuses(response.get("response", ""), 1)["S1"]
            records[task["task_id"]] = {**task, "verdict": verdict, "response": response.get("response", ""),
                                         "response_meta": {k: response.get(k) for k in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count")},
                                         "elapsed_seconds": elapsed, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            atomic_json(record_path, records)
            calls += 1
            print(f"{len(records)}/200 {task['task_id']} -> {verdict['status']}", flush=True)
        except Exception as exc:
            records[task["task_id"]] = {**task, "error": repr(exc)}
            atomic_json(record_path, records)
            raise
    complete = len(records) == len(spec["tasks"]) and all("verdict" in r for r in records.values())
    atomic_json(OUT / "summary.json", summarize(records, complete))
    print(json.dumps(summarize(records, complete), indent=2))


if __name__ == "__main__":
    main()
