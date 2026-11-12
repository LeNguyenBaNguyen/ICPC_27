"""Resumable full main-cohort Llama rejudging of saved claim trajectories.

This is a separate model-conditional detection curve. It uses the archived
explanations/claims and a Llama judge, but it does not turn model agreement into
semantic truth or alter historical Qwen records.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from common import code_without_docstring, facts_summary
from judge import JUDGE, SYSTEM, judge_prompt, order_for, parse_statuses, udiff
from prepare_validation import effective

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifact"
RESULTS = ART / "results"
OUT = ART / "experiments/full_cross_family"
MODEL = "llama3.1:8b"
OPTIONS = {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 440}
ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11435/api/generate")


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def plan():
    funcs = [f for f in read("functions.json") if f["cohort"] == "main"]
    archived = read("verdicts.json")
    tasks = []
    for f in sorted(funcs, key=lambda x: x["id"]):
        ks = effective(f, archived[f["id"]]["commits"])
        tasks.append({"fid": f["id"], "repo": f["repo"], "qualname": f["qualname"], "effective_touches": ks,
                      "maximum_calls": 1 + len(ks)})
    return {
        "experiment": "full_cross_family", "planned_before_fresh_calls": True,
        "model": MODEL, "options": OPTIONS, "endpoint": ENDPOINT,
        "population": "All 129 future-active, trackable main-cohort functions and their archived LLM/docstring claims.",
        "endpoint": "First Llama-judge-detected contradiction, conditional on Llama anchor support; not semantic survival.",
        "design": "Replay anchors then effective touches in order. Llama's own supported claims remain at risk, so later prompts may differ from Qwen's historical alive-only lists. Both claim arms are retained in each prompt; LLM-arm claims define the reported main curve.",
        "limits": "Different model family/runtime/context and model-conditioned risk set. This is not a human validation, causal comparison, or an ensemble correction of the Qwen curve.",
        "functions": len(tasks), "maximum_calls": sum(t["maximum_calls"] for t in tasks), "tasks": tasks,
        "source_sha256": {name: sha(RESULTS / name) for name in ("functions.json", "claims.json", "verdicts.json")},
        "judge_prompt_sha256": digest(SYSTEM + JUDGE),
    }


def call(prompt, num_predict):
    body = {"model": MODEL, "prompt": prompt, "system": SYSTEM, "stream": False,
            "options": {**OPTIONS, "num_predict": num_predict}}
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read())
    return payload, time.perf_counter() - started


def km(records, horizon=6):
    risk = []
    for record in records.values():
        for arm, idx in record["anchor_supported"]:
            if arm != "llm":
                continue
            event = record["first_contradiction"].get(f"{arm}:{idx}")
            risk.append({"event": event if event and event <= horizon else None, "censor": min(horizon, record["last_effective_touch"])})
    survival = 1.0
    table = []
    for k in range(1, horizon + 1):
        n = sum((r["event"] is None or r["event"] >= k) and r["censor"] >= k for r in risk)
        d = sum(r["event"] == k for r in risk)
        if n:
            survival *= 1 - d/n
        table.append({"k": k, "at_risk": n, "events": d, "survival": survival})
    return len(risk), table, survival


def execute_function(f, claims, archived, calls_left):
    ordered = order_for(f["id"], claims[f["id"]])
    anchor = f["anchor"]
    prompt = judge_prompt(f, ordered, anchor["text"], facts_summary(anchor), None,
                          "this exact version of the function", "Function source")
    response, elapsed = call(prompt, 40 + 22 * len(ordered))
    anchor_verdicts = parse_statuses(response.get("response", ""), len(ordered))
    alive = [item for pos, item in enumerate(ordered, 1) if anchor_verdicts[f"S{pos}"]["status"] == "supported"]
    result = {
        "fid": f["id"], "anchor": {"judged": [[a, i] for a, i, _ in ordered],
                                       "verdicts": [{"arm": a, "idx": i, "type": c["type"], **anchor_verdicts[f"S{pos}"]}
                                                    for pos, (a, i, c) in enumerate(ordered, 1)],
                                       "response": response.get("response", ""),
                                       "response_meta": {k: response.get(k) for k in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count")},
                                       "elapsed_seconds": elapsed},
        "commits": [], "anchor_supported": [[a, i] for a, i, _ in alive], "first_contradiction": {},
        "last_effective_touch": 0,
    }
    used = 1
    previous = anchor["text"]
    effective_ks = set(effective(f, archived[f["id"]]["commits"]))
    for k, touch in enumerate(f["touches"], 1):
        current = touch["new"]
        if k not in effective_ks:
            previous = current["text"]
            continue
        if used >= calls_left:
            raise RuntimeError("call budget reached during function; resume is only safe between functions")
        result["last_effective_touch"] = k
        if not alive:
            result["commits"].append({"k": k, "judged": [], "verdicts": [], "skipped": "no_llama_alive_claims"})
            previous = current["text"]
            continue
        prompt = judge_prompt(f, alive, current["text"], facts_summary(current),
                              udiff(code_without_docstring(previous), code_without_docstring(current["text"])),
                              "an earlier version of the function", "Current version of the function")
        response, elapsed = call(prompt, 40 + 22 * len(alive))
        parsed = parse_statuses(response.get("response", ""), len(alive))
        verdicts = [{"arm": a, "idx": i, "type": c["type"], **parsed[f"S{pos}"]}
                    for pos, (a, i, c) in enumerate(alive, 1)]
        result["commits"].append({"k": k, "judged": [[a, i] for a, i, _ in alive], "verdicts": verdicts,
                                  "response": response.get("response", ""),
                                  "response_meta": {key: response.get(key) for key in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count")},
                                  "elapsed_seconds": elapsed})
        used += 1
        next_alive = []
        for item, verdict in zip(alive, verdicts):
            key = f"{item[0]}:{item[1]}"
            if verdict["status"] == "contradicted":
                result["first_contradiction"].setdefault(key, k)
            else:
                next_alive.append(item)
        alive = next_alive
        previous = current["text"]
    return result, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    spec = plan(); spec["plan_sha256"] = digest(spec)
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8"))["plan_sha256"] != spec["plan_sha256"]:
        raise RuntimeError("Frozen plan differs; use a new experiment directory")
    atomic_json(plan_path, spec)
    if not args.run:
        print(json.dumps({"functions": spec["functions"], "maximum_calls": spec["maximum_calls"], "plan": str(plan_path)}, indent=2)); return
    if args.max_calls <= 0:
        raise SystemExit("Fresh inference requires a positive --max-calls budget")
    funcs = {f["id"]: f for f in read("functions.json") if f["cohort"] == "main"}
    claims, archived = read("claims.json"), read("verdicts.json")
    output_path = OUT / "trajectories.json"
    output = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    calls = 0
    for task in spec["tasks"]:
        fid = task["fid"]
        if fid in output:
            continue
        if calls + task["maximum_calls"] > args.max_calls:
            break
        record, used = execute_function(funcs[fid], claims, archived, args.max_calls - calls)
        output[fid] = record
        atomic_json(output_path, output)
        calls += used
        print(f"functions={len(output)}/{spec['functions']} fid={fid} calls_this_run={calls}", flush=True)
    complete = len(output) == spec["functions"]
    if complete:
        n, table, s6 = km(output)
    else:
        n, table, s6 = None, [], None
    summary = {"complete": complete, "completed_functions": len(output), "planned_functions": spec["functions"],
               "calls_this_run": calls, "llm_anchor_supported_claims": n, "km_table": table, "S6": s6,
               "interpretation": "Llama-model-conditional detection curve on the main cohort; not semantic survival or a direct calibration of Qwen."}
    atomic_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
