"""Repair missing Llama verdict lines with one-statement prompts.

The grouped replay remains immutable. Repaired trajectories are written to a
separate file and explicitly marked as a prompt-coverage repair.
"""
import argparse
import copy
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from common import code_without_docstring, facts_summary
from judge import SYSTEM, judge_prompt, parse_statuses, udiff

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifact"
RESULTS = ART / "results"
OUT = ART / "experiments/full_cross_family"
MODEL = "llama3.1:8b"
OPTIONS = {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 100}
ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11435/api/generate")


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def call(prompt):
    body = {"model": MODEL, "prompt": prompt, "system": SYSTEM, "stream": False, "options": OPTIONS}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        payload = json.loads(response.read())
    return payload, time.perf_counter() - started


def missing_tasks(trajectories, funcs, claims):
    tasks = []
    for fid, record in sorted(trajectories.items()):
        f = funcs[fid]
        for block_index, block in [("anchor", record["anchor"])] + [(f"k{c['k']}", c) for c in record["commits"]]:
            judged = block["judged"]
            for pos, verdict in enumerate(block["verdicts"]):
                if verdict.get("parsed"):
                    continue
                arm, idx = judged[pos]
                current = f["anchor"] if block_index == "anchor" else f["touches"][int(block_index[1:]) - 1]["new"]
                if block_index == "anchor":
                    previous, diff = None, None
                else:
                    k = int(block_index[1:])
                    previous = f["anchor"] if k == 1 else f["touches"][k - 2]["new"]
                    diff = udiff(code_without_docstring(previous["text"]), code_without_docstring(current["text"]))
                claim = claims[fid][arm][idx]
                item = (arm, idx, claim)
                prompt = judge_prompt(f, [item], current["text"], facts_summary(current), diff,
                                      "this exact version of the function" if diff is None else "an earlier version of the function",
                                      "Function source" if diff is None else "Current version of the function")
                tasks.append({"task_id": f"{fid}:{block_index}:{arm}:{idx}", "fid": fid, "block": block_index,
                              "position": pos, "arm": arm, "idx": idx, "prompt": prompt})
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    base = json.loads((OUT / "trajectories.json").read_text(encoding="utf-8"))
    funcs = {f["id"]: f for f in read("functions.json") if f["cohort"] == "main"}
    claims = read("claims.json")
    tasks = missing_tasks(base, funcs, claims)
    (OUT / "repair_plan.json").write_text(json.dumps({"tasks": [{k: v for k, v in t.items() if k != "prompt"} for t in tasks],
                                                        "planned_calls": len(tasks), "model": MODEL, "options": OPTIONS}, indent=2) + "\n", encoding="utf-8")
    if not args.run:
        print(json.dumps({"missing_verdicts": len(tasks), "plan": str(OUT / 'repair_plan.json')}, indent=2)); return
    if args.max_calls <= 0:
        raise SystemExit("Fresh inference requires --max-calls")
    repair_path = OUT / "repairs.json"
    repairs = json.loads(repair_path.read_text(encoding="utf-8")) if repair_path.exists() else {}
    for task in tasks:
        if task["task_id"] in repairs:
            continue
        if len(repairs) >= args.max_calls:
            break
        payload, elapsed = call(task["prompt"])
        verdict = parse_statuses(payload.get("response", ""), 1)["S1"]
        repairs[task["task_id"]] = {"status": verdict, "response": payload.get("response", ""),
                                     "elapsed_seconds": elapsed, "response_meta": {k: payload.get(k) for k in ("done_reason", "prompt_eval_count", "eval_count")}}
        atomic_json(repair_path, repairs)
        print(f"{len(repairs)}/{len(tasks)} {task['task_id']} -> {verdict['status']} parsed={verdict['parsed']}", flush=True)
    repaired = copy.deepcopy(base)
    task_map = {task["task_id"]: task for task in tasks}
    for task_id, repair in repairs.items():
        task = task_map[task_id]
        block = repaired[task["fid"]]["anchor"] if task["block"] == "anchor" else next(c for c in repaired[task["fid"]]["commits"] if f"k{c['k']}" == task["block"])
        old = block["verdicts"][task["position"]]
        block["verdicts"][task["position"]] = {**old, **repair["status"], "repair_prompt_single_statement": True}
    atomic_json(OUT / "repaired_trajectories.json", repaired)
    print(json.dumps({"planned": len(tasks), "completed_repairs": len(repairs), "all_repaired": len(repairs) == len(tasks), "output": str(OUT / 'repaired_trajectories.json')}, indent=2))


if __name__ == "__main__":
    main()
