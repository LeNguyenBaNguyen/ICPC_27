"""Sequential local-GPU campaign over frozen supplemental experiment plans.

Only --run executes inference. Existing per-condition observations are resumed.
Errors in one condition are logged and do not conceal other runnable conditions.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifact/src"
EXP = ROOT / "artifact/experiments"
ENDPOINT = "http://127.0.0.1:11435"


def save(value):
    path = EXP / "campaign_run.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def model_ready(name):
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(ENDPOINT + "/api/tags", timeout=3) as response:
                models = json.load(response)
            if any(m["name"] == name for m in models.get("models", [])):
                return
        except OSError:
            pass
        time.sleep(10)
    raise RuntimeError(f"Model readiness timeout: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    order = ["cross_family", "random_redraw_inference", "resurrection", "docstring_free"]
    stages = []
    for name in order:
        plan = json.loads((EXP / "plans" / (name + ".json")).read_text(encoding="utf-8"))
        budget = (plan["generation_calls"] + plan["decomposition_calls"] + plan["max_judge_calls"] + plan["functions"]
                  if name == "docstring_free" else plan["max_fresh_calls"] if name == "random_redraw_inference" else len(plan["tasks"]))
        command = [sys.executable, "-X", "utf8", str(SRC / ("run_redraw_inference.py" if name == "random_redraw_inference" else "run_missing_experiments.py")),
                   "--run", *([] if name == "random_redraw_inference" else [name]), "--endpoint", ENDPOINT, "--max-calls", str(budget)]
        if name == "docstring_free":
            command.append("--fresh-control")
        stages.append({"experiment": name, "model": plan["model"], "budget": budget,
                       "plan_sha256": plan["plan_sha256"], "command": command})
    if not args.run:
        print(json.dumps(stages, indent=2))
        return
    campaign = {"started_unix": time.time(), "complete": False, "stages": [],
                "human_annotation_completed": False, "primary_results_modified": False}
    save(campaign)
    logs = EXP / "campaign_logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for stage in stages:
        current = {**stage, "started_unix": time.time(), "complete": False}
        campaign["stages"].append(current)
        save(campaign)
        try:
            print("Waiting for model", stage["model"], "for", stage["experiment"], flush=True)
            model_ready(stage["model"])
            print("Executing", stage["experiment"], flush=True)
            with (logs / (stage["experiment"] + ".log")).open("a", encoding="utf-8") as log:
                run = subprocess.run(stage["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            current["exit_code"] = run.returncode
            summary = json.loads((EXP / stage["experiment"] / "summary.json").read_text(encoding="utf-8"))
            current["complete"] = bool(summary.get("complete", summary.get("inference_completed", False))) and not summary.get("failed", False)
            current["quality_complete"] = summary.get("quality_complete", False)
            current["summary"] = summary
        except Exception as error:
            current["error"] = repr(error)
        current["ended_unix"] = time.time()
        save(campaign)
        subprocess.run([sys.executable, str(SRC / "experiment_status.py"), "--write"], cwd=ROOT, env=env, check=False)
        print("Condition finished", stage["experiment"], "execution_complete=", current["complete"], flush=True)
    campaign["complete"] = all(s["complete"] for s in campaign["stages"]) and len(campaign["stages"]) == len(stages)
    campaign["ended_unix"] = time.time()
    save(campaign)
    print("Campaign execution complete:", campaign["complete"], flush=True)
    raise SystemExit(0 if campaign["complete"] else 1)


if __name__ == "__main__":
    main()
