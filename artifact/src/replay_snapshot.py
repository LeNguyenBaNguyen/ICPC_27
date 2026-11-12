"""Replay archived cache in an isolated temporary folder; leave originals intact.

Default: mine -> run_llm --second -> rerun_jit -> render (stored statistics).
--analyze additionally recomputes all statistics, including 10,000 H2 refits.
All subprocesses are cache-only. Records logs, SHA-256 and exact commands.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    original = root / "artifact"
    out = args.out or root / "revisions" / ("full-replay" if args.analyze else "cache-replay")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"python": sys.version, "platform": platform.platform(), "cache_only": True,
              "recomputed_statistics": args.analyze, "stages": [], "success": False}
    # uv-created environments may not contain pip; metadata is always available.
    import importlib.metadata
    report["packages"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (out / "environment.txt").write_text("\n".join(report["packages"]) + "\n", encoding="utf-8")
    names = ["functions.json", "random_functions.json", "claims.json", "verdicts.json", "summary.json"]
    report["original_hashes"] = {n: sha(original / "results" / n) for n in names}
    try:
        with tempfile.TemporaryDirectory(prefix="claim-survival-replay-") as tmp:
            work = Path(tmp)
            artifact = work / "artifact"
            shutil.copytree(original / "src", artifact / "src", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(original / "results", artifact / "results")
            if (original / "validation").is_dir():
                shutil.copytree(original / "validation", artifact / "validation")
            if (original / "experiments").is_dir():
                shutil.copytree(original / "experiments", artifact / "experiments")
            shutil.copytree(root / "manuscript", work / "manuscript", ignore=shutil.ignore_patterns("build"))
            stages = [("mine.py", []), ("run_llm.py", ["--second"]), ("rerun_jit.py", [])]
            if args.analyze:
                stages.append(("analyze.py", []))
            stages.extend([("prepare_validation.py", ["--sensitivity-only"]), ("h2_sensitivity.py", [])])
            stages.append(("render.py", []))
            env = dict(os.environ, LLM_CACHE_ONLY="1", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            for script, flags in stages:
                command = [sys.executable, "src/" + script, *flags]
                started = time.monotonic()
                print("Running", script, flush=True)
                with (out / (script + ".log")).open("w", encoding="utf-8") as log:
                    run = subprocess.run(command, cwd=artifact, env=env, stdout=log, stderr=subprocess.STDOUT)
                report["stages"].append({"command": command, "exit_code": run.returncode,
                                         "elapsed_seconds": round(time.monotonic() - started, 2)})
                if run.returncode:
                    raise RuntimeError(f"{script} failed; see {out / (script + '.log')}")
            report["replay_hashes"] = {n: sha(artifact / "results" / n) for n in names}
            report["byte_identical"] = {n: report["original_hashes"][n] == report["replay_hashes"][n] for n in names}
            report["json_value_identical"] = {
                n: json.loads((original / "results" / n).read_text(encoding="utf-8")) ==
                json.loads((artifact / "results" / n).read_text(encoding="utf-8")) for n in names}
            before = json.loads((original / "results" / "summary.json").read_text(encoding="utf-8"))
            after = json.loads((artifact / "results" / "summary.json").read_text(encoding="utf-8"))
            # Numeric drift is reported, never silently copied over published results.
            report["summary_changed_top_level_fields"] = [k for k in sorted(set(before) | set(after))
                                                          if json.dumps(before.get(k), sort_keys=True) != json.dumps(after.get(k), sort_keys=True)]
            report["summary_comparison"] = {k: {"archived": before.get(k), "replay": after.get(k)}
                                             for k in report["summary_changed_top_level_fields"]}
            report["missing_archived_inputs"] = ["results/docstring_audit.json"] if not (original / "results/docstring_audit.json").exists() else []
            shutil.copy2(artifact / "results" / "summary.json", out / "summary.json")
            shutil.copytree(work / "manuscript" / "generated", out / "generated", dirs_exist_ok=True)
            report["success"] = True
    except Exception as exc:
        report["error"] = str(exc)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("packages", "summary_comparison")}, indent=2), flush=True)
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
