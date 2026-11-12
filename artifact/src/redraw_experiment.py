"""Draw the planned 80-function clean cohort and recover pinned trajectories.

This experiment uses the archived anchor population, not a fresh ecosystem census.
Primary results are never modified. Repository code is parsed, never executed.
"""
import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mine
from common import CONFIG

ART = Path(__file__).resolve().parents[1]
OUT = ART / "experiments/random_redraw"
REPOS = ART / "data/repos"
URLS = {
    "pytest": "pytest-dev/pytest", "pydantic": "pydantic/pydantic", "black": "psf/black",
    "rich": "Textualize/rich", "httpx": "encode/httpx", "click": "pallets/click",
    "flask": "pallets/flask", "attrs": "python-attrs/attrs", "pip": "pypa/pip",
    "hypothesis": "HypothesisWorks/hypothesis", "sphinx": "sphinx-doc/sphinx",
    "pylint": "pylint-dev/pylint", "aiohttp": "aio-libs/aiohttp", "textual": "Textualize/textual",
}


def write(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def identity(f):
    return f["repo"], f["path"], f["qualname"]


def git_command(repo, arguments, *, allow_failure=False, timeout=900):
    command = ["git", "-C", str(REPOS / repo), *arguments]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, timeout=timeout,
                            env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    with (OUT / "fetch_logs" / f"{repo}.log").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": ["git", "-C", f"artifact/data/repos/{repo}", *arguments],
                              "exit_code": result.returncode,
                              "elapsed_seconds": round(time.monotonic() - started, 3)}) + "\n")
        if result.stderr:
            log.write(result.stderr.decode("utf-8", "replace") + "\n")
    if result.returncode and not allow_failure:
        raise RuntimeError(f"{repo}: git {' '.join(arguments)}: {result.stderr.decode('utf-8', 'replace')[-1500:]}")
    return result


def provision(repo):
    destination = (REPOS / repo).resolve()
    if destination.parent != REPOS.resolve():
        raise ValueError("Repository path outside intended directory")
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").is_dir():
        git_command(repo, ["init", "--quiet"])
        git_command(repo, ["remote", "add", "origin", f"https://github.com/{URLS[repo]}.git"])
    url = git_command(repo, ["remote", "get-url", "origin"]).stdout.decode().strip()
    if url != f"https://github.com/{URLS[repo]}.git":
        raise ValueError(f"Unexpected existing remote for {repo}: {url}")
    pin = CONFIG["pins"][repo]
    absent = [sha for sha in (pin["head"], pin["anchor"])
              if git_command(repo, ["cat-file", "-e", f"{sha}^{{commit}}"], allow_failure=True).returncode]
    if absent:
        print(f"Fetching pinned history: {repo}", flush=True)
        git_command(repo, ["-c", "http.version=HTTP/1.1", "fetch", "--quiet", "--filter=blob:none", "origin", *absent])
    shallow = git_command(repo, ["rev-parse", "--is-shallow-repository"]).stdout.decode().strip()
    if shallow != "false":
        raise ValueError(f"{repo}: shallow history cannot establish follow-up")
    git_command(repo, ["merge-base", "--is-ancestor", pin["anchor"], pin["head"]])
    first_parent = git_command(repo, ["rev-list", "--count", "--first-parent", f"{pin['anchor']}..{pin['head']}"]).stdout.decode().strip()
    return {"url": url, "anchor": pin["anchor"], "head": pin["head"], "shallow": False,
            "first_parent_commits": int(first_parent)}


def normalize(f, position, source, legacy_ids):
    f = copy.deepcopy(f)
    f["id"] = f"NEW-R{position:03d}"
    f["cohort"] = "random_redraw"
    f["redraw_provenance"] = source
    f["legacy_study_ids"] = legacy_ids.get(identity(f), [])
    prev = f["anchor"]
    for touch in f["touches"]:
        new = touch["new"]
        ot, nt = mine.code_tokens(prev["text"]), mine.code_tokens(new["text"])
        touch["change_types"] = mine.change_types(prev, new, ot, nt)
        touch["format_only"] = mine.code_tokens_norm(prev["text"]) == mine.code_tokens_norm(new["text"])
        touch["paren_only"] = not touch["format_only"] and mine.code_tokens_noparen(prev["text"]) == mine.code_tokens_noparen(new["text"])
        touch["overload_stub"] = bool(mine.OVERLOAD.match(new["text"]))
        prev = new
    return f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fetch_logs").mkdir(exist_ok=True)
    candidates_path = ART / "results/candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    clean = sorted((f for f in candidates["population"] if not mine.VENDOR_PATH.search(f["path"])), key=identity)
    seed = CONFIG["seed"] + 100
    draw = random.Random(seed).sample(clean, CONFIG["random_n"])
    if len({identity(f) for f in draw}) != CONFIG["random_n"]:
        raise ValueError("Duplicate sampled function identities")
    reusable = {}
    for name in ("funcs", "random_round2"):
        for f in candidates[name]:
            reusable[identity(f)] = (f, "archived_candidates." + name)
    legacy_ids = {}
    for name in ("functions.json", "random_functions.json"):
        for f in json.loads((ART / "results" / name).read_text(encoding="utf-8")):
            legacy_ids.setdefault(identity(f), []).append(f["id"])
    manifest = {
        "population_frame": "Archived eligible anchor functions, vendor-path excluded; not a new repository census",
        "seed": seed, "sample_n": len(draw), "population_n": len(clean),
        "inclusion_probability": len(draw) / len(clean), "python": sys.version,
        "source_candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "draw": [{"id": f"NEW-R{i:03d}", **f,
                  "trajectory_source": reusable[identity(f)][1] if identity(f) in reusable else "pinned_repository_history"}
                 for i, f in enumerate(draw)],
    }
    write("sample_manifest.json", manifest)
    output = {}
    for i, item in enumerate(draw):
        if identity(item) in reusable:
            old, source = reusable[identity(item)]
            output[i] = normalize(old, i, source, legacy_ids)
    write("functions.json", [output[i] for i in sorted(output)])
    todo = {}
    for i, item in enumerate(draw):
        if i not in output:
            todo.setdefault(item["repo"], []).append((i, item))
    report = {"selection_completed": True, "trajectory_recovery_completed": False,
              "inference_completed": False, "sample_n": len(draw), "archived_tracks_reused": len(output),
              "new_tracks_required": len(draw) - len(output), "repos_required": sorted(todo),
              "repos": {}, "errors": {}, "source_sha256": manifest["source_candidates_sha256"],
              "primary_results_modified": False}
    write("report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("repos", "errors")}), flush=True)
    if args.selection_only:
        return
    # Do not let mine.git silently convert object/network failures into censored tracks.
    def strict_git(repo, *arguments, binary=False):
        result = git_command(repo, list(arguments))
        return result.stdout if binary else result.stdout.decode("utf-8", "replace")
    mine.git = strict_git
    mine.repo_log = functools.lru_cache(maxsize=14)(mine.repo_log)
    def recover_repo(repo, items):
        evidence = provision(repo)
        recovered = []
        for i, item in items:
            print(f"Tracking {repo}: NEW-R{i:03d} {item['qualname']}", flush=True)
            track = mine.follow_random(item)
            recovered.append((i, normalize(track, i, "pinned_repository_history", legacy_ids)))
        return evidence, recovered
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(recover_repo, repo, items): repo for repo, items in todo.items()}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                evidence, recovered = future.result()
                output.update(recovered)
                report["repos"][repo] = evidence
                print(f"Recovered {repo}: {len(recovered)} tracks ({len(output)}/{len(draw)} total)", flush=True)
            except Exception as exc:
                report["errors"][repo] = str(exc)
                print(f"Failed {repo}: {exc}", flush=True)
            report["recovered_n"] = len(output)
            write("functions.json", [output[i] for i in sorted(output)])
            write("report.json", report)
    report["trajectory_recovery_completed"] = len(output) == len(draw)
    report["functions_sha256"] = hashlib.sha256((OUT / "functions.json").read_bytes()).hexdigest()
    report["effective_touch_n"] = sum(not(t["format_only"] or t["paren_only"]) for f in output.values() for t in f["touches"])
    write("report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["trajectory_recovery_completed"] else 1)


if __name__ == "__main__":
    main()
