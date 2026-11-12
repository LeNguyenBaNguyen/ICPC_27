"""Stage 1: select functions at an anchor commit and follow them through later code-changing commits.

Only git plumbing is executed against the cloned repositories; no repository code is ever run.

Round-2 design (see RUNLOG cycles 9+):
* a *touch* is a mainline commit after which the function's CODE TOKENS differ (docstring, comments and layout removed
  with Python's tokenizer); commits that change only docstring/comments/layout are recorded as *placebo* versions;
* file renames are followed (git -M on the first-parent history); a function that disappears is matched to a unique
  same-name function with similar code in the same file (move), otherwise it is censored with a recorded reason;
* functions with duplicate qualified names in the anchor file (@overload stubs, double definitions) are excluded;
* follow-up up to K_MAX touches or the pinned repository head, with right-censoring (reason recorded);
* two cohorts: MAIN (>= min_touches code touches up to the pinned head, <= 20 per repo, seeded) and RANDOM (a seeded
  simple random sample of all eligible functions at the anchor, not conditioned on later history).
Outputs: results/candidates.json (cache), functions.json, random_functions.json, mining_stats.json, processing_order.json
"""
import difflib, io, keyword, os, random, re, subprocess, sys, textwrap, tokenize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, REPOS, iter_functions, jdump, jload, code_without_docstring

def git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", os.path.join(REPOS, repo), *args], capture_output=True, check=False)
    if r.returncode != 0:
        return None
    return r.stdout if binary else r.stdout.decode("utf8", "replace")

VENDOR_PATH = re.compile(r"(^|/)(_vendor|vendor|vendored|_vendored|third_party|thirdparty|externals?)(/|$)")
EXCLUDE_PATH = re.compile(r"(^|/)(tests?|testing|docs?|examples?|scripts?|benchmarks?|tools|changelog|__pycache__|_vendor|vendor|vendored|_vendored|third_party|thirdparty|externals?)(/|$)|conftest\.py$|test_[^/]*\.py$|_test\.py$|setup\.py$|conf\.py$")
OVERLOAD = re.compile(r"\s*@(typing\.|t\.)?overload\b")

def eligible_file(p):
    return p.endswith(".py") and not EXCLUDE_PATH.search(p)

def eligible_fn(rec):
    n = rec["end_line"] - rec["start_line"] + 1
    if not (CONFIG["min_lines"] <= n <= CONFIG["max_lines"]):
        return False
    if rec["name"].startswith("__"):
        return False
    if re.match(r"\s*@(typing\.)?overload\b", rec["text"]):
        return False
    body = rec["text"].split("\n", 1)[1] if "\n" in rec["text"] else ""
    return len(body.strip()) > 40

# ------------------------------------------------------------------ code tokens and change types
_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}

def code_tokens(text):
    """Code tokens of a function with docstring, comments and layout removed."""
    src = code_without_docstring(text)
    try:
        return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type not in _SKIP]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        src = re.sub(r"#[^\n]*", "", src)
        return [(0, w) for w in re.findall(r"\w+|[^\w\s]", src)]

def callee(c):
    """Callee name of a call expression: drop argument groups (also inside chains), keep the last attribute."""
    prev = None
    while prev != c:
        prev, c = c, re.sub(r"\([^()]*\)", "", c)
    c = c.split("(")[0]
    return c.split(".")[-1].strip()

def _norm_tok(t):
    typ, st = t
    if typ == tokenize.STRING:
        try:
            import ast
            return (typ, repr(ast.literal_eval(st)))
        except Exception:
            return (typ, st.replace("'", '"'))
    if typ in (getattr(tokenize, "FSTRING_START", -1), getattr(tokenize, "FSTRING_END", -2)):
        return (typ, st.replace("'", '"'))
    return t

def code_tokens_norm(text):
    """Code tokens with string-quote style normalised and trailing commas before closers removed."""
    toks = [_norm_tok(t) for t in code_tokens(text)]
    out = []
    for i, t in enumerate(toks):
        if t == (tokenize.OP, ",") and i + 1 < len(toks) and toks[i + 1][0] == tokenize.OP and toks[i + 1][1] in ")]}":
            continue
        out.append(t)
    return out

def code_tokens_noparen(text):
    """Normalised code tokens without grouping parentheses (formatter re-wraps such as parenthesised `with`)."""
    return [t for t in code_tokens_norm(text) if not (t[0] == tokenize.OP and t[1] in "()")]

def change_types(old, new, ot, nt):
    """Heuristic, multi-label change types from tree-sitter facts and code tokens of the old and new versions."""
    types = []
    of, nf = old["facts"], new["facts"]
    if old["params"] != new["params"] or old.get("return_type") != new.get("return_type"):
        types.append("signature")
    oc, nc = {callee(c) for c in of["calls"]}, {callee(c) for c in nf["calls"]}
    if oc != nc:
        types.append("call_change")
    elif set(of["calls"]) != set(nf["calls"]):
        types.append("call_receiver")   # same callee names, different receiver/qualifier text
    norm = lambda xs: [re.sub(r"\s+", "", x) for x in xs]
    if norm(of["raises"]) != norm(nf["raises"]) or of["n_try"] != nf["n_try"]:
        types.append("exception")
    if any(of[k] != nf[k] for k in ("n_if", "n_loop", "n_return", "n_yield", "n_with", "n_await")):
        types.append("control_flow")
    if norm(of["returns"]) != norm(nf["returns"]) and "control_flow" not in types:
        types.append("return_value")
    if not [t for t in types if t != "call_receiver"]:
        if len(ot) == len(nt) and all(a[0] == b[0] for a, b in zip(ot, nt)):
            diff = [(a, b) for a, b in zip(ot, nt) if a != b]
            if all(a[0] == tokenize.NAME and not keyword.iskeyword(a[1]) and not keyword.iskeyword(b[1]) and
                   a[1] not in ("True", "False", "None") and b[1] not in ("True", "False", "None") for a, b in diff):
                types.append("identifier_rename")
            else:
                types.append("literal_or_keyword")
        else:
            types.append("local_edit")
    return types

def fn_lines_changed(a, b):
    a, b = code_without_docstring(a), code_without_docstring(b)
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0))
    return sum(1 for l in d if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))

def placebo_kind(o, n):
    if (o.get("docstring") or "") != (n.get("docstring") or ""):
        return "docstring"
    strip = lambda t: re.sub(r"\s+", "", code_without_docstring(t))
    return "format" if strip(o["text"]) == strip(n["text"]) else "comment"

# ------------------------------------------------------------------ history with file-rename following
def repo_log(repo, anchor, head):
    out = git(repo, "log", "--first-parent", "--reverse", "-M", "--name-status", "--diff-merges=first-parent",
              "--format=@@%H%x09%ct%x09%cs%x09%s", f"{anchor}..{head}") or ""
    commits = []
    for line in out.split("\n"):
        if line.startswith("@@"):
            sha, ct, cs, subj = (line[2:].split("\t", 3) + [""])[:4]
            commits.append({"sha": sha, "ct": int(ct), "cs": cs, "subject": subj[:120], "changes": []})
        elif line.strip() and commits:
            parts = line.split("\t")
            commits[-1]["changes"].append(parts)
    return commits

def file_events(commits, path):
    """Yield (commit, path_after, event) for commits that touch `path`, following renames. event in {M, R, D}."""
    cur = path
    for c in commits:
        for parts in c["changes"]:
            st = parts[0]
            if st.startswith("R") and len(parts) == 3 and parts[1] == cur:
                cur = parts[2]; yield c, cur, "R"; break
            if st in ("M", "T") and parts[1] == cur:
                yield c, cur, "M"; break
            if st == "D" and parts[1] == cur:
                yield c, cur, "D"; return

def track_file(repo, commits, path, src0, want, k_max):
    """Follow functions `want` (qualnames) of `path` from the anchor. Returns {qualname: track}."""
    funcs0 = {}
    allq = {}
    for r in iter_functions(src0):
        allq[r["qualname"]] = allq.get(r["qualname"], 0) + 1
        funcs0[r["qualname"]] = r
    prev = dict(funcs0)
    tracks = {q: {"touches": [], "placebo": [], "censor": None, "qual_now": q, "n_touches_total": 0} for q in want}
    alive = set(want)
    for c, cur, ev in file_events(commits, path):
        if not alive:
            break
        if ev == "D":
            for q in alive:
                tracks[q]["censor"] = {"reason": "file_removed", "commit": c["sha"], "date": c["cs"]}
            alive = set(); break
        src = git(repo, "show", f"{c['sha']}:{cur}", binary=True)
        if src is None:
            continue
        cur_funcs = {}
        for r in iter_functions(src):
            if OVERLOAD.match(r["text"]):   # later-added @overload stubs are never the tracked function
                continue
            cur_funcs.setdefault(r["qualname"], r)
        numstat = git(repo, "diff", "--numstat", "-M", f"{c['sha']}^1", c["sha"], "--", cur) or ""
        try:
            a, d = numstat.split()[:2]; file_lines = int(a) + int(d)
        except Exception:
            file_lines = 0
        for q in sorted(alive):
            tr = tracks[q]; qn = tr["qual_now"]
            o = prev.get(qn)
            n = cur_funcs.get(qn)
            if n is None and o is not None:  # moved/renamed within the file: unique same-name, similar code
                name = qn.split(".")[-1]
                cands = [r for qq, r in cur_funcs.items() if r["name"] == name and qq not in prev]
                if len(cands) == 1 and difflib.SequenceMatcher(None, o["text"], cands[0]["text"]).ratio() >= 0.7:
                    n = cands[0]; tr["qual_now"] = n["qualname"]
            if n is None or o is None:
                tr["censor"] = {"reason": "function_removed_or_renamed", "commit": c["sha"], "date": c["cs"]}
                alive.discard(q); continue
            if o["text"] == n["text"]:
                continue
            ot, nt = code_tokens(o["text"]), code_tokens(n["text"])
            if ot == nt:
                if len(tr["touches"]) < k_max:
                    tr["placebo"].append({"after_k": len(tr["touches"]), "commit": c["sha"], "date": c["cs"],
                                          "kind": placebo_kind(o, n), "new": n})
                continue
            tr["n_touches_total"] += 1
            if len(tr["touches"]) < k_max:
                tr["touches"].append({"commit": c["sha"], "ct": c["ct"], "date": c["cs"], "subject": c["subject"],
                                      "lines_changed_fn": fn_lines_changed(o["text"], n["text"]), "lines_changed_file": file_lines,
                                      "change_types": change_types(o, n, ot, nt), "new": n, "renamed_file": ev == "R"})
        prev = cur_funcs
    for q in alive:
        tracks[q]["censor"] = {"reason": "head", "commit": None, "date": None}
    return tracks, allq

def docstring_age(repo, anchor, path, rec):
    """Age of the anchor docstring: code-changing touches of the function (and days) since the docstring's lines were
    last modified (git blame at the anchor; history before the anchor, same path only)."""
    doc = rec.get("docstring")
    if not doc:
        return None
    lines = rec["text"].split("\n")
    i0 = next((i for i, l in enumerate(lines) if '"""' in l or "'''" in l), None)
    if i0 is None:
        return None
    q = '"""' if '"""' in lines[i0] else "'''"
    i1 = i0 if lines[i0].count(q) >= 2 else next((j for j in range(i0 + 1, len(lines)) if q in lines[j]), i0)
    s, e = rec["start_line"] + i0, rec["start_line"] + i1
    bl = git(repo, "blame", "--porcelain", "-L", f"{s},{e}", anchor, "--", path) or ""
    shas = {}
    for m in re.finditer(r"^([0-9a-f]{40}) \d+ \d+", bl, re.M):
        shas[m.group(1)] = None
    times = {}
    for sha in shas:
        t = (git(repo, "show", "-s", "--format=%ct %cs", sha) or "0 ?").split()
        times[sha] = (int(t[0]), t[1])
    if not times:
        return None
    dsha = max(times, key=lambda x: times[x][0])
    a_ct = int((git(repo, "show", "-s", "--format=%ct", anchor) or "0").strip())
    log = (git(repo, "log", "--first-parent", "--reverse", "--format=%H", f"{dsha}..{anchor}", "--", path) or "").split()
    touches, prev, complete = 0, None, True
    src = git(repo, "show", f"{dsha}:{path}", binary=True)
    if src is not None:
        prev = {r["qualname"]: r for r in iter_functions(src)}.get(rec["qualname"])
    if prev is None:
        complete = False
    for sha in log[-80:]:
        src = git(repo, "show", f"{sha}:{path}", binary=True)
        if src is None:
            complete = False; break
        cur = {r["qualname"]: r for r in iter_functions(src)}.get(rec["qualname"])
        if cur is None:
            complete = False; prev = None; continue
        if prev is not None and code_tokens(prev["text"]) != code_tokens(cur["text"]):
            touches += 1
        prev = cur
    return {"doc_commit": dsha, "doc_date": times[dsha][1], "age_days": (a_ct - times[dsha][0]) // 86400,
            "age_touches": touches, "age_complete": complete and len(log) <= 80}

# ------------------------------------------------------------------ per repo
def mine_repo(repo):
    pin = CONFIG["pins"][repo]
    anchor, head = pin["anchor"], pin["head"]
    commits = repo_log(repo, anchor, head)
    files = [p for p in (git(repo, "ls-tree", "-r", "--name-only", anchor) or "").split("\n") if eligible_file(p)]
    ntouch_file = {}
    for p in files:
        ntouch_file[p] = sum(1 for _ in file_events(commits, p))
    stats = {"anchor": anchor, "head": head, "anchor_date": (git(repo, "show", "-s", "--format=%cs", anchor) or "").strip(),
             "first_parent_commits": len(commits), "eligible_files": len(files), "eligible_functions": 0,
             "duplicate_qualname_excluded": 0}
    population, cand = [], []
    for p in files:
        src0 = git(repo, "show", f"{anchor}:{p}", binary=True)
        if src0 is None:
            continue
        recs = iter_functions(src0)
        cnt = {}
        for r in recs:
            cnt[r["qualname"]] = cnt.get(r["qualname"], 0) + 1
        elig = []
        for r in recs:
            if cnt[r["qualname"]] > 1:
                if eligible_fn(r):
                    stats["duplicate_qualname_excluded"] += 1
                continue
            if eligible_fn(r):
                elig.append(r)
        stats["eligible_functions"] += len(elig)
        for r in elig:
            population.append({"repo": repo, "path": p, "qualname": r["qualname"]})
        if ntouch_file[p] >= CONFIG["min_touches"] and elig:
            tracks, _ = track_file(repo, commits, p, src0, [r["qualname"] for r in elig], CONFIG["k_max"])
            byq = {r["qualname"]: r for r in elig}
            for q, tr in tracks.items():
                if tr["n_touches_total"] >= CONFIG["candidate_min_touches"]:
                    cand.append({"repo": repo, "path": p, "qualname": q, "anchor": byq[q], **tr})
    return repo, cand, population, stats

def follow_random(item):
    repo, path, q = item["repo"], item["path"], item["qualname"]
    pin = CONFIG["pins"][repo]
    commits = repo_log(repo, pin["anchor"], pin["head"])
    src0 = git(repo, "show", f"{pin['anchor']}:{path}", binary=True)
    rec = {r["qualname"]: r for r in iter_functions(src0)}[q]
    tracks, _ = track_file(repo, commits, path, src0, [q], CONFIG["k_max"])
    return {"repo": repo, "path": path, "qualname": q, "anchor": rec, **tracks[q]}

def main():
    from multiprocessing import Pool
    cand = jload("candidates.json")
    if cand is None or "--remine" in sys.argv:
        with Pool(7) as pool:
            res = pool.map(mine_repo, CONFIG["repos"])
        cand = {"funcs": [f for _, fs, _, _ in res for f in fs], "population": [x for _, _, pop, _ in res for x in pop],
                "stats": {r: st for r, _, _, st in res}}
        # RANDOM cohort: seeded simple random sample of the pooled eligible population at the anchor
        rng = random.Random(CONFIG["seed"] + 100)
        pop = sorted(cand["population"], key=lambda x: (x["repo"], x["path"], x["qualname"]))
        sample = rng.sample(pop, CONFIG["random_n"])
        with Pool(7) as pool:
            cand["random"] = pool.map(follow_random, sample)
        # docstring ages for main-cohort candidates with >= min_touches
        jdump(cand, "candidates.json")
    # RANDOM cohort, round 3: vendored third-party code is not part of the studied population. The round-2 draw
    # (seeded simple random sample of all eligible functions) is kept minus its vendored functions. A full re-draw from
    # the cleaned population (--redraw-random) needs the clones (make data); it could not be run in round 3 because
    # re-fetching the repositories was too slow (RUNLOG cycle 11).
    cand.setdefault("random_round2", cand["random"])
    clean = sorted([x for x in cand["population"] if not VENDOR_PATH.search(x["path"])], key=lambda x: (x["repo"], x["path"], x["qualname"]))
    cand["population_clean_n"] = len(clean)
    if "--redraw-random" in sys.argv:
        prev_rand = {(f["repo"], f["path"], f["qualname"]): f for f in cand["random_round2"]}
        sample = random.Random(CONFIG["seed"] + 100).sample(clean, CONFIG["random_n"])
        todo = [x for x in sample if (x["repo"], x["path"], x["qualname"]) not in prev_rand]
        with Pool(7) as pool:
            new = pool.map(follow_random, todo) if todo else []
        newmap = {(f["repo"], f["path"], f["qualname"]): f for f in new}
        cand["random"] = [prev_rand.get((x["repo"], x["path"], x["qualname"])) or newmap[(x["repo"], x["path"], x["qualname"])] for x in sample]
    else:
        cand["random"] = [f for f in cand["random_round2"] if not VENDOR_PATH.search(f["path"])]
    jdump(cand, "candidates.json")
    rng = random.Random(CONFIG["seed"])
    all_funcs, all_stats = [], {}
    for repo in CONFIG["repos"]:
        stats = dict(cand["stats"][repo])
        funcs = sorted([f for f in cand["funcs"] if f["repo"] == repo and f["n_touches_total"] >= CONFIG["min_touches"]],
                       key=lambda f: (f["path"], f["qualname"]))
        stats["functions_with_min_touches"] = len(funcs)
        stats["candidates_ge3"] = sum(1 for f in cand["funcs"] if f["repo"] == repo)
        if len(funcs) > CONFIG["max_funcs_per_repo"]:
            funcs = sorted(rng.sample(funcs, CONFIG["max_funcs_per_repo"]), key=lambda f: (f["path"], f["qualname"]))
        # vendored paths are dropped AFTER the seeded per-project sample and the ID assignment below, so neither the sample
        # of the other functions nor their IDs (and hence cached prompts) change
        stats["vendored_dropped"] = sum(1 for f in funcs if VENDOR_PATH.search(f["path"]))
        stats["sampled"] = len(funcs) - stats["vendored_dropped"]
        hd = git(repo, "show", "-s", "--format=%cs", CONFIG["pins"][repo]["head"]) if os.path.isdir(os.path.join(REPOS, repo)) else None
        prev_stats = (jload("mining_stats.json", {}) or {}).get(repo, {})
        stats["head_date"] = hd.strip() if hd else prev_stats.get("head_date")
        all_stats[repo] = stats
        all_funcs.extend(funcs)
    # stable IDs: functions already studied in round 1 keep their ID (keeps the claim-order shuffle, JIT sampling and
    # cached LLM prompts identical); new functions get fresh IDs after the largest round-1 ID
    r1 = jload(os.path.join("round1", "functions.json"), []) or []
    old_ids = {(g["repo"], g["path"], g["qualname"]): g["id"] for g in r1}
    nxt = 1 + max([int(g["id"][1:]) for g in r1] or [-1])
    for f in all_funcs:
        key = (f["repo"], f["path"], f["qualname"])
        if key in old_ids:
            f["id"] = old_ids[key]
        else:
            f["id"] = f"F{nxt:03d}"; nxt += 1
        f["cohort"] = "main"
    all_funcs.sort(key=lambda f: f["id"])
    all_funcs = [f for f in all_funcs if not VENDOR_PATH.search(f["path"])]
    # random-cohort IDs: functions drawn in round 2 keep their ID (cached prompts); new draws get fresh IDs
    r2ids = {(f["repo"], f["path"], f["qualname"]): f"R{i:03d}" for i, f in enumerate(cand["random_round2"])}
    nxt_r = len(cand["random_round2"])
    rand = []
    for f in cand["random"]:
        f = dict(f); key = (f["repo"], f["path"], f["qualname"])
        if key in r2ids:
            f["id"] = r2ids[key]
        else:
            f["id"] = f"R{nxt_r:03d}"; nxt_r += 1
        f["cohort"] = "random"; rand.append(f)
    rand.sort(key=lambda f: f["id"])
    # relabel touches (labels/flags only; touch texts unchanged, so cached judge prompts are unaffected):
    # callee normalisation for chained calls; format-only touches (quote style / trailing commas); overload stubs
    for f in all_funcs + rand:
        prev = f["anchor"]
        for t in f["touches"]:
            n = t["new"]
            ot, nt = code_tokens(prev["text"]), code_tokens(n["text"])
            t["change_types"] = change_types(prev, n, ot, nt)
            t["format_only"] = code_tokens_norm(prev["text"]) == code_tokens_norm(n["text"])
            t["paren_only"] = (not t["format_only"]) and code_tokens_noparen(prev["text"]) == code_tokens_noparen(n["text"])
            t["overload_stub"] = bool(OVERLOAD.match(n["text"]))
            prev = n
    # docstring age (H4 delayed entry) for main-cohort functions with a docstring
    ages = jload("docstring_age.json", {})
    for f in all_funcs:
        key = f"{f['repo']}:{f['path']}:{f['qualname']}"
        if key not in ages:
            ages[key] = docstring_age(f["repo"], CONFIG["pins"][f["repo"]]["anchor"], f["path"], f["anchor"])
        f["doc_age"] = ages[key]
    jdump(ages, "docstring_age.json")
    order = [f["id"] for f in all_funcs]; random.Random(CONFIG["seed"] + 1).shuffle(order)
    jdump(all_funcs, "functions.json"); jdump(rand, "random_functions.json"); jdump(order, "processing_order.json")
    all_stats["_pooled"] = {"eligible_population": len(cand["population"]), "random_n": len(rand),
                            "eligible_population_clean": cand.get("population_clean_n"),
                            "vendored_in_population": len(cand["population"]) - (cand.get("population_clean_n") or len(cand["population"])),
                            "vendored_in_random_round2": sum(1 for f in cand["random_round2"] if VENDOR_PATH.search(f["path"])),
                            "candidates_ge3": len(cand["funcs"]),
                            "candidates_ge6": sum(1 for f in cand["funcs"] if f["n_touches_total"] >= 6),
                            "candidates_ge8": sum(1 for f in cand["funcs"] if f["n_touches_total"] >= 8)}
    jdump(all_stats, "mining_stats.json")
    print("main", len(all_funcs), "touches", sum(len(f["touches"]) for f in all_funcs),
          "random", len(rand), "random touches", sum(len(f["touches"]) for f in rand), all_stats["_pooled"])

if __name__ == "__main__":
    main()
