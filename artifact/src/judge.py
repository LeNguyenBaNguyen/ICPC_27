"""Stage 4: verify claims at the anchor (initial-error screen) and at each later touching commit (survival),
plus the whole-explanation JIT inconsistency baseline (H3). The judge is grounded with the function source
(docstring removed, so prose cannot vouch for prose), tree-sitter facts, and the commit's diff.

Design choices that keep the number of shared-GPU calls low (see PLAN.md / RUNLOG.md):
* one judge call per (function, version) covering all still-alive claims of both arms, in a seeded shuffled order;
* only anchor-supported claims enter the persistence cohort; a claim leaves the risk set at its first contradiction;
* compact line-oriented verdict format;
* the JIT baseline is asked at every commit where >=1 LLM-arm claim dies (recall) and at a seeded random third of the
  remaining judged commits (false-alarm rate).
A capped second model (claude-haiku-5-5) re-judges a stratified sample of (function, version) pairs for agreement.
"""
import difflib, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, CALLS, jload, jdump, ollama, anthropic_call, facts_summary, code_without_docstring

SYSTEM = ("You are a meticulous code reviewer. You verify whether statements about a Python function hold for the version of the "
          "function shown. Judge only from the shown code, parser facts and diff.")

JUDGE = """Function `{qualname}` in {path}. The statements below were written about {about}.

{version_label} (docstring removed):
```python
{text}
```

Facts extracted from this version by a parser (tree-sitter):
{facts}
{diff_block}
Statements:
{statements}

For each statement decide:
- supported: the shown code makes the statement true (a statement about purpose/intent counts as supported unless the code contradicts it).
- contradicted: the shown code clearly makes the statement false, e.g. a named parameter, call, attribute, return value, exception or condition it describes no longer exists or now behaves differently.
- unresolved: the statement concerns code that is not shown (callers, helper internals, external systems) and the shown code neither confirms nor refutes it.
Be strict about "contradicted": require concrete evidence in the shown code or the diff; renaming a local variable or reformatting does not contradict a statement about behaviour.

Answer with exactly one line per statement and nothing else, in this format:
S1: supported
S2: contradicted -- <evidence, at most 12 words>
S3: unresolved"""

JIT = """A developer wrote this explanation for an earlier version of the Python function `{qualname}` ({path}):
\"\"\"
{explanation}
\"\"\"

The function has since been modified. Diff from the previous version to the current version:
```diff
{diff}
```

Current version (docstring removed):
```python
{text}
```

Is the explanation still consistent with the current code, i.e. does it contain no statement that the current code makes false?
Answer with one word, CONSISTENT or INCONSISTENT, optionally followed by " -- " and a reason of at most 15 words."""

def udiff(a, b):
    d = difflib.unified_diff(a.splitlines(), b.splitlines(), "previous", "current", lineterm="", n=2)
    s = "\n".join(d)
    return s[:5000] + ("\n... (diff truncated)" if len(s) > 5000 else "")

def order_for(fid, claims):
    """Deterministic per-function shuffle so the judge cannot tell LLM claims from docstring claims by position."""
    items = [("llm", i, c) for i, c in enumerate(claims["llm"])] + [("doc", i, c) for i, c in enumerate(claims["doc"])]
    random.Random(f"{CONFIG['seed']}-{fid}").shuffle(items)
    return items

_line = re.compile(r"^\W*S\s*(\d+)\s*[:.)\-]\s*\**\s*(supported|contradicted|unresolved|support|contradict|true|false|refuted)\b\W*(.*)$", re.I)

def parse_statuses(resp, n):
    out = {}
    for line in resp.splitlines():
        m = _line.match(line.strip())
        if not m:
            continue
        k, st, rest = int(m.group(1)), m.group(2).lower(), m.group(3)
        st = "supported" if st.startswith(("support", "true")) else "contradicted" if st.startswith(("contradict", "false", "refuted")) else "unresolved"
        if 1 <= k <= n and f"S{k}" not in out:
            out[f"S{k}"] = {"status": st, "evidence": rest.strip(" -")[:120], "parsed": True}
    for i in range(1, n + 1):
        out.setdefault(f"S{i}", {"status": "unresolved", "evidence": "(no verdict parsed)", "parsed": False})
    return out

def judge_prompt(f, items, text, facts, diff, about, version_label):
    statements = "\n".join(f"S{i+1}. {c['text']}" for i, (_, _, c) in enumerate(items))
    diff_block = f"\nDiff from the previous version to this version:\n```diff\n{diff}\n```\n" if diff else "\n"
    return JUDGE.format(qualname=f["qualname"], path=f["path"], about=about, version_label=version_label,
                        text=code_without_docstring(text), facts=facts, diff_block=diff_block, statements=statements)

def judge_call(f, items, text, facts, diff, about, version_label, backend="ollama"):
    p = judge_prompt(f, items, text, facts, diff, about, version_label)
    n_pred = 40 + 22 * len(items)
    resp = ollama(p, system=SYSTEM, num_predict=n_pred) if backend == "ollama" else anthropic_call(p, system=SYSTEM, max_tokens=n_pred + 200)
    st = parse_statuses(resp, len(items))
    return [{"arm": arm, "idx": i, "type": c["type"], **st[f"S{k+1}"]} for k, (arm, i, c) in enumerate(items)]

def jit_call(f, expl, prev_text, new_text):
    r = ollama(JIT.format(qualname=f["qualname"], path=f["path"], explanation=expl,
                          diff=udiff(code_without_docstring(prev_text), code_without_docstring(new_text)),
                          text=code_without_docstring(new_text)), system=SYSTEM, num_predict=40).strip()
    head = r.upper().lstrip("*# ").split()[0] if r.split() else ""
    cons = True if head.startswith("CONSISTENT") else False if head.startswith("INCONSISTENT") else None
    return cons, r[:200]

def jit_sampled(fid, k):
    return random.Random(f"jit-{CONFIG['seed']}-{fid}-{k}").random() < 1 / 3

def process_function(f, claims, expl):
    """Anchor screen + survival trajectory for one function. Returns the verdict record."""
    items = order_for(f["id"], claims)
    if not items:
        return {"anchor": [], "commits": []}
    a = f["anchor"]
    anchor = judge_call(f, items, a["text"], facts_summary(a), None, "this exact version of the function", "Function source")
    rec = {"anchor": anchor, "commits": []}
    alive = [it for it, v in zip(items, anchor) if v["status"] == "supported"]
    prev_text = a["text"]
    for k, t in enumerate(f["touches"]):
        n = t["new"]
        c = {"k": k + 1, "commit": t["commit"], "judged": [], "verdicts": []}
        if alive:
            diff = udiff(code_without_docstring(prev_text), code_without_docstring(n["text"]))
            v = judge_call(f, alive, n["text"], facts_summary(n), diff, "an earlier version of the function",
                           "Current version of the function")
            c["judged"] = [[arm, i] for arm, i, _ in alive]
            c["verdicts"] = v
            alive = [it for it, vv in zip(alive, v) if vv["status"] != "contradicted"]
        new_dead_llm = sum(1 for vv in c["verdicts"] if vv["arm"] == "llm" and vv["status"] == "contradicted")
        # the JIT baseline (H3) is evaluated on the main cohort within the round-1 horizon only
        c["jit_asked"] = bool(f.get("cohort", "main") == "main" and k + 1 <= CONFIG["follow_k"]
                              and (new_dead_llm > 0 or jit_sampled(f["id"], k + 1)))
        c["jit_stratum"] = "death" if new_dead_llm > 0 else "sample"
        if c["jit_asked"]:
            c["jit_consistent"], c["jit_reason"] = jit_call(f, expl["explanation"], prev_text, n["text"])
        rec["commits"].append(c)
        prev_text = n["text"]
    return rec

def second_model_agreement(funcs, claims, out, n_pos=50, n_neg=50, n_anchor=25):
    """Stratified sample of (function, version) pairs re-judged by the second model on the identical item list."""
    rng = random.Random(CONFIG["seed"])
    fmap = {f["id"]: f for f in funcs}
    pos, neg = [], []
    for fid in sorted(out):
        for c in out[fid]["commits"]:
            if not c["verdicts"]:
                continue
            (pos if any(v["status"] == "contradicted" for v in c["verdicts"]) else neg).append((fid, c["k"]))
    sample = [(x, "pos") for x in rng.sample(pos, min(n_pos, len(pos)))] + [(x, "neg") for x in rng.sample(neg, min(n_neg, len(neg)))]
    anchors = rng.sample(sorted(out), min(n_anchor, len(out)))
    agree = jload("judge_agreement.json", {})
    for fid in anchors:
        key = f"{fid}:0"
        if key in agree or not out[fid]["anchor"]:
            continue
        f = fmap[fid]; items = order_for(fid, claims[fid]); a = f["anchor"]
        try:
            v2 = judge_call(f, items, a["text"], facts_summary(a), None, "this exact version of the function", "Function source", backend="anthropic")
        except RuntimeError as e:
            print("second model:", e); break
        v1 = out[fid]["anchor"]
        agree[key] = {"bulk": [v["status"] for v in v1], "second": [v["status"] for v in v2], "types": [v["type"] for v in v1],
                      "arms": [v["arm"] for v in v1], "stratum": "anchor"}
        jdump(agree, "judge_agreement.json")
    for (fid, k), stratum in sample:
        key = f"{fid}:{k}"
        if key in agree:
            continue
        f = fmap[fid]
        allitems = {(arm, i): (arm, i, c) for arm, i, c in order_for(fid, claims[fid])}
        c1 = next(c for c in out[fid]["commits"] if c["k"] == k)
        items = [allitems[(arm, i)] for arm, i in c1["judged"]]
        t = f["touches"][k - 1]; n = t["new"]
        prev_text = f["anchor"]["text"] if k == 1 else f["touches"][k - 2]["new"]["text"]
        try:
            v2 = judge_call(f, items, n["text"], facts_summary(n), udiff(code_without_docstring(prev_text), code_without_docstring(n["text"])),
                            "an earlier version of the function", "Current version of the function", backend="anthropic")
        except RuntimeError as e:
            print("second model:", e); break
        v1 = c1["verdicts"]
        agree[key] = {"bulk": [v["status"] for v in v1], "second": [v["status"] for v in v2],
                      "types": [v["type"] for v in v1], "arms": [v["arm"] for v in v1], "stratum": stratum}
        jdump(agree, "judge_agreement.json")
    print("judge agreement pairs", len(agree), CALLS)

# ---- JIT baseline v2 (post-hoc repair, RUNLOG cycle 5): v1 flagged all 379 asked commits, mostly for omissions
JIT2 = """Below is an explanation of the Python function `{qualname}` ({path}), the diff of its most recent change, and its current code.

Explanation:
\"\"\"
{explanation}
\"\"\"

Most recent change:
```diff
{diff}
```

Current code (docstring removed):
```python
{text}
```

Does the explanation contain at least one statement that is FALSE for the current code? Omissions are not errors: an explanation that does not mention new behaviour is still consistent if everything it does say is true.
Answer with one word, CONSISTENT or INCONSISTENT, optionally followed by " -- " and a reason of at most 15 words."""

def jit2_call(f, expl, prev_text, new_text):
    r = ollama(JIT2.format(qualname=f["qualname"], path=f["path"], explanation=expl,
                           diff=udiff(code_without_docstring(prev_text), code_without_docstring(new_text)),
                           text=code_without_docstring(new_text)), system=SYSTEM, num_predict=40).strip()
    head = r.upper().lstrip("*# ").split()[0] if r.split() else ""
    cons = True if head.startswith("CONSISTENT") else False if head.startswith("INCONSISTENT") else None
    return cons, r[:200]


def process_placebo(f, claims, rec):
    """Placebo / noise-floor run: re-judge the claims still alive at versions whose CODE TOKENS are unchanged
    (docstring-, comment- or layout-only commits). Any 'contradicted' here is spurious by construction."""
    items_all = {(arm, i): (arm, i, c) for arm, i, c in order_for(f["id"], claims)}
    anchor_ok = [(v["arm"], v["idx"]) for v in rec["anchor"] if v["status"] == "supported"]
    dead_at = {}
    for c in rec["commits"]:
        for (arm, i), v in zip(c["judged"], c["verdicts"]):
            if v["status"] == "contradicted":
                dead_at.setdefault((arm, i), c["k"])
    out = []
    versions = [f["anchor"]["text"]] + [t["new"]["text"] for t in f["touches"]]
    last_text = {}
    for p in f.get("placebo", []):
        j = p["after_k"]
        if j >= len(rec["commits"]) + 1 or j > len(f["touches"]):
            continue
        alive = [items_all[key] for key in anchor_ok if dead_at.get(key, 10 ** 6) > j]
        prev = last_text.get(j, versions[j])
        n = p["new"]
        if not alive:
            last_text[j] = n["text"]; continue
        diff = udiff(code_without_docstring(prev), code_without_docstring(n["text"]))
        v = judge_call(f, alive, n["text"], facts_summary(n), diff or "(no code change)", "an earlier version of the function",
                       "Current version of the function")
        out.append({"after_k": j, "commit": p["commit"], "kind": p["kind"], "judged": [[a, i] for a, i, _ in alive], "verdicts": v})
        last_text[j] = n["text"]
    return out

JIT3 = """Below is an explanation of the Python function `{qualname}` ({path}), the diff of its most recent change, and its current code.

Explanation:
\"\"\"
{explanation}
\"\"\"

Most recent change:
```diff
{diff}
```

Current code (docstring removed):
```python
{text}
```

On a scale from 0 to 10, how likely is it that the explanation contains at least one statement that is FALSE for the current code? (0 = certainly every statement is still true, 10 = certainly at least one statement is false; omissions do not count.) Answer with a single integer."""

def jit3_call(f, expl, prev_text, new_text):
    d = udiff(code_without_docstring(prev_text), code_without_docstring(new_text)) or "(no change)"
    r = ollama(JIT3.format(qualname=f["qualname"], path=f["path"], explanation=expl, diff=d,
                           text=code_without_docstring(new_text)), system=SYSTEM, num_predict=8).strip()
    m = re.search(r"\d+", r)
    return (min(10, int(m.group())) if m else None), r[:40]

def process_retest(f, claims, rec):
    """Test-retest placebo at k=0: re-judge the anchor-supported claims, in a different shuffled order, against the
    UNCHANGED anchor code with the per-commit prompt (diff: no code change). Any 'contradicted' is spurious."""
    items_all = {(arm, i): (arm, i, c) for arm, i, c in order_for(f["id"], claims)}
    alive = [items_all[(v["arm"], v["idx"])] for v in rec["anchor"] if v["status"] == "supported"]
    if not alive:
        return None
    random.Random(f"retest-{CONFIG['seed']}-{f['id']}").shuffle(alive)
    a = f["anchor"]
    v = judge_call(f, alive, a["text"], facts_summary(a), "(no code change)", "an earlier version of the function",
                   "Current version of the function")
    return {"judged": [[arm, i] for arm, i, _ in alive], "verdicts": v}
