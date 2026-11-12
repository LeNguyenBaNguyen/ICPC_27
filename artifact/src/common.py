"""Shared config, paths, deterministic LLM clients with on-disk cache, tree-sitter facts."""
import hashlib, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
DATA = os.path.join(ART, "data")
REPOS = os.path.join(DATA, "repos")
RESULTS = os.path.join(ART, "results")
CACHE = os.path.join(RESULTS, "llm_cache")
GENERATED = os.path.join(os.path.dirname(ART), "manuscript", "generated")
os.makedirs(CACHE, exist_ok=True)

CONFIG = {
    "anchor_before": "2024-11-01",      # anchor = last first-parent commit before this date (~2 years before study)
    "repos": ["pytest", "pydantic", "black", "rich", "httpx", "click", "flask", "attrs",
              "pip", "hypothesis", "sphinx", "pylint", "aiohttp", "textual"],
    "candidate_min_touches": 3,          # mined candidates (candidates.json)
    "min_touches": 6,                    # selection: >= this many later code-changing touching commits
    "follow_k": 6,                       # primary horizon K for S(K) (comparable with round 1)
    "k_max": 12,                         # follow-up cap (touches); right-censoring at head/removal before that
    "random_n": 80,                      # size of the RANDOM (unconditioned) cohort
    "pins": {"aiohttp": {"anchor": "7d0f980af6d725970cda5cb011400f5459699301", "head": "d1e21bf9e819acf4b39dba3a4b0032e0e6d6410e"}, "attrs": {"anchor": "4580a74fc9019786d7fb2e722c890033dfe47517", "head": "9d9f94a043de7fc4012459853a4ed597a5577855"}, "black": {"anchor": "53a219056d1ab092ee2d4e5181c55c2e58c4756c", "head": "b4f1fcae48826790b6f93ca9226d20003bce0eaa"}, "click": {"anchor": "02046e7a19480f85fff7e4577486518abe47e401", "head": "8abb77a8432e03ed14a95f972b44f905279415a7"}, "flask": {"anchor": "62c56e08c43f5eb174d15dd050591cfba9aed548", "head": "49943b5a94413a530761350f04e896f76e8d620e"}, "httpx": {"anchor": "6212e8fa3bf90154b891c7721f6d29737caac861", "head": "b5addb64f0161ff6bfe94c124ef76f6a1fba5254"}, "hypothesis": {"anchor": "d16b1838a00ffea7a7bc880237a6a027beb92464", "head": "9ab0be61aaaa27396fc1bbf4b34e7925cc68d1e2"}, "pip": {"anchor": "420435903ff2fc694d6950a47b896427ecaed78f", "head": "4b71f36fb4dfbc87ce65ac4710eeccf81d379f55"}, "pydantic": {"anchor": "d5f37eb916058ed6896692691901702458570b99", "head": "fe5b5ce5581f10626a16dc7ff4159cf3cb6ddbec"}, "pylint": {"anchor": "15a5ac03c45f6888a5a3b6acc72a72da27495567", "head": "79c8d0c9813b0c402c3890778c6d777b162dde8b"}, "pytest": {"anchor": "a1a491837b35b719d77b4f03be318b505d495386", "head": "bc14038c406563b18a03e15453a10b9eee969e42"}, "rich": {"anchor": "afcc5c5a957c75b325fd7cc45bd70b3ac6413ef4", "head": "3a109ccdffe5b2d0e30684a304fbd388e99ce57b"}, "sphinx": {"anchor": "c3968e9be96d51179e701439fbe7dfa22c8ed204", "head": "035a07c32c3241fd3356f4d827800c1f88525272"}, "textual": {"anchor": "cc4e9c4bcfba47cd86967cb2aae675252822ff7f", "head": "5e5b7ef58b8c5572c13a3ff9027d756623661495"}},
    "min_lines": 6, "max_lines": 120,
    "max_funcs_per_repo": 20,
    "target_funcs": 150,
    "seed": 7,
    "bulk_model": "qwen2.5-coder:14b",
    "second_model": "claude-haiku-5-5",
    "second_model_cap": 300,
}

def jload(name, default=None):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)

def jdump(obj, name):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)

# ---------------------------------------------------------------- LLM clients
CALLS = {"ollama": 0, "ollama_cached": 0, "anthropic": 0, "anthropic_cached": 0}

def _key(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]

def ollama(prompt, system=None, model=None, num_predict=700, seed=None):
    model = model or CONFIG["bulk_model"]; seed = CONFIG["seed"] if seed is None else seed
    path = os.path.join(CACHE, _key("ollama", model, seed, system, prompt) + ".json")
    if os.path.exists(path):
        CALLS["ollama_cached"] += 1
        return json.load(open(path))["response"]
    if os.environ.get("LLM_CACHE_ONLY"):
        raise RuntimeError("cache miss with LLM_CACHE_ONLY set")
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "seed": seed, "num_predict": num_predict}}
    # NB: num_ctx is deliberately not set: the shared Ollama runner uses its default (32768); a different value forces a reload.
    if system:
        body["system"] = system
    for attempt in range(3):
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
            break
        except Exception as e:  # transient server hiccup
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    resp.pop("context", None)
    resp["_meta"] = {"model": model, "seed": seed, "system": system, "prompt_sha": _key(prompt), "prompt": prompt}
    tmp = path + ".tmp"; json.dump(resp, open(tmp, "w")); os.replace(tmp, path)
    CALLS["ollama"] += 1
    return resp["response"]

def anthropic_call(prompt, system=None, model=None, max_tokens=1200):
    """Capped second-model condition (claude-haiku-5-5). Cached; counts live calls."""
    import anthropic
    model = model or CONFIG["second_model"]
    path = os.path.join(CACHE, _key("anthropic", model, system, prompt) + ".json")
    if os.path.exists(path):
        CALLS["anthropic_cached"] += 1
        return json.load(open(path))["text"]
    if os.environ.get("LLM_CACHE_ONLY"):
        raise RuntimeError("cache miss with LLM_CACHE_ONLY set")
    live = jload("anthropic_calls.json", {"n": 0})
    if live["n"] >= CONFIG["second_model_cap"]:
        raise RuntimeError("second-model call cap reached")
    client = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=max_tokens, output_config={"effort": "low"},
              messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = system
    m = client.messages.create(**kw)
    text = "".join(b.text for b in m.content if b.type == "text")
    json.dump({"text": text, "model": m.model, "stop_reason": m.stop_reason,
               "usage": {"in": m.usage.input_tokens, "out": m.usage.output_tokens}}, open(path, "w"))
    live["n"] += 1; jdump(live, "anthropic_calls.json")
    CALLS["anthropic"] += 1
    return text

def extract_json(text):
    """Robustly pull the first JSON object/array out of a model response."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    pairs = sorted((("{", "}"), ("[", "]")), key=lambda oc: (text.find(oc[0]) < 0, text.find(oc[0])))
    for opener, closer in pairs:  # whichever bracket opens first
        i = text.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == opener:
                depth += 1
            elif text[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# ---------------------------------------------------------------- tree-sitter
_parser = None
def parser():
    global _parser
    if _parser is None:
        from tree_sitter import Language, Parser
        import tree_sitter_python
        _parser = Parser(Language(tree_sitter_python.language()))
    return _parser

def _text(node, src):
    return src[node.start_byte:node.end_byte].decode("utf8", "replace")

def iter_functions(src: bytes):
    """Yield dicts for every function/method in a Python source (qualified by enclosing classes)."""
    tree = parser().parse(src)
    out = []
    def walk(node, scope):
        for ch in node.children:
            if ch.type == "class_definition":
                name = ch.child_by_field_name("name")
                body = ch.child_by_field_name("body")
                if name is not None and body is not None:
                    walk(body, scope + [_text(name, src)])
            elif ch.type == "function_definition":
                name = ch.child_by_field_name("name")
                if name is None:
                    continue
                out.append(_fn_record(ch, src, scope + [_text(name, src)]))
                body = ch.child_by_field_name("body")
                if body is not None:
                    walk(body, scope + [_text(name, src)])
            elif ch.type == "decorated_definition":
                inner = ch.child_by_field_name("definition")
                if inner is not None and inner.type in ("function_definition", "class_definition"):
                    # treat decorator as part of definition
                    if inner.type == "function_definition":
                        name = inner.child_by_field_name("name")
                        rec = _fn_record(inner, src, scope + [_text(name, src)])
                        rec["start_line"] = ch.start_point[0] + 1
                        rec["text"] = _text(ch, src)
                        out.append(rec)
                        body = inner.child_by_field_name("body")
                        if body is not None:
                            walk(body, scope + [_text(name, src)])
                    else:
                        name = inner.child_by_field_name("name")
                        body = inner.child_by_field_name("body")
                        if name is not None and body is not None:
                            walk(body, scope + [_text(name, src)])
            elif ch.type in ("if_statement", "try_statement", "block", "else_clause", "except_clause", "finally_clause", "module",
                             "elif_clause", "with_statement", "for_statement", "while_statement"):
                walk(ch, scope)
    walk(tree.root_node, [])
    return out

def _fn_record(fn, src, qual):
    params_node = fn.child_by_field_name("parameters")
    params = []
    if params_node is not None:
        for p in params_node.children:
            if p.type in ("identifier",):
                params.append(_text(p, src))
            elif p.type in ("typed_parameter", "default_parameter", "typed_default_parameter",
                            "list_splat_pattern", "dictionary_splat_pattern", "keyword_separator", "positional_separator"):
                if p.type in ("keyword_separator", "positional_separator"):
                    continue
                ident = None
                for c in p.children:
                    if c.type == "identifier":
                        ident = _text(c, src); break
                    if c.type in ("list_splat_pattern", "dictionary_splat_pattern"):
                        ident = _text(p, src).split(":")[0].strip(); break
                params.append(ident or _text(p, src).split(":")[0].split("=")[0].strip())
    body = fn.child_by_field_name("body")
    facts = {"calls": set(), "returns": [], "raises": [], "attrs_assigned": set(),
             "n_if": 0, "n_loop": 0, "n_try": 0, "n_with": 0, "n_return": 0, "n_yield": 0, "n_await": 0}
    doc = None
    if body is not None:
        first = body.children[0] if body.children else None
        if first is not None and first.type == "expression_statement" and first.children and first.children[0].type == "string":
            doc = _text(first.children[0], src)
        _collect(body, src, facts)
    rt = fn.child_by_field_name("return_type")
    rec = {"qualname": ".".join(qual), "name": qual[-1], "params": params,
           "return_type": _text(rt, src) if rt is not None else None,
           "start_line": fn.start_point[0] + 1, "end_line": fn.end_point[0] + 1,
           "text": _text(fn, src), "docstring": _clean_doc(doc) if doc else None,
           "facts": {k: (sorted(v) if isinstance(v, set) else v) for k, v in facts.items()}}
    return rec

def _collect(node, src, facts):
    for ch in node.children:
        t = ch.type
        if t == "function_definition" or t == "class_definition" or t == "lambda":
            continue  # nested defs are separate records
        if t == "call":
            f = ch.child_by_field_name("function")
            if f is not None:
                facts["calls"].add(_text(f, src)[-60:])
        elif t == "return_statement":
            facts["n_return"] += 1
            facts["returns"].append(_text(ch, src)[:80])
        elif t == "raise_statement":
            facts["raises"].append(_text(ch, src)[:80])
        elif t == "if_statement" or t == "conditional_expression":
            facts["n_if"] += 1
        elif t in ("for_statement", "while_statement"):
            facts["n_loop"] += 1
        elif t == "try_statement":
            facts["n_try"] += 1
        elif t == "with_statement":
            facts["n_with"] += 1
        elif t == "yield":
            facts["n_yield"] += 1
        elif t == "await":
            facts["n_await"] += 1
        elif t == "assignment":
            l = ch.child_by_field_name("left")
            if l is not None and l.type == "attribute":
                facts["attrs_assigned"].add(_text(l, src)[:60])
        _collect(ch, src, facts)

def _clean_doc(s):
    s = s.strip()
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2 * len(q):
            s = s[len(q):-len(q)]; break
    s = re.sub(r"^[rRbBuUfF]+(?=['\"])", "", s)
    lines = [l.rstrip() for l in s.strip("\n").splitlines()]
    import textwrap
    return textwrap.dedent("\n".join(lines)).strip()

def facts_summary(rec):
    f = rec["facts"]
    return (f"signature: {rec['name']}({', '.join(rec['params'])})" + (f" -> {rec['return_type']}" if rec.get('return_type') else "") +
            f"\ncalls: {', '.join(f['calls'][:40]) or '(none)'}"
            f"\nreturn statements: {' | '.join(f['returns'][:8]) or '(none)'}"
            f"\nraise statements: {' | '.join(f['raises'][:8]) or '(none)'}"
            f"\ncontrol flow: if={f['n_if']} loops={f['n_loop']} try={f['n_try']} with={f['n_with']} returns={f['n_return']} yields={f['n_yield']} awaits={f['n_await']}")

def code_without_docstring(text):
    """Return the function source (dedented) with its docstring statement removed, so judges check code, not prose."""
    import textwrap
    src = textwrap.dedent(text)
    b = src.encode("utf8")
    tree = parser().parse(b)
    def find(node):
        for ch in node.children:
            if ch.type == "function_definition":
                return ch
            if ch.type == "decorated_definition":
                d = ch.child_by_field_name("definition")
                if d is not None and d.type == "function_definition":
                    return d
        return None
    fn = find(tree.root_node)
    if fn is None:
        return src
    body = fn.child_by_field_name("body")
    if body is None or not body.children:
        return src
    first = body.children[0]
    if first.type == "expression_statement" and first.children and first.children[0].type == "string":
        start, end = first.start_byte, first.end_byte
        # drop the whole line(s) the docstring occupies
        ls = b.rfind(b"\n", 0, start) + 1
        le = b.find(b"\n", end)
        le = len(b) if le < 0 else le + 1
        out = b[:ls] + b[le:]
        if not body.children[1:]:
            out = b[:ls] + b[ls:start] + b"..." + b[end:]  # docstring-only body: keep valid shape
        return out.decode("utf8", "replace")
    return src
