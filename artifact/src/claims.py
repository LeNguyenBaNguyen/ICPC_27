"""Stage 3: decompose explanations (and anchor docstrings) into typed atomic claims.

Types: structural (signature, parameters, return type, which functions/attributes it uses),
dataflow (how values are computed/transformed, what is returned for which input),
behavioural (exceptions, conditions, early returns, side effects, edge cases),
intent (purpose, rationale, role in the system).
A capped second model (claude-haiku-5-5) re-types a sample and re-extracts a sample for agreement.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, CALLS, jload, jdump, ollama, anthropic_call, extract_json

SYSTEM = "You decompose natural-language descriptions of Python functions into atomic, independently checkable claims. Output JSON only."

PROMPT = """Below is a description of the Python function `{qualname}`.

Description:
\"\"\"
{text}
\"\"\"

Split the description into atomic claims. Each claim must be a single, self-contained factual statement about the function that could be checked against the function's source code (name the function explicitly; no pronouns referring to other claims). Do not add claims that are not in the description; do not merge two facts into one claim. Keep the original wording where possible. Assign each claim exactly one type:
- "structural": the function's signature, parameter names/defaults/types, return type, or which functions, methods, classes or attributes it uses.
- "dataflow": how a value is computed, transformed, passed on, or what is returned for which input.
- "behavioural": exceptions raised or caught, conditions/early returns, side effects, edge-case handling, ordering/iteration behaviour.
- "intent": purpose, rationale, or the role of the function in the larger system.

Return a JSON array of at most 12 objects: [{{"text": "...", "type": "structural|dataflow|behavioural|intent"}}, ...]"""

TYPES = {"structural", "dataflow", "behavioural", "intent"}

def parse_claims(resp):
    arr = extract_json(resp)
    out = []
    if isinstance(arr, dict):
        arr = arr.get("claims") or list(arr.values())[0] if arr else []
    if not isinstance(arr, list):
        return out
    for c in arr[:12]:
        if not isinstance(c, dict):
            continue
        t = str(c.get("type", "")).lower().strip()
        t = {"behavioral": "behavioural", "data-flow": "dataflow", "data_flow": "dataflow", "structure": "structural"}.get(t, t)
        txt = str(c.get("text", "")).strip()
        if txt and t in TYPES:
            out.append({"text": txt, "type": t})
    return out

def extract(qualname, text, backend):
    p = PROMPT.format(qualname=qualname, text=text)
    if backend == "ollama":
        return parse_claims(ollama(p, system=SYSTEM, num_predict=1200))
    return parse_claims(anthropic_call(p, system=SYSTEM, max_tokens=1500))

def claims_one(f, e):
    rec = {"llm": extract(f["qualname"], e["explanation"], "ollama"), "doc": []}
    if e.get("docstring") and len(e["docstring"]) >= 20:
        rec["doc"] = extract(f["qualname"], e["docstring"], "ollama")
    return rec

def second_model_extraction(funcs, expl, out, n_extract=30, n_relabel=50):
    """Extraction-consistency check with a second model on a seeded sample (human annotation pending)."""
    rng = random.Random(CONFIG["seed"])
    ids = sorted(out)
    fmap = {f["id"]: f for f in funcs}
    agree = jload("extraction_agreement.json", {})
    for fid in rng.sample(ids, min(n_extract, len(ids))):
        if fid in agree:
            continue
        f = fmap[fid]
        try:
            second = extract(f["qualname"], expl[fid]["explanation"], "anthropic")
        except RuntimeError as e:
            print("second model:", e); break
        agree[fid] = {"n_bulk": len(out[fid]["llm"]), "n_second": len(second),
                      "types_bulk": [c["type"] for c in out[fid]["llm"]], "types_second": [c["type"] for c in second],
                      "bulk_claims": out[fid]["llm"], "second_claims": second}
        jdump(agree, "extraction_agreement.json")
    relabel = jload("type_relabel.json", {})
    for fid in rng.sample(ids, min(n_relabel, len(ids))):
        if fid in relabel or not out[fid]["llm"]:
            continue
        f = fmap[fid]
        claims = out[fid]["llm"]
        p = ("Classify each claim about the Python function `%s` into exactly one type: structural (signature, parameters, return type, "
             "which functions/attributes it uses), dataflow (how values are computed/transformed/returned), behavioural (exceptions, conditions, "
             "early returns, side effects, edge cases), intent (purpose/rationale/role). Return a JSON array of type strings, one per claim, in order.\n\nClaims:\n"
             % f["qualname"]) + "\n".join(f"{k+1}. {c['text']}" for k, c in enumerate(claims))
        try:
            arr = extract_json(anthropic_call(p, system="Output JSON only.", max_tokens=400))
        except RuntimeError as e:
            print("second model:", e); break
        if isinstance(arr, list) and len(arr) == len(claims):
            norm = [{"behavioral": "behavioural", "data-flow": "dataflow"}.get(str(t).lower().strip(), str(t).lower().strip()) for t in arr]
            relabel[fid] = {"bulk": [c["type"] for c in claims], "second": norm}
            jdump(relabel, "type_relabel.json")
    print("extraction agreement", len(agree), "relabel", len(relabel), CALLS)
