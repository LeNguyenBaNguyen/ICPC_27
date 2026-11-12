"""Stage 2: generate one explanation per function at the anchor commit (bulk local model, T=0, cached)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, CALLS, jload, jdump, ollama

SYSTEM = ("You are an expert Python developer writing documentation for a colleague who has not seen this code. "
          "Explain the function precisely from the source shown. Do not invent behaviour that is not visible in the code.")

PROMPT = """File: {path}
Function: {qualname}

```python
{text}
```

Write an explanation of this function in 5 to 8 plain-prose sentences for a developer. Cover: what it does and why it exists, its parameters and what they mean, what it returns, which helper functions or methods it relies on, how it handles errors or edge cases (exceptions raised, early returns), and any side effects. Do not use bullet points or headings; write prose only."""

def explain_one(f):
    a = f["anchor"]
    resp = ollama(PROMPT.format(path=f["path"], qualname=f["qualname"], text=a["text"]), system=SYSTEM, num_predict=600)
    return {"explanation": resp.strip(), "docstring": a.get("docstring"), "model": CONFIG["bulk_model"]}

def main():
    funcs = jload("functions.json")
    out = jload("explanations.json", {})
    for i, f in enumerate(funcs):
        if f["id"] in out:
            continue
        a = f["anchor"]
        text = a["text"]
        resp = ollama(PROMPT.format(path=f["path"], qualname=f["qualname"], text=text), system=SYSTEM, num_predict=600)
        out[f["id"]] = {"explanation": resp.strip(), "docstring": a.get("docstring"), "model": CONFIG["bulk_model"]}
        if i % 10 == 0:
            jdump(out, "explanations.json"); print(f"explain {i}/{len(funcs)} calls={CALLS}", flush=True)
    jdump(out, "explanations.json")
    print("explanations", len(out), "with docstring", sum(1 for v in out.values() if v["docstring"] and len(v["docstring"]) >= 20), CALLS)

if __name__ == "__main__":
    main()
