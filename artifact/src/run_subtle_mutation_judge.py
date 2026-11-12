"""Run the primary local Qwen judge on the 25 human-audited subtle mutations."""
import csv, json, os, re, urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, "validation")
OUT = os.path.join(ROOT, "experiments", "subtle_mutation_judge")
os.makedirs(OUT, exist_ok=True)
ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11435/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")

def call(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0, "seed": 7}}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

rows = list(csv.DictReader(open(os.path.join(VAL, "subtle_mutations_audit.csv"), encoding="utf-8", newline="")))
observations = []
for row in rows:
    prompt = f"""You are judging whether a saved code claim is contradicted by the current mutated code.
Claim: {row['claim']}
Original code:\n```python\n{row['original_code']}\n```
Mutated code:\n```python\n{row['mutated_code']}\n```
Answer exactly one line: verdict=SUPPORTED or CONTRADICTED or UNRESOLVED; evidence=<at most 20 words>. A known behavior-changing mutation should be CONTRADICTED only when the claim is false of the mutated code."""
    result = call(prompt)
    raw = result.get("response", "")
    m = re.search(r"verdict\s*=\s*(SUPPORTED|CONTRADICTED|UNRESOLVED)", raw, re.I)
    observations.append({"case_id": row["case_id"], "mutation_id": row["mutation_id"], "raw_response": raw, "parsed_verdict": m.group(1).lower() if m else None, "expected_affected": row["expected_affected"], "model": MODEL, "seed": 7, "temperature": 0})
json.dump({"created_at": datetime.now(timezone.utc).isoformat(), "model": MODEL, "endpoint": ENDPOINT, "observations": observations}, open(os.path.join(OUT, "observations.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({"n": len(observations), "parsed": sum(x["parsed_verdict"] is not None for x in observations), "contradicted": sum(x["parsed_verdict"] == "contradicted" for x in observations)}, indent=2))
