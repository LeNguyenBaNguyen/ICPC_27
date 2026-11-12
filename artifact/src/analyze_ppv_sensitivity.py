"""Analyze the event-level human-label sensitivity audit through touch six."""
import csv, json, math, os
from collections import Counter, defaultdict
import numpy as np
from prepare_validation import effective, km, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL, OUT = os.path.join(ROOT, "validation"), os.path.join(ROOT, "experiments", "ppv_sensitivity")
os.makedirs(OUT, exist_ok=True)

def wilson(x, n, z=1.959963984540054):
    p = x / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [c-h, c+h]

rows = list(csv.DictReader(open(os.path.join(VAL, "ppv_weighted_sensitivity_audit.csv"), encoding="utf-8")))
labels = {}
for r in rows:
    key = (r["fid"], int(r["k"]), int(r["idx"]))
    if key in labels:
        raise ValueError(f"duplicate audited event {key}")
    labels[key] = r["verdict"]

funcs, verdicts = load("functions.json"), load("verdicts.json")
if isinstance(funcs, list):
    funcs = {f["id"]: f for f in funcs}
by_fid = defaultdict(lambda: [[], [], []])  # primary, unresolved-censor, unresolved-event
covered = 0
for fid, v in sorted(verdicts.items()):
    if fid not in funcs: continue
    f = funcs[fid]
    if f["cohort"] != "main": continue
    valid = effective(f, v["commits"]); eff = {k: e for e, k in enumerate(valid, 1)}
    for a in v["anchor"]:
        if a["arm"] != "llm" or a["status"] != "supported": continue
        idx = a["idx"]
        traj = [(c["k"], x["status"]) for c in v["commits"]
                for pair, x in zip(c["judged"], c["verdicts"]) if pair == ["llm", idx]]
        dead = next((k for k, s in traj if s == "contradicted"), None)
        base = ((eff[dead], 1) if dead in eff and eff[dead] <= 6 else (min(len(valid), 6), 0))
        human = [(eff[k], labels[(fid,k,idx)]) for k,s in traj if s == "contradicted" and k in eff and eff[k] <= 6 and (fid,k,idx) in labels]
        covered += len(human)
        hcon = next((t for t,s in human if s == "contradicted"), None)
        hunr = next((t for t,s in human if s == "unresolved"), None)
        by_fid[fid][0].append(base)
        by_fid[fid][1].append((hcon,1) if hcon else ((hunr-1,0) if hunr else (min(len(valid),6),0)))
        by_fid[fid][2].append((hcon,1) if hcon else ((hunr,1) if hunr else (min(len(valid),6),0)))

def s6(which): return km([x for values in by_fid.values() for x in values[which]])[6]
def boot(which, seed=20261112, B=2000):
    rng=np.random.RandomState(seed); ids=sorted(by_fid); vals=[]
    for _ in range(B):
        pick=rng.choice(ids,len(ids),replace=True)
        vals.append(km([x for fid in pick for x in by_fid[fid][which]])[6])
    return [float(x) for x in np.percentile(vals,[2.5,97.5])]

counts = Counter(r["verdict"] for r in rows); resolved=counts["supported"]+counts["contradicted"]
s_event, s_censor = s6(2), s6(1)
# One-sided missed-event scenarios: hazard correction assumes homogeneous recall.
haz=-math.log(s_event)
recall = {str(r): math.exp(-haz/r) for r in (0.7,0.88,1.0)}

# Direct overlap with A1/A2 audit: same fid, mining_k, idx.
key_rows=list(csv.DictReader(open(os.path.join(VAL,"private","sampling_key.csv"),encoding="utf-8")))
a1={r["task_id"]:r for r in csv.DictReader(open(os.path.join(VAL,"annotator_1.csv"),encoding="utf-8"))}
a2={r["task_id"]:r for r in csv.DictReader(open(os.path.join(VAL,"annotator_2.csv"),encoding="utf-8"))}
overlap=[]
for r in key_rows:
    k=(r["fid"],int(r["mining_k"]),int(r["idx"]))
    if k in labels: overlap.append((labels[k],a1[r["task_id"]]["status"],a2[r["task_id"]]["status"]))

summary={
 "estimand":"Single-rater human-label sensitivity replacing all observed Qwen-CON rows through effective touch 6; not semantic truth.",
 "audit_event_rows":len(rows),"unique_audited_event_keys":len(labels),"covered_qwen_con_observations":covered,
 "claims":sum(len(v[0]) for v in by_fid.values()),"primary_S6_reproduced":s6(0),
 "human_label_S6_unresolved_censored":s_censor,"human_label_S6_unresolved_event":s_event,
 "function_cluster_bootstrap_95":{"unresolved_censored":boot(1),"unresolved_event":boot(2)},
 "human_verdict_counts":dict(counts),"resolved_contradiction_ppv":counts["contradicted"]/resolved,
 "resolved_contradiction_ppv_wilson_95":wilson(counts["contradicted"],resolved),
 "recall_assumption_S6":{"assumption":"Divide the event-scenario cumulative hazard by homogeneous recall; illustrative, not identified.","values":recall},
 "two_rater_overlap":{"n":len(overlap),"new_rater_contradicted":sum(x[0]=="contradicted" for x in overlap),"a1_a2_both_contradicted":sum(x[1]==x[2]=="contradicted" for x in overlap),"matrix":{f"new_{x[0]}__a1_{x[1]}__a2_{x[2]}":overlap.count(x) for x in set(overlap)}},
 "limitation":"Single-rater labels and homogeneous recall scenarios do not identify semantic truth or rater-error uncertainty."
}
json.dump(summary,open(os.path.join(OUT,"summary.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
