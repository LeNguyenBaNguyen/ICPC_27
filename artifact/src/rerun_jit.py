"""JIT baselines beyond v1 (cached, resumable), on the main cohort:
* v2 (neutral prompt, post-hoc, RUNLOG cycle 6) at every commit where v1 was asked, and at k=0 (sanity: unchanged anchor);
* v3 (graded 0-10 score, round 2) at the same commits and at k=0, for ROC/AUC.
Writes results/jit_v2.json and results/jit_v3.json keyed "<fid>:<k>"."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CALLS, jload, jdump
from judge import jit2_call, jit3_call

def main():
    funcs = {f["id"]: f for f in jload("functions.json")}
    expl = jload("explanations.json"); verdicts = jload("verdicts.json")
    v2 = jload("jit_v2.json", {}); v3 = jload("jit_v3.json", {})
    keep2, keep3 = {}, {}
    n = 0
    for fid in sorted(funcs):
        if fid not in verdicts:
            continue
        f = funcs[fid]
        asks = [0] + [c["k"] for c in verdicts[fid]["commits"] if c.get("jit_asked")]
        for k in asks:
            key = f"{fid}:{k}"
            prev = f["anchor"]["text"] if k <= 1 else f["touches"][k - 2]["new"]["text"]
            new = f["anchor"]["text"] if k == 0 else f["touches"][k - 1]["new"]["text"]
            c2, r2 = jit2_call(f, expl[fid]["explanation"], prev, new)
            keep2[key] = {"consistent": c2, "reason": r2}
            s3, r3 = jit3_call(f, expl[fid]["explanation"], prev, new)
            keep3[key] = {"score": s3, "raw": r3}
            n += 1
            if n % 20 == 0:
                jdump(keep2, "jit_v2.json"); jdump(keep3, "jit_v3.json"); print(time.strftime("%H:%M:%S"), n, CALLS, flush=True)
    jdump(keep2, "jit_v2.json"); jdump(keep3, "jit_v3.json")
    print("jit v2/v3 done", len(keep2), CALLS, flush=True)

if __name__ == "__main__":
    main()
