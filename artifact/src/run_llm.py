"""Driver for all LLM stages, one function at a time (main cohort in a seeded random order, then the random cohort).
Every response is cached (results/llm_cache), so re-running recomputes all records for free and only issues calls for
prompts that are new (e.g. extended follow-up, changed touch sequences, placebo versions, new functions).

Usage: run_llm.py [--second | --second-only]   (--second runs the capped claude-haiku-5-5 agreement checks afterwards)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, CALLS, jload, jdump
from explain import explain_one
from claims import claims_one, second_model_extraction
from judge import process_function, process_placebo, process_retest, second_model_agreement

def main():
    funcs = {f["id"]: f for f in jload("functions.json")}
    rand = jload("random_functions.json", [])
    order = jload("processing_order.json") + [f["id"] for f in rand]
    funcs.update({f["id"]: f for f in rand})
    expl = jload("explanations.json", {}); claims = jload("claims.json", {})
    verdicts = {}
    t0 = time.time()
    if "--second-only" not in sys.argv:
        for n, fid in enumerate(order):
            f = funcs[fid]; t1 = time.time(); c0 = CALLS["ollama"]
            if fid not in expl:
                expl[fid] = explain_one(f); jdump(expl, "explanations.json")
            if fid not in claims:
                claims[fid] = claims_one(f, expl[fid]); jdump(claims, "claims.json")
            rec = process_function(f, claims[fid], expl[fid])
            rec["placebo"] = process_placebo(f, claims[fid], rec) if rec["anchor"] else []
            rec["retest"] = process_retest(f, claims[fid], rec) if (rec["anchor"] and f["cohort"] == "main") else None
            verdicts[fid] = rec
            if CALLS["ollama"] > c0 or n % 25 == 0:
                jdump(verdicts, "verdicts_partial.json")
                print(f"[{time.strftime('%H:%M:%S')}] {fid} ({n+1}/{len(order)}) {time.time()-t1:.0f}s new_calls={CALLS['ollama']-c0} {CALLS}", flush=True)
        # keep only records of current functions, deterministic content
        jdump({k: verdicts[k] for k in sorted(verdicts)}, "verdicts.json")
        jdump({k: expl[k] for k in sorted(expl) if k in funcs}, "explanations.json")
        jdump({k: claims[k] for k in sorted(claims) if k in funcs}, "claims.json")
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "verdicts_partial.json")):
            os.remove(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "verdicts_partial.json"))
        print("LLM stages complete for", len(verdicts), "functions in", f"{time.time()-t0:.0f}s", CALLS, flush=True)
    if "--second" in sys.argv or "--second-only" in sys.argv:
        verdicts = jload("verdicts.json")
        main_ids = [x for x in jload("processing_order.json") if x in verdicts]
        fl = [funcs[x] for x in main_ids]
        if jload("extraction_agreement.json") is None or jload("type_relabel.json") is None:
            second_model_extraction(fl, expl, {x: claims[x] for x in main_ids})
        if jload("judge_agreement.json") is None:  # round-1 sample is kept; entries are re-validated in analyze.py
            second_model_agreement(fl, claims, {x: verdicts[x] for x in main_ids})

if __name__ == "__main__":
    main()
