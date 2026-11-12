"""Frozen inference for the recovered 80-function clean random-cohort redraw.

Default writes a plan only. --run uses an explicit fresh-call budget and the
same isolated Client/settings as the other supplemental experiments. Original
results and archived cached responses remain read-only.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import random
from pathlib import Path

from common import CONFIG, extract_json
from claims import PROMPT as CLAIM_PROMPT, SYSTEM as CLAIM_SYSTEM, parse_claims
from explain import PROMPT as EXPLAIN_PROMPT, SYSTEM as EXPLAIN_SYSTEM
from judge import order_for
from run_missing_experiments import (
    ART, EXPERIMENTS, OPTIONS, BudgetExhausted, Client, atomic_json, digest,
    judge, make_judge_task,
)

NAME = "random_redraw_inference"
MODEL = "qwen2.5-coder:14b"
RECOVERY = EXPERIMENTS / "random_redraw"


def effective_indices(f):
    valid = []
    for k, touch in enumerate(f["touches"], 1):
        if touch.get("overload_stub"):
            break
        if not (touch.get("format_only") or touch.get("paren_only")):
            valid.append(k)
    return valid


def freeze_plan():
    recovery = json.loads((RECOVERY / "report.json").read_text(encoding="utf-8"))
    if not recovery["trajectory_recovery_completed"] or recovery["recovered_n"] != 80:
        raise RuntimeError("All80 recovered trajectories are required before freezing inference")
    funcs = json.loads((RECOVERY / "functions.json").read_text(encoding="utf-8"))
    selection = json.loads((RECOVERY / "sample_manifest.json").read_text(encoding="utf-8"))
    expected = {row["id"]: (row["repo"], row["path"], row["qualname"]) for row in selection["draw"]}
    if len(funcs) != 80 or len({f["id"] for f in funcs}) != 80:
        raise ValueError("Expected80 unique functions")
    if any(expected.get(f["id"]) != (f["repo"], f["path"], f["qualname"]) for f in funcs):
        raise ValueError("Recovered function identities differ from frozen redraw")
    stats = json.loads((ART / "results/mining_stats.json").read_text(encoding="utf-8"))
    tasks = [{"fid": f["id"], "function": f, "effective_followup_touches": effective_indices(f),
              "generation_prompt": EXPLAIN_PROMPT.format(path=f["path"], qualname=f["qualname"], text=f["anchor"]["text"]),
              "generation_system": EXPLAIN_SYSTEM,
              "anchor_date": stats[f["repo"]]["anchor_date"], "head_date": stats[f["repo"]]["head_date"],
              "pins": CONFIG["pins"][f["repo"]]} for f in sorted(funcs, key=lambda f: f["id"])]
    plan = {
        "experiment": NAME, "model": MODEL, "options": OPTIONS,
        "planned_before_fresh_calls": True, "sample_n": len(funcs),
        "population_frame": selection["population_frame"], "population_n": selection["population_n"],
        "selection_seed": selection["seed"], "inclusion_probability": selection["inclusion_probability"],
        "arms": ["llm"], "docstring_claim_comparator_included": False,
        "generation_policy": "Original with-docstring generation prompt, unchanged decomposition; LLM claims only.",
        "followup_policy": "Anchor screen, then all still-uncontradicted anchor-supported claims at every effective touch. Unresolved is not a death. Format/paren-only and overload exposures are excluded before judging.",
        "endpoint": "First fresh judge-detected contradiction, conditional on fresh anchor support; not validated semantic survival.",
        "quality_policy": "Same Client context/output checks as other supplemental experiments. Require1..12complete valid claim objects; reject incomplete explanation. Save every raw failed attempt.",
        "runtime_limitation": "Historical model digest/quantization unknown; this is a fresh backend cohort, not byte-identical regeneration of the archived model.",
        "reliability_rule": {"minimum_claims_at_risk": 30, "minimum_functions_at_risk": 5, "target_k": 6},
        "uncertainty": {"method": "Function-cluster percentile bootstrap", "B": 500, "seed": selection["seed"] + 333},
        "max_fresh_calls": 3 * len(funcs) + sum(len(t["effective_followup_touches"]) for t in tasks),
        "generation_calls": len(funcs), "decomposition_calls": len(funcs), "max_anchor_calls": len(funcs),
        "max_later_calls": sum(len(t["effective_followup_touches"]) for t in tasks),
        "source_sha256": {name: hashlib.sha256((RECOVERY / name).read_bytes()).hexdigest()
                          for name in ("functions.json", "sample_manifest.json")},
        "prompt_template_sha256": {"explain": digest(EXPLAIN_SYSTEM + EXPLAIN_PROMPT),
                                   "claims": digest(CLAIM_SYSTEM + CLAIM_PROMPT)},
        "tasks": tasks,
    }
    plan["plan_sha256"] = digest(plan)
    path = EXPERIMENTS / "plans" / (NAME + ".json")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError("Frozen redraw inference plan changed; use a separately named protocol revision")
    if not path.exists():
        atomic_json(path, plan)
    (EXPERIMENTS / NAME / "cache").mkdir(parents=True, exist_ok=True)
    return plan


def observation_rows(task, rec):
    """Construct rows only after a function's complete frozen follow-up."""
    if not rec.get("complete"):
        return []
    f, valid = task["function"], task["effective_followup_touches"]
    anchor_date = dt.date.fromisoformat(task["anchor_date"])
    stop = next((k for k, t in enumerate(f["touches"], 1) if t.get("overload_stub")), None)
    if stop is not None:
        reason = "overload_stub"
        date = f["touches"][valid[-1] - 1]["date"] if valid else task["anchor_date"]
    elif len(f["touches"]) == CONFIG["k_max"]:
        reason, date = "cap", f["touches"][-1]["date"]
    elif f["censor"]["reason"] == "head":
        reason, date = "head", task["head_date"]
    else:
        reason, date = f["censor"]["reason"], f["censor"]["date"]
    out = []
    for a in rec["anchor"]:
        if a["status"] != "supported":
            continue
        event_e, event_k = None, None
        for e, k in enumerate(valid, 1):
            value = next((x for x in rec["touches"][str(k)] if x["idx"] == a["idx"]), None)
            if value and value["status"] == "contradicted":
                event_e, event_k = e, k
                break
        end_date = f["touches"][event_k - 1]["date"] if event_k else date
        removed = event_k is None and reason in ("file_removed", "function_removed_or_renamed")
        out.append({"fid": task["fid"], "idx": a["idx"], "arm": "llm", "type": a["type"],
                    "event": int(event_k is not None), "time_k": event_e if event_e else len(valid),
                    "days": (dt.date.fromisoformat(end_date) - anchor_date).days,
                    "censor_reason": "" if event_k else reason,
                    "removed": int(removed), "removal_or_contradiction": int(removed or event_k is not None)})
    return out


def survival_at(rows, horizon, time="time_k", event="event"):
    survival = 1.0
    event_times = sorted({r[time] for r in rows if r[event] and r[time] <= horizon})
    for t in event_times:
        at_risk = sum(r[time] >= t for r in rows)
        deaths = sum(r[time] == t and r[event] for r in rows)
        survival *= 1 - deaths / at_risk
    return survival


def summarize(plan, records, client, reason=None, failure=None):
    tasks = {t["fid"]: t for t in plan["tasks"]}
    completed = [fid for fid in tasks if records.get(fid, {}).get("complete")]
    rows = [row for fid in completed for row in observation_rows(tasks[fid], records[fid])]
    complete = len(completed) == plan["sample_n"]
    target = plan["reliability_rule"]["target_k"]
    at_risk = [r for r in rows if r["time_k"] >= target]
    adequate = (len(at_risk) >= plan["reliability_rule"]["minimum_claims_at_risk"] and
                len({r["fid"] for r in at_risk}) >= plan["reliability_rule"]["minimum_functions_at_risk"])
    # No population survival estimates from an incomplete, ordered prefix.
    report = {"experiment": NAME, "plan_sha256": plan["plan_sha256"], "inference_completed": complete, "complete": complete,
              "completed_functions": len(completed), "started_functions": len(records),
              "planned_functions": plan["sample_n"], "stop_reason": reason, "failed": failure is not None,
              "error": failure, "fresh_calls_this_invocation": client.calls, "cache_hits_this_invocation": client.hits,
              "fresh_call_upper_bound": plan["max_fresh_calls"], "quality_counts": client.quality_counts(),
              "anchor_supported_claims_completed_functions": len(rows),
              "unparsed_judge_verdicts": sum(not a.get("parsed", True) for rec in records.values()
                                             for values in [rec.get("anchor", [])] + list(rec.get("touches", {}).values()) for a in values),
              "at_risk_at_6_completed_functions": {"claims": len(at_risk), "functions": len({r["fid"] for r in at_risk})},
              "S6": None, "S6_withheld_reason": "Incomplete cohort" if not complete else "Insufficient at-risk support" if not adequate else None,
              "calendar_365": None,
              "interpretation": "Fresh model-conditional estimates on the archived eligible anchor population; model digest unknown historically. Not semantic survival or human validation."}
    report["quality_complete"] = complete and report["unparsed_judge_verdicts"] == 0 and report["quality_counts"]["cached_length_limited_responses"] == 0
    if complete:
        reliable = [k for k in range(1, CONFIG["k_max"] + 1)
                    if sum(r["time_k"] >= k for r in rows) >= plan["reliability_rule"]["minimum_claims_at_risk"] and
                    len({r["fid"] for r in rows if r["time_k"] >= k}) >= plan["reliability_rule"]["minimum_functions_at_risk"]]
        report["maximum_reliable_touch"] = max(reliable) if reliable else 0
        report["S_at_maximum_reliable_touch"] = survival_at(rows, max(reliable)) if reliable else None
        report["reliable_horizon_reporting"] = "Descriptive point at the largest touch satisfying the frozen30-claim/5-function support rule; does not replace withheld S6."
        if adequate:
            report["S6"] = survival_at(rows, target)
            import numpy as np
            rng = random.Random(plan["uncertainty"]["seed"])
            ids = sorted(tasks)
            by = collections.defaultdict(list)
            for row in rows:
                by[row["fid"]].append(row)
            values = [survival_at([row for fid in rng.choices(ids, k=len(ids)) for row in by[fid]], target)
                      for _ in range(plan["uncertainty"]["B"])]
            report["S6_function_cluster_percentile_interval"] = [float(x) for x in np.percentile(values, [2.5, 97.5])]
        if rows:
            import pandas as pd
            from lifelines import AalenJohansenFitter
            frame = pd.DataFrame(rows)
            frame["cause"] = frame.event.where(frame.event == 1, frame.removed * 2)
            calendar = {"removal_censored_S365": survival_at(rows, 365, time="days"),
                        "removal_as_event_S365": survival_at(rows, 365, time="days", event="removal_or_contradiction"),
                        "contradicted_functions_within_365": int(frame[(frame.event == 1) & (frame.days <= 365)].fid.nunique()),
                        "removed_functions_within_365": int(frame[(frame.removed == 1) & (frame.days <= 365)].fid.nunique()),
                        "uncertainty": "Calendar point estimates are descriptive; no interval claimed."}
            for code, label in ((1, "contradiction_CIF365"), (2, "removal_CIF365")):
                if int((frame.cause == code).sum()) == 0:
                    calendar[label] = 0.0
                else:
                    fit = AalenJohansenFitter(calculate_variance=False, jitter_level=0.0001, seed=CONFIG["seed"])
                    fit.fit(frame.days.clip(lower=0), frame.cause, event_of_interest=code)
                    cdf = fit.cumulative_density_
                    calendar[label] = float(cdf[cdf.index <= 365].iloc[-1, 0]) if (cdf.index <= 365).any() else 0.0
            report["calendar_365"] = calendar
    atomic_json(client.directory / "claim_survival.json", rows)
    atomic_json(client.directory / "summary.json", report)
    return report


def run(client):
    path = client.directory / "observations.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    reason, failure = None, None
    try:
        for task in client.plan["tasks"]:
            fid, f = task["fid"], task["function"]
            rec = records.setdefault(fid, {"touches": {}})
            if rec.get("complete"):
                continue
            if "explanation" not in rec:
                raw = client.generate(fid + ":generate", task["generation_prompt"], task["generation_system"], 600)
                rec["explanation"] = raw["response"].strip()
                atomic_json(path, records)
            if "claims" not in rec:
                prompt = CLAIM_PROMPT.format(qualname=f["qualname"], text=rec["explanation"])
                raw = client.generate(fid + ":decompose", prompt, CLAIM_SYSTEM, 1200)
                parsed, source = parse_claims(raw["response"]), extract_json(raw["response"])
                if isinstance(source, dict):
                    source = source.get("claims") or (list(source.values())[0] if source else [])
                if not parsed or not isinstance(source, list) or len(source) != len(parsed) or len(source) > 12:
                    raise RuntimeError(f"{fid}: incomplete/malformed decomposition; raw output retained")
                rec["claims"] = parsed
                rec["decomposition_claims_at_cap"] = len(parsed) == 12
                atomic_json(path, records)
            ordered = order_for(fid, {"llm": rec["claims"], "doc": []})
            if "anchor" not in rec:
                rec["anchor"] = judge(client, make_judge_task(fid, f, 0, ordered, fid + ":judge:0"))
                atomic_json(path, records)
            supported = {a["idx"] for a in rec["anchor"] if a["status"] == "supported"}
            alive = [item for item in ordered if item[1] in supported]
            for k in task["effective_followup_touches"]:
                if str(k) not in rec["touches"]:
                    rec["touches"][str(k)] = judge(client, make_judge_task(fid, f, k, alive, f"{fid}:judge:{k}")) if alive else []
                    atomic_json(path, records)
                contradicted = {a["idx"] for a in rec["touches"][str(k)] if a["status"] == "contradicted"}
                alive = [item for item in alive if item[1] not in contradicted]
            rec["complete"] = True
            atomic_json(path, records)
    except BudgetExhausted as error:
        reason = str(error)
    except Exception as error:
        reason, failure = str(error), repr(error)
    return summarize(client.plan, records, client, reason, failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    if args.max_calls < 0:
        parser.error("--max-calls must be nonnegative")
    plan = freeze_plan()
    if not args.run:
        print(json.dumps({k: v for k, v in plan.items() if k != "tasks"}, indent=2))
        return
    client = Client(NAME, args.endpoint, args.max_calls)
    report = run(client)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report.get("failed") else 0)


if __name__ == "__main__":
    main()
