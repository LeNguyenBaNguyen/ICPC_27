"""Frozen, resumable supplemental experiments; primary results are read-only.

Default/--plan writes specifications without network or inference. --run NAME
requires a positive --max-calls budget to permit fresh Ollama inference; zero
means cache-only. All caches, model fingerprints and partial results are isolated.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import CONFIG, _key, code_without_docstring, facts_summary, extract_json
from claims import PROMPT as CLAIM_PROMPT, SYSTEM as CLAIM_SYSTEM, parse_claims
from explain import PROMPT as EXPLAIN_PROMPT, SYSTEM as EXPLAIN_SYSTEM
from judge import SYSTEM as JUDGE_SYSTEM, judge_prompt, order_for, parse_statuses, udiff
from prepare_validation import effective, km

ART = Path(__file__).resolve().parents[1]
RESULTS = ART / "results"
EXPERIMENTS = ART / "experiments"
NAMES = ("resurrection", "docstring_free", "cross_family")
MODELS = {"resurrection": "qwen2.5-coder:14b", "docstring_free": "qwen2.5-coder:14b",
          "cross_family": "llama3.1:8b"}
OPTIONS = {"temperature": 0, "seed": 7, "num_ctx": 8192}


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # OneDrive/antivirus may briefly open a file without delete sharing on
    # Windows. Retry the atomic write rather than discard a completed response
    # or fall back to a non-atomic overwrite of the last good observation set.
    for attempt in range(10):
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 1.0))


def source_hashes():
    names = ["functions.json", "random_functions.json", "claims.json", "verdicts.json"]
    hashes = {n: hashlib.sha256((RESULTS / n).read_bytes()).hexdigest() for n in names}
    hashes["judge_prompt_template"] = digest(JUDGE_SYSTEM + __import__("judge").JUDGE)
    hashes["explain_prompt_template"] = digest(EXPLAIN_SYSTEM + EXPLAIN_PROMPT)
    hashes["claims_prompt_template"] = digest(CLAIM_SYSTEM + CLAIM_PROMPT)
    return hashes


def inputs():
    funcs = {f["id"]: f for name in ("functions.json", "random_functions.json") for f in load(name)}
    return funcs, load("claims.json"), load("verdicts.json")


def make_judge_task(fid, f, k, items, task_id, extra=None):
    current = f["anchor"] if k == 0 else f["touches"][k - 1]["new"]
    previous = f["anchor"] if k <= 1 else f["touches"][k - 2]["new"]
    diff = None if k == 0 else udiff(code_without_docstring(previous["text"]), code_without_docstring(current["text"]))
    prompt = judge_prompt(f, items, current["text"], facts_summary(current), diff,
                          "this exact version of the function" if k == 0 else "an earlier version of the function",
                          "Function source" if k == 0 else "Current version of the function")
    return {"task_id": task_id, "fid": fid, "mining_k": k,
            "items": [{"arm": arm, "idx": i, "type": c["type"], "text": c["text"]} for arm, i, c in items],
            "prompt": prompt, "system": JUDGE_SYSTEM, "num_predict": 40 + 22 * len(items),
            **(extra or {})}


def resurrection_plan(funcs, claims, verdicts):
    tasks, excluded = [], []
    claim_keys, pairs, eligible_details = set(), 0, {}
    for fid, v in sorted(verdicts.items()):
        f = funcs[fid]
        valid = effective(f, v["commits"])
        first = {}
        supported = {(a["arm"], a["idx"]) for a in v["anchor"] if a["status"] == "supported"}
        for c in v["commits"]:
            for a in c["verdicts"]:
                key = (a["arm"], a["idx"])
                if key in supported and a["status"] == "contradicted":
                    first.setdefault(key, c["k"])
        for key, death in first.items():
            if death not in valid and any(k > death for k in valid):
                excluded.append({"fid": fid, "arm": key[0], "idx": key[1], "first_contradiction_k": death,
                                 "later_effective_pairs": sum(k > death for k in valid), "reason": "initial contradiction on excluded exposure"})
        for k in valid:
            selected = [(arm, i, c) for arm, i, c in order_for(fid, claims[fid])
                        if (arm, i) in first and first[(arm, i)] in valid and first[(arm, i)] < k]
            if not selected:
                continue
            pairs += len(selected)
            claim_keys.update((fid, arm, i) for arm, i, c in selected)
            for arm, i, c in selected:
                eligible_details[(fid, arm, i)] = {"fid": fid, "arm": arm, "idx": i, "type": c["type"],
                                                   "cohort": f["cohort"], "first_contradiction_k": first[(arm, i)]}
            tasks.append(make_judge_task(fid, f, k, selected, f"{fid}:{k}",
                                        {"role": "followup", "original_first_contradiction": {f"{a}:{i}": first[(a, i)] for a, i, c in selected}}))
    controls = {}
    for key, detail in sorted(eligible_details.items()):
        fid, arm, i = key
        k = detail["first_contradiction_k"]
        if (fid, k) not in controls:
            commit = next(c for c in verdicts[fid]["commits"] if c["k"] == k)
            items = [(a, j, claims[fid][a][j]) for a, j in commit["judged"]]
            controls[(fid, k)] = make_judge_task(fid, funcs[fid], k, items, f"death:{fid}:{k}",
                                               {"role": "death_control", "scored_cases": []})
        task = controls[(fid, k)]
        position = next(j for j, item in enumerate(task["items"]) if item["arm"] == arm and item["idx"] == i)
        task["scored_cases"].append({**detail, "statement_position": position})
    followup_calls = len(tasks)
    tasks = list(controls.values()) + tasks
    return {"population": "Previously anchor-supported claims contradicted on an effective touch, both arms and cohorts, with later effective observations.",
            "endpoint": "At least one later supported verdict after the original first contradiction; not independently established semantic resurrection.",
            "policy": "Recheck each original death using its complete original statement context before follow-up. Do not absorb later contradictions: every frozen claim remains in every planned subsequent effective touch. Later dead-only statement grouping differs from original calls; interpret fresh death strata separately.",
            "eligible_claims": len(claim_keys), "claim_version_pairs": pairs, "grouped_calls": len(tasks),
            "grouped_death_control_calls": len(controls), "grouped_followup_calls": followup_calls,
            "eligible_claim_details": list(eligible_details.values()),
            "excluded_initial_placebo_contradictions": excluded, "tasks": tasks}


def docstring_plan(funcs):
    selected = [f for f in funcs.values() if f["cohort"] == "main" and len(f["anchor"].get("docstring") or "") >= 20]
    selected.sort(key=lambda f: f["id"])
    tasks = []
    for f in selected:
        prompt = EXPLAIN_PROMPT.format(path=f["path"], qualname=f["qualname"], text=code_without_docstring(f["anchor"]["text"]))
        original_key = _key("ollama", CONFIG["bulk_model"], CONFIG["seed"], EXPLAIN_SYSTEM, prompt) + ".json"
        tasks.append({"fid": f["id"], "generation_prompt": prompt, "generation_system": EXPLAIN_SYSTEM,
                      "existing_exact_generation_cache_hit": (RESULTS / "llm_cache" / original_key).exists(),
                      "effective_followup_touches": effective(f, load("verdicts.json")[f["id"]]["commits"])})
    return {"population": "65 main-cohort functions with anchor docstrings of at least20characters; paired to their archived explanations.",
            "endpoint": "Anchor support and first detected contradiction through identical effective follow-up; no causal AI-vs-human authorship claim.",
            "design": "Generate from stripped code, decompose with unchanged prompt, then judge new frozen claims at anchor and all effective touches. Original docstring judgments/claims remain an archived comparator.",
            "warning": "Original model digest/quantization unknown; differences may combine generator-docstring and runtime/weight effects. Claim content and granularity differ.",
            "coverage_proxy": "Case-sensitive lexical mention of anchor parser parameter names and final helper-call identifier components; report fractions separately and pooled. This is not semantic or human task-relevance coverage.",
            "optional_fresh_control": "Regenerate original with-docstring prompt using the same fresh backend for paired lexical coverage only. This control does not add decomposition or survival trajectories, and cannot identify a survival effect.",
            "functions": len(tasks), "generation_calls": len(tasks), "decomposition_calls": len(tasks),
            "max_judge_calls": sum(1 + len(t["effective_followup_touches"]) for t in tasks),
            "tasks": tasks}


def cross_family_plan(funcs, claims, verdicts):
    import csv
    path = ART / "validation" / "private" / "sampling_key.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    strata = collections.defaultdict(list)
    for row in rows:
        strata[(row["stage"], row["bulk_status"])].append(row)
    rng, cases = random.Random(20261111), []
    for stratum in sorted(strata):
        frame = sorted(strata[stratum], key=lambda r: r["task_id"])
        sample = rng.sample(frame, min(10, len(frame)))
        for row in sample:
            fid, arm, i, k = row["fid"], row["arm"], int(row["idx"]), int(row["mining_k"])
            probability = float(row["selection_probability"]) * len(sample) / len(frame)
            cases.append({"task_id": row["task_id"], "fid": fid, "arm": arm, "idx": i, "mining_k": k,
                          "stage": row["stage"], "bulk_status": row["bulk_status"],
                                         "original_400_selection_probability": float(row["selection_probability"]),
                                         "pilot_conditional_selection_probability": len(sample) / len(frame),
                                         "selection_probability": probability, "weight": 1 / probability,
                          "original_stratum_population_n": int(row["population_n"])})
    rng.shuffle(cases)
    groups = {}
    for case in cases:
        fid, k = case["fid"], case["mining_k"]
        key = (fid, k)
        if key not in groups:
            if k == 0:
                items = order_for(fid, claims[fid])
            else:
                commit = next(c for c in verdicts[fid]["commits"] if c["k"] == k)
                items = [(arm, i, claims[fid][arm][i]) for arm, i in commit["judged"]]
            groups[key] = make_judge_task(fid, funcs[fid], k, items, f"{fid}:{k}", {"sampled_cases": []})
        task = groups[key]
        position = next(i for i, item in enumerate(task["items"]) if item["arm"] == case["arm"] and item["idx"] == case["idx"])
        task["sampled_cases"].append({**case, "statement_position": position})
    tasks = list(groups.values())
    return {"population": "Recorded-judgment frame of the400-case stratified validation packet; both arms/cohorts.",
            "endpoint": "Cross-family disagreement with archived bulk verdict, not human-ground-truth accuracy.",
            "pilot": True, "seed": 20261111, "cases": len(cases), "grouped_calls": len(tasks),
            "design": "10cases per six stage/verdict strata, nested in400-case sample. Reconstruct the complete original shuffled anchor or alive-only commit statement list in its original positions and identical parser/diff context. Share identical function/version prompts across selected cases; score only selected cases, other claims are auxiliary.",
            "weighting": "Inverse unconditional inclusion probability=(n400/Nstratum)*(10/n400); preserves unequal stratum sizes. No sampling over unobserved post-death versions.",
            "analysis": {"uncertainty": "Function-cluster bootstrap of inverse-probability-weighted agreement and binary contradiction kappa; descriptive approximation ignores finite-population and full stratified two-stage design complexity.",
                         "bootstrap_seed": 20261112, "bootstrap_B": 1000, "adaptive_early_stopping": False,
                         "outputs": "Three-way weighted confusion, weighted agreement, binary contradiction kappa, both conditional disagreement directions. Neither judge is ground truth."},
            "tasks": tasks}


def freeze_plans(revise_unrun=False):
    funcs, claims, verdicts = inputs()
    specs = {"resurrection": resurrection_plan(funcs, claims, verdicts),
             "docstring_free": docstring_plan(funcs), "cross_family": cross_family_plan(funcs, claims, verdicts)}
    for name, details in specs.items():
        plan = {"experiment": name, "planned_before_fresh_calls": True, "model": MODELS[name],
                "options": OPTIONS, "source_sha256": source_hashes(),
                "quality_policy": "Reject prompt_eval_count+requested_output>=context. Record done_reason=length; reject length-limited explanations, require complete valid claim arrays, retain every raw failed response. Context-count checks cannot prove no upstream truncation.", **details}
        plan["plan_sha256"] = digest(plan)
        path = EXPERIMENTS / "plans" / (name + ".json")
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != plan:
                directory = EXPERIMENTS / name
                if not revise_unrun or (directory / "runtime.json").exists() or (directory / "observations.json").exists() or list((directory / "cache").glob("*.json")):
                    raise RuntimeError(f"Frozen plan changed: {path}; create a separately named protocol revision rather than overwriting.")
                archived = path.with_name(path.stem + ".superseded-precall-" + previous["plan_sha256"][:12] + ".json")
                if archived.exists():
                    raise RuntimeError("Pre-call supersession target already exists")
                os.replace(path, archived)
                atomic_json(path, plan)
        else:
            atomic_json(path, plan)
        (EXPERIMENTS / name / "cache").mkdir(parents=True, exist_ok=True)
    return specs


class BudgetExhausted(Exception):
    pass


class Client:
    def __init__(self, name, endpoint, max_calls):
        self.directory = EXPERIMENTS / name
        self.endpoint = endpoint.rstrip("/")
        self.max_calls, self.calls, self.hits = max_calls, 0, 0
        self.plan = json.loads((EXPERIMENTS / "plans" / (name + ".json")).read_text(encoding="utf-8"))
        self.model = self.plan["model"]
        self.fingerprint = None

    def request(self, route, body=None, timeout=30):
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.endpoint + route, data=payload,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def fingerprint_model(self):
        if self.fingerprint is not None:
            return self.fingerprint
        tags = self.request("/api/tags")
        model = next((m for m in tags.get("models", []) if self.model in (m.get("name"), m.get("model"))), None)
        if model is None or not model.get("digest"):
            raise RuntimeError(f"Required local model/digest unavailable: {self.model}")
        show = self.request("/api/show", {"model": self.model})
        fingerprint = {"model": self.model, "digest": model["digest"], "details": model.get("details"),
                       "show": show, "endpoint": self.endpoint, "options": self.plan["options"],
                       "original_digest_known": False}
        path = self.directory / "runtime.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous["digest"] != fingerprint["digest"] or previous["options"] != fingerprint["options"]:
                raise RuntimeError("Runtime/model identity changed during resume; use a separate experiment namespace")
        else:
            atomic_json(path, fingerprint)
        self.fingerprint = fingerprint
        return fingerprint

    def generate(self, task_id, prompt, system, num_predict):
        # Task-index cache preserves resume even when server is unavailable.
        task_hash = digest({"plan": self.plan["plan_sha256"], "task_id": task_id, "prompt": prompt,
                            "system": system, "num_predict": num_predict})
        path = self.directory / "cache" / (task_hash + ".json")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["task_hash"] != task_hash:
                raise RuntimeError("Cache task identity mismatch")
            self.hits += 1
            return cached["response"]
        if self.calls >= self.max_calls:
            raise BudgetExhausted(f"Fresh-call budget {self.max_calls} exhausted")
        fingerprint = self.fingerprint_model()
        self.calls += 1  # count attempted inference even when network/parsing fails
        started = time.monotonic()
        body = {"model": self.model, "prompt": prompt, "system": system, "stream": False,
                "options": {**self.plan["options"], "num_predict": num_predict}, "keep_alive": "15m"}
        attempt = {"task_id": task_id, "task_hash": task_hash, "model_digest": fingerprint["digest"],
                   "request": body, "started_unix": time.time()}
        try:
            response = self.request("/api/generate", body, timeout=1800)
            attempt["response"] = response
            attempt["quality"] = {"done_reason": response.get("done_reason"),
                                  "output_reached_limit": response.get("done_reason") == "length"}
            if not isinstance(response.get("response"), str) or response.get("done") is not True:
                raise RuntimeError("Incomplete or malformed inference response")
            evaluated = response.get("prompt_eval_count")
            attempt["context_check"] = {"prompt_eval_count": evaluated, "num_ctx": self.plan["options"]["num_ctx"],
                                        "limitation": "Sub-context counts do not prove absence of server-side prompt truncation."}
            if isinstance(evaluated, int) and evaluated + num_predict >= self.plan["options"]["num_ctx"]:
                raise RuntimeError("Prompt evaluation plus output reserve reached context limit; possible truncation, result rejected")
            if response.get("done_reason") == "length" and (task_id.endswith(":generate") or task_id.endswith(":fresh-control")):
                raise RuntimeError("Explanation reached output length limit; raw output retained, incomplete generation rejected")
            attempt.update(response=response, elapsed_seconds=time.monotonic() - started)
            atomic_json(path, attempt)
            print(f"{self.model} {task_id} done, fresh_calls={self.calls}, seconds={attempt['elapsed_seconds']:.1f}", flush=True)
            return response
        except Exception as error:
            attempt.update(error=repr(error), elapsed_seconds=time.monotonic() - started)
            atomic_json(self.directory / "errors" / (task_hash + ".json"), attempt)
            raise

    def quality_counts(self):
        valid = [json.loads(p.read_text(encoding="utf-8")) for p in (self.directory / "cache").glob("*.json")]
        failed = [json.loads(p.read_text(encoding="utf-8")) for p in (self.directory / "errors").glob("*.json")]
        return {"cached_responses": len(valid), "cached_length_limited_responses": sum(r.get("quality", {}).get("output_reached_limit", False) for r in valid),
                "failed_request_records": len(failed), "rejected_length_limited_responses": sum(r.get("quality", {}).get("output_reached_limit", False) for r in failed)}


def judge(client, task):
    raw = client.generate(task["task_id"], task["prompt"], task["system"], task["num_predict"])
    parsed = parse_statuses(raw["response"], len(task["items"]))
    return [{**item, **parsed[f"S{i + 1}"]} for i, item in enumerate(task["items"])]


def weighted_agreement_stats(rows):
    """Rows are (fid, bulk_status, second_status, inverse_inclusion_weight)."""
    weight = sum(row[3] for row in rows)
    if not weight:
        return {"agreement": None, "binary_contradiction_kappa": None}
    agreement = sum(w * (a == b) for fid, a, b, w in rows) / weight
    bulk_con = sum(w for fid, a, b, w in rows if a == "contradicted") / weight
    second_con = sum(w for fid, a, b, w in rows if b == "contradicted") / weight
    binary_po = sum(w * ((a == "contradicted") == (b == "contradicted")) for fid, a, b, w in rows) / weight
    pe = bulk_con * second_con + (1 - bulk_con) * (1 - second_con)
    return {"agreement": agreement, "binary_contradiction_kappa": (binary_po - pe) / (1 - pe) if pe < 1 else None}


def cluster_agreement_intervals(rows, seed, count):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row[0]].append(row)
    ids = sorted(groups)
    rng = random.Random(seed)
    values = {"agreement": [], "binary_contradiction_kappa": []}
    for _ in range(count):
        sample = [row for fid in rng.choices(ids, k=len(ids)) for row in groups[fid]]
        result = weighted_agreement_stats(sample)
        for key in values:
            if result[key] is not None:
                values[key].append(result[key])
    def quantile(vals, q):
        vals = sorted(vals)
        position = (len(vals) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(vals) - 1)
        return vals[lower] + (position - lower) * (vals[upper] - vals[lower])
    return {key: {"low": quantile(vals, .025), "high": quantile(vals, .975), "valid_replicates": len(vals)} if vals else None
            for key, vals in values.items()}


def run_judgments(name, client):
    plan = client.plan
    path = client.directory / "observations.json"
    observations = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    reason, failure = None, None
    try:
        for task in plan["tasks"]:
            if task["task_id"] in observations:
                continue
            observations[task["task_id"]] = {"fid": task["fid"], "mining_k": task["mining_k"],
                                             "verdicts": judge(client, task)}
            atomic_json(path, observations)
    except BudgetExhausted as error:
        reason = str(error)
    except Exception as error:
        reason, failure = str(error), repr(error)
    report = {"experiment": name, "plan_sha256": plan["plan_sha256"], "complete": len(observations) == len(plan["tasks"]),
              "completed_tasks": len(observations), "planned_tasks": len(plan["tasks"]),
              "fresh_calls_this_invocation": client.calls, "cache_hits_this_invocation": client.hits,
              "stop_reason": reason, "failed": failure is not None, "error": failure, "quality_counts": client.quality_counts(),
              "unparsed_verdicts": sum(not a.get("parsed", True) for v in observations.values() for a in v["verdicts"])}
    if name == "resurrection":
        eligible = {(c["fid"], c["arm"], c["idx"]) for c in plan["eligible_claim_details"]}
        observed, regained = set(), set()
        controls = {}
        for task in plan["tasks"]:
            if task.get("role") != "death_control" or task["task_id"] not in observations:
                continue
            for case in task["scored_cases"]:
                value = observations[task["task_id"]]["verdicts"][case["statement_position"]]
                controls[(case["fid"], case["arm"], case["idx"])] = value["status"]
        followup_ids = {t["task_id"] for t in plan["tasks"] if t.get("role") == "followup"}
        for task_id, value in observations.items():
            if task_id not in followup_ids:
                continue
            for a in value["verdicts"]:
                key = (value["fid"], a["arm"], a["idx"])
                observed.add(key)
                if a["status"] == "supported" and a.get("parsed"):
                    regained.add(key)
        strata = {}
        for field in ("arm", "cohort", "type", "fresh_death_status"):
            groups = collections.defaultdict(set)
            for detail in plan["eligible_claim_details"]:
                key = (detail["fid"], detail["arm"], detail["idx"])
                label = controls.get(key, "not_observed") if field == "fresh_death_status" else detail[field]
                groups[label].add(key)
            strata[field] = {label: {"eligible_claims": len(keys), "observed_followup_claims": len(keys & observed),
                                     "later_supported_claims": len(keys & regained)} for label, keys in groups.items()}
        report.update(eligible_claims=len(eligible), claims_with_fresh_death_recheck=len(controls),
                      claims_with_at_least_one_followup=len(observed), stratified_claim_counts=strata,
                      claims_with_later_supported_verdict=len(regained),
                      observed_claim_fraction_later_supported=len(regained) / len(observed) if observed else None,
                      interpretation="Later judge support after original contradiction; partial fractions are exposure-dependent and are not semantic resurrection rates.")
    else:
        weighted, total_weight, count, scored, unparsed = 0, 0, 0, 0, 0
        matrix = {a + "->" + b: 0.0 for a in ("supported", "contradicted", "unresolved")
                  for b in ("supported", "contradicted", "unresolved")}
        scored_rows = []
        for task in plan["tasks"]:
            if task["task_id"] not in observations:
                continue
            for case in task["sampled_cases"]:
                value = observations[task["task_id"]]["verdicts"][case["statement_position"]]
                status = value["status"]
                matrix[case["bulk_status"] + "->" + status] += case["weight"]
                weighted += case["weight"] * (status == case["bulk_status"])
                total_weight += case["weight"]
                count += status == case["bulk_status"]
                scored += 1
                unparsed += not value.get("parsed", True)
                scored_rows.append((task["fid"], case["bulk_status"], status, case["weight"]))
        binary = weighted_agreement_stats(scored_rows)
        bulk_con_total = sum(w for fid, a, b, w in scored_rows if a == "contradicted")
        bulk_not_total = sum(w for fid, a, b, w in scored_rows if a != "contradicted")
        report.update(pilot=True, completed_cases=scored, planned_cases=plan["cases"], unparsed_scored_verdicts=unparsed,
                      weighted_agreement=weighted / total_weight if total_weight else None,
                      raw_agreement=count / scored if scored else None,
                      weighted_confusion_totals=dict(matrix),
                      weighted_binary_contradiction_kappa=binary["binary_contradiction_kappa"],
                      second_not_con_given_bulk_con=sum(w for fid, a, b, w in scored_rows if a == "contradicted" and b != "contradicted") / bulk_con_total if bulk_con_total else None,
                      second_con_given_bulk_not_con=sum(w for fid, a, b, w in scored_rows if a != "contradicted" and b == "contradicted") / bulk_not_total if bulk_not_total else None,
                      descriptive_function_cluster_intervals=(cluster_agreement_intervals(scored_rows, plan["analysis"]["bootstrap_seed"], plan["analysis"]["bootstrap_B"]) if report["complete"] else None),
                      interval_method=plan["analysis"]["uncertainty"],
                      interpretation="LLM agreement, not human accuracy. Partial pilot summaries are descriptive, not design-complete estimates.")
    report["quality_complete"] = report["complete"] and report["unparsed_verdicts"] == 0 and report["quality_counts"]["cached_length_limited_responses"] == 0
    if name == "cross_family":
        # The frozen design scores only sampled cases; auxiliary statements
        # preserve original prompt context and retain their separate quality flag.
        report["scored_quality_complete"] = (report["complete"] and report["completed_cases"] == report["planned_cases"]
                                             and report["unparsed_scored_verdicts"] == 0
                                             and report["quality_counts"]["cached_length_limited_responses"] == 0)
    atomic_json(client.directory / "summary.json", report)
    return report


def new_claim_survival(f, record):
    valid = effective(f, load("verdicts.json")[f["id"]]["commits"])
    anchor = record.get("anchor", [])
    rows = []
    for i, a in enumerate(anchor):
        if a["status"] != "supported":
            continue
        death = next((e for e, k in enumerate(valid, 1)
                      if str(k) in record.get("touches", {}) and record["touches"][str(k)][i]["status"] == "contradicted"), None)
        observed = sum(str(k) in record.get("touches", {}) for k in valid)
        rows.append({"idx": i, "time_k": death if death else observed, "event": int(death is not None)})
    return rows


def lexical_coverage(f, text):
    parameters = sorted(set(f["anchor"].get("params", [])))
    helpers = sorted({re.findall(r"[A-Za-z_]\w*", call)[-1] for call in f["anchor"].get("facts", {}).get("calls", [])
                      if re.findall(r"[A-Za-z_]\w*", call)})
    result = {"interpretation": "Lexical parser-identifier proxy, not semantic or task-relevant information coverage."}
    for name, identifiers in (("parameters", parameters), ("helpers", helpers), ("pooled", sorted(set(parameters + helpers)))):
        mentioned = [identifier for identifier in identifiers if re.search(r"(?<!\w)" + re.escape(identifier) + r"(?!\w)", text)]
        result[name] = {"total": len(identifiers), "mentioned": len(mentioned),
                        "fraction": len(mentioned) / len(identifiers) if identifiers else None,
                        "identifiers": identifiers, "mentioned_identifiers": mentioned}
    return result


def run_docstrings(client, limit_functions=None, fresh_control=False):
    funcs, original_claims, original_verdicts = inputs()
    path = client.directory / "observations.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    reason, failure = None, None
    tasks = client.plan["tasks"] if limit_functions is None else client.plan["tasks"][:limit_functions]
    try:
        for task in tasks:
            fid, f = task["fid"], funcs[task["fid"]]
            rec = records.get(fid, {"touches": {}})
            if "explanation" not in rec:
                response = client.generate(fid + ":generate", task["generation_prompt"], task["generation_system"], 600)
                rec["explanation"] = response["response"].strip()
                records[fid] = rec
                atomic_json(path, records)
            if fresh_control and "fresh_with_docstring_explanation" not in rec:
                prompt = EXPLAIN_PROMPT.format(path=f["path"], qualname=f["qualname"], text=f["anchor"]["text"])
                response = client.generate(fid + ":fresh-control", prompt, EXPLAIN_SYSTEM, 600)
                rec["fresh_with_docstring_explanation"] = response["response"].strip()
                atomic_json(path, records)
            rec["coverage_docstring_free"] = lexical_coverage(f, rec["explanation"])
            rec["coverage_archived_with_docstring"] = lexical_coverage(f, load("explanations.json")[fid]["explanation"])
            if "fresh_with_docstring_explanation" in rec:
                rec["coverage_fresh_with_docstring"] = lexical_coverage(f, rec["fresh_with_docstring_explanation"])
            if "claims" not in rec:
                prompt = CLAIM_PROMPT.format(qualname=f["qualname"], text=rec["explanation"])
                response = client.generate(fid + ":decompose", prompt, CLAIM_SYSTEM, 1200)
                parsed = parse_claims(response["response"])
                if not parsed:
                    raise RuntimeError(f"No claims parsed for {fid}; raw output retained, not silently dropped")
                raw_claims = extract_json(response["response"])
                if isinstance(raw_claims, dict):
                    raw_claims = raw_claims.get("claims") or (list(raw_claims.values())[0] if raw_claims else [])
                if not isinstance(raw_claims, list) or len(raw_claims) > 12 or len(raw_claims) != len(parsed):
                    raise RuntimeError(f"Malformed or excess claim objects for {fid}; raw output retained, no silent omission")
                rec["claims"] = parsed
                rec["decomposition_claims_at_cap"] = len(parsed) == 12
                atomic_json(path, records)
            items = [("docstring_free", i, c) for i, c in enumerate(rec["claims"])]
            # Keep the new claims fixed at every version: no disappearance after contradiction.
            for k in [0] + task["effective_followup_touches"]:
                if (k == 0 and "anchor" in rec) or str(k) in rec["touches"]:
                    continue
                job = make_judge_task(fid, f, k, items, f"{fid}:judge:{k}")
                values = judge(client, job)
                if k == 0:
                    rec["anchor"] = values
                else:
                    rec["touches"][str(k)] = values
                atomic_json(path, records)
            rec["complete"] = True
            rec["new_claim_survival"] = new_claim_survival(f, rec)
            rec["archived_comparator"] = {"explanation": load("explanations.json")[fid],
                                           "claims": original_claims[fid], "verdicts": original_verdicts[fid]}
            atomic_json(path, records)
    except BudgetExhausted as error:
        reason = str(error)
    except Exception as error:
        reason, failure = str(error), repr(error)
    completed = [fid for fid, rec in records.items() if rec.get("complete")]
    anchor_total = sum(len(records[fid]["anchor"]) for fid in completed)
    anchor_supported = sum(a["status"] == "supported" for fid in completed for a in records[fid]["anchor"])
    pairs = [{"fid": fid,
              "docstring_free_fraction": records[fid]["coverage_docstring_free"]["pooled"]["fraction"],
              "archived_with_docstring_fraction": records[fid]["coverage_archived_with_docstring"]["pooled"]["fraction"],
              "fresh_with_docstring_fraction": records[fid].get("coverage_fresh_with_docstring", {}).get("pooled", {}).get("fraction")}
             for fid in completed]
    new_rows = [(row["time_k"], row["event"]) for fid in completed for row in records[fid]["new_claim_survival"]]
    import csv
    with (RESULTS / "claims_survival.csv").open(encoding="utf-8", newline="") as stream:
        old_rows = [row for row in csv.DictReader(stream) if row["fid"] in completed and row["arm"] == "llm" and row["anchor_status"] == "supported"]
    original_rows = [(int(row["time_k"]), int(row["event"])) for row in old_rows]
    def mean_difference(comparator):
        deltas = [p["docstring_free_fraction"] - p[comparator] for p in pairs
                  if p["docstring_free_fraction"] is not None and p[comparator] is not None]
        return {"paired_functions": len(deltas), "mean_docstring_free_minus_comparator": sum(deltas) / len(deltas) if deltas else None}
    control_complete = [fid for fid in completed if "fresh_with_docstring_explanation" in records[fid]]
    report = {"experiment": "docstring_free", "plan_sha256": client.plan["plan_sha256"],
              "complete": len(completed) == len(client.plan["tasks"]) and (not fresh_control or len(control_complete) == len(client.plan["tasks"])), "planned_functions": len(client.plan["tasks"]),
              "survival_target_complete": len(completed) == len(client.plan["tasks"]),
              "fresh_generation_control_completed_functions": len(control_complete),
              "completed_functions": len(completed), "started_functions": len(records),
              "requested_function_limit": limit_functions, "requested_scope_complete": all(records.get(t["fid"], {}).get("complete") and (not fresh_control or "fresh_with_docstring_explanation" in records[t["fid"]]) for t in tasks),
              "fresh_control_requested": fresh_control, "paired_lexical_coverage": pairs,
              "new_claims_at_anchor": anchor_total, "new_anchor_supported": anchor_supported,
              "new_anchor_support_fraction": anchor_supported / anchor_total if anchor_total else None,
              "descriptive_S6_completed_functions": {"docstring_free": km(new_rows)[6] if new_rows else None,
                                                       "archived_with_docstring": km(original_rows)[6] if original_rows else None,
                                                       "new_claims": len(new_rows), "archived_claims": len(original_rows),
                                                       "limitation": "Descriptive claim-mix comparison on completed functions, not content-matched or independently validated; partial prefixes are not representative."},
              "coverage_comparison_archived": mean_difference("archived_with_docstring_fraction"),
              "coverage_comparison_fresh_control": mean_difference("fresh_with_docstring_fraction"),
              "fresh_calls_this_invocation": client.calls, "cache_hits_this_invocation": client.hits, "stop_reason": reason,
              "failed": failure is not None, "error": failure,
              "quality_counts": client.quality_counts(),
              "new_claims_completed_functions": sum(len(records[fid]["claims"]) for fid in completed),
              "unparsed_judge_verdicts": sum(not a.get("parsed", True) for rec in records.values()
                                             for values in [rec.get("anchor", [])] + list(rec["touches"].values()) for a in values),
              "interpretation": "Paired archived-vs-fresh docstring-free descriptions. Not a clean causal docstring ablation: original digest unknown and runtime or weights may differ. Human validation absent."}
    report["quality_complete"] = report["complete"] and report["unparsed_judge_verdicts"] == 0 and report["quality_counts"]["cached_length_limited_responses"] == 0
    atomic_json(client.directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", choices=NAMES)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--limit-functions", type=int, help="Explicit docstring pilot prefix; full-target completion remains separate")
    parser.add_argument("--fresh-control", action="store_true", help="Add fresh with-docstring generation for matched lexical coverage only")
    parser.add_argument("--revise-unrun-plans", action="store_true", help="Archive and supersede changed plans only before any runtime or result exists")
    args = parser.parse_args()
    if args.max_calls < 0:
        parser.error("--max-calls must be nonnegative")
    if args.limit_functions is not None and args.limit_functions < 1:
        parser.error("--limit-functions must be positive")
    plans = freeze_plans(args.revise_unrun_plans)
    if not args.run:
        print(json.dumps({name: {k: v for k, v in spec.items() if k not in ("tasks", "eligible_claim_details")} for name, spec in plans.items()}, indent=2))
        return
    client = Client(args.run, args.endpoint, args.max_calls)
    try:
        report = run_docstrings(client, args.limit_functions, args.fresh_control) if args.run == "docstring_free" else run_judgments(args.run, client)
        print(json.dumps(report, indent=2))
        if report.get("failed"):
            atomic_json(client.directory / "last_failure.json", {"error": report["error"], "fresh_calls_this_invocation": client.calls,
                                                                 "plan_sha256": client.plan["plan_sha256"]})
            raise SystemExit(1)
    except Exception as error:
        atomic_json(client.directory / "last_failure.json", {"error": repr(error), "fresh_calls_this_invocation": client.calls,
                                                             "plan_sha256": client.plan["plan_sha256"]})
        raise


if __name__ == "__main__":
    main()
