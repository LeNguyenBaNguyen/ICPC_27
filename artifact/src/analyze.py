"""Stage 5 (round 2): claim-level survival analysis for the MAIN (churn-selected) and RANDOM (unconditioned) cohorts.

Primary model: discrete-time complementary log-log regression (grouped-time proportional hazards; touch-index period
effects) with function-clustered sandwich covariance (stats_core.fit_cloglog). Robustness: unpenalised lifelines Cox
(counting-process form with time-varying covariates, Efron ties, function-clustered robust SE). Kaplan-Meier with right
censoring at the follow-up cap, repository head, or function/file removal. Function-level cluster bootstrap elsewhere.
Writes results/*.csv and results/summary.json."""
import json, math, os, re, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, RESULTS, jload, jdump
from stats_core import fit_cloglog, wald, hr_ci, km_int, quantile_time, holm, auc, kappa
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

K = CONFIG["follow_k"]; KMAX = CONFIG["k_max"]
TYPES = ["structural", "dataflow", "behavioural", "intent"]
CT = ["signature", "call_change", "call_receiver", "exception", "control_flow", "return_value",
      "identifier_rename", "literal_or_keyword", "local_edit"]
B = 1000
rng = np.random.RandomState(CONFIG["seed"])

def is_compound(text):
    """Enumerative claim: names >= 3 code items, or lists parameters/helpers with >= 2 commas."""
    if len(re.findall(r"`[^`]+`", text)) >= 3:
        return True
    return bool(re.search(r"\b(parameters?|arguments?|params|helpers?|methods|functions|attributes)\b", text, re.I)) and text.count(",") >= 2

def period_cat(k):
    return str(k) if k <= 6 else ("7-9" if k <= 9 else "10-12")

# ------------------------------------------------------------------ tables
def effective_touches(f, n_judged):
    """Mining touch indices (1-based) that count as touches: format-only touches (quote style, trailing commas) are
    placebo exposures, and follow-up ends before a touch whose tracked version is a later-added @overload stub."""
    valid = []
    for k in range(1, n_judged + 1):
        t = f["touches"][k - 1]
        if t.get("overload_stub"):
            return valid, "overload_stub"
        if not (t.get("format_only") or t.get("paren_only")):
            valid.append(k)
    return valid, None

def build(funcs, verdicts, claims, stats):
    import datetime as dt
    rows, periods = [], []
    for fid in sorted(verdicts):
        if fid not in funcs:
            continue
        v = verdicts[fid]; f = funcs[fid]; st = stats[f["repo"]]
        a_date = dt.date.fromisoformat(st["anchor_date"]); h_date = dt.date.fromisoformat(st["head_date"])
        valid, stop = effective_touches(f, len(v["commits"]))
        eff = {k: e for e, k in enumerate(valid, start=1)}
        n_obs = len(valid)
        if stop == "overload_stub":
            reason = "overload_stub"; c_date = dt.date.fromisoformat(f["touches"][valid[-1] - 1]["date"]) if valid else a_date
        elif len(v["commits"]) == KMAX:
            reason = "cap"; c_date = dt.date.fromisoformat(f["touches"][KMAX - 1]["date"])
        elif f["censor"]["reason"] == "head":
            reason = "head"; c_date = h_date
        else:
            reason = f["censor"]["reason"]; c_date = dt.date.fromisoformat(f["censor"]["date"])
        for a in v["anchor"]:
            key = [a["arm"], a["idx"]]
            cl = claims[fid][a["arm"]][a["idx"]]
            traj = {c["k"]: c["verdicts"][c["judged"].index(key)]["status"] for c in v["commits"] if key in c["judged"]}
            first_dead_k = next((k for k in sorted(traj) if traj[k] == "contradicted"), None)
            sup = a["status"] == "supported"
            event = int(sup and first_dead_k is not None and first_dead_k in eff)
            spurious_format_death = int(sup and first_dead_k is not None and first_dead_k not in eff and (stop is None or first_dead_k <= (valid[-1] if valid else 0)))
            if event:
                t = eff[first_dead_k]
            elif first_dead_k is not None and sup:   # contradicted at a format-only touch or after the follow-up end
                t = len([k for k in valid if k < first_dead_k])
            else:
                t = n_obs
            death_touch = f["touches"][first_dead_k - 1] if event else None
            days = (dt.date.fromisoformat(death_touch["date"]) - a_date).days if event else (c_date - a_date).days
            rows.append({"fid": fid, "cohort": f["cohort"], "repo": f["repo"], "arm": a["arm"], "idx": a["idx"], "type": a["type"],
                         "compound": int(is_compound(cl["text"])), "anchor_status": a["status"], "anchor_parsed": a.get("parsed", True),
                         "event": event, "time_k": t, "n_obs": n_obs, "censor_reason": "" if event else reason, "days": days,
                         "format_death": spurious_format_death,
                         "n_unresolved": sum(1 for s_ in traj.values() if s_ == "unresolved"),
                         "n_unparsed": sum(1 for c in v["commits"] if key in c["judged"] and not c["verdicts"][c["judged"].index(key)].get("parsed", True)),
                         "death_change_types": "|".join(death_touch["change_types"]) if death_touch else "", "text": cl["text"]})
            if sup:
                for e in range(1, t + 1):
                    tch = f["touches"][valid[e - 1] - 1]
                    p = {"fid": fid, "cohort": f["cohort"], "repo": f["repo"], "claim": f"{fid}:{a['arm']}:{a['idx']}", "arm": a["arm"],
                         "type": a["type"], "compound": int(is_compound(cl["text"])), "k": e, "pcat": period_cat(e),
                         "event": int(event and e == t), "log_lines_fn": math.log1p(tch["lines_changed_fn"]),
                         "log_lines_file": math.log1p(tch["lines_changed_file"]), "lines_fn": tch["lines_changed_fn"],
                         "lines_file": tch["lines_changed_file"]}
                    for ct in CT:
                        p["ct_" + ct] = int(ct in tch["change_types"])
                    periods.append(p)
    return pd.DataFrame(rows), pd.DataFrame(periods)

def design(P, covs, pcats=None):
    pcats = pcats or sorted(P.pcat.unique(), key=lambda x: (len(x), x))
    cols = ["const"] + ["k_" + c for c in pcats[1:]] + covs
    X = np.column_stack([np.ones(len(P))] + [(P.pcat == c).astype(float).values for c in pcats[1:]] +
                        [P[c].astype(float).values for c in covs])
    return X, cols

def cloglog(P, covs, groups="fid"):
    X, cols = design(P, covs)
    fit = fit_cloglog(X, P.event.values, P[groups].values)
    fit["cols"] = cols
    return fit

def idx(fit, names):
    return [fit["cols"].index(n) for n in names if n in fit["cols"]]

def cluster_boot(df, stat, groups="fid", nb=B, return_vals=False):
    ids = df[groups].unique(); n = len(ids)
    by = {g: sub for g, sub in df.groupby(groups)}
    vals = []
    for _ in range(nb):
        pick = rng.choice(ids, n, replace=True)
        try:
            vals.append(stat(pd.concat([by[g] for g in pick], ignore_index=True)))
        except Exception:
            vals.append(np.nan)
    vals = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if return_vals:
        return vals
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)

def S_at(df, k):
    return km_int(df.time_k.values, df.event.values, KMAX)[0][k]

def km_block(df, label, S):
    """KM summary for one claim set (anchor-supported claims): S(K) + CI, t_0.9 + CI, extrapolated half-life."""
    curve, haz = km_int(df.time_k.values, df.event.values, KMAX)
    lo, hi = cluster_boot(df, lambda s: S_at(s, K))
    t90 = quantile_time(curve, 0.10)
    bt = cluster_boot(df, lambda s: (quantile_time(km_int(s.time_k.values, s.event.values, KMAX)[0], 0.10) or (KMAX + 1)), return_vals=True)
    exposures = int(df.time_k.clip(lower=0).sum()); deaths = int(df.event.sum())
    h = deaths / exposures if exposures else float("nan")
    hl = math.log(2) / -math.log(1 - h) if h and h > 0 else float("nan")
    bh = cluster_boot(df, lambda s: math.log(2) / -math.log(1 - s.event.sum() / s.time_k.clip(lower=0).sum()), return_vals=True)
    S[f"{label}_n"] = int(len(df)); S[f"{label}_deaths"] = deaths; S[f"{label}_exposures"] = exposures
    S[f"{label}_S_K"] = curve[K]; S[f"{label}_S_K_lo"] = lo; S[f"{label}_S_K_hi"] = hi
    S[f"{label}_S_Kmax"] = curve[KMAX]
    S[f"{label}_t90"] = t90
    S[f"{label}_t90_lo"] = float(np.percentile(bt, 2.5)) if len(bt) else float("nan")
    S[f"{label}_t90_hi"] = float(np.percentile(bt, 97.5)) if len(bt) else float("nan")
    S[f"{label}_hazard_per_touch"] = 100 * h
    S[f"{label}_halflife_extrap"] = hl
    S[f"{label}_halflife_extrap_lo"] = float(np.percentile(bh, 2.5)) if len(bh) else float("nan")
    S[f"{label}_halflife_extrap_hi"] = float(np.percentile(bh, 97.5)) if len(bh) else float("nan")
    S[f"{label}_at_risk_K"] = int(haz[K - 1][1]); S[f"{label}_at_risk_Kmax"] = int(haz[KMAX - 1][1])
    S[f"{label}_at_risk_1"] = int(haz[0][1])
    S[f"{label}_funcs_at_risk_1"] = int(df[df.time_k >= 1].fid.nunique()); S[f"{label}_funcs_at_risk_K"] = int(df[df.time_k >= K].fid.nunique())
    # reliability rule for t_0.9: report only if >= 30 claims from >= 5 functions are at risk at that touch
    if t90 is not None:
        ok = int(np.sum(df.time_k.values >= t90)) >= 30 and int(df[df.time_k >= t90].fid.nunique()) >= 5
        S[f"{label}_t90_reliable"] = bool(ok)
        if not ok:
            S[f"{label}_t90"] = None
    else:
        S[f"{label}_t90_reliable"] = True
    return curve

# ------------------------------------------------------------------ identifier-existence comparator (no LLM)
_tick = re.compile(r"`([^`]+)`")
_codeword = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|\b[a-z0-9]+_[a-z0-9_]+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b")

def names_in(text):
    from mine import code_tokens
    return {s for t, s in code_tokens(text) if re.match(r"[A-Za-z_]\w*$", s)}

def claim_identifiers(text, anchor_names):
    raw = set()
    for m in _tick.findall(text):
        raw |= set(re.findall(r"[A-Za-z_]\w*", m))
    for m in _codeword.findall(text):
        raw |= set(re.findall(r"[A-Za-z_]\w*", m))
    return {x for x in raw if x in anchor_names and len(x) >= 3}

def identifier_flags(funcs, verdicts, claims):
    """For each main-cohort function: per version k, the set of LLM claim indices with an identifier missing."""
    out = {}
    for fid in sorted(verdicts):
        f = funcs.get(fid)
        if f is None or f["cohort"] != "main":
            continue
        an = names_in(f["anchor"]["text"])
        ids = {i: claim_identifiers(c["text"], an) for i, c in enumerate(claims[fid]["llm"])}
        flags = {}
        for k, t in enumerate(f["touches"], start=1):
            nn = names_in(t["new"]["text"])
            flags[k] = {i for i, s in ids.items() if s and not s <= nn}
        out[fid] = {"ids": ids, "flags": flags}
    return out

H2_BOOT_B = 2000

def _h2_boot_one(args):
    P, covs, tcols, b0, seed, nb = args
    from scipy.stats import chi2 as _chi2
    brs = np.random.RandomState(seed); ids = P.fid.unique(); byP = {g: d for g, d in P.groupby("fid")}
    bb, bbp = [], []; skips = {"no_intent_death": 0, "period_only": 0, "diverged": 0}
    for _ in range(nb):
        Pb = pd.concat([byP[g] for g in brs.choice(ids, len(ids), replace=True)], ignore_index=True)
        if Pb.loc[Pb.ty_intent == 1, "event"].sum() == 0:
            skips["no_intent_death"] += 1; continue
        fb = cloglog(Pb, covs)
        if not np.all(np.abs(fb["beta"][idx(fb, covs)]) < 15):
            skips["diverged"] += 1; continue
        if not np.all(np.abs(fb["beta"][1:]) < 15):   # separation only on a period effect (nuisance)
            skips["period_only"] += 1; bbp.append(fb["beta"][idx(fb, tcols)]); continue
        bb.append(fb["beta"][idx(fb, tcols)])
    bb = np.array(bb); V = np.cov(bb.T); st = float(b0 @ np.linalg.solve(V, b0))
    bb2 = np.vstack([bb] + ([np.array(bbp)] if bbp else [])); V2 = np.cov(bb2.T); st2 = float(b0 @ np.linalg.solve(V2, b0))
    return {"seed": seed, "stat": st, "p": float(_chi2.sf(st, len(tcols))), "stat2": st2, "p2": float(_chi2.sf(st2, len(tcols))), "skips": skips}

def h2_boot_parallel(P, covs, tcols, b0, seeds, nb):
    from multiprocessing import Pool
    with Pool(min(len(seeds), 5)) as pool:
        return pool.map(_h2_boot_one, [(P, covs, tcols, b0, sd, nb) for sd in seeds])

# ------------------------------------------------------------------ main
def main():
    S = {"config": {k: v for k, v in CONFIG.items() if k != "pins"}}
    main_f = {f["id"]: f for f in jload("functions.json")}
    rand_f = {f["id"]: f for f in jload("random_functions.json")}
    funcs = {**main_f, **rand_f}
    verdicts = jload("verdicts.json"); claims = jload("claims.json"); stats = jload("mining_stats.json")
    df, pdf = build(funcs, verdicts, claims, stats)
    df.drop(columns=["text"]).to_csv(os.path.join(RESULTS, "claims_survival.csv"), index=False)
    pdf.to_csv(os.path.join(RESULTS, "claim_periods.csv"), index=False)
    M = df[df.cohort == "main"]; R = df[df.cohort == "random"]
    Mc = M[M.anchor_status == "supported"]; Rc = R[R.anchor_status == "supported"]
    llm = Mc[Mc.arm == "llm"]; doc = Mc[Mc.arm == "doc"]; rllm = Rc[Rc.arm == "llm"]
    P = pdf[(pdf.cohort == "main") & (pdf.arm == "llm")].copy()

    # ---------------- dataset
    mf = [main_f[x] for x in sorted(main_f) if x in verdicts]; rf = [rand_f[x] for x in sorted(rand_f) if x in verdicts]
    pooled = stats["_pooled"]
    S.update({"n_functions": len(mf), "n_repos": len({f["repo"] for f in mf}), "n_pairs": int(sum(len(f["touches"]) for f in mf)),
              "n_pairs_K": int(sum(min(K, len(f["touches"])) for f in mf)),
              "n_distinct_commits": len({(f["repo"], t["commit"]) for f in mf for t in f["touches"]}),
              "n_distinct_commits_K": len({(f["repo"], t["commit"]) for f in mf for t in f["touches"][:K]}),
              "n_llm_claims": int((M.arm == "llm").sum()), "n_doc_claims": int((M.arm == "doc").sum()),
              "n_funcs_with_doc": int(M[M.arm == "doc"].fid.nunique()),
              "claims_per_expl": float(M[M.arm == "llm"].groupby("fid").size().mean()),
              "claims_per_doc": float(M[M.arm == "doc"].groupby("fid").size().mean()),
              "n_expl_at_claim_cap": int(sum(1 for x in mf if len(claims[x["id"]]["llm"]) >= 12)),
              "touch_dist": {str(k): int(v) for k, v in pd.Series([len(f["touches"]) for f in mf]).value_counts().sort_index().items()},
              "n_funcs_cap": int(sum(1 for f in mf if len(f["touches"]) == KMAX)),
              "eligible_population": pooled["eligible_population"], "n_candidates_ge3": pooled["candidates_ge3"],
              "n_candidates_ge6": pooled["candidates_ge6"], "n_candidates_ge8": pooled["candidates_ge8"],
              "duplicate_qualname_excluded": int(sum(stats[r].get("duplicate_qualname_excluded", 0) for r in CONFIG["repos"])),
              "median_days_followup": int(np.median([(pd.Timestamp(stats[f["repo"]]["head_date"]) - pd.Timestamp(stats[f["repo"]]["anchor_date"])).days for f in mf])),
              "n_comment_only_placebo": int(sum(1 for f in mf for p in f.get("placebo", []) if p["kind"] == "comment")),
              "n_placebo_versions": int(sum(len(f.get("placebo", [])) for f in mf)),
              "n_identifier_rename": int(sum("identifier_rename" in t["change_types"] for f in mf for t in f["touches"])),
              "n_call_receiver": int(sum("call_receiver" in t["change_types"] for f in mf for t in f["touches"])),
              "n_call_change": int(sum("call_change" in t["change_types"] for f in mf for t in f["touches"])),
              "n_multilabel": int(sum(len([c for c in t["change_types"] if c != "call_receiver"]) > 1 for f in mf for t in f["touches"])),
              "n_file_rename_followed": int(sum(t.get("renamed_file", False) for f in mf for t in f["touches"])),
              "n_main_censored_removed": int(sum(1 for f in mf if f["censor"]["reason"] != "head" and len(f["touches"]) < KMAX)),
              "n_random": len(rf), "n_random_touched": int(sum(1 for f in rf if f["touches"])),
              "n_random_pairs": int(sum(len(f["touches"]) for f in rf)),
              "random_censor": {r: int(n) for r, n in pd.Series([("cap" if len(f["touches"]) == KMAX else f["censor"]["reason"]) for f in rf]).value_counts().items()},
              "n_random_llm_claims": int((R.arm == "llm").sum())})
    tabrows = []
    for repo in CONFIG["repos"]:
        fl = [f for f in mf if f["repo"] == repo]
        if not fl:
            continue
        sub = M[M.repo == repo]
        tabrows.append({"repo": repo, "anchor": CONFIG["pins"][repo]["anchor"][:7], "functions": len(fl),
                        "pairs": int(sum(len(f["touches"]) for f in fl)),
                        "commits": len({t["commit"] for f in fl for t in f["touches"]}),
                        "with_docstring": int(sub[sub.arm == "doc"].fid.nunique()),
                        "llm_claims": int((sub.arm == "llm").sum()), "doc_claims": int((sub.arm == "doc").sum()),
                        "random": int(sum(1 for f in rf if f["repo"] == repo))})
    others = [r for r in CONFIG["repos"] if r not in {x["repo"] for x in tabrows} and any(f["repo"] == r for f in rf)]
    if others:
        tabrows.append({"repo": "others", "anchor": "", "functions": 0, "pairs": 0, "commits": 0,
                        "with_docstring": 0, "llm_claims": 0, "doc_claims": 0, "random": int(sum(1 for f in rf if f["repo"] in others))})
    pd.DataFrame(tabrows).to_csv(os.path.join(RESULTS, "tab_dataset.csv"), index=False)
    S["eligible_population_clean"] = pooled.get("eligible_population_clean"); S["vendored_in_population"] = pooled.get("vendored_in_population")
    S["vendored_in_random_round2"] = pooled.get("vendored_in_random_round2")
    S["n_format_only_touches"] = int(sum(1 for f in mf for t in f["touches"] if t.get("format_only")))
    S["n_overload_stub_touches"] = int(sum(1 for f in mf + rf for t in f["touches"] if t.get("overload_stub")))
    S["n_effective_pairs"] = int(sum(len(effective_touches(f, len(verdicts[f["id"]]["commits"]))[0]) for f in mf))
    S["n_random_repos"] = len({f["repo"] for f in rf})

    # ---------------- anchor screen
    for arm in ("llm", "doc"):
        a = M[M.arm == arm]
        S[f"anchor_contradicted_pct_{arm}"] = 100 * float((a.anchor_status == "contradicted").mean())
        S[f"anchor_unresolved_pct_{arm}"] = 100 * float((a.anchor_status == "unresolved").mean())
        S[f"anchor_contradicted_n_{arm}"] = int((a.anchor_status == "contradicted").sum())
        S[f"cohort_n_{arm}"] = int((a.anchor_status == "supported").sum())
    for arm, sub in (("llm", llm), ("doc", doc)):
        for ty in TYPES:
            S[f"typemix_{arm}_{ty}_pct"] = 100 * float((sub.type == ty).mean()) if len(sub) else 0.0

    # ---------------- survival: main and random cohorts (Kaplan-Meier, touch time)
    kmrows = []
    for label, sub in (("main", llm), ("random", rllm)):
        curve = km_block(sub, label, S)
        if label == "main":
            bvals = []
            ids = sub.fid.unique(); by = {g: d for g, d in sub.groupby("fid")}
            brs = np.random.RandomState(CONFIG["seed"] + 11)
            for _ in range(500):
                samp = pd.concat([by[g] for g in brs.choice(ids, len(ids), replace=True)], ignore_index=True)
                bvals.append(km_int(samp.time_k.values, samp.event.values, KMAX)[0])
            bvals = np.array(bvals)
            kmrows += [{"series": label, "k": t, "S": curve[t], "lo": float(np.percentile(bvals[:, t], 2.5)),
                        "hi": float(np.percentile(bvals[:, t], 97.5))} for t in range(KMAX + 1)]
        else:
            kmrows += [{"series": label, "k": t, "S": curve[t]} for t in range(KMAX + 1)]
    for ty in TYPES:
        sub = llm[llm.type == ty]
        curve = km_block(sub, f"main_{ty}", S)
        kmrows += [{"series": ty, "k": t, "S": curve[t]} for t in range(KMAX + 1)]
    S["dead_pct_K_main"] = 100 * float(((llm.event == 1) & (llm.time_k <= K)).mean())
    kr = [k for k in range(1, KMAX + 1) if int(np.sum(rllm.time_k.values >= k)) >= 30 and rllm[rllm.time_k >= k].fid.nunique() >= 5]
    S["random_k_reliable"] = max(kr) if kr else 0
    if kr:
        kk = S["random_k_reliable"]
        S["random_S_kr"] = km_int(rllm.time_k.values, rllm.event.values, KMAX)[0][kk]
        S["random_S_kr_lo"], S["random_S_kr_hi"] = cluster_boot(rllm, lambda s_: km_int(s_.time_k.values, s_.event.values, KMAX)[0][kk])
        S["random_at_risk_kr"] = int(np.sum(rllm.time_k.values >= kk)); S["random_funcs_at_risk_kr"] = int(rllm[rllm.time_k >= kk].fid.nunique())
    S["random_at_risk_3"] = int(np.sum(rllm.time_k.values >= 3)); S["random_funcs_at_risk_3"] = int(rllm[rllm.time_k >= 3].fid.nunique())
    # calendar time (days since anchor): KM at 365 days
    from lifelines import KaplanMeierFitter
    from lifelines import AalenJohansenFitter
    REMOVAL = ("file_removed", "function_removed_or_renamed")
    for label, sub in (("main", llm), ("random", rllm)):
        sub = sub.copy(); sub["removed"] = sub.censor_reason.isin(REMOVAL).astype(int)
        sub["ev_rm"] = ((sub.event == 1) | (sub.removed == 1)).astype(int)
        sub["code"] = np.where(sub.event == 1, 1, np.where(sub.removed == 1, 2, 0))
        d = sub.days.clip(lower=0)
        S[f"{label}_S_365d"] = float(KaplanMeierFitter().fit(d, sub.event).predict(365))
        lo, hi = cluster_boot(sub, lambda s: float(KaplanMeierFitter().fit(s.days.clip(lower=0), s.event).predict(365)), nb=300)
        S[f"{label}_S_365d_lo"], S[f"{label}_S_365d_hi"] = lo, hi
        S[f"{label}_S_365d_rmdeath"] = float(KaplanMeierFitter().fit(d, sub.ev_rm).predict(365))
        lo, hi = cluster_boot(sub, lambda s: float(KaplanMeierFitter().fit(s.days.clip(lower=0), s.ev_rm).predict(365)), nb=300)
        S[f"{label}_S_365d_rmdeath_lo"], S[f"{label}_S_365d_rmdeath_hi"] = lo, hi
        for code, nm in ((1, "death"), (2, "removal")):
            aj = AalenJohansenFitter(calculate_variance=False, jitter_level=0.0001, seed=CONFIG["seed"])
            aj.fit(d, sub.code, event_of_interest=code)
            cif = aj.cumulative_density_
            S[f"{label}_CIF_{nm}_365d"] = 100 * float(cif[cif.index <= 365].iloc[-1, 0]) if (cif.index <= 365).any() else 0.0
        S[f"{label}_removed_funcs"] = int(sub[sub.removed == 1].fid.nunique())
        S[f"{label}_death_funcs_365d"] = int(sub[(sub.event == 1) & (sub.days <= 365)].fid.nunique())
        if label == "random":
            def cif_diff(s_):
                d_ = s_.days.clip(lower=0); out = []
                for code in (2, 1):
                    aj = AalenJohansenFitter(calculate_variance=False, jitter_level=0.0001, seed=CONFIG["seed"])
                    aj.fit(d_, s_.code, event_of_interest=code); c = aj.cumulative_density_
                    out.append(float(c[c.index <= 365].iloc[-1, 0]) if (c.index <= 365).any() else 0.0)
                return 100 * (out[0] - out[1])
            S["random_CIF_diff_365d"] = cif_diff(sub)
            dr = np.random.RandomState(CONFIG["seed"] + 51); byr = {g: d_ for g, d_ in sub.groupby("fid")}; idr = sorted(byr); dv = []
            for _ in range(1000):
                ss = pd.concat([byr[g] for g in dr.choice(idr, len(idr), replace=True)], ignore_index=True)
                try:
                    dv.append(cif_diff(ss))
                except Exception:
                    pass
            S["random_CIF_diff_365d_lo"], S["random_CIF_diff_365d_hi"] = float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))
        S[f"{label}_removed_funcs_365d"] = int(sub[(sub.removed == 1) & (sub.days <= 365)].fid.nunique())
    # random-vs-main per-touch hazard on comparable exposure (touch time)
    S["random_functions_with_cohort_claims"] = int(rllm.fid.nunique())
    pd.DataFrame(kmrows).to_csv(os.path.join(RESULTS, "km_curves.csv"), index=False)

    # ---------------- placebo noise floor (code tokens unchanged)
    pl_rows = []
    for fid in sorted(verdicts):
        f = funcs.get(fid); v = verdicts[fid]
        if f is None:
            continue
        for p in v.get("placebo") or []:
            for (arm, i), x in zip(p["judged"], p["verdicts"]):
                pl_rows.append({"fid": fid, "cohort": f["cohort"], "source": "placebo_" + p["kind"], "arm": arm, "con": int(x["status"] == "contradicted")})
        rt = v.get("retest")
        if rt:
            for (arm, i), x in zip(rt["judged"], rt["verdicts"]):
                pl_rows.append({"fid": fid, "cohort": f["cohort"], "source": "retest_anchor", "arm": arm, "con": int(x["status"] == "contradicted")})
    PL = pd.DataFrame(pl_rows); PL.to_csv(os.path.join(RESULTS, "placebo.csv"), index=False)
    PLl = PL[PL.arm == "llm"]
    S["placebo_n_looks"] = int(len(PLl)); S["placebo_n_con"] = int(PLl.con.sum())
    S["placebo_rate"] = 100 * float(PLl.con.mean())
    lo, hi = cluster_boot(PLl, lambda s: float(s.con.mean())); S["placebo_rate_lo"], S["placebo_rate_hi"] = 100 * lo, 100 * hi
    for src in ("retest_anchor", "placebo_docstring", "placebo_comment", "placebo_format"):
        sub = PLl[PLl.source == src]
        S[f"placebo_{src}_n"] = int(len(sub)); S[f"placebo_{src}_rate"] = 100 * float(sub.con.mean()) if len(sub) else float("nan")
    sub = PLl[PLl.source != "retest_anchor"]
    S["placebo_versions_n"] = int(len(sub)); S["placebo_versions_rate"] = 100 * float(sub.con.mean()) if len(sub) else float("nan")
    fl = PLl.con.mean()
    curve, haz = km_int(llm.time_k.values, llm.event.values, KMAX)
    def corrected(fr):
        s = 1.0
        for k in range(K):
            h = haz[k][0]; s *= 1 - max(0.0, (h - fr) / (1 - fr))
        return s
    S["main_S_K_corrected"] = corrected(fl)
    # epsilon-sensitivity range (epsilon at its CI bounds, hazards fixed) -- NOT a confidence interval
    S["main_S_K_corrected_epshi"] = corrected(hi); S["main_S_K_corrected_epslo"] = corrected(lo)
    share = lambda sc: 100 * ((1 - S["main_S_K"]) - (1 - sc)) / (1 - S["main_S_K"])
    S["main_dead_K_spurious_share"] = share(S["main_S_K_corrected"])
    S["main_dead_K_spurious_share_hi"] = share(S["main_S_K_corrected_epshi"])
    S["main_dead_K_spurious_share_lo"] = share(S["main_S_K_corrected_epslo"])
    # joint function-cluster bootstrap: claims and the same functions' placebo looks resampled together
    PLm = PLl[PLl.cohort == "main"]; byc = {g: d for g, d in llm.groupby("fid")}; byp = {g: d for g, d in PLm.groupby("fid")}
    ids = sorted(byc); jrs = np.random.RandomState(CONFIG["seed"] + 21); vals = []
    for _ in range(B):
        pick = jrs.choice(ids, len(ids), replace=True)
        c = pd.concat([byc[g] for g in pick], ignore_index=True)
        pl = [byp[g] for g in pick if g in byp]
        e = float(pd.concat(pl).con.mean()) if pl else 0.0
        _, hz_ = km_int(c.time_k.values, c.event.values, KMAX); sc = 1.0
        for k in range(K):
            sc *= 1 - max(0.0, (hz_[k][0] - e) / (1 - e))
        vals.append(sc)
    S["main_S_K_corrected_lo"], S["main_S_K_corrected_hi"] = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    PLd = PL[(PL.arm == "doc") & (PL.source == "retest_anchor")]
    S["format_only_looks"] = int(sum(1 for fid in verdicts for c in verdicts[fid]["commits"]
                                     if fid in funcs and (funcs[fid]["touches"][c["k"] - 1].get("format_only") or funcs[fid]["touches"][c["k"] - 1].get("paren_only"))
                                     for x in c["verdicts"]))
    S["n_paren_only_touches"] = int(sum(1 for f in mf for t in f["touches"] if t.get("paren_only")))
    S["placebo_doc_retest_n"] = int(len(PLd)); S["placebo_doc_retest_con"] = int(PLd.con.sum())
    S["placebo_doc_retest_rate"] = 100 * float(PLd.con.mean()) if len(PLd) else float("nan")

    # ---------------- primary model: discrete-time cloglog (main, LLM arm)
    ct_cols = ["ct_" + c for c in CT if c != "local_edit" and P.loc[P["ct_" + c] == 1, "event"].sum() >= 5]
    S["ct_excluded_lt5_events"] = [c for c in CT if c != "local_edit" and "ct_" + c not in ct_cols]
    P["ty_structural"] = (P.type == "structural").astype(int); P["ty_behavioural"] = (P.type == "behavioural").astype(int)
    P["ty_intent"] = (P.type == "intent").astype(int)  # reference: dataflow (largest category)
    tcols = ["ty_structural", "ty_behavioural", "ty_intent"]
    covs = tcols + ["log_lines_fn", "log_lines_file"] + ct_cols
    fitA = cloglog(P, covs)
    S["cloglog"] = {c: hr_ci(fitA, fitA["cols"].index(c)) for c in covs}
    S["cloglog_events"] = fitA["events"]; S["cloglog_periods"] = fitA["n"]; S["cloglog_clusters"] = fitA["clusters"]
    S["H2_wald_stat"], S["H2_wald_p"] = wald(fitA, idx(fitA, tcols))
    # cluster-bootstrap covariance of the type coefficients (refits). Pre-stated: B = H2_BOOT_B, seed = CONFIG seed + 41;
    # seed-sensitivity: the same B with seeds + 41 ... + 45 (5 seeds, run in parallel)
    from scipy.stats import chi2 as _chi2
    b0 = fitA["beta"][idx(fitA, tcols)]
    res = h2_boot_parallel(P, covs, tcols, b0, [CONFIG["seed"] + 41 + i for i in range(5)], H2_BOOT_B)
    r0 = res[0]
    S["H2_boot_B"] = H2_BOOT_B
    S["H2_wald_boot_stat"], S["H2_wald_boot_p"] = r0["stat"], r0["p"]
    S["H2_wald_boot2_stat"], S["H2_wald_boot2_p"] = r0["stat2"], r0["p2"]
    S["H2_boot_skipped"] = sum(r0["skips"].values())
    S["H2_boot_skip_no_intent"] = r0["skips"]["no_intent_death"]; S["H2_boot_skip_period"] = r0["skips"]["period_only"]; S["H2_boot_skip_diverged"] = r0["skips"]["diverged"]
    ps = [r["p"] for r in res]
    S["H2_boot_p_seed_min"], S["H2_boot_p_seed_max"] = float(min(ps)), float(max(ps))
    S["H2_boot_p_mcse"] = float(np.std(ps, ddof=1))   # Monte-Carlo SE estimated from the seed spread at this B
    it = llm[(llm.type == "intent") & (llm.event == 1)]
    S["H2_intent_deaths"] = int(len(it)); S["H2_intent_death_functions"] = int(it.fid.nunique())
    S["H2_intent_deaths_click"] = int((it.repo == "click").sum())
    fitU = cloglog(P, tcols)
    S["H2_wald_unadj_stat"], S["H2_wald_unadj_p"] = wald(fitU, idx(fitU, tcols))
    S["cloglog_unadj"] = {c: hr_ci(fitU, fitU["cols"].index(c)) for c in tcols}
    # compound (enumerative) claims
    fitC = cloglog(P, covs + ["compound"])
    S["H2_wald_compound_stat"], S["H2_wald_compound_p"] = wald(fitC, idx(fitC, tcols))
    S["cloglog_compound"] = {c: hr_ci(fitC, fitC["cols"].index(c)) for c in tcols + ["compound"]}
    st_ = llm[llm.type == "structural"]
    for flag, name in ((1, "compound"), (0, "atomic")):
        sub = st_[st_.compound == flag]
        S[f"structural_{name}_n"] = int(len(sub)); S[f"structural_{name}_dead_K"] = 100 * float(((sub.event == 1) & (sub.time_k <= K)).mean())
    for ty in TYPES:
        sub = llm[(llm.type == ty) & (llm.compound == 0)]
        S[f"atomic_{ty}_dead_K"] = 100 * float(((sub.event == 1) & (sub.time_k <= K)).mean()) if len(sub) else float("nan")
    S["compound_pct_llm"] = 100 * float(llm.compound.mean())
    # repo-clustered sensitivity
    fitRepo = cloglog(P, covs, groups="repo")
    S["H2_wald_repo_p"] = wald(fitRepo, idx(fitRepo, tcols))[1]
    # proportional-hazards check: covariate x log(k) interactions, joint robust Wald
    for c in tcols + ["log_lines_fn"] + ct_cols:
        P[c + "_xlogk"] = P[c] * np.log(P.k)
    fitPH = cloglog(P, covs + [c + "_xlogk" for c in tcols + ["log_lines_fn"] + ct_cols])
    S["PH_wald_stat"], S["PH_wald_p"] = wald(fitPH, idx(fitPH, [c + "_xlogk" for c in tcols + ["log_lines_fn"] + ct_cols]))
    S["PH_df"] = len(tcols) + 1 + len(ct_cols)
    from scipy.stats import chi2 as _chi2b
    S["PH_lr_stat"] = 2 * (fitPH["ll"] - fitA["ll"]); S["PH_lr_p"] = float(_chi2b.sf(S["PH_lr_stat"], S["PH_df"]))
    S["PH_type_p"] = wald(fitPH, idx(fitPH, [c + "_xlogk" for c in tcols]))[1]
    S["PH_churn_p"] = wald(fitPH, idx(fitPH, ["log_lines_fn_xlogk"]))[1]
    S["PH_ct_p"] = wald(fitPH, idx(fitPH, [c + "_xlogk" for c in ct_cols]))[1]
    # period-restricted sensitivity: k <= K only
    fitK = cloglog(P[P.k <= K], covs)
    S["H2_wald_kK_p"] = wald(fitK, idx(fitK, tcols))[1]
    # unpenalised Cox (robustness), counting-process form with time-varying covariates
    from lifelines import CoxPHFitter
    Pc = P.copy(); Pc["entry"] = Pc.k - 1; Pc["stop"] = Pc.k
    try:
        cph = CoxPHFitter(penalizer=0.0).fit(Pc[covs + ["entry", "stop", "event", "fid"]], duration_col="stop", event_col="event",
                                             entry_col="entry", cluster_col="fid", robust=True)
        sm = cph.summary
        S["cox_unpen"] = {c: {"HR": float(np.exp(sm.loc[c, "coef"])), "lo": float(np.exp(sm.loc[c, "coef lower 95%"])),
                              "hi": float(np.exp(sm.loc[c, "coef upper 95%"]))} for c in covs}
        S["cox_unpen_ok"] = True
    except Exception as e:
        S["cox_unpen_ok"] = False; S["cox_unpen_error"] = repr(e)[:200]
    # KM by type table
    hz = []
    for ty in TYPES:
        sub = llm[llm.type == ty]
        c = {"structural": "ty_structural", "behavioural": "ty_behavioural", "intent": "ty_intent"}.get(ty)
        r = {"type": ty, "n": len(sub), "deaths": int(sub.event.sum()), "deaths_K": int(((sub.event == 1) & (sub.time_k <= K)).sum()),
             "S_K": S[f"main_{ty}_S_K"], "S_K_lo": S[f"main_{ty}_S_K_lo"], "S_K_hi": S[f"main_{ty}_S_K_hi"],
             "t90": S[f"main_{ty}_t90"], "dead_K": 100 * float(((sub.event == 1) & (sub.time_k <= K)).mean())}
        if c:
            r.update({"HR": S["cloglog"][c]["HR"], "HR_lo": S["cloglog"][c]["lo"], "HR_hi": S["cloglog"][c]["hi"]})
        else:
            r.update({"HR": 1.0, "HR_lo": float("nan"), "HR_hi": float("nan")})
        hz.append(r)
    pd.DataFrame(hz).to_csv(os.path.join(RESULTS, "hazard_by_type.csv"), index=False)
    order = sorted(hz, key=lambda h: -h["dead_K"])
    S["type_order_by_death"] = [h["type"] for h in order]
    S["H2_intent_lowest"] = bool(order[-1]["type"] == "intent")
    S["H2_behavioural_highest"] = bool(order[0]["type"] == "behavioural")
    # change types
    ctrows = []
    for c in CT:
        sub = P[P["ct_" + c] == 1]
        if len(sub) == 0:
            continue
        lo, hi = cluster_boot(sub, lambda s: float(s.event.mean()), nb=300)
        row = {"change_type": c, "exposures": len(sub), "pairs": int(sub.groupby(["fid", "k"]).ngroups), "deaths": int(sub.event.sum()),
               "death_rate_pct": 100 * float(sub.event.mean()), "lo": 100 * lo, "hi": 100 * hi}
        if "ct_" + c in S["cloglog"]:
            h = S["cloglog"]["ct_" + c]; row.update({"HR": h["HR"], "HR_lo": h["lo"], "HR_hi": h["hi"]})
        ctrows.append(row)
    pd.DataFrame(ctrows).to_csv(os.path.join(RESULTS, "changetype.csv"), index=False)
    S["changetype"] = {r["change_type"]: r for r in ctrows}
    # churn on a comparable scale: HR across the IQR of log(1+lines_fn), file
    for c in ("log_lines_fn", "log_lines_file"):
        q1, q3 = np.percentile(P[c], [25, 75]); b = fitA["beta"][fitA["cols"].index(c)]; se = math.sqrt(fitA["V"][fitA["cols"].index(c)] [fitA["cols"].index(c)])
        S[f"{c}_IQR_HR"] = {"HR": math.exp(b * (q3 - q1)), "lo": math.exp((b - 1.96 * se) * (q3 - q1)), "hi": math.exp((b + 1.96 * se) * (q3 - q1)),
                            "q1_lines": float(np.expm1(q1)), "q3_lines": float(np.expm1(q3))}

    # ---------------- H1: churn vs death (commit level, main cohort, LLM arm)
    fc = P.groupby(["fid", "k"]).agg(deaths=("event", "sum"), alive=("event", "size"), lines_fn=("lines_fn", "first"),
                                    lines_file=("lines_file", "first"), sig=("ct_signature", "first"), call=("ct_call_change", "first")).reset_index()
    fc["death_frac"] = fc.deaths / fc.alive; fc["any"] = (fc.deaths > 0).astype(int)
    S["H1_rho_fn"] = float(spearmanr(fc.lines_fn, fc.death_frac)[0]); S["H1_rho_file"] = float(spearmanr(fc.lines_file, fc.death_frac)[0])
    def boot_rho(seed, nb):
        r_ = np.random.RandomState(seed); ids = fc.fid.unique(); by = {g: d for g, d in fc.groupby("fid")}; out = []
        for _ in range(nb):
            c = pd.concat([by[g] for g in r_.choice(ids, len(ids), replace=True)], ignore_index=True)
            out.append(spearmanr(c.lines_fn, c.death_frac)[0])
        return np.array(out)
    vals = boot_rho(CONFIG["seed"] + 31, 10000)   # confirmatory: dedicated seed, B = 10,000
    S["H1_rho_fn_lo"], S["H1_rho_fn_hi"] = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    S["H1_p_weak"] = (1 + int(np.sum(vals >= 0.3))) / (len(vals) + 1)
    S["H1_p_weak_mcse"] = math.sqrt(S["H1_p_weak"] * (1 - S["H1_p_weak"]) / len(vals))
    spread = [(1 + int(np.sum(boot_rho(1000 + i, 1000) >= 0.3))) / 1001 for i in range(20)]
    S["H1_p_seed_min"], S["H1_p_seed_max"] = float(min(spread)), float(max(spread))
    S["H1_seeds_ge05_pct"] = 100 * float(np.mean([x >= 0.05 for x in spread]))
    S["H1_rho_file_lo"], S["H1_rho_file_hi"] = cluster_boot(fc, lambda s: spearmanr(s.lines_file, s.death_frac)[0])
    S["H1_auc_fn"] = auc(fc.lines_fn.values, fc["any"].values)
    S["H1_auc_fn_lo"], S["H1_auc_fn_hi"] = cluster_boot(fc, lambda s: auc(s.lines_fn.values, s["any"].values), nb=300)
    S["H1_n_pairs"] = int(len(fc)); S["H1_pairs_with_death_pct"] = 100 * float(fc["any"].mean())
    S["H1_zero_share"] = 100 * float((fc.death_frac == 0).mean())
    # trigger comparison at matched trigger rate (descriptive; flags share evidence with the judge)
    trig = ((fc.sig + fc.call) > 0).astype(int)
    r_ = float(trig.mean()); thr = np.quantile(fc.lines_fn, 1 - r_)
    churn_trig = (fc.lines_fn >= thr).astype(int)
    for name, t in (("flag", trig), ("churn", churn_trig)):
        tp = int(((t == 1) & (fc["any"] == 1)).sum())
        S[f"trigger_{name}_rate"] = 100 * float(t.mean())
        S[f"trigger_{name}_recall"] = 100 * tp / int(fc["any"].sum())
        S[f"trigger_{name}_precision"] = 100 * tp / max(1, int(t.sum()))
    S["trigger_churn_threshold_lines"] = float(thr)

    # ---------------- H3: whole-explanation JIT checks (degenerate) + graded v3 + identifier-existence comparator
    v2 = jload("jit_v2.json", {}); v3 = jload("jit_v3.json", {})
    idf = identifier_flags(funcs, verdicts, claims)
    jrows = []
    for fid in sorted(verdicts):
        f = funcs.get(fid)
        if f is None or f["cohort"] != "main":
            continue
        v = verdicts[fid]
        sup = {a["idx"] for a in v["anchor"] if a["arm"] == "llm" and a["status"] == "supported"}
        nonsup_anchor = any(a["arm"] == "llm" and a["status"] != "supported" for a in v["anchor"])
        dead_at = {}
        for c in v["commits"]:
            for (arm, i), x in zip(c["judged"], c["verdicts"]):
                if arm == "llm" and x["status"] == "contradicted" and i in sup and i not in dead_at:
                    dead_at[i] = c["k"]
        for c in v["commits"]:
            if not c.get("jit_asked"):
                continue
            k = c["k"]; key = f"{fid}:{k}"
            n_new = sum(1 for kk in dead_at.values() if kk == k); n_cum = sum(1 for kk in dead_at.values() if kk <= k)
            fl_ids = idf[fid]["flags"].get(k, set())
            jrows.append({"fid": fid, "k": k, "stratum": c["jit_stratum"], "new_dead": n_new, "any_dead_cum": int(n_cum > 0),
                          "anchor_nonsupported": int(nonsup_anchor),
                          "v1": int(c.get("jit_consistent") is False), "v2": int(v2.get(key, {}).get("consistent") is False),
                          "v3": v3.get(key, {}).get("score"), "ident": int(len(fl_ids & sup) > 0),
                          "ident_dying": int(len({i for i, kk in dead_at.items() if kk == k} & fl_ids) > 0),
                          "lines_fn": f["touches"][k - 1]["lines_changed_fn"]})
    J = pd.DataFrame(jrows); J.to_csv(os.path.join(RESULTS, "jit_commits.csv"), index=False)
    pos = J[J.new_dead > 0]; neg = J[(J.stratum == "sample") & (J.any_dead_cum == 0)]
    neg_strict = neg[neg.anchor_nonsupported == 0]
    S.update({"H3_n_asked": int(len(J)), "H3_n_pos": int(len(pos)), "H3_n_neg": int(len(neg)), "H3_n_neg_strict": int(len(neg_strict)),
              "H3_n_neg_anchor_wrong": int((neg.anchor_nonsupported == 1).sum()), "H3_n_dead_claims": int(pos.new_dead.sum())})
    for col in ("v1", "v2", "ident"):
        S[f"H3_{col}_recall"] = 100 * float((pos[col] * pos.new_dead).sum() / pos.new_dead.sum())
        S[f"H3_{col}_fa"] = 100 * float(neg[col].mean()); S[f"H3_{col}_fa_strict"] = 100 * float(neg_strict[col].mean()) if len(neg_strict) else float("nan")
    lo, hi = cluster_boot(pos, lambda s: float((s.ident * s.new_dead).sum() / s.new_dead.sum())); S["H3_ident_recall_lo"], S["H3_ident_recall_hi"] = 100 * lo, 100 * hi
    lo, hi = cluster_boot(neg, lambda s: float(s.ident.mean())); S["H3_ident_fa_lo"], S["H3_ident_fa_hi"] = 100 * lo, 100 * hi
    # k = 0 sanity (unchanged anchor)
    k0 = [key for key in v2 if key.endswith(":0")]
    S["H3_k0_n"] = len(k0); S["H3_k0_v2_flag"] = 100 * float(np.mean([v2[x]["consistent"] is False for x in k0])) if k0 else float("nan")
    s0 = [v3[x]["score"] for x in k0 if x in v3 and v3[x]["score"] is not None]
    S["H3_k0_v3_mean"] = float(np.mean(s0)) if s0 else float("nan")
    Jv3 = J.dropna(subset=["v3"])
    p3, n3 = Jv3[Jv3.new_dead > 0], Jv3[(Jv3.stratum == "sample") & (Jv3.any_dead_cum == 0)]
    S["H3_v3_mean_pos"] = float(p3.v3.mean()) if len(p3) else float("nan"); S["H3_v3_mean_neg"] = float(n3.v3.mean()) if len(n3) else float("nan")
    yy = np.r_[np.ones(len(p3)), np.zeros(len(n3))]; sc = np.r_[p3.v3.values, n3.v3.values]
    S["H3_v3_auc"] = auc(sc, yy) if len(p3) and len(n3) else float("nan")
    both = pd.concat([p3.assign(y=1), n3.assign(y=0)])
    S["H3_v3_auc_lo"], S["H3_v3_auc_hi"] = cluster_boot(both, lambda s: auc(s.v3.values, s.y.values), nb=500) if len(both) else (np.nan, np.nan)
    yy0 = np.r_[np.ones(len(p3)), np.zeros(len(s0))]; sc0 = np.r_[p3.v3.values, np.array(s0, float)]
    S["H3_v3_auc_vs_k0"] = auc(sc0, yy0) if len(p3) and len(s0) else float("nan")
    S["H3_v3_parsed_pct"] = 100 * float(J.v3.notna().mean())
    S["H3_degenerate"] = bool(S["H3_v1_recall"] >= 95 and S["H3_v1_fa"] >= 95 and S["H3_v2_recall"] >= 95 and S["H3_v2_fa"] >= 95)
    # claim-level identifier comparator over all claim-periods (main, LLM, anchor-supported)
    cl_rows = []
    for _, r in llm.iterrows():
        fl_ = idf[r.fid]
        has = bool(fl_["ids"].get(int(r.idx)))
        vk, _ = effective_touches(funcs[r.fid], len(verdicts[r.fid]["commits"]))
        first_flag = next((k for k in vk[:int(r.time_k)] if int(r.idx) in fl_["flags"].get(k, set())), None)
        cl_rows.append({"has_ids": int(has), "event": int(r.event), "flag_by_death": int(first_flag is not None)})
    CL = pd.DataFrame(cl_rows)
    S["ident_claims_with_ids_pct"] = 100 * float(CL.has_ids.mean())
    d_ = CL[CL.event == 1]; nd = CL[CL.event == 0]
    S["ident_claim_recall"] = 100 * float(d_.flag_by_death.mean()) if len(d_) else float("nan")
    S["ident_claim_flag_nondead"] = 100 * float(nd.flag_by_death.mean()) if len(nd) else float("nan")

    # ---------------- H4: LLM explanation vs anchor docstring (main cohort, functions with both arms)
    bothf = sorted(set(llm.fid) & set(doc.fid))
    L = llm[llm.fid.isin(bothf)]; D = doc[doc.fid.isin(bothf)]
    deadK = lambda s: ((s.event == 1) & (s.time_k <= K))
    S.update({"H4_n_functions": len(bothf), "H4_n_llm_claims": int(len(L)), "H4_n_doc_claims": int(len(D)),
              "H4_deaths_llm": int(deadK(L).sum()), "H4_deaths_doc": int(deadK(D).sum()),
              "H4_doc_death_functions": int(D[deadK(D)].fid.nunique()),
              "H4_dead_pct_llm": 100 * float(deadK(L).mean()), "H4_dead_pct_doc": 100 * float(deadK(D).mean())})
    allL = M[(M.arm == "llm") & M.fid.isin(bothf)]; allD = M[(M.arm == "doc") & M.fid.isin(bothf)]
    for arm, sub in (("llm", allL), ("doc", allD)):
        S[f"H4_false_at_K_{arm}"] = 100 * float(((sub.anchor_status == "contradicted") | deadK(sub)).mean())
    pf = pd.DataFrame({"fid": bothf, "llm": [float(deadK(L[L.fid == x]).mean()) for x in bothf],
                       "doc": [float(deadK(D[D.fid == x]).mean()) for x in bothf]})
    pf["diff"] = pf.llm - pf.doc
    S["H4_paired_diff_pct"] = 100 * float(pf["diff"].mean())
    S["H4_paired_diff_lo"], S["H4_paired_diff_hi"] = [100 * x for x in cluster_boot(pf, lambda s: float(s["diff"].mean()))]
    prs = np.random.RandomState(CONFIG["seed"] + 7); obs = pf["diff"].mean(); d = pf["diff"].values; NP = 20000
    flips = prs.choice([-1, 1], size=(NP, len(d)))
    S["H4_p"] = float((1 + np.sum((flips * d).mean(axis=1) >= obs)) / (NP + 1))
    S["H4_functions_llm_worse"] = int((pf["diff"] > 0).sum()); S["H4_functions_doc_worse"] = int((pf["diff"] < 0).sum())
    PB = pdf[(pdf.cohort == "main") & pdf.fid.isin(bothf)].copy(); PB = PB[PB.k <= K]
    PB["arm_llm"] = (PB.arm == "llm").astype(int)
    for ty in ("structural", "behavioural", "intent"):
        PB["ty_" + ty] = (PB.type == ty).astype(int)
    f4 = cloglog(PB, ["arm_llm", "ty_structural", "ty_behavioural", "ty_intent"])
    S["H4_HR"] = hr_ci(f4, f4["cols"].index("arm_llm"))
    # delayed entry by docstring age (touches since the docstring was last edited); complete histories only
    ages = {f["id"]: f.get("doc_age") for f in mf}
    rows4 = []
    for _, r in pd.concat([L, D]).iterrows():
        tk = int(min(r.time_k, K)); ev = int(r.event and r.time_k <= K)
        if r.arm == "doc":
            ag = ages.get(r.fid)
            if not ag or not ag.get("age_complete"):
                continue
            e0 = int(ag["age_touches"])
        else:
            e0 = 0
        rows4.append({"fid": r.fid, "arm_llm": int(r.arm == "llm"), "entry": e0, "stop": e0 + tk, "event": ev,
                      "ty_structural": int(r.type == "structural"), "ty_behavioural": int(r.type == "behavioural"), "ty_intent": int(r.type == "intent")})
    D4 = pd.DataFrame(rows4); D4 = D4[D4.stop > D4.entry]
    S["H4_age_n_doc_functions"] = int(D4[D4.arm_llm == 0].fid.nunique()); S["H4_age_n_doc_claims"] = int((D4.arm_llm == 0).sum())
    S["H4_age_median_doc"] = float(np.median([a["age_touches"] for a in ages.values() if a and a.get("age_complete")])) if any(a and a.get("age_complete") for a in ages.values()) else float("nan")
    try:
        c4 = CoxPHFitter(penalizer=0.0).fit(D4, duration_col="stop", event_col="event", entry_col="entry", cluster_col="fid", robust=True)
        s4 = c4.summary.loc["arm_llm"]
        S["H4_HR_delayed"] = {"HR": float(np.exp(s4["coef"])), "lo": float(np.exp(s4["coef lower 95%"])), "hi": float(np.exp(s4["coef upper 95%"]))}
    except Exception as e:
        S["H4_HR_delayed"] = {"HR": float("nan"), "lo": float("nan"), "hi": float("nan"), "error": repr(e)[:120]}
    # same functions in both arms (those with a complete docstring age), stratified by function
    agefs = set(D4[D4.arm_llm == 0].fid)
    D4s = D4[D4.fid.isin(agefs)]
    S["H4_age_n_llm_claims_restricted"] = int((D4s.arm_llm == 1).sum())
    try:
        D4s = D4s.assign(cl=D4s.fid)
        c4s = CoxPHFitter(penalizer=0.0).fit(D4s, duration_col="stop", event_col="event", entry_col="entry", strata=["fid"],
                                             cluster_col="cl", robust=True)
        s4 = c4s.summary.loc["arm_llm"]
        S["H4_HR_delayed_strat"] = {"HR": float(np.exp(s4["coef"])), "lo": float(np.exp(s4["coef lower 95%"])), "hi": float(np.exp(s4["coef upper 95%"]))}
    except Exception as e:
        S["H4_HR_delayed_strat"] = {"HR": float("nan"), "lo": float("nan"), "hi": float("nan"), "error": repr(e)[:120]}
    young = {fid for fid, a in ages.items() if a and a.get("age_complete") and a["age_touches"] <= K}
    Ly, Dy = L[L.fid.isin(young)], D[D.fid.isin(young)]
    S["H4_young_n_functions"] = len(young & set(bothf)); S["H4_young_dead_llm"] = 100 * float(deadK(Ly).mean()) if len(Ly) else float("nan")
    S["H4_young_dead_doc"] = 100 * float(deadK(Dy).mean()) if len(Dy) else float("nan"); S["H4_young_deaths_doc"] = int(deadK(Dy).sum())
    # audit of anchor-contradicted docstring claims (AI build assistant, results/docstring_audit.json)
    audit = jload("docstring_audit.json", {})
    cur_con = {f"{r.fid}:{r.idx}" for r in M[(M.arm == "doc") & (M.anchor_status == "contradicted")].itertuples()}
    audit = {k: v for k, v in audit.items() if k in cur_con}   # only claims contradicted at the anchor in this cohort
    S["doc_audit_coverage"] = f"{len(audit)}/{len(cur_con)}"
    if audit:
        labs = pd.Series([v["label"] for v in audit.values()])
        S["doc_audit_n"] = int(len(labs)); S["doc_audit_counts"] = {k: int(v) for k, v in labs.value_counts().items()}
        S["doc_audit_real_pct"] = 100 * float((labs == "real_stale").mean())
        dec = pd.Series([k.split(":")[0] for k, v in audit.items() if v["label"] == "decomposer_error"]).value_counts()
        S["doc_audit_decomposer_max_func"] = int(dec.iloc[0]) if len(dec) else 0
        S["anchor_contradicted_pct_doc_audited"] = S["anchor_contradicted_pct_doc"] * S["doc_audit_real_pct"] / 100
    pd.DataFrame([{"series": "llm_paired", "k": t, "S": km_int(L.time_k.values, L.event.values, KMAX)[0][t]} for t in range(K + 1)] +
                 [{"series": "doc_paired", "k": t, "S": km_int(D.time_k.values, D.event.values, KMAX)[0][t]} for t in range(K + 1)]).to_csv(
        os.path.join(RESULTS, "km_arms.csv"), index=False)

    # ---------------- reliability proxy: second-model agreement, re-validated and stratum-weighted
    S["agreement"] = agreement(main_f, verdicts)
    # ---------------- heterogeneity
    S["dead_pct_K_by_repo"] = {r: 100 * float(deadK(llm[llm.repo == r]).mean()) for r in sorted(llm.repo.unique())}
    loo = [S_at(llm[llm.repo != r], K) for r in sorted(llm.repo.unique())]
    S["loo_repo_S_K_min"], S["loo_repo_S_K_max"] = float(min(loo)), float(max(loo))
    S["n_unparsed_verdicts"] = int(sum(1 for v in verdicts.values() for c in v["commits"] for x in c["verdicts"] if not x.get("parsed", True)))
    S["n_verdicts"] = int(sum(len(c["verdicts"]) for v in verdicts.values() for c in v["commits"]))
    S["unresolved_period_pct"] = 100 * float(M[M.arm == "llm"].n_unresolved.sum() / max(1, len(P)))

    # ---------------- confirmatory family: H1, H2, H4 (Holm, m=3); sensitivity with H3 (m=4, p=1)
    raw = {"H1": S["H1_p_weak"], "H2": S["H2_wald_p"], "H4": S["H4_p"]}
    adj = holm(raw); adj4 = holm({**raw, "H3": 1.0})
    S["holm"] = {k: {"p_raw": raw[k], "p_holm": adj[k], "p_holm_m4": adj4[k]} for k in raw}
    S["H1_verdict"] = "supported" if adj["H1"] < 0.05 else ("inconclusive" if S["H1_rho_fn"] < 0.3 else "not supported")
    S["H1_verdict_m4"] = "supported" if adj4["H1"] < 0.05 else ("inconclusive" if S["H1_rho_fn"] < 0.3 else "not supported")
    if adj["H2"] < 0.05 and S["H2_intent_lowest"] and S["H2_behavioural_highest"]:
        S["H2_verdict"] = "supported"
    elif adj["H2"] < 0.05:
        S["H2_verdict"] = "type effect, predicted ordering not supported"
    else:
        S["H2_verdict"] = "not supported"
    S["H3_verdict"] = "inconclusive" if S["H3_degenerate"] else ("supported" if S["H3_v2_recall"] < 50 else "not supported")
    # H4 is "supported" only if the death-share test survives Holm AND the left-truncation (delayed-entry) analysis agrees
    de = S["H4_HR_delayed"]
    if adj["H4"] < 0.05 and de["lo"] > 1:
        S["H4_verdict"] = "supported"
    elif adj["H4"] < 0.05:
        S["H4_verdict"] = "inconclusive"
    else:
        S["H4_verdict"] = "not supported"
    # call accounting
    cdir = os.path.join(RESULTS, "llm_cache")
    kinds = [("eval_count" in json.load(open(os.path.join(cdir, x)))) for x in sorted(os.listdir(cdir)) if x.endswith(".json")]
    S["n_ollama_calls"] = int(sum(kinds)); S["n_anthropic_calls"] = int(len(kinds) - sum(kinds))
    S["note"] = ("Real repository histories and real LLM outputs (bulk: %s via Ollama, seed %d, T=0; reliability proxy: %s). "
                 "Human annotation is planned, not done." % (CONFIG["bulk_model"], CONFIG["seed"], CONFIG["second_model"]))
    jdump(S, "summary.json")
    print(json.dumps({k: v for k, v in S.items() if not isinstance(v, (dict, list))}, indent=0)[:6000])

def agreement(main_f, verdicts):
    out = {}
    ex = jload("extraction_agreement.json", {})
    if ex:
        nb = [v["n_bulk"] for v in ex.values()]; ns = [v["n_second"] for v in ex.values()]
        out["extraction_n"] = len(ex); out["extraction_count_rho"] = float(spearmanr(nb, ns)[0])
    rl = jload("type_relabel.json", {})
    if rl:
        a = [t for v in rl.values() for t in v["bulk"]]; b = [t for v in rl.values() for t in v["second"]]
        out["type_kappa"] = kappa(a, b); out["type_n_claims"] = len(a)
    ja = jload("judge_agreement.json", {})
    # population frame for the commit strata (main cohort, k <= K, commits with >= 1 judged claim)
    Npos = sum(1 for fid, v in verdicts.items() if fid in main_f for c in v["commits"] if c["k"] <= K and c["verdicts"]
               and any(x["status"] == "contradicted" for x in c["verdicts"]))
    Nneg = sum(1 for fid, v in verdicts.items() if fid in main_f for c in v["commits"] if c["k"] <= K and c["verdicts"]
               and not any(x["status"] == "contradicted" for x in c["verdicts"]))
    valid = {}
    for key, e in ja.items():
        fid, k = key.split(":"); k = int(k)
        if fid not in verdicts:
            continue
        cur = verdicts[fid]["anchor"] if k == 0 else next((c["verdicts"] for c in verdicts[fid]["commits"] if c["k"] == k), None)
        if cur is not None and [x["status"] for x in cur] == e["bulk"]:
            valid[key] = e
    out["judge_pairs_total"] = len(ja); out["judge_pairs_valid"] = len(valid)
    n_s = {"pos": sum(1 for e in valid.values() if e["stratum"] == "pos"), "neg": sum(1 for e in valid.values() if e["stratum"] == "neg")}
    W = {"pos": Npos / max(1, n_s["pos"]), "neg": Nneg / max(1, n_s["neg"])}
    out["frame_pos"], out["frame_neg"] = Npos, Nneg
    for tag, sel, weighted in (("_anchor", lambda e: e["stratum"] == "anchor", False), ("_commit", lambda e: e["stratum"] != "anchor", True)):
        a, b, w = [], [], []
        for e in valid.values():
            if not sel(e):
                continue
            for x, y in zip(e["bulk"], e["second"]):
                a.append(x); b.append(y); w.append(W[e["stratum"]] if weighted else 1.0)
        if not a:
            continue
        w = np.array(w); ab = np.array([x == "contradicted" for x in a]); bb = np.array([y == "contradicted" for y in b])
        out[f"judge{tag}_pairs"] = sum(1 for e in valid.values() if sel(e)); out[f"judge{tag}_n"] = len(a)
        out[f"judge{tag}_agree_pct"] = 100 * float(np.sum(w * (np.array(a) == np.array(b))) / w.sum())
        out[f"judge{tag}_kappa3"] = kappa(a, b, w); out[f"judge{tag}_kappa_dead"] = kappa(list(ab), list(bb), w)
        out[f"judge{tag}_dead_bulk"] = 100 * float(np.sum(w * ab) / w.sum()); out[f"judge{tag}_dead_second"] = 100 * float(np.sum(w * bb) / w.sum())
        out[f"judge{tag}_precision"] = 100 * float(np.sum(w * (ab & bb)) / max(1e-9, np.sum(w * ab)))
        out[f"judge{tag}_recall"] = 100 * float(np.sum(w * (ab & bb)) / max(1e-9, np.sum(w * bb)))
        out[f"judge{tag}_second_con_where_bulk_not"] = 100 * float(np.sum(w * (~ab & bb)) / max(1e-9, np.sum(w * ~ab)))
        out[f"judge{tag}_kappa_dead_unweighted"] = kappa(list(ab), list(bb))
    return out

if __name__ == "__main__":
    main()
