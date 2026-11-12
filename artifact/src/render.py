"""Stage 6: render results CSV/JSON into LaTeX fragments (tables, pgfplots figures, macros) under manuscript/generated/."""
import csv, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, GENERATED, CONFIG

def rd(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        return list(csv.DictReader(f))

def write(name, text):
    os.makedirs(GENERATED, exist_ok=True)
    with open(os.path.join(GENERATED, name), "w") as f:
        f.write(text)

def isnan(x):
    try:
        return x in ("", None) or math.isnan(float(x))
    except (TypeError, ValueError):
        return True
def f1(x): return "--" if isnan(x) else f"{float(x):.1f}"
def f2(x): return "--" if isnan(x) else f"{float(x):.2f}"
def ti(x): return "n.r." if isnan(x) else f"{float(x):.0f}"
def pfmt(p):
    """p-value as a math relation for $p\\Macro$: '{<}0.001' or '{=}0.012'."""
    return "{\\le}0.001" if p <= 0.001 else f"{{=}}{p:.3f}"
def ci(d, fmt=f2):
    return f"{fmt(d['HR'])} [{fmt(d['lo'])}, {fmt(d['hi'])}]"

TYPE_LABEL = {"structural": "Structural", "dataflow": "Data-flow", "behavioural": "Behavioural", "intent": "Intent"}
CT_LABEL = {"signature": "Signature", "call_change": "Call set", "call_receiver": "Receiver only", "exception": "Exception/try",
            "control_flow": "Control flow", "return_value": "Return expr.", "identifier_rename": "Identifier subst.",
            "literal_or_keyword": "Literal/keyword", "local_edit": "Other local edit"}

def tab_dataset(S):
    rows = rd("tab_dataset.csv")
    dash = lambda r, k: "--" if r["functions"] == "0" else r[k]
    body = "\n".join(f"{r['repo']} & {dash(r,'functions')} & {dash(r,'pairs')} & {dash(r,'commits')} & {dash(r,'with_docstring')} & {dash(r,'llm_claims')} & {dash(r,'doc_claims')} & {r['random']} \\\\" for r in rows)
    tot = (f"\\midrule\nTotal & {S['n_functions']} & {S['n_pairs']} & {S['n_distinct_commits']} & {S['n_funcs_with_doc']} & "
           f"{S['n_llm_claims']} & {S['n_doc_claims']} & {S['n_random']} \\\\")
    write("tab_dataset.tex", r"""\begin{table}[t]
\centering
\caption{Subjects. Main cohort: functions, function--commit touches (up to %d each) and distinct commits; Rand.: random-cohort functions (others: rich, flask, httpx).}
\label{tab:dataset}
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Project & Func. & Touches & Commits & w/ doc & LLM cl. & Doc cl. & Rand. \\
\midrule
%s
%s
\bottomrule
\end{tabular}
\end{table}
""" % (CONFIG["k_max"], body, tot))

def tab_survival(S):
    rows = []
    def cohort_row(name, lab):
        t90 = ti(S[lab + "_t90"]) if S.get(lab + "_t90_reliable", True) else "--"
        return (f"{name} & {S[lab + '_n']} & {S[lab + '_deaths']} & {f2(S[lab + '_S_K'])} [{f2(S[lab + '_S_K_lo'])}, {f2(S[lab + '_S_K_hi'])}] & "
                f"{t90} & -- \\\\")
    rows.append(cohort_row("Main (all)", "main"))
    rows.append(f"\\quad placebo scenario & & & {f2(S['main_S_K_corrected'])} [{f2(S['main_S_K_corrected_lo'])}, {f2(S['main_S_K_corrected_hi'])}] & & -- \\\\")
    rows.append(f"Random ($k{{=}}1$) & {S['random_at_risk_1']} & {S['random_deaths']} & n.r. & -- & -- \\\\")
    rows.append("\\midrule")
    hz = {r["type"]: r for r in rd("hazard_by_type.csv")}
    for ty in ("structural", "dataflow", "behavioural", "intent"):
        r = hz[ty]; lab = "main_" + ty
        hr = "1 (ref.)" if ty == "dataflow" else f"{f2(r['HR'])} [{f2(r['HR_lo'])}, {f2(r['HR_hi'])}]"
        rows.append(f"{TYPE_LABEL[ty]} & {r['n']} & {r['deaths']} & {f2(r['S_K'])} [{f2(r['S_K_lo'])}, {f2(r['S_K_hi'])}] & {ti(r['t90'])} & "
                    f"{hr} \\\\")
    write("tab_survival.tex", r"""\begin{table}[t]
\centering
\caption{Freedom from judge-detected contradiction among anchor-supported claims. Main cohort selects future-active, trackable functions; placebo adjustment is a scenario. HRs vs.\ data-flow.}
\label{tab:survival}
\footnotesize
\setlength{\tabcolsep}{1.3pt}
\begin{tabular}{@{}lrrlrl@{}}
\toprule
Claims & $n$ & Events$_{\le 12}$ & $\hat S(%d)$ [95\%% CI] & $t_{0.9}$ & HR [95\%% CI] \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (CONFIG["follow_k"], "\n".join(rows)))

def tab_tests(S):
    H = S["holm"]
    def ver(h):
        return f"$p_{{\\mathrm{{Holm}}}}{pfmt(H[h]['p_holm'])}$: {S[h + '_verdict']}"
    rows = [
        "\\multicolumn{2}{@{}l}{\\textit{H1: churn (" + str(S["H1_n_pairs"]) + " touches)}} \\\\",
        "$\\rho$(function lines, share dying) & " + f"{f2(S['H1_rho_fn'])} [{f2(S['H1_rho_fn_lo'])}, {f2(S['H1_rho_fn_hi'])}]",
        "AUC, function lines $\\to$ any death & " + f"{f2(S['H1_auc_fn'])} [{f2(S['H1_auc_fn_lo'])}, {f2(S['H1_auc_fn_hi'])}]",
        "HR per IQR of function lines & " + ci(S["log_lines_fn_IQR_HR"]),
        "Test $\\rho<0.3$ (post hoc) & " + ver("H1"),
        "\\midrule",
        "\\multicolumn{2}{@{}l}{\\textit{H3: detectors (" + str(S["H3_n_pos"]) + " death / " + str(S["H3_n_neg"]) + " clean touches)}} \\\\",
        "JIT v1, v2 flagged & " + f"all; v2 at $k{{=}}0$: {S['H3_k0_v2_flag']:.0f}\\%",
        "JIT v3 graded, AUC & " + f"{f2(S['H3_v3_auc'])} [{f2(S['H3_v3_auc_lo'])}, {f2(S['H3_v3_auc_hi'])}]",
        "Identifier check: touch / claim recall & " + f"{f1(S['H3_ident_recall'])}\\% / {f1(S['ident_claim_recall'])}\\%",
        "Identifier check: false alarms (touch) & " + f"{f1(S['H3_ident_fa'])}\\%",
        "\\midrule",
        "\\multicolumn{2}{@{}l}{\\textit{H4: " + str(S["H4_n_functions"]) + " functions with both arms (" + str(S["H4_deaths_llm"]) + " vs.\\ " + str(S["H4_deaths_doc"]) + " deaths)}} \\\\",
        "Dead by $K$, LLM vs.\\ doc & " + f"{f1(S['H4_dead_pct_llm'])}\\% vs.\\ {f1(S['H4_dead_pct_doc'])}\\%",
        "Contradicted by $K$ incl.\\ anchor & " + f"{f1(S['H4_false_at_K_llm'])}\\% vs.\\ {f1(S['H4_false_at_K_doc'])}\\%",
        "HR LLM vs.\\ doc, cloglog & " + f"{f2(S['H4_HR']['HR'])} [{f2(S['H4_HR']['lo'])}, {f2(S['H4_HR']['hi'])}]",
        "HR, delayed-entry Cox (" + str(S["H4_age_n_doc_functions"]) + " doc fns) & " + f"{f2(S['H4_HR_delayed']['HR'])} [{f2(S['H4_HR_delayed']['lo'])}, {f2(S['H4_HR_delayed']['hi'])}]",
        "Sign-flip test; verdict (coded rule) & " + ver("H4"),
    ]
    write("tab_tests.tex", r"""\begin{table}[t]
\centering
\caption{Tests of H1, H3 and H4 (95\%% CIs in brackets).}
\label{tab:tests}
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}p{0.50\columnwidth}p{0.47\columnwidth}@{}}
\toprule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(r if (r.endswith("\\midrule") or r.endswith("\\\\")) else r + " \\\\" for r in rows))

def fig_km(S):
    rows = rd("km_curves.csv")
    styles = {"structural": "blue,mark=square*", "dataflow": "teal,mark=triangle*", "behavioural": "red,mark=*",
              "intent": "black!70,mark=diamond*", "random": "orange,dash pattern=on 5pt off 2pt,mark=none,very thick"}
    kr = S["random_k_reliable"]
    names = dict(TYPE_LABEL, random=f"Random cohort ($k\\le{kr}$)")
    main = [r for r in rows if r["series"] == "main"]
    lo = " ".join(f"({r['k']},{float(r['lo']):.4f})" for r in main); hi = " ".join(f"({r['k']},{float(r['hi']):.4f})" for r in main)
    mc = " ".join(f"({r['k']},{float(r['S']):.4f})" for r in main)
    plots = [f"\\addplot[name path=lo,const plot,draw=none,forget plot] coordinates {{{lo}}};",
             f"\\addplot[name path=hi,const plot,draw=none,forget plot] coordinates {{{hi}}};",
             "\\addplot[black!15,forget plot] fill between[of=lo and hi];",
             f"\\addplot+[const plot, black, mark=none, ultra thick] coordinates {{{mc}}};\n\\addlegendentry{{Main, all claims (95\\% band)}}"]
    for t in ("structural", "dataflow", "behavioural", "intent", "random"):
        coords = " ".join(f"({r['k']},{float(r['S']):.4f})" for r in rows if r["series"] == t and (t != "random" or int(r["k"]) <= kr))
        plots.append(f"\\addplot+[const plot, {styles[t]}, mark size=1.1pt, thick] coordinates {{{coords}}};\n\\addlegendentry{{{names[t]}}}")
    write("fig_km.tex", r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=4.0cm,xlabel={code-changing touches since anchor $k$},ylabel={freedom from detection $\hat S(k)$},
  xmin=0,xmax=%d,ymin=0.6,ymax=1.0,xtick={0,2,...,%d},grid=major,grid style={dashed,gray!30},
  legend style={at={(0.5,-0.42)},anchor=north,font=\tiny,cells={anchor=west},draw=none},legend columns=3,
  tick label style={font=\scriptsize},label style={font=\footnotesize}]
%s
\end{axis}
\end{tikzpicture}
\caption{Kaplan--Meier freedom from judge-detected contradiction: future-active main cohort (all, cluster-bootstrap band, by type) and random cohort.}
\label{fig:km}
\end{figure}
""" % (CONFIG["k_max"], CONFIG["k_max"], "\n".join(plots)))

def fig_forest(S):
    items = [(f"Function lines (IQR)", S["log_lines_fn_IQR_HR"]), ("File lines (IQR)", S["log_lines_file_IQR_HR"])]
    ct = {r["change_type"]: r for r in rd("changetype.csv")}
    for c in ("signature", "call_change", "call_receiver", "exception", "control_flow", "return_value", "identifier_rename", "literal_or_keyword"):
        r = ct.get(c)
        if r and not isnan(r.get("HR")):
            items.append((f"{CT_LABEL[c]} ({r['pairs']})", {"HR": float(r["HR"]), "lo": float(r["HR_lo"]), "hi": float(r["HR_hi"])}))
    n = len(items)
    coords = " ".join(f"({d['HR']:.3f},{n - i}) -= ({d['HR'] - d['lo']:.3f},0) += ({d['hi'] - d['HR']:.3f},0)" for i, (_, d) in enumerate(items))
    ticks = ",".join(str(n - i) for i in range(n)); labels = ",".join("{" + lab + "}" for lab, _ in items)
    write("fig_forest.tex", r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.78\columnwidth,height=4.0cm,xmode=log,log basis x=2,xmin=0.25,xmax=8,xtick={0.25,0.5,1,2,4,8},
  xticklabels={0.25,0.5,1,2,4,8},ytick={%s},yticklabels={%s},ymin=0.3,ymax=%d.7,
  y tick label style={font=\scriptsize},x tick label style={font=\scriptsize},xlabel={hazard ratio (log scale)},label style={font=\footnotesize},
  xmajorgrids,grid style={dashed,gray!30}]
\addplot[only marks,mark=*,mark size=1.6pt,blue!70!black,error bars/.cd,x dir=both,x explicit,error bar style={black}] coordinates {%s};
\draw[gray,dashed] (axis cs:1,0.3) -- (axis cs:1,%d.7);
\end{axis}
\end{tikzpicture}
\caption{Hazard ratios with 95\%% CIs (cloglog model): churn per inter-quartile range; change types (touches with the flag) vs.\ other local edits.}
\label{fig:forest}
\end{figure}
""" % (ticks, labels, n, coords, n))

WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
def digits_to_words(name):
    return "".join(WORDS[int(ch)] if ch.isdigit() else ch for ch in name)

def macros(S):
    m = {}
    def put(name, val, fmt="%.1f"):
        if isinstance(val, bool):
            m[name] = str(val)
        elif isinstance(val, int):
            m[name] = str(val)
        elif val is None or (isinstance(val, float) and math.isnan(val)):
            m[name] = "n.r."
        else:
            m[name] = fmt % val
    A = S["agreement"]
    for k, v in S.items():
        if isinstance(v, (dict, list)) or k in ("note",):
            continue
        name = "".join(p.capitalize() for p in k.replace("-", "_").split("_"))
        name = "".join(ch for ch in name if ch.isalpha() or ch.isdigit())
        name = digits_to_words(name)
        if isinstance(v, str):
            m["X" + name] = v
        elif isinstance(v, float) and re.search(r"(^|_)p($|_)", k):
            m["X" + name] = pfmt(v)
        elif isinstance(v, float):
            put("X" + name, v, "%.2f" if abs(v) < 10 and ("_S_" in k or k.startswith("H1_rho") or "auc" in k.lower()) else "%.1f")
        else:
            put("X" + name, v)
    for k, v in A.items():
        name = "XA" + "".join(p.capitalize() for p in k.split("_")); name = "".join(ch for ch in name if ch.isalpha())
        put(name, v, "%.2f" if "kappa" in k or "rho" in k else "%.1f")
    for grp in ("cloglog", "cloglog_unadj", "cloglog_compound", "cox_unpen"):
        for c, d in S.get(grp, {}).items():
            nm = "X" + "".join(p.capitalize() for p in (grp + "_" + c).split("_")); nm = "".join(ch for ch in nm if ch.isalpha())
            put(nm, d["HR"], "%.2f"); put(nm + "Lo", d["lo"], "%.2f"); put(nm + "Hi", d["hi"], "%.2f")
    for key in ("H4_HR", "H4_HR_delayed", "H4_HR_delayed_strat", "log_lines_fn_IQR_HR", "log_lines_file_IQR_HR"):
        d = S[key]; nm = "X" + "".join(p.capitalize() for p in key.split("_")); nm = "".join(ch for ch in nm if ch.isalpha())
        put(nm, d["HR"], "%.2f"); put(nm + "Lo", d["lo"], "%.2f"); put(nm + "Hi", d["hi"], "%.2f")
    for h in ("H1", "H2", "H4"):
        nm = {"H1": "One", "H2": "Two", "H4": "Four"}[h]
        m[f"XH{nm}PHolm"] = pfmt(S["holm"][h]["p_holm"]); m[f"XH{nm}PRaw"] = pfmt(S["holm"][h]["p_raw"]); m[f"XH{nm}PHolmMFour"] = pfmt(S["holm"][h]["p_holm_m4"])
    for c, r in S["changetype"].items():
        nm = "XCT" + "".join(p.capitalize() for p in c.split("_"))
        put(nm + "Rate", r["death_rate_pct"]); put(nm + "N", r["pairs"])
        if "HR" in r:
            put(nm + "HR", r["HR"], "%.2f"); put(nm + "HRLo", r["HR_lo"], "%.2f"); put(nm + "HRHi", r["HR_hi"], "%.2f")
    for r, v in S["dead_pct_K_by_repo"].items():
        put("XDeadRepo" + "".join(ch for ch in r.title() if ch.isalpha()), v)
    for r, v in S.get("doc_audit_counts", {}).items():
        put("XAudit" + "".join(p.capitalize() for p in r.split("_")), v)
    m["XHOneSeedMin"] = f"{S['H1_p_seed_min']:.3f}"; m["XHOneSeedMax"] = f"{S['H1_p_seed_max']:.3f}"; m["XHOneMcse"] = f"{S['H1_p_weak_mcse']:.3f}"
    m["XHTwoBootMin"] = f"{S['H2_boot_p_seed_min']:.3f}"; m["XHTwoBootMax"] = f"{S['H2_boot_p_seed_max']:.3f}"; m["XHTwoBootMcse"] = f"{S['H2_boot_p_mcse']:.3f}"
    m["XBulkModel"] = CONFIG["bulk_model"]; m["XSecondModel"] = CONFIG["second_model"]
    put("XFollowK", CONFIG["follow_k"]); put("XKMax", CONFIG["k_max"]); put("XMinTouches", CONFIG["min_touches"]); m["XAnchorDate"] = CONFIG["anchor_before"]
    put("XRandomCohortSize", CONFIG["random_n"])
    validation = os.path.join(os.path.dirname(RESULTS), "validation")
    with open(os.path.join(validation, "unresolved_sensitivity.json"), encoding="utf-8") as stream:
        u = json.load(stream)
    with open(os.path.join(validation, "h2_sensitivity.json"), encoding="utf-8") as stream:
        h = json.load(stream)["without_click_pagers"]
    put("XUnresolvedClaims", u["claims_with_unresolved_before_endpoint"])
    put("XUnresolvedCensorS", u["censor_before_first_unresolved_S6"], "%.3f")
    put("XUnresolvedEventS", u["first_unresolved_as_event_S6"], "%.3f")
    put("XHtwoRemainingIntent", h["intent_events"])
    put("XHtwoPagerP", h["joint_type_wald_p"], "%.3f")
    for key, suffix in (("HR", "HR"), ("lo", "Lo"), ("hi", "Hi")):
        put("XHtwoPager" + suffix, h["intent_HR"][key], "%.2f")
    experiments = os.path.join(os.path.dirname(RESULTS), "experiments")
    fresh = {}
    for name in ("resurrection", "cross_family", "random_redraw_inference", "docstring_free"):
        path = os.path.join(experiments, name, "summary.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as stream:
                fresh[name] = json.load(stream)
    if len(fresh) == 4 and all(s.get("complete") and not s.get("failed") for s in fresh.values()):
        r, c = fresh["resurrection"], fresh["cross_family"]
        put("XFreshResN", r["eligible_claims"])
        put("XFreshResSupported", r["claims_with_later_supported_verdict"])
        put("XFreshResUnparsed", r["unparsed_verdicts"])
        put("XFreshCrossN", c["completed_cases"])
        put("XFreshCrossKappa", c["weighted_binary_contradiction_kappa"], "%.2f")
        put("XFreshRawAgree", 100 * c["raw_agreement"], "%.0f")
        put("XFreshWeightedAgree", 100 * c["weighted_agreement"], "%.1f")
        put("XFreshRedrawN", fresh["random_redraw_inference"]["completed_functions"])
        put("XFreshDocN", fresh["docstring_free"]["completed_functions"])
    human_path = os.path.join(validation, "human_analysis", "summary.json")
    if os.path.exists(human_path):
        from analyze_human_annotation import current_summary
        human = current_summary()
        put("XHumanN", human["validated_rows"])
        put("XHumanRawAgree", 100*human["raw"]["three_way_agreement"], "%.2f")
        put("XHumanWeightedAgree", 100*human["weighted"]["three_way_agreement"], "%.2f")
        put("XHumanKappa", human["weighted"]["three_way_kappa"], "%.2f")
        put("XHumanUnresolved", human["status_counts"].get("unresolved",0))
        put("XHumanCompound", human["decomposition_counts"].get("no",0))
    two_rater_path = os.path.join(validation, "two_rater_analysis", "summary.json")
    adjudicated_path = os.path.join(validation, "two_rater_analysis", "adjudicated_summary.json")
    if os.path.exists(two_rater_path):
        two = json.load(open(two_rater_path, encoding="utf-8"))
        put("XTwoRaterN", two["validated_rows"])
        put("XTwoRaterAgree", 100 * two["raw"]["agreement"], "%.2f")
        put("XTwoRaterKappa", two["raw"]["kappa"], "%.3f")
        put("XTwoRaterWeightedAgree", 100 * two["weighted"]["agreement"], "%.3f")
        put("XTwoRaterWeightedKappa", two["weighted"]["kappa"], "%.3f")
        put("XTwoRaterDisagreements", two["disagreements"])
    if os.path.exists(adjudicated_path):
        adjudicated = json.load(open(adjudicated_path, encoding="utf-8"))
        put("XAdjudicatedN", adjudicated["adjudicated_rows"])
    cross_path = os.path.join(experiments, "full_cross_family", "repaired_analysis.json")
    if os.path.exists(cross_path):
        cross = json.load(open(cross_path, encoding="utf-8"))
        put("XLlamaFullN", cross["llama_anchor_supported_claims"])
        put("XLlamaFullSsix", cross["llama_S6"], "%.3f")
        put("XLlamaFullSsixLo", cross["llama_S6_cluster_bootstrap_95"][0], "%.3f")
        put("XLlamaFullSsixHi", cross["llama_S6_cluster_bootstrap_95"][1], "%.3f")
        put("XLlamaAnchorAgree", 100 * cross["qwen_llama_anchor_agreement"]["agreement"], "%.2f")
        put("XLlamaAnchorKappa", cross["qwen_llama_anchor_agreement"]["kappa"], "%.3f")
    mutation_path = os.path.join(experiments, "mutation_recall", "summary.json")
    if os.path.exists(mutation_path):
        mutation = json.load(open(mutation_path, encoding="utf-8"))
        put("XMutationRecallPct", 100 * mutation["synthetic_recall"], "%.1f")
        put("XMutationRecallLo", 100 * mutation["synthetic_recall_wilson_95"][0], "%.1f")
        put("XMutationRecallHi", 100 * mutation["synthetic_recall_wilson_95"][1], "%.1f")
        put("XMutationFprPct", 100 * mutation["control_false_positive_rate"], "%.1f")
    audit_path = os.path.join(validation, "cross_model_audit_summary.json")
    if os.path.exists(audit_path):
        audit = json.load(open(audit_path, encoding="utf-8"))
        put("XDisagreementN", audit["disagreement_cases"])
        put("XQwenSupportedN", audit["qwen_supported_cases"])
        put("XQwenSupportedHumanPct", 100 * audit["qwen_supported_human_supported_rate"], "%.1f")
        put("XQwenContradictedN", audit["qwen_contradicted_cases"])
        put("XQwenContradictedHumanPct", 100 * audit["qwen_contradicted_human_contradicted_rate"], "%.1f")
        put("XSubtleMutationN", audit["subtle_mutation_cases"])
    ppv_path = os.path.join(experiments, "ppv_sensitivity", "summary.json")
    if os.path.exists(ppv_path):
        ppv = json.load(open(ppv_path, encoding="utf-8"))
        put("XHumanEventAuditN", ppv["audit_event_rows"])
        put("XHumanLabelSsixLo", ppv["human_label_S6_unresolved_event"], "%.3f")
        put("XHumanLabelSsixHi", ppv["human_label_S6_unresolved_censored"], "%.3f")
        put("XHumanLabelConN", ppv["human_verdict_counts"]["contradicted"])
        put("XHumanLabelSupN", ppv["human_verdict_counts"]["supported"])
        put("XHumanLabelUnrN", ppv["human_verdict_counts"]["unresolved"])
        put("XHumanPpv", 100 * ppv["resolved_contradiction_ppv"], "%.1f")
        put("XHumanPpvLo", 100 * ppv["resolved_contradiction_ppv_wilson_95"][0], "%.1f")
        put("XHumanPpvHi", 100 * ppv["resolved_contradiction_ppv_wilson_95"][1], "%.1f")
        put("XHumanSsixBootLo", ppv["function_cluster_bootstrap_95"]["unresolved_event"][0], "%.3f")
        put("XHumanSsixBootHi", ppv["function_cluster_bootstrap_95"]["unresolved_event"][1], "%.3f")
        put("XHumanOverlapN", ppv["two_rater_overlap"]["n"])
        put("XHumanOverlapCon", ppv["two_rater_overlap"]["new_rater_contradicted"])
        put("XHumanRecallSeven", ppv["recall_assumption_S6"]["values"]["0.7"], "%.3f")
        put("XHumanRecallEightEight", ppv["recall_assumption_S6"]["values"]["0.88"], "%.3f")
    write("macros.tex", "% generated by artifact/src/render.py -- do not edit\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())) + "\n")

def main():
    S = json.load(open(os.path.join(RESULTS, "summary.json")))
    for old in ("tab_hazard.tex", "tab_churn_jit.tex", "fig_changetype.tex"):
        p = os.path.join(GENERATED, old)
        if os.path.exists(p):
            os.remove(p)
    tab_dataset(S); tab_survival(S); tab_tests(S); fig_km(S); fig_forest(S); macros(S)
    print("rendered", sorted(os.listdir(GENERATED)))

if __name__ == "__main__":
    main()
