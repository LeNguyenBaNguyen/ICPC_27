"""Exploratory removal of click pager functions contributing four intent events.

Uses the archived period table and the original model specification. No new
bootstrap, model selection or confirmatory verdict. Writes separate results.
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import CT, cloglog, idx
from stats_core import hr_ci, wald


def main():
    root = Path(__file__).resolve().parents[1]
    p = pd.read_csv(root / "results" / "claim_periods.csv")
    p = p[(p.cohort == "main") & (p.arm == "llm")].copy()
    p["pcat"] = p.pcat.astype(str)
    for ty in ("structural", "behavioural", "intent"):
        p["ty_" + ty] = (p.type == ty).astype(int)
    tcols = ["ty_structural", "ty_behavioural", "ty_intent"]
    ct = ["ct_" + c for c in CT if c != "local_edit" and p.loc[p["ct_" + c] == 1, "event"].sum() >= 5]
    covs = tcols + ["log_lines_fn", "log_lines_file"] + ct
    funcs = json.loads((root / "results" / "functions.json").read_text(encoding="utf-8"))
    names = {"_nullpager", "_pipepager", "_tempfilepager"}
    excluded = sorted(f["id"] for f in funcs if f["repo"] == "click" and f["qualname"] in names)
    if len(excluded) != 3:
        raise ValueError("Unexpected pager exclusion set")
    results = {"exploratory": True, "excluded_fids": excluded,
               "reason": "Three click pager functions account for four of six recorded intent events; chosen after review.",
               "covariates_fixed_from_full_sample": covs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, data in (("full", p), ("without_click_pagers", p[~p.fid.isin(excluded)])):
            fit = cloglog(data, covs)
            finite = bool(np.isfinite(fit["beta"]).all() and np.isfinite(fit["V"]).all())
            results[name] = {"functions": int(data.fid.nunique()), "events": int(data.event.sum()),
                             "intent_events": int(data.loc[data.type == "intent", "event"].sum()),
                             "iterations": fit["iter"], "finite_fit": finite,
                             "max_abs_covariate_beta": float(np.abs(fit["beta"][idx(fit, covs)]).max()),
                             "joint_type_wald_p": wald(fit, idx(fit, tcols))[1],
                             "intent_HR": hr_ci(fit, fit["cols"].index("ty_intent"))}
    archived = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
    if abs(results["full"]["joint_type_wald_p"] - archived["H2_wald_p"]) > 1e-7:
        raise ValueError("Full-model replication does not match archived Wald p")
    results["interpretation"] = "Sparse-event, exploratory sensitivity. Wald inference remains fragile; this is not independent endpoint validation."
    out = root / "validation"
    out.mkdir(exist_ok=True)
    (out / "h2_sensitivity.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
