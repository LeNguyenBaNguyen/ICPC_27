"""Small, dependency-free statistical core: discrete-time complementary log-log (grouped-time proportional hazards)
regression with cluster-robust (sandwich) covariance, Wald tests, Kaplan-Meier on integer time, Holm adjustment."""
import math
import numpy as np
from scipy.stats import chi2, norm

def fit_cloglog(X, y, groups, max_iter=100, tol=1e-9):
    """Fisher scoring for P(y=1) = 1 - exp(-exp(X b)). Returns dict with beta, naive and cluster-robust covariance.
    Robust covariance: H^-1 (sum_g s_g s_g') H^-1 * G/(G-1)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    b = np.zeros(X.shape[1]); b[0] = math.log(-math.log(1 - max(1e-4, y.mean()))) if X.shape[1] else 0.0
    for it in range(max_iter):
        eta = np.clip(X @ b, -30, 5); mu = np.exp(eta); p = np.clip(1 - np.exp(-mu), 1e-12, 1 - 1e-12)
        dp = mu * np.exp(-mu)
        w = dp ** 2 / (p * (1 - p))
        z = eta + (y - p) / dp
        XtW = X.T * w
        H = XtW @ X
        try:
            b_new = np.linalg.solve(H, XtW @ z)
        except np.linalg.LinAlgError:
            b_new = np.linalg.lstsq(H, XtW @ z, rcond=None)[0]
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new; break
        b = b_new
    eta = np.clip(X @ b, -30, 5); mu = np.exp(eta); p = np.clip(1 - np.exp(-mu), 1e-12, 1 - 1e-12)
    dp = mu * np.exp(-mu)
    w = dp ** 2 / (p * (1 - p))
    H = (X.T * w) @ X
    Hinv = np.linalg.pinv(H)
    score_i = X * ((y - p) * dp / (p * (1 - p)))[:, None]
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    S = np.zeros((len(uniq), X.shape[1]))
    np.add.at(S, inv, score_i)
    G = len(uniq)
    meat = S.T @ S * (G / (G - 1) if G > 1 else 1.0)
    V = Hinv @ meat @ Hinv
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"beta": b, "V": V, "V_naive": Hinv, "ll": ll, "n": len(y), "events": int(y.sum()), "clusters": G, "iter": it + 1}

def wald(fit, idx):
    b = fit["beta"][idx]; V = fit["V"][np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    return stat, float(chi2.sf(stat, len(idx)))

def hr_ci(fit, i):
    b = fit["beta"][i]; se = math.sqrt(max(fit["V"][i, i], 0))
    return {"HR": math.exp(b), "lo": math.exp(b - 1.96 * se), "hi": math.exp(b + 1.96 * se), "p": float(2 * norm.sf(abs(b / se))) if se > 0 else float("nan")}

def km_int(times, events, kmax):
    """Kaplan-Meier on integer time 1..kmax. times = last observed touch (event or censoring)."""
    times = np.asarray(times); events = np.asarray(events)
    S, out, haz = 1.0, [1.0], []
    for t in range(1, kmax + 1):
        n = int(np.sum(times >= t)); d = int(np.sum((times == t) & (events == 1)))
        h = d / n if n else 0.0
        S *= 1 - h
        out.append(S); haz.append((h, n, d))
    return out, haz

def quantile_time(curve, q):
    """First t with S(t) <= 1 - q (e.g. q=0.1 -> time to 10% claim death); None if not reached."""
    for t, s in enumerate(curve):
        if s <= 1 - q + 1e-12:
            return t
    return None

def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)); out[k] = running
    return out

def auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y); score = np.asarray(score, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def kappa(a, b, w=None):
    a, b = list(a), list(b)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    if len(a) == 0:
        return float("nan")
    cats = sorted(set(a) | set(b)); W = w.sum()
    po = sum(wi for x, z, wi in zip(a, b, w) if x == z) / W
    pa = {c: sum(wi for x, wi in zip(a, w) if x == c) / W for c in cats}
    pb = {c: sum(wi for z, wi in zip(b, w) if z == c) / W for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
