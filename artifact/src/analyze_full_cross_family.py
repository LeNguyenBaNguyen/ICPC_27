"""Offline analysis for the completed full Llama cross-family replay."""
import csv
import hashlib
import json
import random
import argparse
from pathlib import Path

from common import CONFIG

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifact"
RESULTS = ART / "results"
OUT = ART / "experiments/full_cross_family"


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def km_s6(records):
    rows = []
    for record in records:
        for arm, idx in record["anchor_supported"]:
            if arm != "llm":
                continue
            event = record["first_contradiction"].get(f"{arm}:{idx}")
            last = record["last_effective_touch"]
            rows.append((event if event and event <= 6 else None, min(6, last)))
    survival = 1.0
    for k in range(1, 7):
        n = sum((event is None or event >= k) and censor >= k for event, censor in rows)
        d = sum(event == k for event, _ in rows)
        if n:
            survival *= 1 - d / n
    return survival, len(rows)


def cluster_bootstrap(records, B=2000, seed=20261112):
    rng = random.Random(seed)
    records = list(records)
    draws = []
    for _ in range(B):
        sample = [records[i] for i in rng.choices(range(len(records)), k=len(records))]
        draws.append(km_s6(sample)[0])
    draws.sort()
    return [draws[int((len(draws)-1)*p)] for p in (0.025, 0.975)]


def model_anchor_agreement(trajectories, qwen):
    labels_1, labels_2 = [], []
    for fid, record in trajectories.items():
        q = {(v["arm"], v["idx"]): v["status"] for v in qwen[fid]["anchor"]}
        l = {(v["arm"], v["idx"]): v["status"] for v in record["anchor"]["verdicts"]}
        for key in sorted(set(q) & set(l)):
            labels_1.append(q[key]); labels_2.append(l[key])
    labels = ("supported", "contradicted", "unresolved")
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(labels_1, labels_2): matrix[a][b] += 1
    n = len(labels_1); observed = sum(matrix[x][x] for x in labels) / n
    left = {x: sum(matrix[x].values()) / n for x in labels}
    right = {x: sum(matrix[y][x] for y in labels) / n for x in labels}
    expected = sum(left[x] * right[x] for x in labels)
    return {"n": n, "agreement": observed, "kappa": (observed-expected)/(1-expected),
            "matrix_qwen_rows_llama_columns": matrix}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="trajectories.json")
    args = parser.parse_args()
    trajectories = json.loads((OUT / args.trajectories).read_text(encoding="utf-8"))
    qwen = read("verdicts.json")
    records = [trajectories[fid] for fid in sorted(trajectories)]
    s6, n = km_s6(records)
    analysis = {
        "complete_functions": len(records), "llama_anchor_supported_claims": n,
        "llama_S6": s6, "llama_S6_cluster_bootstrap_95": cluster_bootstrap(records),
        "qwen_llama_anchor_agreement": model_anchor_agreement(trajectories, qwen),
        "unparsed_verdicts": sum(1 for record in records for v in record["anchor"]["verdicts"] if not v["parsed"])
            + sum(1 for record in records for commit in record["commits"] for v in commit["verdicts"] if not v["parsed"]),
        "source_sha256": {name: sha(RESULTS / name) for name in ("functions.json", "claims.json", "verdicts.json")},
        "interpretation": "Full main-cohort Llama replay is a model-conditional detection curve. Agreement with Qwen is not human accuracy and does not identify semantic survival.",
    }
    output_name = "repaired_analysis.json" if args.trajectories.startswith("repaired_") else "analysis.json"
    (OUT / output_name).write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
