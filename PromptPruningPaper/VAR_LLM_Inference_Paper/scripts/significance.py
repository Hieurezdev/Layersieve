"""Paired comparison of two methods on the baseline-comparison sets (same examples, per-example official scores as in
scripts/score_bench.py): mean difference with a 95% paired-bootstrap interval (10k resamples, seed 0) and the two-sided
bootstrap p-value, per benchmark (InfiniteBench; RULER per length and pooled).
    python scripts/significance.py ls_unit_d0.4_f0.25_r22 m_asl_t0.3_f0.375 [more pairs as A:B ...]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, "/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/eval/infinite_bench")
from compute_scores import get_score_one  # noqa: E402

R = os.environ.get("R") or os.path.join(ROOT, "results/bench")
IB = os.environ.get("IB") or os.path.join(ROOT, "data/infbench.jsonl")
RU = os.environ.get("RU") or os.path.join(ROOT, "data/ruler_eval.jsonl")


def gold(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def ruler_score(pred, refs, metric):
    hits = [r.lower() in pred.lower() for r in refs]
    return 100.0 * (max(hits) if metric == "part" else sum(hits) / len(hits))


ib, ru = gold(IB), gold(RU)


def scores(bench, name):
    p = f"{R}/{bench}/{name}.jsonl"
    out = {}
    if not os.path.exists(p):
        return out
    for x in map(json.loads, open(p)):
        if bench == "infbench" and x["id"] in ib:
            g = ib[x["id"]]
            out[x["id"]] = 100.0 * get_score_one(x["pred"], g["answers"], g["task"], "qwen")
        elif bench == "ruler" and x["id"] in ru:
            g = ru[x["id"]]
            out[x["id"]] = ruler_score(x["pred"], g["answers"], g["metric"])
    return out


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return d.mean(), lo, hi, min(p, 1.0)


def compare(a, b):
    print(f"\n### {a} - {b}\n\n| Set | n | diff | 95% CI | p |\n|---|---|---|---|---|")
    for bench in ("infbench", "ruler"):
        sa, sb = scores(bench, a), scores(bench, b)
        ids = sorted(set(sa) & set(sb))
        groups = {"InfiniteBench": ids} if bench == "infbench" else {}
        if bench == "ruler":
            for L in ("32768", "65536", "131072"):
                groups[f"RULER {int(L) // 1024}k"] = [i for i in ids if i.startswith(f"ruler{L}_")]
            groups["RULER all"] = ids
            groups["RULER all w/o cwe"] = [i for i in ids if "_cwe_" not in i]
        for g, gi in groups.items():
            if not gi:
                continue
            d = np.array([sa[i] - sb[i] for i in gi])
            m, lo, hi, p = boot(d)
            print(f"| {g} | {len(gi)} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.3f} |")


pairs = sys.argv[1:] or ["ls_unit_d0.4_f0.25_r22:m_asl_t0.3_f0.375", "ls_unit_d0.4_f0.25_r22:dense",
                         "ls_unit_d0.4_f0.25_noref:m_asl_t0.3_f0.375", "ls_unit_d0.4_f0.25_r22:m_fastkv_l15_f0.375",
                         "ls_unit_d0.4_f0.25_r22:m_slim_0.75_0.55_0.35", "m_asl_t0.3_f0.375:dense"]
for pr in pairs:
    compare(*pr.split(":"))
