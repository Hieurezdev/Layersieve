"""Score the baseline comparison (scripts/eval_bench.py, baselines/asl_driver.py outputs) with the official metrics:
InfiniteBench = compute_scores.get_score_one of the ASL repo (same as the InfiniteBench repo), RULER =
string_match_all / string_match_part. Prints one markdown table per benchmark, one row per result file (method).
    python scripts/score_bench.py                     # results/bench/{infbench,ruler}/*.jsonl
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--asl", default="/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/eval/infinite_bench")
ap.add_argument("--results", default=os.path.join(ROOT, "results/bench"))
ap.add_argument("--infbench", default=os.path.join(ROOT, "data/infbench.jsonl"))
ap.add_argument("--ruler", default=os.path.join(ROOT, "data/ruler_eval.jsonl"))
a = ap.parse_args()
sys.path.insert(0, a.asl)
from compute_scores import get_score_one  # noqa: E402

IB_TASKS = ["math_find", "code_debug", "longdialogue_qa_eng", "kv_retrieval"]
RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
               "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]


def ruler_score(pred, refs, metric):
    hits = [r.lower() in pred.lower() for r in refs]
    return 100.0 * (max(hits) if metric == "part" else sum(hits) / len(hits))


def gold(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def table(bench, title, ref, cols, score_fn, col_of):
    """One row per result file of results/<bench>; cell = mean score (0-100) over the examples of that column;
    a cell computed on fewer examples than the eval set has shows the count; Avg only when every cell is complete."""
    files = sorted(glob.glob(f"{a.results}/{bench}/*.jsonl"))
    if not files or not ref:
        return
    n_ref = defaultdict(int)
    for r in ref.values():
        n_ref[col_of(r)] += 1
    print(f"## {title}\n")
    print("| Method | " + " | ".join(cols) + " | Avg | done |")
    print("|---|" + "---|" * (len(cols) + 2))
    for p in files:
        s, n = defaultdict(float), defaultdict(int)
        for x in map(json.loads, open(p)):
            g = ref.get(x["id"])
            if g is None:
                continue
            s[col_of(g)] += score_fn(x["pred"], g)
            n[col_of(g)] += 1
        vals = [s[c] / n[c] if n[c] else None for c in cols]
        full = [v for v, c in zip(vals, cols) if v is not None and n[c] == n_ref[c]]
        avg = f"{sum(full) / len(full):.1f}" if len(full) == len(cols) else "-"
        cells = " | ".join("-" if v is None else f"{v:.1f}" + ("" if n[c] == n_ref[c] else f" ({n[c]})")
                           for v, c in zip(vals, cols))
        print(f"| {os.path.basename(p)[:-6]} | {cells} | {avg} | {sum(n.values())}/{len(ref)} |")
    print()


ib = gold(a.infbench)
table("infbench", "InfiniteBench (100 / task, seed 42)", ib, IB_TASKS,
      lambda pred, g: 100.0 * get_score_one(pred, g["answers"], g["task"], "qwen"), lambda g: g["task"])
ru = gold(a.ruler)
lens = sorted({g["task"].split("@")[1] for g in ru.values()}, key=int)
score = lambda pred, g: ruler_score(pred, g["answers"], g["metric"])  # noqa: E731
table("ruler", "RULER (13 task x 100, seed 42): average per length", ru, [f"{int(L) // 1024}k" for L in lens], score,
      lambda g: f"{int(g['task'].split('@')[1]) // 1024}k")
for L in lens:
    sub = {k: g for k, g in ru.items() if g["task"].endswith("@" + L)}
    table("ruler", f"RULER {int(L) // 1024}k per task", sub, RULER_TASKS, score, lambda g: g["task"].split("@")[0])
