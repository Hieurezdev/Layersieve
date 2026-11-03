"""Summarize scripts/run_floor.sh (results/floor): how the floor phi (fraction of the prunable prompt left at the last
layer) should be set across model sizes and prompt lengths. Prints markdown tables.
    python scripts/summarize_floor.py
"""
import json
import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results/floor")
MODELS = [("3b", "3B", 36), ("7b", "7B", 28), ("7b1m", "7B-1M", 28), ("14b", "14B", 48), ("14b1m", "14B-1M", 48),
          ("32b", "32B", 64)]
FLOORS = ["0.05", "0.10", "0.15", "0.25", "0.35"]
BINS = [(0, 10000, "<10k"), (10000, 17000, "10-17k"), (17000, 10 ** 9, ">17k")]
RU_TASKS = ["cwe", "fwe", "niah_multikey_3"]


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pct(rows):
    return f"{100 * sum(r['ok'] for r in rows) / len(rows):.0f}" if rows else "-"


print("## A. Hard set: % of dense-correct answers kept (l0 = 0.5L, rho = 0.4; depth-scaled: rho = 1 - 0.6^(28/L))\n")
print("| Model | " + " | ".join(f"phi={f}" for f in FLOORS) + " | phi=.25, depth-scaled rho | l0=.61L, depth-scaled rho |")
print("|---|" + "---|" * (len(FLOORS) + 2))
by_len = defaultdict(list)
for key, name, L in MODELS:
    cells = []
    for f in FLOORS:
        rows = load(f"{R}/hard/{key}_f{f}.jsonl")
        cells.append(pct(rows))
        for r in rows:
            by_len[(f, next(b[2] for b in BINS if b[0] <= r["len"] < b[1]))].append(r)
    cells.append(pct(load(f"{R}/hard/{key}_scaled_f0.25.jsonl")) if L != 28 else "= .25")
    cells.append(pct(load(f"{R}/hard/{key}_rel_f0.25.jsonl")))
    print(f"| {name} ({L}) | " + " | ".join(cells) + " |")
print("\n### By prompt length (all models pooled)\n")
print("| Length | " + " | ".join(f"phi={f}" for f in FLOORS) + " |")
print("|---|" + "---|" * len(FLOORS))
for _, _, b in BINS:
    print(f"| {b} | " + " | ".join(f"{pct(by_len[(f, b)])} (n={len(by_len[(f, b)])})" for f in FLOORS) + " |")


def ruler_score(pred, refs, metric):
    hits = [r.lower() in pred.lower() for r in refs]
    return 100.0 * (max(hits) if metric == "part" else sum(hits) / len(hits))


gold = {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, "data/ruler_eval.jsonl")))}


def ruler(path):
    s, n = defaultdict(float), defaultdict(int)
    for x in load(path):
        g = gold[x["id"]]
        k = (g["task"].split("@")[0], int(g["task"].split("@")[1]) // 1024)
        s[k] += ruler_score(x["pred"], g["answers"], g["metric"])
        n[k] += 1
    return {k: (s[k] / n[k], n[k]) for k in s}


def cell(d, k):
    return "-" if k not in d else f"{d[k][0]:.0f}" + ("" if d[k][1] == 100 else f" ({d[k][1]})")


print("\n## B. RULER 32k (l0 = 0.5L, rho = 0.4), score per task\n")
print("| Model | Task | dense | " + " | ".join(f"phi={f}" for f in FLOORS) + " | phi=.25, depth-scaled rho | l0=.61L, depth-scaled rho |")
print("|---|---|---|" + "---|" * (len(FLOORS) + 2))
for key, name, L in MODELS:
    res = {c: ruler(f"{R}/ruler/{key}_{c}.jsonl") for c in ["dense"] + [f"f{f}" for f in FLOORS] + ["scaled_f0.25", "rel_f0.25"]}
    if key == "7b1m":                                   # l0 .61L = 17, rho .4: the main config, run in results/bench
        res["rel_f0.25"] = ruler(os.path.join(ROOT, "results/bench/ruler/ls_unit_d0.4_f0.25_noref.jsonl"))
    if not any(res.values()):
        continue
    for t in RU_TASKS:
        print(f"| {name} | {t} | " + " | ".join(cell(res[c], (t, 32)) for c in res) + " |")

print("\n## C. 7B-1M, l0 = 17 (main config): cwe / fwe at 32k vs 128k\n")
print("| Task | Length | " + " | ".join(f"phi={f}" for f in FLOORS) + " |")
print("|---|---|" + "---|" * len(FLOORS))
res = {f: ruler(f"{R}/ruler/7b1m17_f{f}.jsonl") for f in FLOORS}
res["0.25"] = ruler(os.path.join(ROOT, "results/bench/ruler/ls_unit_d0.4_f0.25_noref.jsonl"))
for t in ("cwe", "fwe"):
    for ln in (32, 128):
        print(f"| {t} | {ln}k | " + " | ".join(cell(res[f], (t, ln)) for f in FLOORS) + " |")
