"""Accuracy-cost frontier of results/frontier (run_night6.sh, block F): progressive vs one cut (both unit-level) at
C = .55 / .62 / .72 on the hard set (% of 120 dense-correct prompts kept) and RULER 32k (6 hard tasks, mean).
    python scripts/summarize_frontier.py
"""
import json
import os
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FR = os.path.join(ROOT, "results/frontier")
gold = {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, "data/ruler_eval.jsonl")))}


def ruler(p):
    s, n = defaultdict(float), defaultdict(int)
    for x in map(json.loads, open(p)):
        g = gold[x["id"]]
        hits = [a.lower() in x["pred"].lower() for a in g["answers"]]
        s[g["task"].split("@")[0]] += 100.0 * (max(hits) if g["metric"] == "part" else sum(hits) / len(hits))
        n[g["task"].split("@")[0]] += 1
    return {t: s[t] / n[t] for t in s}, sum(n.values())


rows = []
for f in sorted(os.listdir(f"{FR}/hard")):
    name = f[:-6]
    h = [json.loads(l) for l in open(f"{FR}/hard/{f}")]
    rp = f"{FR}/ruler/{name}.jsonl"
    r, nr = ruler(rp) if os.path.exists(rp) else ({}, 0)
    c, kind = re.match(r"c(\d+)_(one|prog|acc)", name).groups()
    rows.append((int(c), kind, name, 100 * sum(x["ok"] for x in h) / len(h), r, nr))
tasks = ["cwe", "fwe", "niah_multikey_3", "vt", "niah_multivalue", "qa_2"]
print("| C | kind | config | hard % | RULER32k mean | " + " | ".join(tasks) + " | n |")
print("|---|---|---|---|---|" + "---|" * (len(tasks) + 1))
for c, kind, name, h, r, nr in sorted(rows):
    m = f"{sum(r.values()) / len(r):.1f}" if len(r) == len(tasks) else "-"
    print(f"| .{c} | {kind} | {name.split('_', 2)[2]} | {h:.0f} | {m} | " + " | ".join(f"{r[t]:.0f}" if t in r else "-" for t in tasks) + f" | {nr} |")
