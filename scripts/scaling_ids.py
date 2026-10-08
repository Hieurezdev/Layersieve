"""Per-model sample for the scaling sweep: from a dense run on data/hard.jsonl, pick up to --k ids per task that the
model answers correctly (random, seed 0), so pruning is only judged on questions the dense model can answer.
    python scripts/scaling_ids.py results/scaling/dense/14b.jsonl results/scaling/ids/14b.txt
"""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("dense")
ap.add_argument("out")
ap.add_argument("--k", type=int, default=30)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.dense)]
ids = []
for t in sorted({r["task"] for r in rows}):
    ok = sorted(r["id"] for r in rows if r["task"] == t and r["ok"])
    ids += random.Random(0).sample(ok, min(a.k, len(ok)))
    print(t, f"dense {sum(r['ok'] for r in rows if r['task'] == t)}/{sum(r['task'] == t for r in rows)}",
          "picked", min(a.k, len(ok)))
open(a.out, "w").write("\n".join(ids) + "\n")
