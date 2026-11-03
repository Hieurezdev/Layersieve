"""So diem cac file ket qua kv (results/kv50) voi dense va cat dan: so cau dung, TTFT, McNemar tren cung id."""
import glob
import json
import os
import sys
from math import comb

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/kv50")
load = lambda f: {r["id"]: r for r in map(json.loads, open(f))}
base = {"dense": load(f"{d}/kv50_dense.jsonl"), "catdan": load(f"{d}/kv50_catdan.jsonl")}


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


print(f"{'file':40s} {'dung':>8s} {'ttft':>6s}  vs dense (thang:thua, p)   vs catdan")
for f in sorted(glob.glob(f"{d}/*.jsonl")):
    s = load(f)
    t = [r.get("ttft") for r in s.values() if r.get("ttft") is not None]
    row = f"{os.path.basename(f):40s} {sum(r['ok'] for r in s.values()):3d}/{len(s):<4d} {sum(t) / len(t) if t else float('nan'):6.1f}"
    for b in base.values():
        w = sum(s[i]["ok"] and not b[i]["ok"] for i in s)
        l = sum(b[i]["ok"] and not s[i]["ok"] for i in s)
        row += f"  {w:2d}:{l:<2d} p={mcnemar(w, l):.3f}     "
    print(row)
