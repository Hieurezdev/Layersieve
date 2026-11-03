"""Score RULER / InfiniteBench runs on common ids. usage: rscore.py base run1 run2 ...  (names under results/selfext,
or a path). RULER: metric field (all = fraction of answers present, part = any). InfiniteBench kv: any answer in pred;
math_find: first number == answer; code_debug: option letter or name. Paired bootstrap vs base."""
import json, re, sys, random
R = "results/selfext/"
G = {}
for f in ["data/ruler_eval.jsonl", "data/infbench.jsonl", "data/infbench_kv86.jsonl"]:
    for l in open(f):
        r = json.loads(l); r.pop("prompt_ids", None); G[r["id"]] = r
def load(n):
    p = n if "/" in n else R + n + ".jsonl"
    return {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(p)}
def score(i, pred):
    g = G[i]; a = g.get("answers") or g.get("outputs"); a = a if isinstance(a, list) else [a]; t = g["task"].split("@")[0]
    pl = pred.lower()
    if t == "math_find":
        m = re.findall(r"-?\d+\.?\d*", pred); return float(bool(m) and float(m[0]) == float(a[0]))
    if t == "code_debug":
        opts = ["A", "B", "C", "D"]; m = re.search(r"\b([ABCD])\b", pred)
        return float(any(x in pred for x in a) or (m is not None and m.group(1) in a))
    if g.get("metric", "").endswith("part") or t in ("kv_retrieval",) or t.startswith("qa"):
        return float(any(str(x).lower() in pl for x in a))
    return sum(str(x).lower() in pl for x in a) / len(a)
runs = {n: load(n) for n in sys.argv[1:]}
ids = sorted(set.intersection(*map(set, runs.values())))
base = sys.argv[1]
sc = {n: [score(i, r[i]) for i in ids] for n, r in runs.items()}
print("n =", len(ids))
for n, s in sc.items():
    d = [x - y for x, y in zip(s, sc[base])]
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(2000)) if d else [0]
    print(f"{n:32s} {100*sum(s)/len(s):6.1f}  diff {100*sum(d)/len(d):+6.1f}  CI[{100*bs[50]:+.1f},{100*bs[1949]:+.1f}]"
          f"  worse {sum(x<0 for x in d)} better {sum(x>0 for x in d)}")
