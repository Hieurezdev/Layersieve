"""Score --select runs: plain (pred_dense), recalled (pred_recall), selected (pred), oracle; paired bootstrap of
selected - plain. usage: selscore.py run1 run2 ... (names under results/selfext). Uses only the answer fields."""
import json, re, sys, random, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
G = {}
for f in ["data/ruler_eval.jsonl", "data/infbench.jsonl", "data/infbench_kv86.jsonl", "data/infbench_ld200.jsonl"]:
    for l in open(f):
        r = json.loads(l); r.pop("prompt_ids", None); G[r["id"]] = r
def first(pred):
    m = re.search(r"\*\*([^*]+)\*\*", pred)
    if m: return m.group(1)
    m = re.search(r"(?:is likely|is probably|appears to be|is)\s*:?\s*([A-Za-z][A-Za-z .'-]{0,40})", pred)
    return m.group(1) if m else pred[:40]
def score(i, pred):
    g = G[i]; a = [str(x) for x in (g.get("answers") or g.get("outputs"))]; t = g["task"].split("@")[0]; pl = pred.lower()
    if t == "longdialogue_qa_eng":
        w = re.sub(r"[^A-Za-z ]", " ", first(pred)).upper().split()
        return float(any(x.upper() in w or x.upper() == " ".join(w) for x in a))
    if t == "math_find":
        m = re.findall(r"-?\d+\.?\d*", pred); return float(bool(m) and float(m[0]) == float(a[0]))
    if t == "code_debug":
        m = re.search(r"\b([ABCD])\b", pred); return float(any(x in pred for x in a) or (m is not None and m.group(1) in a))
    if t == "kv_retrieval" or t.startswith("qa"):
        return float(any(x.lower() in pl for x in a))
    return sum(x.lower() in pl for x in a) / len(a)
print(f"{'run':24s} {'n':>3s} {'dense':>6s} {'recall':>6s} {'select':>6s} {'oracle':>6s}  sel-dense CI          chose recall")
for n in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f"results/selfext/{n}.jsonl")]
    d = [score(r["id"], r["pred_dense"]) for r in rs]; c = [score(r["id"], r["pred_recall"]) for r in rs]
    s = [score(r["id"], r["pred"]) for r in rs]; o = [max(x, y) for x, y in zip(d, c)]
    diff = [x - y for x, y in zip(s, d)]
    bs = sorted(sum(random.choice(diff) for _ in diff) / len(diff) for _ in range(2000))
    m = lambda v: 100 * sum(v) / len(v)
    print(f"{n:24s} {len(rs):3d} {m(d):6.1f} {m(c):6.1f} {m(s):6.1f} {m(o):6.1f}  {m(diff):+5.1f} [{100*bs[50]:+.1f},{100*bs[1949]:+.1f}]  "
          f"{sum(r['chose'] == 'recall' for r in rs)}/{len(rs)}")
