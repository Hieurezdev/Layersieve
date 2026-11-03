"""LongDialogue scored on the FIRST name the model commits to (bold span, else text after 'is likely'/'is'), with
asterisks stripped, vs the official word-match scorer; paired bootstrap vs ld200_g1w0, by nearest-MASK distance."""
import json, re, sys, os, numpy as np
sys.path.insert(0, "/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/eval/infinite_bench")
from compute_scores import get_score_one
S = os.path.dirname(os.path.abspath(__file__)); R = "results/selfext"
dist = json.load(open(S + "/ld_maskdist.json"))
ib = {json.loads(l)["id"]: json.loads(l) for l in open("data/infbench_ld200.jsonl")}
def first(pred):
    m = re.search(r"\*\*([^*]+)\*\*", pred)
    if m: return m.group(1)
    m = re.search(r"(?:is likely|is probably|appears to be|is)\s*:?\s*([A-Za-z][A-Za-z .'-]{0,40})", pred)
    return m.group(1) if m else pred[:40]
def sc_first(pred, ans):
    w = re.sub(r"[^A-Za-z ]", " ", first(pred)).upper().split()
    return float(any(a.upper() in w or a.upper() == " ".join(w) for a in ans))
def load(n):
    out = {}
    for f in n.split("+"):
        for l in open(f"{R}/{f}.jsonl"):
            r = json.loads(l); g = ib[r["id"]]
            out[r["id"]] = (100 * get_score_one(r["pred"], g["answers"], g["task"], "qwen"), 100 * sc_first(r["pred"], g["answers"]))
    return out
names = sys.argv[1:]; sc = {n: load(n) for n in ["ld200_g1w0"] + names}
ids = sorted(set.intersection(*map(set, sc.values())))
B = [(0, 32e3), (32e3, 48e3), (48e3, 1.4e5)]
print(f"n={len(ids)}  buckets {[sum(a <= dist[i] < b for i in ids) for a, b in B]}  (official | first-name)")
for k, lab in ((0, "official"), (1, "first-name")):
    print("---", lab)
    for n, s in sc.items():
        row = [np.mean([s[i][k] for i in ids if a <= dist[i] < b]) for a, b in B]
        d = np.array([s[i][k] - sc["ld200_g1w0"][i][k] for i in ids]); m = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
        print(f"{n:34s} all={np.mean([s[i][k] for i in ids]):5.1f}  " + "  ".join(f"{x:5.1f}" for x in row),
              f"  diff={d.mean():+.1f} CI[{np.percentile(m, 2.5):+.1f},{np.percentile(m, 97.5):+.1f}] p={2 * min((m <= 0).mean(), (m >= 0).mean()):.3f}")
