"""Compare answer-selection rules on --select runs with cross-context log-probs. usage: gates.py run1 run2 ..."""
import json, sys, random, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, names = sys.argv[:1], sys.argv[1:]
import selscore as S  # noqa  (its loop runs on no names)
RULES = {
    "own":    lambda r: r["lp_recall"] > r["lp_dense"],
    "verify": lambda r: r["lp_recall_in_dense"] > r["lp_dense"],
    "sum":    lambda r: r["lp_recall"] + r["lp_recall_in_dense"] > r["lp_dense"] + r["lp_dense_in_recall"],
}
tot = {k: [] for k in ["dense", "recall", *RULES, "oracle"]}
print(f"{'run':24s} {'n':>3s} " + " ".join(f"{k:>7s}" for k in tot))
for n in names:
    rs = [json.loads(l) for l in open(f"results/selfext/{n}.jsonl")]
    rs = [r for r in rs if "lp_recall_in_dense" in r]
    d = [S.score(r["id"], r["pred_dense"]) for r in rs]; c = [S.score(r["id"], r["pred_recall"]) for r in rs]
    row = {"dense": d, "recall": c, "oracle": [max(x, y) for x, y in zip(d, c)]}
    for k, f in RULES.items():
        row[k] = [y if f(r) else x for r, x, y in zip(rs, d, c)]
    for k in tot: tot[k].append(sum(row[k]) / len(rs))
    print(f"{n:24s} {len(rs):3d} " + " ".join(f"{100*sum(row[k])/len(rs):7.1f}" for k in tot))
print(f"{'mean over tasks':24s} {'':3s} " + " ".join(f"{100*sum(v)/len(v):7.1f}" for v in tot.values()))
