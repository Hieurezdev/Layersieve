"""Summarize scripts/probe_action.py.
  1. Trajectories: mean cost (token x layer), final keep, accuracy, mean loss increase, per kind and per task.
  2. Equal-cost check: for every oracle trajectory, the best fixed schedule at <= its cost, and the cheapest fixed
     schedule at >= its accuracy. Controls that only adapt per prompt (not per layer): for each prompt pick the cheapest
     fixed schedule whose loss increase is <= eps ("prompt-loss"), or that answers correctly ("prompt-outcome").
  3. Branch labels: how well do pre-cut features flag harmful cuts (L_cut - L_keep > delta)? AUC of single features,
     and logistic regression (5-fold, split by prompt) on attention-only vs attention + state/history features.
  Extra directories (closed-loop policy runs, --only-policies) add their trajectories to sections 1-2; the
  "pred_*" policies are compared with the fixed schedules like the oracles.
Usage: python scripts/summarize_actprobe.py results/actprobe/7b1m [results/actprobe/7b1m_policy]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

d = sys.argv[1]
T = [json.loads(l) for x in sys.argv[1:] for l in open(os.path.join(x, "traj.jsonl"))]
B = [json.loads(l) for l in open(os.path.join(d, "branches.jsonl"))]
skipped = {t["id"] for t in T if t.get("skipped")}
T = [t for t in T if t["id"] not in skipped]
ids = sorted({t["id"] for t in T})
tasks = sorted({t["task"] for t in T})
by = defaultdict(dict)                                     # id -> traj -> row
for t in T:
    by[t["id"]][t["traj"]] = t
ids = [i for i in ids if len(by[i]) == max(map(len, by.values()))]   # complete prompts only
n_task = {k: sum(by[i]["dense"]["task"] == k for i in ids) for k in tasks}
print(f"{len(ids)} prompts {n_task}, skipped (HF dense wrong) {len(skipped)}")
names = list(by[ids[0]])


def agg(name, sel=ids):
    rows = [by[i][name] for i in sel]
    return (np.mean([r["cost"] for r in rows]), np.mean([r["final_keep"] for r in rows]),
            np.mean([r["ok"] for r in rows]), np.mean([r["dloss"] for r in rows]))


print("\n## 1. Trajectories (cost = token x layer fraction; acc over dense-correct prompts)")
print(f"{'traj':20s} {'cost':>5s} {'keepL':>6s} {'acc':>5s} {'dloss':>7s}  " + " ".join(f"{k[:6]:>6s}" for k in tasks))
stats = {}
for n in names:
    c, k, acc, dl = agg(n)
    stats[n] = (c, acc)
    per = [agg(n, [i for i in ids if by[i][n]["task"] == t])[2] for t in tasks]
    print(f"{n:20s} {c:5.3f} {k:6.3f} {acc:5.3f} {dl:+7.3f}  " + " ".join(f"{p:6.2f}" for p in per))

print("\n## 2. Equal cost: oracles and closed-loop policies vs fixed schedules")
fixed = [n for n in names if n.startswith("fixed")]
for n in names:
    if not n.startswith(("local", "global", "pred")):
        continue
    c, acc = stats[n]
    at_cost = [stats[f][1] for f in fixed if stats[f][0] <= c + 1e-9]
    at_acc = [stats[f][0] for f in fixed if stats[f][1] >= acc - 1e-9]
    print(f"{n:20s} cost {c:.3f} acc {acc:.3f} | best fixed at <= cost: "
          f"{max(at_cost) if at_cost else float('nan'):.3f} | cheapest fixed at >= acc: "
          f"{min(at_acc) if at_acc else float('nan'):.3f}")
for eps in (0.01, 0.03, 0.1, 0.3):
    pick = [min((by[i][f] for f in fixed if by[i][f]["dloss"] <= eps), key=lambda r: r["cost"],
                default=by[i]["dense"]) for i in ids]
    print(f"prompt-loss eps={eps:<5g} cost {np.mean([p['cost'] for p in pick]):.3f} acc {np.mean([p['ok'] for p in pick]):.3f}")
pick = [min((by[i][f] for f in fixed if by[i][f]["ok"]), key=lambda r: r["cost"], default=by[i]["dense"]) for i in ids]
print(f"prompt-outcome   cost {np.mean([p['cost'] for p in pick]):.3f} acc 1.000 (upper bound of per-prompt schedule choice)")


def auc(score, y):
    """P(score of a positive > score of a negative), ties = 1/2."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable").astype(float)
    s = np.asarray(score)
    for v in np.unique(s):                                   # average ranks of ties
        m = s == v
        r[m] = r[m].mean()
    return (r[y].sum() - y.sum() * (y.sum() - 1) / 2) / (y.sum() * (~y).sum())


def logreg_cv(X, y, groups, k=5, l2=1e-2, iters=2000, lr=0.5):
    """Out-of-fold probabilities of an L2 logistic regression, folds split by group (prompt)."""
    ug = np.unique(groups)
    rng = np.random.RandomState(0)
    fold = dict(zip(ug, rng.permutation(len(ug)) % k))
    f = np.array([fold[g] for g in groups])
    p = np.zeros(len(y))
    for j in range(k):
        tr, te = f != j, f == j
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w, b = np.zeros(X.shape[1]), 0.0
        yt = y[tr].astype(float)
        pw = (1 - yt.mean()) / max(yt.mean(), 1e-9)          # balance classes
        sw = np.where(yt > 0, pw, 1.0)
        for _ in range(iters):
            z = 1 / (1 + np.exp(-(Xt @ w + b)))
            g = sw * (z - yt)
            w -= lr * (Xt.T @ g / sw.sum() + l2 * w)
            b -= lr * g.sum() / sw.sum()
        p[te] = Xe @ w + b
    return p


print(f"\n## 3. Branch labels ({len(B)} branches, {len({b['id'] for b in B})} prompts)")
B = [b for b in B if b["id"] in set(ids) and b["n_tok"] > 0]
dl = np.array([b["loss_cut"] - b["loss_keep"] for b in B])
g = np.array([b["id"] for b in B])
L = max(b["layer"] for b in B) + 1


def col(k, fill=0.5):
    return np.array([fill if b.get(k) is None else b[k] for b in B], float)


feat = {
    "frac": col("frac"), "n_tok": np.log(col("n_tok")), "mass_max": col("mass_max"), "mass_mean": col("mass_mean"),
    "dmax_pct": col("dmax_pct"), "dmax_rel": np.log(col("dmax_rel", 1.0) + 1e-9),
    "layer": col("layer") / L, "alive": col("alive"),
    "prev_pct_mean": col("prev_pct_mean"), "prev_pct_max": col("prev_pct_max"),
    "rank_jump": col("prev_pct_max") - col("dmax_pct"),
}
ATTN = ["frac", "n_tok", "mass_max", "mass_mean", "dmax_pct", "dmax_rel"]
STATE = ATTN + ["layer", "alive", "prev_pct_mean", "prev_pct_max", "rank_jump"]
print("dloss quantiles (50/90/99/max):", np.round(np.quantile(dl, [0.5, 0.9, 0.99, 1.0]), 3))
for delta in (0.05, 0.3, 1.0):
    y = dl > delta
    print(f"\ndelta={delta}: harmful {y.sum()}/{len(y)} ({y.mean():.1%}) in {len(set(g[y]))} prompts;"
          f" by task " + ", ".join(f"{t} {y[np.array([b['task'] == t for b in B])].mean():.1%}" for t in tasks))
    if y.sum() < 5:
        continue
    print("  single-feature AUC:", ", ".join(f"{k} {auc(v, y):.3f}" for k, v in feat.items()))
    for name, fs in (("attention-only", ATTN), ("attention+state", STATE)):
        X = np.stack([feat[k] for k in fs], 1)
        print(f"  logreg {name:16s} AUC {auc(logreg_cv(X, y, g), y):.3f}")
ev = np.array([b["ev_in_d"] if b.get("ev_in_d") is not None else np.nan for b in B])
has = ~np.isnan(ev)
if has.any():
    hit = ev[has] > 0
    print(f"\nkv/vt: cuts that remove evidence {hit.sum()}/{has.sum()};"
          f" harmful (dloss>0.3) among them {np.mean(dl[has][hit] > 0.3) if hit.any() else float('nan'):.2f},"
          f" among the rest {np.mean(dl[has][~hit] > 0.3):.3f}")
    print(f"  mass_max AUC for 'removes evidence': {auc(feat['mass_max'][has], hit):.3f};"
          f" by layer (evidence-removing cuts / harmful):",
          {int(l): (int((hit & (col('layer')[has] == l)).sum()), int((hit & (dl[has] > 0.3) & (col('layer')[has] == l)).sum()))
           for l in np.unique(col('layer')[has]) if (hit & (col('layer')[has] == l)).any()})
