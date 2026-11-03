"""Train the harm predictors used by the closed-loop policies of scripts/probe_action.py (--policies, --harm-model).
Label: L_cut - L_keep > delta on the branches of a probe run. 5 folds split by prompt; each fold's model is used only on
its held-out prompts. Models: logistic regression on attention features (lr_attn) and on attention + state features
(lr_state), and a 2-layer MLP on attention + state (mlp_state). Prints out-of-fold AUC and, per threshold, the share of
harmful / safe branches a policy would accept.
Usage: python scripts/train_harm.py results/actprobe/7b1m/branches.jsonl results/actprobe/7b1m/harm.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inmodel_prune.harm import ATTN, STATE, feature_vec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("branches")
ap.add_argument("out")
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()

B = [json.loads(l) for l in open(a.branches)]
B = [b for b in B if b["n_tok"] > 0]
L = max(b["layer"] for b in B) + 1
y = np.array([b["loss_cut"] - b["loss_keep"] > a.delta for b in B], dtype=np.float64)
pids = sorted({b["id"] for b in B})
perm = np.random.RandomState(0).permutation(len(pids))
fold_of = {p: int(perm[i] % a.folds) for i, p in enumerate(pids)}
f = np.array([fold_of[b["id"]] for b in B])
print(f"{len(B)} branches, {len(pids)} prompts, harmful (dloss > {a.delta}) {y.mean():.1%}")


def auc(s, yy):
    yy = yy.astype(bool)
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(len(s))
    inv = np.unique(s, return_inverse=True)[1]
    r = (np.bincount(inv, r) / np.bincount(inv))[inv]                     # average ranks of ties
    return (r[yy].sum() - yy.sum() * (yy.sum() - 1) / 2) / (yy.sum() * (~yy).sum())


def fit_lr(X, yy, l2=1e-3, iters=3000, lr=0.5):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        g = 1 / (1 + np.exp(-(X @ w + b))) - yy
        w -= lr * (X.T @ g / len(yy) + l2 * w)
        b -= lr * g.mean()
    return {"w": w.tolist(), "b": float(b)}


def fit_mlp(X, yy, hidden=64, epochs=60, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(yy, dtype=torch.float32)
    for _ in range(epochs):
        for idx in torch.randperm(len(yt)).split(512):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx])[:, 0], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    p = [t.detach().numpy() for t in net.parameters()]
    return {"W1": p[0].T.tolist(), "b1": p[1].tolist(), "W2": p[2].T.tolist(), "b2": p[3].tolist(),
            "W3": p[4][0].tolist(), "b3": float(p[5][0])}


out = {"delta": a.delta, "n_layers": L, "fold_of": fold_of, "models": {}}
for name, feats, kind in (("lr_attn", ATTN, "lr"), ("lr_state", STATE, "lr"), ("mlp_state", STATE, "mlp")):
    X = np.stack([feature_vec(b, feats, L) for b in B])
    oof = np.zeros(len(B))
    folds = {}
    for k in range(a.folds):
        tr = f != k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xs = (X - mu) / sd
        m = fit_lr(Xs[tr], y[tr]) if kind == "lr" else fit_mlp(Xs[tr], y[tr])
        m.update(mu=mu.tolist(), sd=sd.tolist())
        folds[str(k)] = m
        if kind == "lr":
            oof[~tr] = Xs[~tr] @ np.array(m["w"]) + m["b"]
        else:
            h = np.maximum(Xs[~tr] @ np.array(m["W1"]) + m["b1"], 0)
            h = np.maximum(h @ np.array(m["W2"]) + m["b2"], 0)
            oof[~tr] = h @ np.array(m["W3"]) + m["b3"]
    out["models"][name] = {"type": kind, "features": feats, "folds": folds}
    p = 1 / (1 + np.exp(-oof))
    print(f"{name:10s} out-of-fold AUC {auc(oof, y):.3f}  | threshold: accepted harmful / accepted safe")
    for th in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        acc = p <= th
        print(f"   p<={th:<5g} {acc[y > 0].mean():6.1%} / {acc[y == 0].mean():6.1%}")
mass = np.array([b["mass_max"] for b in B])
print(f"mass_max   AUC {auc(mass, y):.3f}  | threshold: accepted harmful / accepted safe")
for th in np.quantile(mass, [0.05, 0.1, 0.2, 0.3, 0.5]):
    acc = mass <= th
    print(f"   mass<={th:.4f} {acc[y > 0].mean():6.1%} / {acc[y == 0].mean():6.1%}")
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
