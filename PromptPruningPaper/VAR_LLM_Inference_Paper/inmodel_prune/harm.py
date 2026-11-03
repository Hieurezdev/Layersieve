"""Harm predictor for direction A (scripts/probe_action.py, scripts/train_harm.py): probability that dropping a candidate
set D before layer l raises the teacher-forced answer loss by more than delta, from features available before the cut.
Models are cross-fitted by prompt: the model used on a prompt never saw that prompt's branches in training.
"""
import json
import math

import numpy as np

ATTN = ["frac", "n_tok", "mass_max", "mass_mean", "dmax_pct", "dmax_rel"]
STATE = ATTN + ["layer", "alive", "prev_pct_mean", "prev_pct_max", "rank_jump"]


def feature_vec(row, names, n_layers):
    """row: a branch row of probe_action.py (or the same keys built online). Missing previous-layer ranks (first
    decision point) are filled with 0.5."""
    def g(k, fill):
        return fill if row.get(k) is None else row[k]
    v = {"frac": row["frac"], "n_tok": math.log(max(row["n_tok"], 1)), "mass_max": row["mass_max"],
         "mass_mean": row["mass_mean"], "dmax_pct": g("dmax_pct", 0.5), "dmax_rel": math.log(g("dmax_rel", 1.0) + 1e-9),
         "layer": row["layer"] / n_layers, "alive": row["alive"], "prev_pct_mean": g("prev_pct_mean", 0.5),
         "prev_pct_max": g("prev_pct_max", 0.5), "rank_jump": g("prev_pct_max", 0.5) - g("dmax_pct", 0.5)}
    return np.array([v[n] for n in names], dtype=np.float64)


def forward(m, x):
    """Logit of one model fold (dict with mu, sd and either w, b (lr) or W1, b1, W2, b2, W3, b3 (mlp))."""
    x = (x - np.array(m["mu"])) / np.array(m["sd"])
    if "w" in m:
        return float(x @ np.array(m["w"]) + m["b"])
    h = np.maximum(x @ np.array(m["W1"]) + np.array(m["b1"]), 0)
    h = np.maximum(h @ np.array(m["W2"]) + np.array(m["b2"]), 0)
    return float(h @ np.array(m["W3"]) + m["b3"])


class HarmModel:
    def __init__(self, path):
        self.d = json.load(open(path)) if path else {"models": {}, "fold_of": {}, "n_layers": 1}

    def prob(self, name, row, prompt_id):
        """name 'mass': the attention mass of D itself (no learning); otherwise P(harm) from the fold that held out
        this prompt."""
        if name == "mass":
            return row["mass_max"]
        m = self.d["models"][name]
        fold = m["folds"][str(self.d["fold_of"][prompt_id])]
        z = forward(fold, feature_vec(row, m["features"], self.d["n_layers"]))
        return 1 / (1 + math.exp(-z))
