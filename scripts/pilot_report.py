"""Summarize the evidence-loss pilot (scripts/run_pilot.sh) into markdown tables (stdout) and results/pilot/summary.json.
  A  per schedule: full-evidence retention, failures on dense-correct prompts, how many lose evidence, causal test
  B  where evidence is lost: first-loss layer, rank / percentile / score margin of the lost unit right before the cut
  C  "lost evidence has unstable scores": features of every dropped unit at its drop layer, computed only from layers at
     which it was still alive (pruned run) -> lost-evidence vs other dropped units, AUC, and the offline deferral budget
     needed to catch the lost evidence at its first cut (selectivity only, not accuracy)
  D  dense counterfactual: rank of the lost units in a separate unpruned run at the layers after their cut
    python scripts/pilot_report.py [--configs ls_l17,ls_l14,ls_l11]
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(ROOT, "results/pilot"))
ap.add_argument("--configs", default="ls_l17,ls_l14,ls_l11")
ap.add_argument("--L", type=int, default=28)
a = ap.parse_args()
D = a.dir


def load(name):
    p = os.path.join(D, name + ".jsonl")
    return {(r := json.loads(l))["id"]: r for l in open(p)} if os.path.exists(p) else {}


dense = load("dense")
dtrace = load("dense_trace")
summary = {}
TASKS = ["kv", "vt", "mk", "qa2"]


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def fmt(x, n=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


print("## A. Retention, failures and causal test (dense-correct prompts)\n")
print("| schedule | C | prompts | full evidence kept (all) | dense-correct | LS correct | failures | "
      "failures that lost evidence | force-keep fixes | placebo fixes | failures with evidence kept | "
      "lost evidence but still correct |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows_task = []
for cfg in a.configs.split(","):
    R = load(cfg)
    if not R:
        continue
    C = np.mean([sum(r["kept"]) / (a.L * r["len"]) for r in R.values()])
    dc = [r for r in R.values() if dense.get(r["id"], {}).get("ok")]
    fail = [r for r in dc if not r["ok"]]
    fl = [r for r in fail if not r["ev_all_kept"]]
    fk = [r for r in fail if r["ev_all_kept"]]
    fix = [r for r in fl if r.get("force", {}).get("ok")]
    plc = [r for r in fl if r.get("placebo", {}).get("ok")]
    still = [r for r in dc if r["ok"] and not r["ev_all_kept"]]
    full = sum(r["ev_all_kept"] for r in R.values())
    print(f"| {cfg} | {C:.3f} | {len(R)} | {full}/{len(R)} ({100 * full / len(R):.0f}%) | {len(dc)} | "
          f"{len(dc) - len(fail)} | {len(fail)} | {len(fl)} | {len(fix)}/{len(fl)} | {len(plc)}/{len(fl)} | {len(fk)} | "
          f"{len(still)} |")
    summary[cfg] = {"C": C, "prompts": len(R), "full_kept": full, "dense_correct": len(dc), "fail": len(fail),
                    "fail_lost": len(fl), "force_fix": len(fix), "placebo_fix": len(plc), "fail_kept": len(fk),
                    "lost_still_ok": len(still),
                    "fail_ids": {r["id"]: {"lost": not r["ev_all_kept"], "first_loss": r["first_loss_layer"],
                                           "pred": r["pred"][:60], "force": r.get("force"),
                                           "placebo": r.get("placebo")} for r in fail}}
    for t in TASKS:
        Rt = [r for r in R.values() if r["task"] == t]
        dct = [r for r in Rt if dense.get(r["id"], {}).get("ok")]
        ft = [r for r in dct if not r["ok"]]
        rows_task.append((cfg, t, len(Rt), sum(r["ev_all_kept"] for r in Rt), len(dct), len(ft),
                          sum(not r["ev_all_kept"] for r in ft), sum(r.get("force", {}).get("ok", False) for r in ft),
                          sum(r.get("placebo", {}).get("ok", False) for r in ft)))

print("\n### A2. Per task\n")
print("| schedule | task | prompts | full evidence kept | dense-correct | failures | ...lost evidence | force fixes | "
      "placebo fixes |")
print("|---|---|---|---|---|---|---|---|---|")
for row in rows_task:
    print("| " + " | ".join(map(str, row)) + " |")

print("\n## B. Where evidence is lost (every prompt, every lost evidence unit, at its drop layer)\n")
print("| schedule | lost units (prompts) | first-loss layer: count | pct at cut (median, min-max) | rank at cut "
      "(median) | score / best dropped score (median) | lost by task |")
print("|---|---|---|---|---|---|---|")
for cfg in a.configs.split(","):
    R = load(cfg)
    if not R:
        continue
    first = Counter(r["first_loss_layer"] for r in R.values() if r["first_loss_layer"] is not None)
    pcts, ranks, ratios, n_units = [], [], [], 0
    for r in R.values():
        for u, l in r["ev_drop_layer"].items():
            if l < 0:
                continue
            info = next(x for x in r["layers"] if x["layer"] == l)
            e = info["ev"][u]
            n_units += 1
            pcts.append(e["pct"])
            ranks.append(e["rank"])
            ratios.append(e["score"] / info["cut"])
    bytask = Counter(r["task"] for r in R.values() if not r["ev_all_kept"])
    if n_units:
        print(f"| {cfg} | {n_units} ({sum(first.values())}) | "
              f"{', '.join(f'L{k}:{v}' for k, v in sorted(first.items()))} | {np.median(pcts):.3f} "
              f"({min(pcts):.3f}-{max(pcts):.3f}) | {np.median(ranks):.0f} | {np.median(ratios):.2f} | {dict(bytask)} |")
    else:
        print(f"| {cfg} | 0 | – | – | – | – | – |")

print("\n## C. Are lost evidence units unstable before their cut? (pruned run, alive layers only)\n")
FEATS = {"vol4": "std of percentile over the last 4 scored layers (incl. cut)",
         "fall": "best earlier percentile - percentile at cut",
         "peak": "best percentile over the scored layers before the cut (dense prefix included)",
         "dlast": "percentile change from the previous layer (negative = falling)",
         "head_spread": "best-head percentile - median-head percentile at cut",
         "head_top": "fraction of heads ranking the unit in their top 25% at cut",
         "margin": "score / best dropped score at cut (near-threshold baseline)",
         "pct_cut": "percentile at cut"}
groups = defaultdict(lambda: defaultdict(list))   # cfg -> feature/group -> values
defer = defaultdict(lambda: defaultdict(list))    # cfg -> feature -> per-(prompt, layer) catch info
for cfg in a.configs.split(","):
    for f in sorted(glob.glob(os.path.join(D, "npz", cfg, "*.npz"))):
        pid = os.path.basename(f)[:-4]
        if pid not in dense:
            continue
        z = np.load(f)
        layers, pct, score = z["layers"], z["pct"].astype(float), z["score"]
        ht, hs = z["head_top"].astype(float), z["head_spread"].astype(float)
        dl, ev, size = z["drop_layer"], z["is_ev"], z["usize"]
        for li, l in enumerate(layers):
            dropped = np.nonzero(dl == l)[0]
            if len(dropped) == 0:
                continue
            cut = np.nanmax(score[li, dropped])
            lo = max(0, li - 3)
            P = pct[:, dropped]
            feats = {"vol4": np.nanstd(P[lo:li + 1], 0),
                     "fall": (np.nanmax(P[:li], 0) - P[li]) if li > 0 else np.zeros(len(dropped)),
                     "peak": np.nanmax(P[:li], 0) if li > 0 else P[li],
                     "dlast": (P[li] - P[li - 1]) if li > 0 else np.zeros(len(dropped)),
                     "head_spread": hs[li, dropped], "head_top": ht[li, dropped],
                     "margin": score[li, dropped] / cut, "pct_cut": P[li]}
            is_e = ev[dropped]
            for k, v in feats.items():
                groups[cfg][(k, "ev")] += list(v[is_e])
                groups[cfg][(k, "ne")] += list(v[~is_e])
                near = feats["margin"] >= 0.5
                groups[cfg][(k, "ne_near")] += list(v[~is_e & near])
            # kept evidence units at the same layer (alive after it): same features, for reference
            kept_e = np.nonzero(ev & ((dl < 0) | (dl > l)) & ~np.isnan(pct[li]))[0]
            if len(kept_e):
                Pk = pct[:, kept_e]
                groups[cfg][("vol4", "ev_kept")] += list(np.nanstd(Pk[lo:li + 1], 0))
                groups[cfg][("fall", "ev_kept")] += list((np.nanmax(Pk[:li], 0) - Pk[li]) if li > 0 else [0] * len(kept_e))
                groups[cfg][("peak", "ev_kept")] += list(np.nanmax(Pk[:li], 0) if li > 0 else Pk[li])
                groups[cfg][("head_spread", "ev_kept")] += list(hs[li, kept_e])
                groups[cfg][("head_top", "ev_kept")] += list(ht[li, kept_e])
                groups[cfg][("pct_cut", "ev_kept")] += list(pct[li, kept_e])
            if is_e.any():
                # offline deferral: defer the dropped units with the most "unstable" value of a feature, as a share of
                # the dropped tokens; does it include the evidence units dropped here?
                tok = size[dropped].astype(float)
                for k, sign in (("vol4", 1), ("fall", 1), ("peak", 1), ("head_spread", 1), ("head_top", 1), ("margin", 1),
                                ("dlast", -1)):
                    v = np.nan_to_num(sign * feats[k], nan=-1e9)
                    order = np.argsort(-v, kind="stable")
                    share = np.cumsum(tok[order]) / tok.sum()
                    need = share[np.nonzero(is_e[order])[0]]           # deferral share needed per lost unit
                    defer[cfg][k] += list(need)

for cfg in a.configs.split(","):
    g = groups[cfg]
    n_ev = len(g[("vol4", "ev")])
    if n_ev == 0:
        continue
    print(f"\n### {cfg}: {n_ev} lost evidence units vs {len(g[('vol4', 'ne')])} other dropped units "
          f"({len(g[('vol4', 'ne_near')])} near the cut, margin >= 0.5)\n")
    print("| feature | lost evidence (median) | evidence kept at same layer (median) | other dropped (median) | "
          "AUC lost-ev vs other dropped | AUC vs near-cut dropped |")
    print("|---|---|---|---|---|---|")
    for k, desc in FEATS.items():
        ev_, ne_, nn_, ek = g[(k, "ev")], g[(k, "ne")], g[(k, "ne_near")], g.get((k, "ev_kept"), [])
        print(f"| {k}: {desc} | {fmt(np.nanmedian(ev_), 3)} | {fmt(np.nanmedian(ek) if len(ek) else None, 3)} | "
              f"{fmt(np.nanmedian(ne_), 3)} | {fmt(auc(ev_, ne_))} | {fmt(auc(ev_, nn_))} |")
    summary.setdefault(cfg, {})["instability"] = {
        k: {"ev_median": float(np.nanmedian(g[(k, "ev")])), "ne_median": float(np.nanmedian(g[(k, "ne")])),
            "auc": auc(g[(k, "ev")], g[(k, "ne")]), "auc_near": auc(g[(k, "ev")], g[(k, "ne_near")])} for k in FEATS}
    print("\nOffline deferral at the first cut: share of the tokens dropped at that layer that must be deferred "
          "(ranked by the feature) to also defer the lost evidence unit; lower is better, random = 0.5 on average.\n")
    print("| rank by | median share needed | caught at 5% | at 10% | at 20% |")
    print("|---|---|---|---|---|")
    for k, need in defer[cfg].items():
        need = np.array(need)
        print(f"| {k} | {np.median(need):.3f} | {np.mean(need <= .05):.2f} | {np.mean(need <= .10):.2f} | "
              f"{np.mean(need <= .20):.2f} |")
        summary[cfg].setdefault("defer", {})[k] = {"median": float(np.median(need)),
                                                     "at5": float(np.mean(need <= .05)),
                                                     "at10": float(np.mean(need <= .10)),
                                                     "at20": float(np.mean(need <= .20))}

print("\n## D. Dense counterfactual: percentile of each lost unit in an unpruned run, at layers after its cut\n")
print("| schedule | lost units | dense pct at cut layer (median) | mean dense pct over later layers (median) | "
      "share of later layers with dense pct >= 0.9 (median) |")
print("|---|---|---|---|---|")
for cfg in a.configs.split(","):
    R = load(cfg)
    at, later, hi = [], [], []
    for r in R.values():
        f = os.path.join(D, "npz", "dense_trace", r["id"] + ".npz")
        if r["ev_all_kept"] or not os.path.exists(f):
            continue
        z = np.load(f)
        layers, pct = list(z["layers"]), z["pct"].astype(float)
        for u, l in r["ev_drop_layer"].items():
            if l < 0:
                continue
            li = layers.index(l)
            at.append(pct[li, int(u)])
            after = pct[li + 1:, int(u)]
            later.append(np.mean(after))
            hi.append(np.mean(after >= 0.9))
    if at:
        print(f"| {cfg} | {len(at)} | {np.median(at):.3f} | {np.median(later):.3f} | {np.median(hi):.2f} |")

print("\n## E. Cost of the measurements (wall clock, shared GPU)\n")
print("| run | prompts | generate s / prompt (mean) | analysis s / prompt (CPU, mean) | interventions (generate s, total) |")
print("|---|---|---|---|---|")
for name in ["dense"] + a.configs.split(",") + ["dense_trace"]:
    R = load(name)
    if not R:
        continue
    iv = sum(r[k]["sec"] for r in R.values() for k in ("force", "placebo") if k in r)
    an = [r["analyse_sec"] for r in R.values() if "analyse_sec" in r]
    print(f"| {name} | {len(R)} | {np.mean([r['sec'] for r in R.values()]):.2f} | {fmt(np.mean(an) if an else None)} | "
          f"{iv:.0f} |")

json.dump(summary, open(os.path.join(D, "summary.json"), "w"), indent=1, default=str)
