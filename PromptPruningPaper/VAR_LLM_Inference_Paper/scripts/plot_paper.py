"""Figures for paper/layersieve.
    python scripts/plot_paper.py   -> paper/layersieve/figures/{start_layer,pareto}.pdf
start_layer: start layer across model sizes (results/scaling).
(a) accuracy (% of dense-correct hard cases kept) vs l0/L, schedules with the same final keep (24.5%);
(b) one-shot probe: fraction of evidence in the top 25% units if selection were done at layer l; plotted as the
worst value over layers >= l (last 2 layers excluded), i.e. whether selecting anywhere from l on is safe.
pareto: accuracy (RULER 128k, InfiniteBench; results/bench) vs prefill speedup at 128k (results/bench/speed_vs.jsonl,
the schedule replayed in vLLM; ASL weighted by the cut layers it chose)."""
import sys
import glob
import json
import os
import re
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results", "scaling")
MODELS = [("3b", "3B", 36), ("7b", "7B", 28), ("7b1m", "7B-1M", 28), ("14b", "14B", 48), ("14b1m", "14B-1M", 48),
          ("32b", "32B", 64)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6, fixed order
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "font.family": "serif"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
for (key, name, L), c, m in zip(MODELS, COLORS, MARKERS):
    ids = open(f"{R}/ids/{key}.txt").read().split()
    pts = []
    for p in glob.glob(f"{R}/sweep/{key}_s*_d*.jsonl"):
        s, d = re.search(r"_s(\d+)_d([\d.]+)\.jsonl", p).groups()
        s, d = int(s), float(d)
        if abs((1 - d) ** (L - s) - 0.245) > 0.03:          # keep only the equal-final-keep schedules
            continue
        rows = load(p)
        if len(rows) == len(ids):
            pts.append((s / L, 100 * sum(r["ok"] for r in rows) / len(ids)))
    pts.sort()
    ax1.plot(*zip(*pts), color=c, marker=m, ms=4, lw=1.4, label=name)
    rows = load(f"{R}/probe/{key}.jsonl")
    curve = [mean(r["unit_rec"][l] for r in rows) for l in range(L - 2)]          # last 2 layers excluded
    worst = [min(curve[l:]) for l in range(L - 2)]                                # worst recall from layer l on
    ax2.plot([l / L for l in range(L - 2)], worst, color=c, marker=m, ms=2.5, lw=1.2, label=name)
for ax in (ax1, ax2):
    ax.axvline(0.5, color="#52514e", lw=0.8, ls="--")
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    ax.set_xlabel(r"relative depth $\ell/L$")
ax1.set_ylabel("accuracy (% of dense)")
ax1.set_title(r"(a) progressive sieving from $\ell_0$", fontsize=8)
ax2.set_ylabel(r"min recall over layers $\geq \ell$")
ax2.set_title(r"(b) evidence kept by top-25% selection", fontsize=8)
ax1.legend(ncol=2, fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout()
out = os.path.join(ROOT, "paper", "layersieve", "figures", "start_layer.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print(out)

# ---- pareto
sys.path.insert(0, "/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/eval/infinite_bench")
from compute_scores import get_score_one  # noqa: E402

B = os.path.join(ROOT, "results", "bench")
gold = {r["id"]: r for f in ("infbench", "ruler_eval") for r in load(os.path.join(ROOT, "data", f + ".jsonl"))}


def acc(name):
    """(RULER 128k avg, InfiniteBench avg), averaged per task as in scripts/score_bench.py."""
    out = []
    for bench, keep in (("ruler", lambda g: g["task"].endswith("@131072")), ("infbench", lambda g: True)):
        per = {}
        for x in load(f"{B}/{bench}/{name}.jsonl"):
            g = gold[x["id"]]
            if not keep(g):
                continue
            if bench == "ruler":
                hits = [a.lower() in x["pred"].lower() for a in g["answers"]]
                v = max(hits) if g["metric"] == "part" else sum(hits) / len(hits)
            else:
                v = get_score_one(x["pred"], g["answers"], g["task"], "qwen")
            per.setdefault(g["task"], []).append(100 * v)
        assert sum(map(len, per.values())) in (400, 1300), (name, bench)
        out.append(mean(mean(v) for v in per.values()))
    return out


speed = {}
for r in load(f"{B}/speed_vs.jsonl"):
    if r["len"] < 200000:
        k = (r["start_layer"], r["drop_ratio"], r["decode_ref"], r["oneshot_keep"], r.get("min_keep") or None)
        speed.setdefault(k, []).append(r["ttft"])
speed = {k: mean(v) for k, v in speed.items()}
dense_t = speed[(28, 0.0, None, None, None)]
up = lambda k: dense_t / speed[k]  # noqa: E731
# ASL: expected TTFT over its realized cut layers (no cut = dense); its monitoring cost is not charged. TTFT of a cut at
# layer s is linear in s; the few cut layers without a measured schedule (12, 13, 16, 24-27: <1% of prompts) use the
# least-squares line through the measured ones (15, 17-23)
cut = {k[0]: v for k, v in speed.items() if k[3] == 0.375}
mx, my = mean(cut), mean(cut.values())
slope = sum((x - mx) * (y - my) for x, y in cut.items()) / sum((x - mx) ** 2 for x in cut)
asl_rows = {"ruler": [r for r in load(f"{B}/ruler/m_asl_t0.3_f0.375.jsonl") if r["id"].startswith("ruler131072_")],
            "infbench": load(f"{B}/infbench/m_asl_t0.3_f0.375.jsonl")}
asl_up = {b: dense_t / mean(dense_t if r["select_layer"] is None else cut.get(r["select_layer"], my + slope * (
    r["select_layer"] - mx)) for r in rows) for b, rows in asl_rows.items()}
LS = [("ls_unit_d0.12_r27", (17, 0.12, 27, None, None)), ("ls_unit_d0.25_f0.3_r22", (17, 0.25, 22, None, 0.3)),
      ("ls_unit_d0.25_r22", (17, 0.25, 22, None, None)), ("ls_unit_d0.4_f0.25_r22", (17, 0.4, 22, None, 0.25))]
OTHER = [("Dense", "dense", 1.0, "o"), ("ASL", "m_asl_t0.3_f0.375", None, "s"),
         ("FastKV", "m_fastkv_l15_f0.375", up((15, 0.0, None, 0.375, None)), "^"),
         ("one cut L15, units", "abl_oneshot_unit_l15_k0.375", up((15, 0.0, None, 0.375, None)), "D"),
         ("one cut L15, tokens", "abl_oneshot_tok_l15_k0.375", up((15, 0.0, None, 0.375, None)), "v")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
ls_pts = sorted((up(k), *acc(n)) for n, k in LS)
for ax, j, title in ((axes[0], 1, "(a) RULER 128k"), (axes[1], 2, "(b) InfiniteBench")):
    ax.axhline(acc("dense")[j - 1], color="#52514e", lw=0.8, ls=":", zorder=1)
    ax.scatter([p[0] for p in ls_pts], [p[j] for p in ls_pts], color=COLORS[0], marker="o", s=22, zorder=3,
               label="LayerSieve (variants)", edgecolors="white", linewidths=0.6)
    main = max(ls_pts)                                                       # rho .4, floor .25 is the fastest variant
    ax.scatter([main[0]], [main[j]], color=COLORS[0], marker="*", s=90, zorder=5, edgecolors="white",
               linewidths=0.6, label="LayerSieve (default)")
    for (name, f, x, m), c in zip(OTHER, ["#52514e"] + COLORS[1:]):
        x = x or asl_up["ruler" if j == 1 else "infbench"]               # ASL: its cut layers on this benchmark
        ax.scatter([x], [acc(f)[j - 1]], color=c, marker=m, s=22, label=name, zorder=4, edgecolors="white",
                   linewidths=0.6)
    ax.grid(color="#e4e3df", lw=0.5)
    ax.set_xlabel(r"prefill speedup at 128k ($\times$)")
    ax.set_title(title, fontsize=8)
axes[0].set_ylabel("accuracy")
axes[1].legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout()
out = os.path.join(ROOT, "paper", "layersieve", "figures", "pareto.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print(out, "points:", ls_pts, "ASL speedup", asl_up)
