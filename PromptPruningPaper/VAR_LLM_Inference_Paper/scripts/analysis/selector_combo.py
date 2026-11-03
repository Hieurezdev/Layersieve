"""Combined selector: per block, the max over scoring variants of (score / top score of that variant); blocks above a
relative threshold are picked. Reports gold coverage, picked blocks and distractor units picked (analysis only)."""
import glob, os, sys
import numpy as np
sys.argv, dirs = sys.argv[:1], sys.argv[1:]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selector_eval.py")).read().split("rows = defaultdict(list)")[0]
ns = {}; exec(src, ns)


def variant(z, v):
    if "_contrast_" in v:
        l, w = v.split("_")[0], v.split("_w")[1]
        qn = z[f"{l}_mean_w{w}"]; nn = z[f"{l}_neutral"]; s = qn / qn.sum() - nn / nn.sum()
    else:
        s = z[v].astype(float).copy()
    s[0] = -np.inf
    return s / s.max()


COMBOS = {"c16_L17 + c128_L14": ["L17_contrast_w16", "L14_contrast_w128"],
          "max16_L17 + c128_L14": ["L17_max_w16", "L14_contrast_w128"],
          "max16_L14 + c128_L14": ["L14_max_w16", "L14_contrast_w128"],
          "c16_L17 only": ["L17_contrast_w16"], "c128_L14 only": ["L14_contrast_w128"]}
TH = [0.5, 0.3, 0.2]
print("combo                     thr | " + " | ".join(f"{os.path.basename(d):>22s}" for d in dirs) + "   (gold covered / blocks picked / distractor units picked)")
cache = {d: [(np.load(f), ns["PROMPTS"][os.path.basename(f)[:-4]]) for f in sorted(glob.glob(os.path.join(d, "*.npz")))] for d in dirs}
ev = {d: [ns["evidence"](r) for _, r in cache[d]] for d in dirs}
for name, vs in COMBOS.items():
    for t in TH:
        cells = []
        for d in dirs:
            gc, nbk, dp = [], [], []
            for (z, r), (gold, dis) in zip(cache[d], ev[d]):
                s = np.max([variant(z, v) for v in vs], 0); B = int(z["block"]); nb = len(s)
                pick = set(np.nonzero(s >= t)[0].tolist())
                gb = ns["blocks_of"](gold, B, nb); db = ns["blocks_of"](dis, B, nb)
                gc.append(np.mean([any(b in pick for b in u) for u in gb if u]) if gb else np.nan)
                nbk.append(len(pick)); dp.append(sum(any(b in pick for b in u) for u in db if u))
            cells.append(f"{np.nanmean(gc):.2f}/{np.median(nbk):3.0f}/{np.mean(dp):4.1f}")
        print(f"{name:24s} {t:4.1f} | " + " | ".join(f"{c:>22s}" for c in cells))
