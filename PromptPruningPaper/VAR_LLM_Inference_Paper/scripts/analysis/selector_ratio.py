"""For a few scoring variants: score of each gold / distractor unit's best block relative to the top block of the
prompt, so a relative threshold can be judged (analysis only; uses selector_eval's evidence locator)."""
import glob, os, sys
import numpy as np
sys.argv, dirs = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("se_eval", os.path.join(os.path.dirname(os.path.abspath(__file__)), "selector_eval.py"))
src = open(spec.origin).read().split("rows = defaultdict(list)")[0]          # helpers only, not the main loop
ns = {}; exec(src, ns)
VARS = ["L17_max_w16", "L14_max_w16", "L14_contrast_w128", "L17_contrast_w16", "L17_mean_w16"]
TH = [0.5, 0.3, 0.2, 0.1, 0.05]
for d in dirs:
    stats = {v: {"g": [], "d": [], "n_above": {t: [] for t in TH}, "gold_in": {t: [] for t in TH}} for v in VARS}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f); r = ns["PROMPTS"][os.path.basename(f)[:-4]]
        B = int(z["block"]); gold, dis = ns["evidence"](r)
        for v in VARS:
            if v.split("_")[1] == "contrast":
                l, w = v.split("_")[0], v.split("_w")[1]
                qn = z[f"{l}_mean_w{w}"]; nn = z[f"{l}_neutral"]; s = qn / qn.sum() - nn / nn.sum()
            else:
                s = z[v].copy()
            s[0] = -np.inf; top = s.max(); nb = len(s)
            gb = ns["blocks_of"](gold, B, nb); db = ns["blocks_of"](dis, B, nb)
            gr = [max(s[b] for b in u) / top for u in gb if u]; dr = [max(s[b] for b in u) / top for u in db if u]
            stats[v]["g"] += gr; stats[v]["d"] += dr
            for t in TH:
                stats[v]["n_above"][t].append(int((s >= t * top).sum()))
                stats[v]["gold_in"][t].append(np.mean([x >= t for x in gr]) if gr else np.nan)
    print(f"\n== {os.path.basename(d)}")
    for v in VARS:
        g, dd = np.array(stats[v]["g"]), np.array(stats[v]["d"])
        line = f"  {v:18s} gold ratio p10 {np.percentile(g,10):.2f} med {np.median(g):.2f}"
        if len(dd): line += f" | distractor ratio med {np.median(dd):.2f} p90 {np.percentile(dd,90):.2f}"
        line += " | thr: " + " ".join(f"{t}:gold {np.nanmean(stats[v]['gold_in'][t]):.2f}/blk {np.median(stats[v]['n_above'][t]):.0f}" for t in TH)
        print(line)
