"""Read scripts/use_probe.py npz files: dense-right vs dense-wrong prompts (first-name metric), at the decision token.
    python scripts/analysis/use_probe_eval.py results/selfext/useprobe/ld200 results/selfext/ld200_g1w0.jsonl"""
import glob, json, os, sys
import numpy as np

d, dense = sys.argv[1], sys.argv[2]
lf = {"__file__": "scripts/analysis/ld_first.py"}
exec(open("scripts/analysis/ld_first.py").read().split("names = sys.argv")[0], lf)
pred = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(dense)}
hq = np.array(json.load(open("results/repos/heads_7b1m.json"))["query_head"])
RH = hq > 0.3                                                                # retrieval heads [L, H]
Z = {}
for f in sorted(glob.glob(d + "/*.npz")):
    i = os.path.basename(f)[:-4]
    Z[i] = (dict(np.load(f)), lf["sc_first"](pred[i], lf["ib"][i]["answers"]) > 0)
ok = [i for i in Z if Z[i][1]]; bad = [i for i in Z if not Z[i][1]]
print(f"n={len(Z)}  dense right {len(ok)}  wrong {len(bad)}")


def m(ids, f):
    return np.mean([f(Z[i][0]) for i in ids], 0) if ids else np.nan


print("\n1) decision token: log p(gold) | log p(rival)   (rival = best non-gold first token)")
for lab, ids in (("right", ok), ("wrong", bad)):
    print(f"  {lab:5s} gold {m(ids, lambda z: z['gold_lp']):6.2f}  rival {m(ids, lambda z: z['rival_lp']):6.2f}  "
          f"median gold rank-gap {np.median([z['rival_lp'] - z['gold_lp'] for z in (Z[i][0] for i in ids)]):.2f}")

print("\n2) logit lens  gold - rival  per layer (mean); share of prompts where gold > rival at that layer")
L = len(next(iter(Z.values()))[0]["lens_gold"])
for lab, ids in (("right", ok), ("wrong", bad)):
    g = np.array([Z[i][0]["lens_gold"] - Z[i][0]["lens_rival"] for i in ids])
    print(f"  {lab:5s} " + " ".join(f"{x:5.1f}" for x in g.mean(0)))
    print(f"  {'':5s} " + " ".join(f"{x:5.2f}" for x in (g > 0).mean(0)))
    if lab == "wrong":
        print(f"  wrong prompts where gold beats rival at SOME layer >= 14: {(g[:, 14:] > 0).any(1).mean():.2f}")

print("\n3) direct logit attribution to gold - rival, summed per layer: attention | MLP")
for lab, ids in (("right", ok), ("wrong", bad)):
    at = m(ids, lambda z: z["dla_head"].sum(1)); ml = m(ids, lambda z: z["dla_mlp"])
    print(f"  {lab:5s} attn " + " ".join(f"{x:5.1f}" for x in at))
    print(f"  {'':5s} mlp  " + " ".join(f"{x:5.1f}" for x in ml))
    print(f"  {'':5s} total attn {at.sum():6.1f}  mlp {ml.sum():6.1f}   retrieval heads {m(ids, lambda z: z['dla_head'][RH].sum()):6.1f}")
hd = np.array([Z[i][0]["dla_head"] for i in ok + bad])
top = np.dstack(np.unravel_index(np.argsort(-np.abs(hd.mean(0)).ravel())[:12], hd.shape[1:]))[0]
print("  top heads by |mean DLA| (layer,head): right | wrong")
for l, h in top:
    print(f"   L{l:2d}H{h:2d} retr={RH[l, h]!s:5s} {m(ok, lambda z: z['dla_head'][l, h]):6.2f} | {m(bad, lambda z: z['dla_head'][l, h]):6.2f}")

print("\n4) attention mass at the decision token (retrieval heads mean | all heads mean)")
for k in ("near", "far", "rival", "mask"):
    for lab, ids in (("right", ok), ("wrong", bad)):
        ids2 = [i for i in ids if Z[i][0]["mass_" + k].sum() > 0 or k != "rival"]
        if not ids2:
            continue
        print(f"  {k:5s} {lab:5s} n={len(ids2):3d}  retr {m(ids2, lambda z: z['mass_' + k][RH].mean()):.4f}  "
              f"all {m(ids2, lambda z: z['mass_' + k].mean()):.4f}  max-head {m(ids2, lambda z: z['mass_' + k].max()):.3f}")
