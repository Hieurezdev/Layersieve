"""LD50 oracle-window probes vs dense / copy (first-name metric), paired bootstrap vs dense, split by mask distance.
usage: ld_cause.py run1 run2 ..."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); names, sys.argv = sys.argv[1:], sys.argv[:1]
import selscore as S
R = "results/selfext/"
L = lambda n: {json.loads(l)["id"]: json.loads(l) for l in open(R + n + ".jsonl")}
v = L("ld50_v"); M = L("ld50_move_mask")
runs = {"dense": {i: v[i]["pred_dense"] for i in v}, "copy (selected)": {i: v[i]["pred_recall"] for i in v}}
for n in names:
    runs[n] = {i: r["pred"] for i, r in L(n).items()}
ids = [i for i in v if all(i in r for r in runs.values())]
far = [i for i in ids if (M[i].get("moved_from_distance") or 0) > 48000]
d0 = [S.score(i, runs["dense"][i]) for i in ids]
print(f"n={len(ids)} (far >48k: {len(far)})")
for n, r in runs.items():
    s = [S.score(i, r[i]) for i in ids]; df = [a - b for a, b in zip(s, d0)]
    bs = sorted(sum(random.choice(df) for _ in df) / len(df) for _ in range(4000))
    sf = [S.score(i, r[i]) for i in far]
    print(f"{n:24s} all {100*sum(s)/len(s):5.1f}  far {100*sum(sf)/max(len(sf),1):5.1f}  vs dense {100*sum(df)/len(df):+5.1f} [{100*bs[100]:+.0f},{100*bs[3899]:+.0f}]")
