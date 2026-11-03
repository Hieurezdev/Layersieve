"""LD prompts 51-100 (data/infbench_ld50b.jsonl): probes vs the HF dense run ld200_g1w0 (first-name metric).
usage: ld50b_cause.py run1 run2 ..."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); names, sys.argv = sys.argv[1:], sys.argv[:1]
import selscore as S
R = "results/selfext/"
L = lambda n: {json.loads(l)["id"]: json.loads(l) for l in open(R + n + ".jsonl")}
want = {json.loads(l)["id"] for l in open("data/infbench_ld50b.jsonl")}
d = {i: r["pred"] for i, r in L("ld200_g1w0").items() if i in want}
runs = {"dense": d}
for n in names:
    runs[n] = {i: r["pred"] for i, r in L(n).items()}
ids = [i for i in d if all(i in r for r in runs.values())]
d0 = [S.score(i, d[i]) for i in ids]
print(f"n={len(ids)}")
for n, r in runs.items():
    s = [S.score(i, r[i]) for i in ids]; df = [a - b for a, b in zip(s, d0)]
    bs = sorted(sum(random.choice(df) for _ in df) / len(df) for _ in range(4000))
    print(f"{n:22s} {100*sum(s)/len(s):5.1f}  vs dense {100*sum(df)/len(df):+5.1f} [{100*bs[100]:+.0f},{100*bs[3899]:+.0f}]")
