"""Tong hop results/hard: do chinh xac cua SpecPrefill va LayerSieve tren tap KHO (draft dense sai, target dense dung)
so voi tap DE (ca hai dung), va ty le giu evidence (kv/vt: moi span con >= 1/2 token). McNemar SpecPrefill vs LayerSieve."""
import json
import os
import sys
from math import comb

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/hard")
load = lambda n: {r["id"]: r for r in map(json.loads, open(f"{d}/{n}.jsonl"))} if os.path.exists(f"{d}/{n}.jsonl") else None  # noqa: E731
GROUPS = ["kv", "vt", "hotpotqa", "2wikimqa"]


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def acc(run, ids):
    ids = [i for i in ids if i in run]
    return f"{sum(run[i]['ok'] for i in ids):3d}/{len(ids):<3d}" if ids else "  -    "


def ev(run, ids):
    ids = [i for i in ids if i in run and run[i].get("ev_kept")]
    return f"{sum(all(run[i]['ev_kept']) for i in ids) / len(ids):5.0%}" if ids else "   - "


tgt = load("target_dense")
ls = load("layersieve")
print("Dense:", "  ".join(f"{n} " + " ".join(f"{g}={acc(r, [i for i in r if r[i]['task'] == g])}" for g in GROUPS)
                          for n in ["target_dense", "draft15_dense", "draft05_dense"] if (r := load(n))))
for draft, sps in [("draft15_dense", ["sp15_p1", "sp15_p3"]), ("draft05_dense", ["sp05_p1"])]:
    dr = load(draft)
    if not (dr and tgt):
        continue
    print(f"\n== tap kho theo {draft} ==")
    print(f"{'group':10s} {'subset':6s} {'n':>4s} | " + " | ".join(f"{m:>9s} acc  ev_all" for m in sps + ["layersieve"])
          + " | SP vs LS (thang:thua, p)")
    for g in GROUPS:
        ids = [i for i in tgt if tgt[i]["task"] == g and i in dr]
        subsets = {"hard": [i for i in ids if tgt[i]["ok"] and not dr[i]["ok"]],
                   "easy": [i for i in ids if tgt[i]["ok"] and dr[i]["ok"]]}
        for sname, sub in subsets.items():
            row = f"{g:10s} {sname:6s} {len(sub):4d} | "
            row += " | ".join(f"{acc(r, sub):>9s} {ev(r, sub)}" if (r := load(m)) else f"{'-':>9s}     -"
                              for m in sps + ["layersieve"])
            sp = load(sps[0])
            if sp and ls:
                c = [i for i in sub if i in sp and i in ls]
                w = sum(sp[i]["ok"] and not ls[i]["ok"] for i in c)
                l_ = sum(ls[i]["ok"] and not sp[i]["ok"] for i in c)
                row += f" | {w}:{l_} p={mcnemar(w, l_):.3f}"
            print(row)

# tren tap kho kv/vt: SpecPrefill sai vi mat evidence hay giu evidence ma van sai?
sp, dr = load("sp15_p1"), load("draft15_dense")
if sp and dr and tgt:
    print("\n== sp15_p1, tap kho kv/vt: nguyen nhan sai ==")
    for g in ["kv", "vt"]:
        hard = [i for i in sp if sp[i]["task"] == g and tgt[i]["ok"] and not dr[i]["ok"]]
        wrong = [i for i in hard if not sp[i]["ok"]]
        lost = sum(not all(sp[i]["ev_kept"]) for i in wrong)
        print(f"{g}: sai {len(wrong)}/{len(hard)}, trong do mat evidence {lost}, giu du evidence ma van sai {len(wrong) - lost}")
    print("TTFT trung binh (s):", "  ".join(f"{n}={sum(r[i]['ttft'] for i in r) / len(r):.2f}" for n in
                                         ["target_dense", "layersieve", "sp15_p1", "sp15_p3", "sp05_p1"] if (r := load(n))))
