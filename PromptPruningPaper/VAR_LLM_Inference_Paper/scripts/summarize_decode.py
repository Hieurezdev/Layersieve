"""Tong hop sang KV cho decode (scripts/run_decode.sh): ty le KV doc moi buoc decode so voi dense (ly thuyet, lich
l0 = 17, rho = 0.12), so cau dung tung task tren tap case kho (chi mau target dense dung) va so voi LayerSieve khong sang
KV decode (McNemar), TTFT / TPOT do tren kv50, va kv50 voi unit + r = 27."""
import json
import os
from collections import defaultdict
from math import comb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R, M = os.path.join(ROOT, "results/decode"), os.path.join(ROOT, "results/moe")
TASKS = ["kv", "vt", "hotpotqa", "2wikimqa"]
L, L0, RHO = 28, 17, 0.12


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def kv_read(ref, rho=RHO):
    """KV read per decode step / dense: layer i holds min(|S_i|, |S_ref|) tokens, |S_i| / T = (1 - rho)^(i - l0 + 1)."""
    s = [1.0 if i < L0 else (1 - rho) ** (i - L0 + 1) for i in range(L)]
    r = s[ref] if ref is not None else 1.0
    return sum(min(x, r) for x in s) / L


target = {**load(f"{M}/hard/target_dense.jsonl"), **load(f"{M}/hard_vt/target_dense_all.jsonl")}
ok_ids = [i for i in target if target[i]["ok"]]
both = lambda n: {**load(f"{M}/hard/{n}.jsonl"), **load(f"{M}/hard_vt/{n}.jsonl")}  # noqa: E731
rows = [("base (khong sang KV)", None, both("base")), ("r17", 17, load(f"{R}/hard/r17.jsonl")),
        ("r22", 22, load(f"{R}/hard/r22.jsonl")), ("r27", 27, load(f"{R}/hard/r27.jsonl")),
        ("unit (khong sang KV)", None, both("unit")),
        ("unit + r27", 27, load(f"{R}/hard/unit_r27.jsonl")),
        ("unit rho=.25", None, load(os.path.join(ROOT, "results/sweep/hard_unit/s17_d0.25.jsonl")), 0.25),
        ("unit rho=.25 + r27", 27, load(f"{R}/hard/unit_d0.25_r27.jsonl"), 0.25),
        ("unit rho=.25 + r22", 22, load(f"{R}/hard/unit_d0.25_r22.jsonl"), 0.25)]
print("== Do chinh xac, tap case kho ==")
print(f"{'cau hinh':22s} {'KV doc/buoc':>11s} " + " ".join(f"{t:>9s}" for t in TASKS) + "   vs cung lich khong sang KV")
for k, (name, ref, r, *rho) in enumerate(rows):
    if not r:
        continue
    ids = [i for i in ok_ids if i in r]
    row = f"{name:22s} {kv_read(ref, *rho):11.0%} "
    for t in TASKS:
        tid = [i for i in ids if r[i]["task"] == t]
        n_ok = sum(r[i]["ok"] for i in tid)
        row += f"{f'{n_ok}/{len(tid)}' if tid else '-':>9s} "
    ref_run = rows[0][2] if k < 4 else rows[6][2] if rho else rows[4][2]
    if ref is not None and ref_run:
        c = [i for i in ids if i in ref_run]
        w = sum(r[i]["ok"] and not ref_run[i]["ok"] for i in c)
        l_ = sum(ref_run[i]["ok"] and not r[i]["ok"] for i in c)
        row += f"  {w}:{l_} p={mcnemar(w, l_):.3f}"
    print(row)


SPEED = os.path.join(ROOT, "results/sweep/speed.jsonl")                       # scripts/run_speed_all.sh
sp = [json.loads(l) for l in open(SPEED)] if os.path.exists(SPEED) else []
if sp:
    agg = defaultdict(lambda: [[], []])
    for x in sp:
        key = (x["start_layer"], x["drop_ratio"], x.get("decode_ref"))
        agg[key][0].append(x["ttft"])
        agg[key][1].append(x["tpot"])
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    dense = agg.get((28, 0.0, None))
    print(f"\n== Toc do tren kv50 (~170k token) ==")
    print(f"{'l0':>3s} {'rho':>5s} {'r':>4s} {'n':>2s} {'TTFT s':>7s} {'nhanh x':>7s} {'TPOT ms':>8s} {'nhanh x':>7s}")
    for (s, d, ref), (t, p) in sorted(agg.items(), key=lambda x: (-x[0][0], x[0][1], x[0][2] or 99)):
        sx = f"{mean(dense[0]) / mean(t):7.2f} {1000 * mean(p):8.1f} {mean(dense[1]) / mean(p):7.2f}" if dense else ""
        print(f"{s:3d} {d:5.2f} {str(ref or '-'):>4s} {len(t):2d} {mean(t):7.1f} " + (sx or f"{'':7s} {1000 * mean(p):8.1f}"))

kv = load(f"{R}/kv50_unit_r27.jsonl")
if kv:
    old = os.path.join(ROOT, "results/kv50")
    refs = [("cung lich khong sang KV", load(f"{M}/kv50/unit.jsonl")),
            ("LayerSieve goc", load(f"{old}/kv50_catdan.jsonl")), ("dense", load(f"{old}/kv50_dense.jsonl"))]
    print(f"\n== kv50: unit_r27 {sum(kv[i]['ok'] for i in kv)}/{len(kv)} ==")
    for rn, base in refs:
        c = [i for i in kv if i in base]
        if not c:
            continue
        w = sum(kv[i]["ok"] and not base[i]["ok"] for i in c)
        l_ = sum(base[i]["ok"] and not kv[i]["ok"] for i in c)
        print(f"  vs {rn}: {sum(kv[i]['ok'] for i in c)}/{len(c)} vs {sum(base[i]['ok'] for i in c)}/{len(c)},"
              f" thang:thua {w}:{l_} p={mcnemar(w, l_):.3f}")
