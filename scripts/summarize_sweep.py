"""Tong hop quet lich sang (scripts/run_sweep.sh): moi cau hinh (l0, rho) -> token x layer so voi dense (ly thuyet,
cung la ty le KV doc khi decode), ty le token giu o layer cuoi (do), so cau dung tung task tren tap case kho
(chi mau target dense dung) va so voi base (l0=17, rho=0.12) bang McNemar; TTFT / TPOT do tren kv50 so voi dense."""
import json
import os
from collections import defaultdict
from math import comb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R, M = os.path.join(ROOT, "results/sweep"), os.path.join(ROOT, "results/moe")
TASKS = ["kv", "vt", "hotpotqa", "2wikimqa"]
L = 28


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def cost(s, d):
    return (s + sum((1 - d) ** k for k in range(1, L - s + 1))) / L


target = {**load(f"{M}/hard/target_dense.jsonl"), **load(f"{M}/hard_vt/target_dense_all.jsonl")}
ok_ids = [i for i in target if target[i]["ok"]]


def table(title, first, sub):
    """first = the (17, 0.12) run of this scoring; sub = results/sweep/<sub>/s<l0>_d<rho>.jsonl."""
    runs = {(17, 0.12): first}
    for f in sorted(os.listdir(f"{R}/{sub}")) if os.path.isdir(f"{R}/{sub}") else []:
        s, d = f[1:-6].split("_d")
        runs[(int(s), float(d))] = load(f"{R}/{sub}/{f}")
    print(f"== {title} ==")
    print(f"{'l0':>3s} {'rho':>5s} {'tokxlayer':>9s} {'giu cuoi':>8s} " + " ".join(f"{t:>9s}" for t in TASKS)
          + "   vs base token (thang:thua, p)")
    for (s, d), r in sorted(runs.items(), key=lambda x: (-x[0][0], x[0][1])):
        print_row(s, d, r)
    print()


def print_row(s, d, r):
    ids = [i for i in ok_ids if i in r]
    kf = [r[i]["kept_frac"] for i in ids if "kept_frac" in r[i]]
    row = f"{s:3d} {d:5.2f} {cost(s, d):9.2f} {sum(kf) / len(kf) if kf else float('nan'):8.0%} "
    for t in TASKS:
        tid = [i for i in ids if r[i]["task"] == t]
        n_ok = sum(r[i]["ok"] for i in tid)
        row += f"{f'{n_ok}/{len(tid)}' if tid else '-':>9s} "
    c = [i for i in ids if i in base]
    w = sum(r[i]["ok"] and not base[i]["ok"] for i in c)
    l_ = sum(base[i]["ok"] and not r[i]["ok"] for i in c)
    print(row + (f"  {w}:{l_} p={mcnemar(w, l_):.3f}" if r is not base else "  (base)"))


base = {**load(f"{M}/hard/base.jsonl"), **load(f"{M}/hard_vt/base.jsonl")}
table("Do chinh xac, chon tung token (tap case kho, chi mau target dense dung)", base, "hard")
table("Do chinh xac, chon theo don vi (unit)", {**load(f"{M}/hard/unit.jsonl"), **load(f"{M}/hard_vt/unit.jsonl")},
      "hard_unit")

sp = [json.loads(l) for l in open(f"{R}/speed.jsonl")] if os.path.exists(f"{R}/speed.jsonl") else []
if sp:
    agg = defaultdict(lambda: [[], []])
    for x in sp:
        if x.get("decode_ref") is not None:                                  # sang KV decode: summarize_decode.py
            continue
        agg[(x["start_layer"], x["drop_ratio"])][0].append(x["ttft"])
        agg[(x["start_layer"], x["drop_ratio"])][1].append(x["tpot"])
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    dt, dp = mean(agg[(28, 0.0)][0]), mean(agg[(28, 0.0)][1])
    print(f"\n== Toc do tren kv50 (~170k token, {len(agg[(28, 0.0)][0])} prompt) ==")
    print(f"{'l0':>3s} {'rho':>5s} {'TTFT s':>7s} {'nhanh x':>7s} {'TPOT ms':>8s} {'nhanh x':>7s}")
    for (s, d), (t, p) in sorted(agg.items(), key=lambda x: (-x[0][0], x[0][1])):
        print(f"{s:3d} {d:5.2f} {mean(t):7.1f} {dt / mean(t):7.2f} {1000 * mean(p):8.1f} {dp / mean(p):7.2f}")
