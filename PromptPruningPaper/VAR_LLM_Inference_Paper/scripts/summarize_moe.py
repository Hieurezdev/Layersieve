"""Tong hop ablation cac cai tien cua MOE tren LayerSieve (scripts/run_moe.sh): so cau dung theo task, so voi base
(LayerSieve goc chay cung may) bang McNemar tren cung id, ty le giu evidence (kv/vt) va ty le token giu o layer cuoi.
Tap case kho (gop results/moe/hard va results/moe/hard_vt): chi lay mau target dense dung, tach theo draft 0.5B / 1.5B
dense chay tren cung du lieu: kho = draft sai, de = draft dung. SpecPrefill (scripts/run_sp_check.sh, results/moe/sp)
in cung bang, so voi base LayerSieve."""
import json
import os
from math import comb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results/moe")
NAMES = ["base", "unit", "accum", "bm25", "all", "bm25only"]
SP_NAMES = ["sp15_p9", "sp15_p3", "sp15_p1", "sp05_p1"]
TASKS = ["kv", "vt", "hotpotqa", "2wikimqa"]


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else None


def load_hard(name):
    """results/moe/hard plus the extra vt samples of scripts/run_moe_vt.sh (results/moe/hard_vt)."""
    a, b = load(f"{R}/hard/{name}.jsonl"), load(f"{R}/hard_vt/{name}.jsonl")
    return None if a is None and b is None else {**(a or {}), **(b or {})}


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def vs(run, base, ids):
    ids = [i for i in ids if i in run and i in base]
    w = sum(run[i]["ok"] and not base[i]["ok"] for i in ids)
    l_ = sum(base[i]["ok"] and not run[i]["ok"] for i in ids)
    return f"{w}:{l_} p={mcnemar(w, l_):.2f}"


def acc(run, ids):
    ids = [i for i in ids if i in run]
    return f"{sum(run[i]['ok'] for i in ids)}/{len(ids)}" if ids else "-"


print("== Case kho (results/moe/hard) ==")
draft = load_hard("draft05_dense") or {}
draft15 = load(f"{R}/sp/draft15_dense.jsonl") or {}
target = {**(load(f"{R}/hard/target_dense.jsonl") or {}), **(load(f"{R}/hard_vt/target_dense_all.jsonl") or {})} or None
runs = {n: r for n in NAMES if (r := load_hard(n))}
runs.update({n: r for n in SP_NAMES if (r := load(f"{R}/sp/{n}.jsonl"))})
base = runs.get("base")
for t in TASKS:
    ids = sorted({i for r in runs.values() for i in r
                  if r[i]["task"] == t and (target is None or target.get(i, {}).get("ok"))})
    groups = {"tat ca": ids}
    for dn, d in [("0.5B", draft), ("1.5B", draft15)]:
        if d:
            groups[f"kho {dn}"] = [i for i in ids if i in d and not d[i]["ok"]]
            groups[f"de {dn}"] = [i for i in ids if i in d and d[i]["ok"]]
    print(f"\n{t}")
    print(f"  {'bien the':9s} " + " ".join(f"{g:>9s}" for g in groups) + "   ev_all  giu   vs base (tat ca)")
    for n, r in runs.items():
        evs = [all(r[i]["ev_kept"]) for i in ids if i in r and r[i].get("ev_kept")]
        kf = [r[i]["kept_frac"] for i in ids if i in r and "kept_frac" in r[i]]
        row = f"  {n:9s} " + " ".join(f"{acc(r, g):>9s}" for g in groups.values())
        row += f"   {sum(evs) / len(evs):5.0%}" if evs else "       -"
        row += f"  {sum(kf) / len(kf):4.0%}" if kf else "     -"
        row += f"   {vs(r, base, ids)}" if base and n != "base" else ""
        print(row)

print("\n== kv50 (results/moe/kv50) ==")
runs = {n: r for n in NAMES if (r := load(f"{R}/kv50/{n}.jsonl"))}
base = runs.get("base")
for n, r in runs.items():
    kl = [x["kept_last"] for x in r.values()]
    print(f"  {n:9s} {acc(r, list(r)):>7s}  giu o layer cuoi TB {sum(kl) / len(kl):8.0f} token"
          + (f"   vs base {vs(r, base, list(r))}" if base and n != "base" else ""))
