"""Summarize scripts/run_scaling.sh (results/scaling): Q1 start layer per model (probe + unit sweep), Q2 prefill speed per
model size, Q3 decode speed with KV sieving up to 1M tokens. Prints markdown tables.
    python scripts/summarize_scaling.py
"""
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, median

R = os.environ.get("R") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "scaling")
MODELS = [("3b", "3B", 36), ("7b", "7B", 28), ("7b1m", "7B-1M", 28), ("14b", "14B", 48), ("14b1m", "14B-1M", 48),
          ("32b", "32B", 64)]
TASKS = ["kv", "vt", "hotpotqa", "2wikimqa"]


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def q1():
    print("## Q1. Layer bắt đầu sàng\n")
    print("### Probe: evidence còn trong top 25% nếu chọn một lần ở layer ℓ (unit_rec, trung bình kv+vt)\n")
    for key, name, L in MODELS:
        rows = load(f"{R}/probe/{key}.jsonl")
        if not rows:
            continue
        for metric in ("unit_rec", "tok_rec"):
            curve = [mean(r[metric][l] for r in rows) for l in range(L)]
            # first layer from which the recall stays >= 0.9 to the end
            stable = next((l for l in range(L) if min(curve[l:]) >= 0.9), None)
            print(f"- {name} ({L} layer, {len(rows)} prompt) {metric}: ổn định ≥0.9 từ L{stable}"
                  f" ({stable / L:.2f}L)" if stable is not None else f"- {name} {metric}: không ổn định")
        curve = [mean(r["unit_rec"][l] for r in rows) for l in range(L)]
        print("  " + " ".join(f"{l}:{c:.2f}" for l, c in enumerate(curve)))
    print("\n### Sweep unit (layer cuối giữ 24.5%), đúng / số câu dense đúng\n")
    print("| Model | ℓ₀ | ℓ₀/L | ρ | " + " | ".join(TASKS) + " | Tổng | % dense | token×layer |")
    print("|---|---|---|---|" + "---|" * (len(TASKS) + 3))
    for key, name, L in MODELS:
        p = f"{R}/ids/{key}.txt"
        ids = open(p).read().split() if os.path.exists(p) else []
        if not ids:
            continue
        n = {t: sum(i.startswith(t + "_") for i in ids) for t in TASKS}
        runs = []
        for p in glob.glob(f"{R}/sweep/{key}_s*_d*.jsonl"):
            s, d = re.search(r"_s(\d+)_d([\d.]+)\.jsonl", p).groups()
            runs.append((int(s), float(d), load(p)))
        for s, d, rows in sorted(runs, key=lambda x: (x[0], x[1])):
            ok = defaultdict(int)
            for r in rows:
                ok[r["task"]] += r["ok"]
            tot = sum(ok.values())
            done = len(rows) == len(ids)
            tl = (s + sum((1 - d) ** k for k in range(1, L - s + 1))) / L      # prunable part only, as in README
            print(f"| {name} | {s} | {s / L:.2f} | {d:.4f} | " + " | ".join(f"{ok[t]}/{n[t]}" for t in TASKS)
                  + f" | {tot}/{len(ids)}{'' if done else ' (đang chạy: ' + str(len(rows)) + ')'} | "
                  f"{100 * tot / max(len(rows), 1):.0f}% | {tl:.2f} |")
    print()


def speed_table(rows, title):
    """Medians per (context, schedule); speedups vs dense at the same context. token×layer = prefill compute,
    KV đọc/bước = KV kept after prefill (incl. decode-time sieving) over dense. Graph runs (bench_speed --graph): TPOT =
    wall clock with decode in a CUDA graph. Eager runs: GPU/bước = GPU kernel time per decode step (no CPU launch
    overhead), TPOT eager = wall clock (launch bound, noisy). "bẩn" = runs with another GPU process."""
    by = defaultdict(list)
    for r in rows:
        tgt = int(r["id"].split("_")[1]) if str(r["id"]).startswith("speed_") else r["len"]   # speed_<target>_<k>
        by[(round(tgt / 1024) if tgt < 1_000_000 else 1024, r["start_layer"], r["drop_ratio"], r.get("decode_ref"))].append(r)
    lens = sorted({k[0] for k in by})
    print(title)
    graph = any(r.get("graph") for r in rows)
    tp = "TPOT graph (ms) | TPOT nhanh" if graph else "GPU/bước (ms) | GPU nhanh | TPOT eager (ms)"
    print(f"| Context | Lịch | token×layer | KV đọc/bước | TTFT (s) | TTFT nhanh | {tp} | n | bẩn |")
    print("|" + "---|" * (10 if graph else 11))
    for Lk in lens:
        keys = sorted((k for k in by if k[0] == Lk), key=lambda k: (-k[1], k[2], k[3] or 0))
        dense = [k for k in keys if k[2] == 0.0]
        if not dense:
            continue
        d = by[dense[0]]
        t0, g0 = median(x["ttft"] for x in d), median(x.get("gpu_step") or 0 for x in d)
        p0 = median(x["tpot"] for x in d)
        for k in keys:
            xs = by[k]
            nl = xs[0]["n_layers"]
            ttft, tpot = median(x["ttft"] for x in xs), median(x["tpot"] for x in xs)
            g = median(x.get("gpu_step") or 0 for x in xs)
            kept = xs[0]["kept"]
            kv = sum(kept) / (nl * xs[0]["len"]) if kept else 1.0
            tl = 1.0 if k[2] == 0.0 else (k[1] + sum((1 - k[2]) ** j for j in range(1, nl - k[1] + 1))) / nl
            name = "dense" if k[2] == 0.0 else f"ℓ₀={k[1]} ρ={k[2]:g}" + (f" r={k[3]}" if k[3] is not None else "")
            dirty = sum(bool(x.get("other_gpu_procs")) for x in xs)
            print(f"| {'1M' if Lk == 1024 else f'{Lk}k'} | {name} | {tl:.2f} | {kv:.2f} | "
                  f"{ttft:.2f} | {t0 / ttft:.2f}x | "
                  + (f"{1000 * tpot:.2f} | {p0 / tpot:.2f}x | " if graph else
                     f"{1000 * g:.2f} | {g0 / g if g else 0:.2f}x | {1000 * tpot:.1f} | ")
                  + f"{len(xs)} | {dirty or ''} |")
    print()


def q2():
    print("## Q2. Tốc độ prefill theo cỡ model\n")
    for key, name, L in MODELS:
        rows = load(f"{R}/speed/{key}.jsonl")
        if rows:
            speed_table(rows, f"### {name} ({L} layer)\n")


def q3():
    print("## Q3. Decode khi sàng KV theo độ dài context\n")
    for key, name, L in MODELS:
        rows = load(f"{R}/decode/{key}.jsonl")
        if rows:
            speed_table(rows, f"### {name} ({L} layer)\n")


q1()
q2()
q3()
