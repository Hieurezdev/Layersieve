"""Tables for the multi-turn experiments.
  accuracy: scripts/multiturn_hf.py output (per task x question type x mode), the fresh-vs-resieve check and HF timings
  latency:  scripts/bench_multiturn.py outputs (layers + prefix), composed into per-turn and per-session latency
    python scripts/analysis/multiturn_table.py --acc results/multiturn/hf.jsonl \
        --layers results/multiturn/layers.jsonl --prefix results/multiturn/prefix.jsonl
"""
import argparse
import json
import statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--acc", default="")
ap.add_argument("--layers", default="")
ap.add_argument("--prefix", default="")
ap.add_argument("--start-layer", type=int, default=17)
ap.add_argument("--out-tokens", type=int, default=256)
ap.add_argument("--turns", default="1,2,3,5,10")
a = ap.parse_args()
med = st.median

if a.acc:
    rows = [json.loads(l) for l in open(a.acc)]
    q = [r for r in rows if "mode" in r]
    docs = [r for r in rows if "doc_time" in r]
    acc = defaultdict(list)
    for r in q:
        acc[r["task"], r["type"], r["mode"]].append(r["correct"])
    tasks = sorted({r["task"] for r in q})
    types = ["orig", "same", "switch"]
    print("## Accuracy (% correct, n)\n")
    print("| mode | " + " | ".join(f"{t} {y}" for t in tasks for y in types) + " |")
    print("|---" * (1 + len(tasks) * len(types)) + "|")
    for mode in ["dense", "resieve", "naive", "fresh", "dense_fresh"]:
        cells = []
        for t in tasks:
            for y in types:
                v = acc.get((t, y, mode))
                cells.append(f"{100 * sum(v) / len(v):.0f} ({len(v)})" if v else "–")
        print(f"| {mode} | " + " | ".join(cells) + " |")
    # paired: naive vs resieve on the same (document, question)
    by = {(r["id"], r["type"], r["mode"]): r["correct"] for r in q}
    print("\n## naive vs resieve, paired (both / only resieve / only naive / neither)\n")
    for t in tasks:
        for y in ["same", "switch"]:
            ids = [i for (i, yy, m) in by if yy == y and m == "naive" and any(r["id"] == i and r["task"] == t for r in q)]
            c = defaultdict(int)
            for i in ids:
                c[(by[i, y, "resieve"], by[i, y, "naive"])] += 1
            print(f"- {t} {y}: {c[True, True]} / {c[True, False]} / {c[False, True]} / {c[False, False]}")
    f = [r for r in q if r["mode"] == "fresh"]
    d = [r for r in q if r["mode"] == "dense_fresh"]
    if f:
        print(f"\n## resieve vs fresh LayerSieve ({len(f)} questions)\n")
        print(f"- same prediction: {sum(r['same_pred'] for r in f)}/{len(f)}; same first token: "
              f"{sum(r['same_first_token'] for r in f)}/{len(f)}; identical kept set: {sum(r['same_kept'] for r in f)}/{len(f)}")
        print(f"- kept-set Jaccard: median {med(r['kept_jaccard'] for r in f):.3f}, min {min(r['kept_jaccard'] for r in f):.3f}")
        print(f"- first-token logit max |diff|: median {med(r['logit_maxdiff'] for r in f):.2f}, max {max(r['logit_maxdiff'] for r in f):.2f}")
    if d:
        print(f"- noise floor, dense with vs without prefix cache ({len(d)}): same prediction {sum(r['same_pred'] for r in d)}/{len(d)}, "
              f"logit max |diff| median {med(r['logit_maxdiff'] for r in d):.2f}, max {max(r['logit_maxdiff'] for r in d):.2f}")
    print("\n## HF prefill time per question (s, median; shared GPU)\n")
    for mode in ["dense", "resieve", "naive", "fresh", "dense_fresh"]:
        v = [r["ttft"] for r in q if r["mode"] == mode]
        if v:
            print(f"- {mode}: {med(v):.2f}")
    if docs:
        L = len(docs[0]["layer_time"])
        print(f"- document dense prefill: {med(r['doc_time'] for r in docs):.2f} "
              f"(layers >= {a.start_layer}: {med(sum(r['layer_time'][a.start_layer:]) for r in docs):.2f}); "
              f"peak memory {max(r['peak_gib'] for r in docs):.1f} GiB")

if a.layers:
    lay = [json.loads(l) for l in open(a.layers)]
    pre = [json.loads(l) for l in open(a.prefix)] if a.prefix else []
    n = a.out_tokens
    S = a.start_layer
    print("\n## Latency in vLLM (s, median over prompts)\n")
    for ln in sorted({r["len"] // 1000 for r in lay}):
        g = lambda cfg, f: med(f(r) for r in lay if r["cfg"] == cfg and r["len"] // 1000 == ln)  # noqa: E731
        d_ttft, d_tpot = g("dense", lambda r: r["ttft"]), g("dense", lambda r: r["tpot"])
        l_ttft, l_tpot = g("layersieve", lambda r: r["ttft"]), g("layersieve", lambda r: r["tpot"])
        L = len(lay[0]["layer_ms"])
        b_fill = g("dense", lambda r: sum(r["layer_ms"][S:]) / 1e3)
        a_deep = g("layersieve", lambda r: sum(r["layer_ms"][S:]) / 1e3)
        pp = [r for r in pre if r["len"] // 1000 == ln and r["turn"] > 0]
        eps = med(r["ttft"] for r in pp) if pp else float("nan")
        pc_meas = med(r["tpot"] for r in pp) if pp else float("nan")
        pc_tpot = d_tpot                     # decode over a full dense cache = dense decode (pc_meas is noisier)
        a_follow = a_deep + eps * S / L
        print(f"### ~{ln}k tokens\n")
        print(f"- dense TTFT {d_ttft:.2f}, TPOT {1e3 * d_tpot:.1f} ms; LayerSieve TTFT {l_ttft:.2f}, TPOT {1e3 * l_tpot:.1f} ms")
        print(f"- follow-up prefill: dense + prefix cache (eps) {eps:.3f}; mode A {a_follow:.3f} "
              f"(sieve layers {a_deep:.3f} + question through {S} dense layers); mode B background fill {b_fill:.2f}; "
              f"TPOT over the prefix cache as measured {1e3 * pc_meas:.1f} ms (dense TPOT used below)")
        print("\n| turns | dense + prefix cache | LayerSieve, discard | A: resieve | B: fill (off critical path) |")
        print("|---|---|---|---|---|")
        for N in map(int, a.turns.split(",")):
            dense = d_ttft + n * d_tpot + (N - 1) * (eps + n * pc_tpot)
            disc = N * (l_ttft + n * l_tpot)
            A = l_ttft + n * l_tpot + (N - 1) * (a_follow + n * l_tpot)
            B = l_ttft + n * l_tpot + (N - 1) * (eps + n * pc_tpot)
            print(f"| {N} | {dense:.1f} | {disc:.1f} | {A:.1f} | {B:.1f} |")
        print()
