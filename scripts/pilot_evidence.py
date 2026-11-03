"""Evidence-loss pilot for LayerSieve (prompts from data/build_pilot.py). One vLLM process per configuration.
  --mode dense        answers of the unpruned model (gate: only dense-correct prompts are analysed)
  --mode dense-trace  unpruned forward with the LayerSieve score of every unit at every layer >= --trace-from
                      (counterfactual scores after a cut; never mixed with the pruned run's own scores)
  --mode ls           LayerSieve prefill (start, rate, floor, units; no KV sieving). The scores of the dense layers
                      [trace-from, start) and of every pruned layer (survivors only) are recorded, so every statistic
                      of a unit uses only layers at which it was still alive. With --dense, every prompt that dense
                      answers correctly and LayerSieve answers wrongly after losing evidence is re-run twice at the
                      same per-layer token budget: (force) the evidence units can never be dropped, other low-scoring
                      units are dropped in their place; (placebo) the same number of tokens of randomly chosen
                      non-evidence units that were dropped in the original run are protected instead.
Per prompt it writes one json line (answers, per-layer evidence trajectory, interventions) and, with --npz-dir, the
per-unit arrays used by scripts/pilot_report.py (percentile rank per layer, head agreement, drop layer).
    python scripts/pilot_evidence.py --mode dense --out results/pilot/dense.jsonl
    python scripts/pilot_evidence.py --mode ls --start-layer 17 --drop-ratio 0.4 --min-keep 0.25 \
        --dense results/pilot/dense.jsonl --out results/pilot/ls_l17.jsonl --npz-dir results/pilot/npz/ls_l17
"""
import argparse
import json
import os
import random
import re
import string
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--prompts", default=os.path.join(ROOT, "data/pilot.jsonl"))
ap.add_argument("--mode", required=True, choices=["dense", "dense-trace", "ls"])
ap.add_argument("--start-layer", type=int, default=17)
ap.add_argument("--drop-ratio", type=float, default=0.4)
ap.add_argument("--min-keep", type=float, default=0.25)
ap.add_argument("--trace-from", type=int, default=6)
ap.add_argument("--dense", default="", help="dense answers (jsonl): enables the interventions")
ap.add_argument("--ids", default="", help="comma list or file of ids to run")
ap.add_argument("--max-model-len", type=int, default=65536)
ap.add_argument("--gpu-util", type=float, default=0.32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--npz-dir", default="")
a = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from inmodel_prune import units  # noqa: E402
from inmodel_prune import vllm_patch as vp  # noqa: E402

N_SINK = N_OBS = 128
L = AutoConfig.from_pretrained(a.model).num_hidden_layers
if a.mode == "ls":
    vp.enable_inmodel_prune(start_layer=a.start_layer, drop_ratio=a.drop_ratio, min_keep=a.min_keep, unit=True,
                            trace_from=a.trace_from, mlp_chunk=16384, tokenizer=a.model)
elif a.mode == "dense-trace":
    vp.enable_inmodel_prune(start_layer=L, drop_ratio=0.0, unit=True, trace_from=a.trace_from, mlp_chunk=16384,
                            tokenizer=a.model)
boundary = units.boundary_token_ids(AutoTokenizer.from_pretrained(a.model))

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(a.model, enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_num_seqs=1,
          max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def correct(task, pred, answers):
    if task == "qa2":
        return any(norm(x) in norm(pred) for x in answers)
    return any(x in pred for x in answers)


def generate(r):
    sp = SamplingParams(temperature=0.0, max_tokens=r["max_new_tokens"])
    torch.cuda.synchronize()
    t0 = time.time()
    pred = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"])], sp, use_tqdm=False)[0].outputs[0].text
    return pred.strip(), time.time() - t0


def pct_rank(x):
    """Percentile rank along the last dim in [0, 1], 1 = highest score."""
    r = torch.empty_like(x)
    n = x.shape[-1]
    r.scatter_(-1, torch.argsort(x, -1), (torch.arange(n, dtype=x.dtype) / max(n - 1, 1)).expand_as(x).contiguous())
    return r


def analyse(r, trace):
    """Per-unit arrays over the traced layers (NaN where the unit was not alive) and the per-layer evidence summary."""
    ids = torch.tensor(r["prompt_ids"])
    T = len(ids)
    unit = units.unit_ids(ids, boundary)
    U = int(unit.max()) + 1
    pr = torch.zeros(T, dtype=torch.bool)
    pr[N_SINK:T - N_OBS] = True
    ev_tok = torch.zeros(T, dtype=torch.bool)
    for s, e in r["evidence"]:
        ev_tok[s:e] = True
    is_ev = torch.zeros(U, dtype=torch.bool)
    is_ev[unit[ev_tok]] = True
    span_units = [sorted(set(unit[s:e].tolist())) for s, e in r["evidence"]]
    usize = torch.bincount(unit[pr], minlength=U)
    n_tr = len(trace)
    pct = torch.full((n_tr, U), float("nan"))
    score = torch.full((n_tr, U), float("nan"))
    head_top = torch.full((n_tr, U), float("nan"))      # fraction of heads ranking the unit in their top 25%
    head_spread = torch.full((n_tr, U), float("nan"))   # best-head percentile - median-head percentile
    drop_layer = torch.full((U,), -1, dtype=torch.long)
    layers = []
    for t, tr in enumerate(trace):
        pos = tr["pos"].long()
        sel = pr[pos]
        u = unit[pos[sel]]
        uniq, inv = torch.unique(u, return_inverse=True)
        us = torch.full((len(uniq),), float("-inf")).scatter_reduce(0, inv, tr["imp"][sel], reduce="amax")
        pct[t, uniq] = pct_rank(us)
        score[t, uniq] = us
        hd = tr["head"].float()[:, sel]
        H = hd.shape[0]
        hs = torch.full((H, len(uniq)), float("-inf")).scatter_reduce(1, inv.expand(H, -1), hd, reduce="amax")
        hp = pct_rank(hs)
        head_top[t, uniq] = (hp >= 0.75).float().mean(0)
        head_spread[t, uniq] = hp.max(0).values - hp.median(0).values
        info = {"layer": tr["layer"], "alive_tok": int(len(pos)), "alive_units": int(len(uniq)),
                "ev_tok_alive": round(float(ev_tok[pos].sum()) / max(int(ev_tok.sum()), 1), 4)}
        if tr["dropped"] is not None and len(tr["dropped"]):
            du = torch.unique(unit[tr["dropped"].long()])
            drop_layer[du] = tr["layer"]
            dmask = torch.zeros(U, dtype=torch.bool)
            dmask[du] = True
            info["cut"] = float(score[t][dmask].max())                      # best score among dropped units
            info["n_drop_units"] = int(len(du))
        ev_alive = [int(x) for x in uniq[is_ev[uniq]]]
        info["ev"] = {str(x): {"pct": round(float(pct[t, x]), 4), "score": float(score[t, x]),
                               "rank": int((score[t] > score[t, x]).sum()) + 1,
                               "head_top": round(float(head_top[t, x]), 3),
                               "head_spread": round(float(head_spread[t, x]), 3),
                               "dropped_here": bool(drop_layer[x] == tr["layer"])} for x in ev_alive}
        layers.append(info)
    ev_drop = {str(int(x)): int(drop_layer[x]) for x in torch.nonzero(is_ev).squeeze(1)}
    lost = [l for l in ev_drop.values() if l >= 0]
    summ = {"span_units": span_units, "ev_drop_layer": ev_drop, "first_loss_layer": min(lost) if lost else None,
            "spans_lost": sum(any(drop_layer[x] >= 0 for x in su) for su in span_units),
            "ev_all_kept": not lost, "layers": layers}
    arrays = {"layers": np.array([tr["layer"] for tr in trace]), "pct": pct.half().numpy(),
              "score": score.numpy(), "head_top": head_top.half().numpy(), "head_spread": head_spread.half().numpy(),
              "drop_layer": drop_layer.numpy(), "is_ev": is_ev.numpy(), "usize": usize.numpy()}
    return summ, arrays, unit, is_ev, drop_layer


def ev_kept_now(r):
    """After an intervention: fraction of evidence tokens that reached the last layer."""
    kept = set(vp.last_kept_positions())
    tok = [i for s, e in r["evidence"] for i in range(s, e)]
    return sum(i in kept for i in tok) / len(tok)


only = None
if a.ids:
    only = set(open(a.ids).read().split()) if os.path.exists(a.ids) else set(a.ids.split(","))
dense = {}
if a.dense:
    dense = {(d := json.loads(l))["id"]: d for l in open(a.dense)}
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
if a.npz_dir:
    os.makedirs(a.npz_dir, exist_ok=True)
rng = random.Random(a.seed)
for line in open(a.prompts):
    r = json.loads(line)
    if r["id"] in done or (only is not None and r["id"] not in only):
        continue
    pred, sec = generate(r)
    rec = {"id": r["id"], "task": r["task"], "len": r["len"], "pred": pred, "ok": correct(r["task"], pred, r["answers"]),
           "sec": round(sec, 3)}
    if a.mode != "dense":
        t0 = time.time()
        summ, arrays, unit, is_ev, drop_layer = analyse(r, vp.last_trace())
        rec["analyse_sec"] = round(time.time() - t0, 3)
        rec.update(summ)
        if a.mode == "ls":
            rec["kept"] = vp.last_kept()
        if a.npz_dir:
            np.savez_compressed(os.path.join(a.npz_dir, r["id"] + ".npz"), **arrays)
    d = dense.get(r["id"])
    if a.mode == "ls" and d is not None and d["ok"] and not rec["ok"] and not rec["ev_all_kept"]:
        budget = rec["kept"]
        ev_pos = torch.nonzero(torch.isin(unit, torch.nonzero(is_ev).squeeze(1))).squeeze(1).tolist()
        vp.set_intervention(protect_pos=ev_pos, budget=budget)
        p, s = generate(r)
        rec["force"] = {"pred": p, "ok": correct(r["task"], p, r["answers"]), "ev_kept": ev_kept_now(r),
                        "kept": vp.last_kept()[-1], "sec": round(s, 3)}
        # placebo: protect randomly chosen dropped non-evidence units with about as many tokens as the evidence
        cand = torch.nonzero((drop_layer >= 0) & ~is_ev).squeeze(1).tolist()
        rng.shuffle(cand)
        size = torch.bincount(unit, minlength=len(is_ev))
        pick, n = [], 0
        for c in cand:
            if n >= len(ev_pos):
                break
            pick.append(c)
            n += int(size[c])
        vp.set_intervention(protect_pos=torch.nonzero(torch.isin(unit, torch.tensor(pick))).squeeze(1).tolist(),
                            budget=budget)
        p, s = generate(r)
        rec["placebo"] = {"pred": p, "ok": correct(r["task"], p, r["answers"]), "ev_kept": ev_kept_now(r),
                          "kept": vp.last_kept()[-1], "protected_tok": n, "sec": round(s, 3)}
        vp.set_intervention()
    print(r["id"], rec["ok"], "first_loss", rec.get("first_loss_layer"), repr(pred[:60]),
          {k: rec[k]["ok"] for k in ("force", "placebo") if k in rec}, flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
