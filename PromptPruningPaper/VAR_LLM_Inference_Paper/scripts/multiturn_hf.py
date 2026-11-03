"""Several questions over one document (data/build_multiq.py) with the modes of inmodel_prune/multiturn.py:
dense (= mode B, dense with a prefix cache), resieve (= mode A), naive (the first question's pruned cache reused),
partial (= mode C, only what the new question adds is computed) and,
for the first --verify documents, fresh (LayerSieve from scratch) to check that resieve reproduces it.
One line per (document, question, mode): {"id", "task", "type", "mode", "pred", "correct", "ttft", "kept", ...};
one line per document with "doc_time" and the dense per-layer prefill times.
    python scripts/multiturn_hf.py --data data/multiq.jsonl --verify 3 --out results/multiturn/hf.jsonl
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer  # noqa: E402

from inmodel_prune.multiturn import MultiTurn, SieveConfig  # noqa: E402
from inmodel_prune.prune import InModelPruneConfig, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--data", default="data/multiq.jsonl")
ap.add_argument("--tasks", default="", help="comma list of kv, mk3")
ap.add_argument("--n", type=int, default=0, help="documents per task (0 = all)")
ap.add_argument("--modes", default="dense,resieve,naive", help="naive / partial reuse the orig question's resieve")
ap.add_argument("--verify", type=int, default=3, help="also run fresh LayerSieve on the first N documents per task")
ap.add_argument("--start-layer", type=int, default=17, help="schedule (paper Eq. 4 for other depths: l0 = 0.61L, "
                "rho = 1 - 0.6^(28/L), decode_ref ~ 0.79L)")
ap.add_argument("--drop-ratio", type=float, default=0.4)
ap.add_argument("--decode-ref", type=int, default=22)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = load_model(a.model, InModelPruneConfig(mlp_chunk=16384))
mt = MultiTurn(model, tok, SieveConfig(start_layer=a.start_layer, drop_ratio=a.drop_ratio, decode_ref=a.decode_ref))
EOS = {tok.eos_token_id} | {i for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>") if (i := tok.convert_tokens_to_ids(t))
                            not in (None, tok.unk_token_id)}


def score(pred, q):
    """RULER metrics: "all" = fraction of the answers found, "part" = 1 if any is found (kv / niah: one answer)."""
    hit = [ans.lower() in pred.lower() for ans in q["answers"]]
    return sum(hit) / len(hit) if q.get("metric", "all") == "all" else float(any(hit))
done = {json.loads(l)["id"] for l in open(a.out) if '"doc_time"' in l} if os.path.exists(a.out) else set()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
tasks = set(a.tasks.split(",")) if a.tasks else None
seen = {}
for line in open(a.data):
    r = json.loads(line)
    if tasks and r["task"] not in tasks:
        continue
    seen[r["task"]] = seen.get(r["task"], 0) + 1
    if (a.n and seen[r["task"]] > a.n) or r["id"] in done:
        continue
    verify = seen[r["task"]] <= a.verify
    doc_time = mt.load_document(torch.tensor(r["prefix_ids"], device="cuda"))
    rows = []
    for q in r["questions"]:
        suf = torch.tensor(q["suffix_ids"], device="cuda")
        modes = [m for m in a.modes.split(",") if q["type"] != "orig" or m not in ("naive", "partial")
                 and not m.startswith("fill")] + \
            (["fresh"] if verify else []) + (["dense_fresh"] if verify and q["type"] == "orig" else [])
        res = {}
        for mode in modes:
            ids, ttft, info = mt.answer(mode, suf, r["max_new_tokens"], EOS)
            res[mode] = info
            pred = tok.decode(ids, skip_special_tokens=True).strip()
            rec = {"id": r["id"], "task": r["task"], "type": q["type"], "mode": mode, "pred": pred,
                   "correct": score(pred, q), "ttft": ttft,
                   "kept": int(info["kept"].shape[0]) if "kept" in info else None,
                   "prompt_len": len(r["prefix_ids"]) + len(q["suffix_ids"]),
                   **{k: info[k] for k in ("new", "deep_tokens", "fill_tokens") if k in info}}
            if mode == "dense_fresh":                            # noise floor: dense with vs without a prefix cache
                rec["logit_maxdiff"] = float((res["dense"]["logits"] - info["logits"]).abs().max())
                rec["same_pred"] = pred == next(x["pred"] for x in rows if x["type"] == q["type"] and x["mode"] == "dense")
            if mode == "fresh":                                  # does resieve reproduce LayerSieve from scratch?
                ka, kb = res["resieve"]["kept"], info["kept"]
                rec["same_kept"] = bool(ka.shape == kb.shape and torch.equal(ka, kb))
                rec["kept_jaccard"] = len(set(ka.tolist()) & set(kb.tolist())) / len(set(ka.tolist()) | set(kb.tolist()))
                la, lb = res["resieve"]["logits"], info["logits"]
                rec["logit_maxdiff"] = float((la - lb).abs().max())
                rec["same_first_token"] = bool(la.argmax() == lb.argmax())
                rec["same_pred"] = pred == next(x["pred"] for x in rows if x["type"] == q["type"] and x["mode"] == "resieve")
            rows.append(rec)
            print(r["id"], q["type"], mode, rec["correct"], f"{ttft:.2f}s", repr(pred[:60]),
                  {k: rec[k] for k in ("same_kept", "kept_jaccard", "logit_maxdiff", "same_pred") if k in rec}, flush=True)
            if mode == "resieve" and q["type"] == "orig":
                mt.keep_for_naive()
    with open(a.out, "a") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"id": r["id"], "task": r["task"], "doc_time": doc_time, "layer_time": mt.layer_time,
                            "prefix_len": len(r["prefix_ids"]),
                            "peak_gib": torch.cuda.max_memory_allocated() / 2**30}) + "\n")
    del mt.dense, mt.h_ckpt, mt.naive, mt.last_pruned, mt.deep
    mt.turn1 = None
    torch.cuda.empty_cache()
