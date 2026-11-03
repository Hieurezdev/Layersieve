"""Multi-turn sessions (data/build_scbench.py: shared document, follow-up turns with the golden previous answer as
history) with the modes of inmodel_prune/multiturn.py in conversation mode: dense (= B), resieve (= A), naive, partial
(= C); for the first --verify sessions per task also fresh (LayerSieve from scratch on the whole conversation).
Scored with SCBench's own get_score_one (../ext/MInference/scbench/compute_scores.py).
One line per (session, turn, mode): {"id", "task", "turn", "mode", "pred", "score", "ttft", "kept", ...}.
    python scripts/session_hf.py --data data/scbench128k.jsonl --verify 2 --out results/multiturn/scbench.jsonl
"""
import argparse
import json
import os
import sys
import types

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer  # noqa: E402

from inmodel_prune.multiturn import MultiTurn, SieveConfig  # noqa: E402
from inmodel_prune.prune import InModelPruneConfig, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--data", default="data/scbench128k.jsonl")
ap.add_argument("--scbench", default=os.path.join(ROOT, "../ext/MInference/scbench"))
ap.add_argument("--tasks", default="")
ap.add_argument("--n", type=int, default=0, help="sessions per task (0 = all)")
ap.add_argument("--modes", default="dense,resieve,naive,partial")
ap.add_argument("--verify", type=int, default=2)
ap.add_argument("--start-layer", type=int, default=17)
ap.add_argument("--drop-ratio", type=float, default=0.4)
ap.add_argument("--decode-ref", type=int, default=22)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, a.scbench)
for name in ("args", "repo_qa_utils", "evaluate"):        # imported by compute_scores, unused for these tasks
    sys.modules[name] = types.ModuleType(name)
sys.modules["args"].parse_args = lambda: None
sys.modules["repo_qa_utils"].compute_score = None
sys.modules["evaluate"].load = lambda *x, **k: None
from compute_scores import get_score_one  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
model = load_model(a.model, InModelPruneConfig(mlp_chunk=16384))
mt = MultiTurn(model, tok, SieveConfig(start_layer=a.start_layer, drop_ratio=a.drop_ratio, decode_ref=a.decode_ref))
EOS = {tok.eos_token_id} | {i for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>") if (i := tok.convert_tokens_to_ids(t))
                            not in (None, tok.unk_token_id)}
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
    doc_time = mt.load_document(torch.tensor(r["prefix_ids"], device="cuda"), history=True)
    rows = []
    for k, turn in enumerate(r["turns"]):
        delta = torch.tensor(turn["delta_ids"], device="cuda")
        modes = [m for m in a.modes.split(",") if k > 0 or m not in ("naive", "partial")] + (["fresh"] if verify else [])
        res = {}
        for mode in modes:
            ids, ttft, info = mt.answer(mode, delta, turn["max_new_tokens"], EOS)
            res[mode] = info
            pred = tok.decode(ids, skip_special_tokens=True).strip()
            rec = {"id": r["id"], "task": r["task"], "turn": k, "mode": mode, "pred": pred,
                   "score": get_score_one(pred, turn["label"], r["score_task"], a.model), "ttft": ttft,
                   "kept": int(info["kept"].shape[0]) if "kept" in info else None, "conv_len": int(mt.P.shape[0]) + len(delta),
                   **{x: info[x] for x in ("new", "deep_tokens") if x in info}}
            if mode == "fresh":
                ka, kb = res["resieve"]["kept"], info["kept"]
                rec["kept_jaccard"] = len(set(ka.tolist()) & set(kb.tolist())) / len(set(ka.tolist()) | set(kb.tolist()))
                rec["logit_maxdiff"] = float((res["resieve"]["logits"] - info["logits"]).abs().max())
                rec["same_pred"] = pred == next(x["pred"] for x in rows if x["turn"] == k and x["mode"] == "resieve")
            rows.append(rec)
            print(r["id"], k, mode, rec["score"], f"{ttft:.2f}s", repr(pred[:50]),
                  {x: rec[x] for x in ("kept_jaccard", "logit_maxdiff", "same_pred") if x in rec}, flush=True)
            if mode == "resieve" and k == 0:
                mt.keep_for_naive()
        mt.extend(delta)                                   # the turn (previous answer + question) joins the history
    with open(a.out, "a") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"id": r["id"], "task": r["task"], "doc_time": doc_time, "prefix_len": len(r["prefix_ids"]),
                            "peak_gib": torch.cuda.max_memory_allocated() / 2**30}) + "\n")
    del mt.dense, mt.h_ckpt, mt.naive, mt.last_pruned, mt.deep
    mt.turn1 = None
    torch.cuda.empty_cache()
