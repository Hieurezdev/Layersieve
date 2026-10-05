"""Hard-case eval (prompts from data/build_hard.py) on vLLM: dense (draft or target), SpecPrefill (Qwen port,
baselines/specprefill_qwen.py) or LayerSieve. Records answer correctness, TTFT and, for pruning methods, how much of
the evidence survived (a span counts as kept if >= half of its tokens are kept).
Usage:
    python scripts/eval_hard.py --model Qwen/Qwen2.5-1.5B-Instruct --method dense --out results/hard/draft15_dense.jsonl
    python scripts/eval_hard.py --method specprefill --spec-model Qwen/Qwen2.5-1.5B-Instruct \
        --spec-config baselines/SpecPrefill/configs/config_p1_full_lah8.yaml --out results/hard/sp15_p1.jsonl
    python scripts/eval_hard.py --method layersieve --out results/hard/layersieve.jsonl
"""
import argparse
import json
import os
import re
import string
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--prompts", default=os.path.join(ROOT, "data/hard.jsonl"))
ap.add_argument("--method", default="dense", choices=["dense", "specprefill", "layersieve"])
ap.add_argument("--spec-model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--spec-config", default=os.path.join(ROOT, "baselines/SpecPrefill/configs/config_p1_full_lah8.yaml"))
ap.add_argument("--start-layer", type=int, default=17)
ap.add_argument("--drop-ratio", type=float, default=0.12)
ap.add_argument("--tasks", default="", help="comma list of tasks to run (default all)")
ap.add_argument("--ids", default="", help="file with one id per line: run only these")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.method == "specprefill":
    import specprefill_qwen
    specprefill_qwen.enable(a.spec_model, a.spec_config)
    kept_positions = specprefill_qwen.last_kept
elif a.method == "layersieve":
    from inmodel_prune.vllm_patch import enable_inmodel_prune, last_kept_positions
    enable_inmodel_prune(start_layer=a.start_layer, drop_ratio=a.drop_ratio)
    kept_positions = last_kept_positions
else:
    kept_positions = lambda: None  # noqa: E731

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MAX_TOKENS = {"kv": 100, "vt": 50}


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def correct(task, pred, answers):
    if task in ("kv", "vt"):
        return any(x in pred for x in answers)
    return any(norm(x) in norm(pred) for x in answers)


done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
tasks = set(a.tasks.split(",")) if a.tasks else None
only = {l.strip() for l in open(a.ids)} if a.ids else None

llm = LLM(a.model, enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_num_seqs=1,
          max_model_len=32768, gpu_memory_utilization=0.85)
first = SamplingParams(temperature=0.0, max_tokens=1)
for line in open(a.prompts):
    r = json.loads(line)
    if r["id"] in done or (tasks and r["task"] not in tasks) or (only is not None and r["id"] not in only):
        continue
    p = TokensPrompt(prompt_token_ids=r["prompt_ids"])
    t0 = time.time(); llm.generate([p], first, use_tqdm=False); ttft = time.time() - t0
    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS.get(r["task"], 32))
    pred = llm.generate([p], greedy, use_tqdm=False)[0].outputs[0].text
    rec = {"id": r["id"], "task": r["task"], "len": r["len"], "pred": pred, "ok": correct(r["task"], pred, r["answers"]),
           "ttft": ttft}
    kept = kept_positions()
    if kept is not None:
        ks = set(kept)
        rec["kept_frac"] = len(kept) / r["len"]
        span_kept = [sum(i in ks for i in range(s, e)) >= (e - s) / 2 for s, e in r["evidence"]]
        rec["ev_kept"] = span_kept
    print(rec, flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
