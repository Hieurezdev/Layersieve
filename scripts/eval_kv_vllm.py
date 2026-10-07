"""Run InfiniteBench kv_retrieval prompts (built by data/build_kv.py) on vLLM: dense, "cat dan" (in-model
progressive pruning) or SlimInfer (re-implemented, --method slim). Records TTFT (1-token generate) and greedy answers.
Usage: python scripts/eval_kv_vllm.py --prompts prompts.jsonl [--dense] [--start-layer 17 --drop-ratio 0.12]
prompts.jsonl lines: {"id": ..., "prompt_ids": [...], "answers": [...]}"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--prompts", required=True)
ap.add_argument("--dense", action="store_true")
ap.add_argument("--start-layer", type=int, default=17)
ap.add_argument("--drop-ratio", type=float, default=0.12)
ap.add_argument("--max-tokens", type=int, default=100)
ap.add_argument("--method", default="ours", choices=["ours", "slim"])
ap.add_argument("--slim-ref-len", type=int, default=0, help=">0: scale SlimInfer kept counts by prompt_len / this")
ap.add_argument("--slim-frac", default="", help="e.g. 0.8,0.6,0.4: SlimInfer keeps these fractions of the prompt after L8/L17/L25")
ap.add_argument("--unit", action="store_true", help="MOE variant: drop whole units (lines/sentences/clauses)")
ap.add_argument("--accum", action="store_true", help="MOE variant: rank by importance summed over the layers scored so far")
ap.add_argument("--bm25", type=float, default=0.0, help="MOE variant: protect the BM25-best units, up to this fraction of the prompt")
ap.add_argument("--score", default="attn", choices=["attn", "bm25"], help="bm25: BM25 replaces attention importance (ablation)")
ap.add_argument("--gpu-util", type=float, default=0.85)
ap.add_argument("--max-model-len", type=int, default=180000)
ap.add_argument("--decode-ref", type=int, default=None, help="layers < R keep only the KV of tokens that reached R")
ap.add_argument("--mlp-chunk", type=int, default=0, help=">0: MLP in token chunks (exact, less memory; pruning runs only)")
ap.add_argument("--no-ttft", action="store_true", help="skip the separate 1-token TTFT run (accuracy only)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if not a.dense:
    from inmodel_prune.vllm_patch import enable_inmodel_prune, last_kept
    enable_inmodel_prune(start_layer=a.start_layer, drop_ratio=a.drop_ratio, method=a.method, slim_ref_len=a.slim_ref_len,
                         unit=a.unit, accum=a.accum, bm25_frac=a.bm25, score=a.score, mlp_chunk=a.mlp_chunk,
                         decode_ref=a.decode_ref,
                         slim_layers=dict(zip((8, 17, 25), map(float, a.slim_frac.split(",")))) if a.slim_frac else None)

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(a.model, enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_num_seqs=1,
          max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util)
greedy = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
first = SamplingParams(temperature=0.0, max_tokens=1)
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
for line in open(a.prompts):
    r = json.loads(line)
    if r["id"] in done:
        continue
    p = TokensPrompt(prompt_token_ids=r["prompt_ids"])
    ttft = None
    if not a.no_ttft:
        t0 = time.time(); llm.generate([p], first, use_tqdm=False); ttft = time.time() - t0
    out = llm.generate([p], greedy, use_tqdm=False)[0].outputs[0].text
    kept = None if a.dense else last_kept()
    rec = {"id": r["id"], "pred": out, "ok": any(x in out for x in r["answers"]), "ttft": ttft,
           "kept_last": kept[-1] if kept else len(r["prompt_ids"])}
    print(rec, flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
