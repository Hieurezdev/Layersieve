"""TTFT and TPOT of LayerSieve schedules vs dense on long kv prompts (data/kv50.jsonl), vLLM 0.6.3, one process.
The model is loaded once; enable_inmodel_prune is re-called to switch schedule. "dense" = start_layer 28 (nothing
dropped) through the same patched code path, so every config runs the same chunked MLP. Configs are interleaved per
prompt (after a warmup) so slow drift of a shared GPU hits all of them alike. Decode runs --decode tokens with
ignore_eos; TTFT / TPOT come from vLLM request metrics (first_token_time - first_scheduled_time, and
(finished_time - first_token_time) / (decode - 1); last_token_time is only kept up to date when stats logging is on).
A config is start_layer:drop_ratio[:decode_ref] (decode_ref: decode-time KV sieving, see vllm_patch).
Usage: python scripts/bench_speed.py --configs 28:0,17:0.12,14:0.1,17:0.12:27 --n 3 --out results/sweep/speed.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--prompts", default="data/kv50.jsonl")
ap.add_argument("--configs", required=True, help="comma list of start_layer:drop_ratio[:decode_ref] (28:0 = dense)")
ap.add_argument("--n", type=int, default=3, help="number of prompts")
ap.add_argument("--decode", type=int, default=256)
ap.add_argument("--gpu-util", type=float, default=0.68)
ap.add_argument("--max-model-len", type=int, default=170000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from inmodel_prune.vllm_patch import enable_inmodel_prune  # noqa: E402

configs = [(int(c[0]), float(c[1]), int(c[2]) if len(c) > 2 else None) for c in (x.split(":") for x in a.configs.split(","))]
enable_inmodel_prune(start_layer=28, drop_ratio=0.0, mlp_chunk=16384)

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(a.model, enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_num_seqs=1,
          max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util)
sp = SamplingParams(temperature=0.0, max_tokens=a.decode, ignore_eos=True)
rows = [json.loads(l) for _, l in zip(range(a.n), open(a.prompts))]
done = {(x["id"], x["start_layer"], x["drop_ratio"], x.get("decode_ref")) for x in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
llm.generate([TokensPrompt(prompt_token_ids=rows[0]["prompt_ids"][:8192])], sp, use_tqdm=False)       # warmup
for r in rows:
    p = TokensPrompt(prompt_token_ids=r["prompt_ids"])
    for s, d, ref in configs:
        if (r["id"], s, d, ref) in done:
            continue
        enable_inmodel_prune(start_layer=s, drop_ratio=d, mlp_chunk=16384, decode_ref=ref)
        out = llm.generate([p], sp, use_tqdm=False)[0]
        m = out.metrics
        rec = {"id": r["id"], "len": len(r["prompt_ids"]), "start_layer": s, "drop_ratio": d, "decode_ref": ref,
               "ttft": m.first_token_time - m.first_scheduled_time,
               "tpot": (m.finished_time - m.first_token_time) / (a.decode - 1),
               "n_out": len(out.outputs[0].token_ids)}
        print(rec, flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
