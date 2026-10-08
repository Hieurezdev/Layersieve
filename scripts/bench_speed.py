"""TTFT and TPOT of LayerSieve schedules vs dense on long kv prompts (data/kv50.jsonl or data/speed.jsonl), vLLM 0.6.3,
one process. The model is loaded once; enable_inmodel_prune is re-called to switch schedule. "dense" (or start_layer =
number of layers) = nothing dropped, through the same patched code path, so every config runs the same chunked MLP. Configs are interleaved per
prompt (after a warmup) so slow drift of a shared GPU hits all of them alike. Decode runs --decode tokens with
ignore_eos; TTFT / TPOT come from vLLM request metrics (first_token_time - first_scheduled_time, and
(finished_time - first_token_time) / (decode - 1); last_token_time is only kept up to date when stats logging is on).
A config is start_layer:drop_ratio[:decode_ref] (decode_ref: decode-time KV sieving, see vllm_patch).
--graph: decode runs in a CUDA graph (wall-clock TPOT is then the real step time); prefill is always eager.
gpu_step (eager only): summed GPU kernel time per decode step over the last --prof-steps steps (CUDA profiler); eager decode is
CPU-launch bound on fast GPUs, so this is the step time without launch overhead (~ what CUDA graphs would give).
--lengths L1,L2: use the first --n prompts of data/speed.jsonl (data/build_speed.py) with id speed_<L>_*, per length.
--yarn F: YaRN rope scaling by F over the native 32k (Qwen2.5 non-1M models; speed only, quality is not measured).
Usage: python scripts/bench_speed.py --configs 28:0,17:0.12,14:0.1,17:0.12:27 --n 3 --out results/sweep/speed.jsonl
       python scripts/bench_speed.py --model Qwen/Qwen2.5-14B-Instruct --yarn 4 --prompts data/speed.jsonl \
           --lengths 32768,131072 --configs dense,17:0.12 --n 2 --out results/scaling/speed/14b.jsonl
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--prompts", default="data/kv50.jsonl")
ap.add_argument("--configs", required=True, help="comma list of dense | start_layer:drop_ratio[:decode_ref]")
ap.add_argument("--lengths", default="", help="comma list: prompts speed_<L>_* of --prompts, --n per length")
ap.add_argument("--yarn", type=float, default=0.0, help="YaRN factor over original_max_position_embeddings 32768")
ap.add_argument("--n", type=int, default=3, help="number of prompts")
ap.add_argument("--decode", type=int, default=256)
ap.add_argument("--gpu-util", type=float, default=0.68)
ap.add_argument("--max-model-len", type=int, default=0, help="0 = longest prompt + decode")
ap.add_argument("--prof-steps", type=int, default=32, help="eager only: profile GPU kernels of the last N decode steps (0 = off)")
ap.add_argument("--graph", action="store_true", help="decode in a CUDA graph (prefill stays eager)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from transformers import AutoConfig  # noqa: E402

from inmodel_prune.vllm_patch import decode_kernel_time, enable_inmodel_prune, last_kept  # noqa: E402

L = AutoConfig.from_pretrained(a.model).num_hidden_layers
configs = [(L, 0.0, None) if x == "dense" else (int(c[0]), float(c[1]), int(c[2]) if len(c) > 2 else None)
           for x, c in ((x, x.split(":")) for x in a.configs.split(","))]
enable_inmodel_prune(start_layer=L, drop_ratio=0.0, mlp_chunk=16384)
if a.lengths:
    allr = [json.loads(l) for l in open(a.prompts)]
    rows = [r for n in a.lengths.split(",") for r in [x for x in allr if x["id"].startswith(f"speed_{n}_")][:a.n]]
else:
    rows = [json.loads(l) for _, l in zip(range(a.n), open(a.prompts))]
max_len = a.max_model_len or max(len(r["prompt_ids"]) for r in rows) + a.decode + 16

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

rope = {"type": "yarn", "factor": a.yarn, "original_max_position_embeddings": 32768} if a.yarn else None
llm = LLM(a.model, enforce_eager=not a.graph, max_seq_len_to_capture=max_len, enable_chunked_prefill=False,
          enable_prefix_caching=False, max_num_seqs=1, max_model_len=max_len, gpu_memory_utilization=a.gpu_util,
          rope_scaling=rope)
if a.graph:
    a.prof_steps = 0                                     # the patched Python forward does not run on graph replay
sp = SamplingParams(temperature=0.0, max_tokens=a.decode, ignore_eos=True)


def others():
    """PIDs of other processes on the GPU (a measurement taken while one runs is not clean)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.split()
    return sorted(int(x) for x in out if x.isdigit() and int(x) != os.getpid())


done = {(x["id"], x["start_layer"], x["drop_ratio"], x.get("decode_ref")) for x in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
llm.generate([TokensPrompt(prompt_token_ids=rows[0]["prompt_ids"][:8192])], sp, use_tqdm=False)       # warmup
for r in rows:
    p = TokensPrompt(prompt_token_ids=r["prompt_ids"])
    for s, d, ref in configs:
        if (r["id"], s, d, ref) in done:
            continue
        enable_inmodel_prune(start_layer=s, drop_ratio=d, mlp_chunk=16384, decode_ref=ref,
                             prof_from=max(a.decode - a.prof_steps, 1) if a.prof_steps else 0)
        busy = others()
        out = llm.generate([p], sp, use_tqdm=False)[0]
        busy = sorted(set(busy) | set(others()))
        gpu_step = decode_kernel_time()
        m = out.metrics
        rec = {"model": a.model, "graph": a.graph, "n_layers": L, "id": r["id"], "len": len(r["prompt_ids"]), "start_layer": s,
               "drop_ratio": d, "decode_ref": ref, "kept": last_kept(),
               "ttft": m.first_token_time - m.first_scheduled_time,
               "tpot": (m.finished_time - m.first_token_time) / (a.decode - 1),
               "gpu_step": gpu_step, "n_out": len(out.outputs[0].token_ids), "other_gpu_procs": busy}
        print(rec, flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
