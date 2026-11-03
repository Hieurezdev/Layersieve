"""Timing of the multi-turn modes (inmodel_prune/multiturn.py) in vLLM 0.6.3, measured in parts:
  --mode layers   patched vLLM, prefix caching off: TTFT, TPOT and per-layer prefill times (CUDA events, scoring
                  included) of dense and LayerSieve. Mode A's follow-up prefill is LayerSieve's layers >= start_layer
                  plus the question through the dense layers; mode B's background completion is dense layers >= start_layer.
  --mode prefix   stock vLLM with prefix caching: TTFT and TPOT of a follow-up question over a cached document
                  (the last --suffix tokens of the prompt replaced), i.e. dense with a prefix cache.
Prompts: the first --n prompts of data/speed.jsonl per length. Every config runs after a warmup, interleaved per prompt.
    python scripts/bench_multiturn.py --mode layers --lengths 131072,262144 --out results/multiturn/layers.jsonl
    python scripts/bench_multiturn.py --mode prefix --lengths 131072,262144 --out results/multiturn/prefix.jsonl
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--mode", required=True, choices=["layers", "prefix"])
ap.add_argument("--prompts", default="data/speed.jsonl")
ap.add_argument("--lengths", default="131072")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--decode", type=int, default=256)
ap.add_argument("--suffix", type=int, default=100, help="prefix mode: question tokens replaced for the follow-up")
ap.add_argument("--turns", type=int, default=3, help="prefix mode: follow-up questions per document")
ap.add_argument("--gpu-util", type=float, default=0.3)
ap.add_argument("--out", required=True)
a = ap.parse_args()

allr = [json.loads(l) for l in open(a.prompts)]
rows = [r for n in a.lengths.split(",") for r in [x for x in allr if x["id"].startswith(f"speed_{n}_")][:a.n]]
max_len = max(len(r["prompt_ids"]) for r in rows) + a.decode + 16
LS = dict(start_layer=17, drop_ratio=0.4, min_keep=0.25, decode_ref=22, unit=True)
if a.mode == "layers":
    from inmodel_prune.vllm_patch import enable_inmodel_prune, last_kept, last_layer_ms
    enable_inmodel_prune(start_layer=28, drop_ratio=0.0, mlp_chunk=16384, tokenizer=a.model)
else:                                 # stock forward; only the MLP runs in 16k-token chunks (exact), as in the layers mode
    from inmodel_prune import vllm_patch
    from vllm.model_executor.models import qwen2
    vllm_patch._CFG["mlp_chunk"] = 16384
    vllm_patch._ORIG.setdefault("mlp", {})[qwen2.Qwen2MLP] = qwen2.Qwen2MLP.forward
    qwen2.Qwen2MLP.forward = vllm_patch._mlp_chunked

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(a.model, enforce_eager=False, max_seq_len_to_capture=max_len, enable_chunked_prefill=False,
          enable_prefix_caching=a.mode == "prefix", max_num_seqs=1, max_model_len=max_len,
          gpu_memory_utilization=a.gpu_util, dtype="bfloat16")
sp = SamplingParams(temperature=0.0, max_tokens=a.decode, ignore_eos=True)


def run(ids):
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
    m = out.metrics
    return m.first_token_time - m.first_scheduled_time, (m.finished_time - m.first_token_time) / (a.decode - 1)


def write(rec):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.split()
    rec["other_gpu_procs"] = sorted(int(x) for x in out if x.isdigit() and int(x) != os.getpid())   # not clean if any
    print(rec, flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
run(rows[0]["prompt_ids"][:8192])                                                   # warmup
for r in rows:
    ids = r["prompt_ids"]
    if a.mode == "layers":
        for name, kw in (("dense", dict(start_layer=28, drop_ratio=0.0)), ("layersieve", LS)):
            enable_inmodel_prune(mlp_chunk=16384, tokenizer=a.model, layer_times=True, **kw)
            ttft, tpot = run(ids)
            write({"mode": "layers", "id": r["id"], "len": len(ids), "cfg": name, "ttft": ttft, "tpot": tpot,
                   "kept": last_kept(), "layer_ms": last_layer_ms()})
    else:
        doc = ids[:-a.suffix]
        for t in range(a.turns + 1):                    # turn 0 fills the cache; then questions differing in the suffix
            q = [(x + 7919 * (t + 1)) % 150000 for x in ids[-a.suffix:]] if t else ids[-a.suffix:]
            ttft, tpot = run(doc + q)
            write({"mode": "prefix", "id": r["id"], "len": len(ids), "turn": t, "ttft": ttft, "tpot": tpot})
