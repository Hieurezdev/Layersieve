"""Per-layer probe: how well does the importance score at layer l (attention from the last n_obs prompt tokens, as used
by LayerSieve) rank the evidence of kv / vt prompts (data/build_hard.py)? Dense forward (no pruning), vLLM 0.6.3.
For every prompt and layer, over the prunable tokens (not sink, not observation window):
  tok_pct    mean percentile of the evidence tokens' avg-pooled (13) importance (1 = top)
  unit_pct   mean percentile of the evidence units' score (max token importance in the unit) among all units
  tok_rec    fraction of evidence tokens in the top `keep` (default 25%) by pooled importance
  unit_rec   fraction of evidence tokens whose unit is in the top units covering `keep` of the tokens
Usage: python scripts/probe_layers.py --model Qwen/Qwen2.5-14B-Instruct --ids results/scaling/ids/14b.txt \
           --out results/scaling/probe/14b.jsonl
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", default=os.path.join(ROOT, "data/hard.jsonl"))
ap.add_argument("--ids", default="", help="file with one id per line: run only these (kv / vt only are probed)")
ap.add_argument("--keep", type=float, default=0.25)
ap.add_argument("--gpu-util", type=float, default=0.85)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from inmodel_prune import units  # noqa: E402
from inmodel_prune.vllm_patch import enable_inmodel_prune, last_probe  # noqa: E402

N_SINK = N_OBS = 128
L = AutoConfig.from_pretrained(a.model).num_hidden_layers
enable_inmodel_prune(start_layer=L, drop_ratio=0.0, probe=True)
boundary = units.boundary_token_ids(AutoTokenizer.from_pretrained(a.model))

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(a.model, enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, max_num_seqs=1,
          max_model_len=32768, gpu_memory_utilization=a.gpu_util)
one = SamplingParams(temperature=0.0, max_tokens=1)


def pct(x):
    """Percentile rank in [0, 1] of every element (1 = largest)."""
    r = torch.empty_like(x)
    r[torch.argsort(x)] = torch.arange(len(x), dtype=x.dtype) / max(len(x) - 1, 1)
    return r


done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
only = {l.strip() for l in open(a.ids)} if a.ids else None
for line in open(a.prompts):
    r = json.loads(line)
    if r["task"] not in ("kv", "vt") or r["id"] in done or (only is not None and r["id"] not in only):
        continue
    ids = torch.tensor(r["prompt_ids"])
    T = len(ids)
    llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"])], one, use_tqdm=False)
    pr = torch.zeros(T, dtype=torch.bool)
    pr[N_SINK:T - N_OBS] = True
    ev = torch.zeros(T, dtype=torch.bool)
    for s, e in r["evidence"]:
        ev[s:e] = True
    ev &= pr
    unit = units.unit_ids(ids, boundary)
    uid = unit[pr]
    _, uinv = torch.unique(uid, return_inverse=True)
    U = int(uinv.max()) + 1
    usize = torch.bincount(uinv, minlength=U)
    n_keep = int(a.keep * int(pr.sum()))
    ev_p = ev[pr]
    ev_units = torch.unique(uinv[ev_p])
    rec = {"id": r["id"], "task": r["task"], "len": T, "tok_pct": [], "unit_pct": [], "tok_rec": [], "unit_rec": []}
    for imp in last_probe():
        imp = imp.float()
        pooled = F.avg_pool1d(imp[None, None], 13, 1, 6, count_include_pad=False)[0, 0][pr]
        tp = pct(pooled)
        rec["tok_pct"].append(round(float(tp[ev_p].mean()), 4))
        rec["tok_rec"].append(round(float((tp[ev_p] >= 1 - n_keep / len(tp)).float().mean()), 4))
        us = torch.full((U,), float("-inf")).scatter_reduce(0, uinv, imp[pr], reduce="amax")
        up = pct(us)
        rec["unit_pct"].append(round(float(up[ev_units].mean()), 4))
        order = torch.argsort(us, descending=True)
        top = torch.zeros(U, dtype=torch.bool)
        top[order[torch.cumsum(usize[order], 0) - usize[order] < n_keep]] = True
        rec["unit_rec"].append(round(float(top[uinv[ev_p]].float().mean()), 4))
    print(r["id"], "tok_pct", rec["tok_pct"][::4], flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
