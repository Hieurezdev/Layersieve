"""How precisely can the model's own attention locate the evidence? One prefill per prompt, then the block scores
(64-token blocks of the context) under many scoring variants are saved for an offline comparison with the known
evidence positions (scripts/analysis/selector_eval.py). Only the prompt is read here; answers are never used.

Context = all but the last QREG = 128 prompt tokens (the question region). Per query row of that region and per layer in
--layers, the softmax attention over the context is aggregated over heads two ways (mean as in recall_eval.py, max as in
LayerSieve) and summed into blocks. Windows: the last w rows for w in --windows. Also: the 16 retrieval heads
(results/repos/heads_7b1m.json, score > 0.3) in their own layers, and a neutral question (the same chat frame with a
generic request, NEUTRAL) run at the same positions, for contrastive scoring.

    python scripts/selector_probe.py --prompts data/infbench_kv86.jsonl --limit 20 --out results/selfext/selprobe/kv20
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--tasks", default="")
ap.add_argument("--lengths", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--layers", default="14,17,20,24")
ap.add_argument("--windows", default="1,8,16,32,128")
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--heads", default="results/repos/heads_7b1m.json")
ap.add_argument("--out", required=True, help="directory; one .npz per prompt")
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
QREG = 128
LAYERS = [int(x) for x in a.layers.split(",")]
WINS = [int(x) for x in a.windows.split(",")]
L = len(se.model.model.layers)
NEUTRAL = tok.encode("<|im_end|>\n<|im_start|>user\nPlease answer the question about the text above.<|im_end|>\n"
                     "<|im_start|>assistant\n")
RH = torch.tensor(json.load(open(a.heads))["query_head"]) > 0.3                 # [L, H] retrieval heads
RH_LAYERS = sorted({int(l) for l in torch.nonzero(RH)[:, 0]})


def row_scores(q, k, qpos, n, heads=None):
    """Per query row: attention over context keys 0..n-1, aggregated over heads (mean, max) -> two [T, n] tensors.
    heads: optional bool [H] (mean over those heads only; max unused)."""
    rep = se.H // se.Hk
    qr = se.rope(q, qpos).float()
    kr = se.rope(k[:n], torch.arange(n, device="cuda")).float()
    T = q.shape[0]
    mean = torch.zeros(T, n, device="cuda")
    mx = torch.zeros(T, n, device="cuda")
    cnt = 0
    for g in range(se.Hk):
        hs = list(range(g * rep, (g + 1) * rep))
        if heads is not None:
            hs = [h for h in hs if heads[h]]
            if not hs:
                continue
        p = torch.softmax(torch.einsum("trd,nd->rtn", qr[:, hs], kr[:, g]) * se.D ** -0.5, -1)   # [h, T, n]
        mean += p.sum(0)
        mx = torch.maximum(mx, p.amax(0))
        cnt += len(hs)
        del p
    return mean / max(cnt, 1), mx


def blocks(x, n):
    nb = n // a.block
    return x[..., :nb * a.block].reshape(*x.shape[:-1], nb, a.block).sum(-1)


@torch.no_grad()
def probe(ids):
    T = len(ids)
    q0 = T - QREG
    K = [torch.empty(T + len(NEUTRAL), se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(k) for k in K]
    for s in range(0, q0, 8192):
        se.forward(ids[s:min(s + 8192, q0)], s, K, V)
    got, orig = [], se.attend

    def grab(q, k, v, qpos, remap=True):
        got.append(q.clone())
        return orig(q, k, v, qpos, remap)
    se.attend = grab
    se.forward(ids[q0:], q0, K, V)                                              # the real question region
    qreal = got[:]
    got.clear()
    se.forward(NEUTRAL, q0, K, V)                                               # neutral request, same positions
    qneu = got[:]
    se.attend = orig
    out = {}
    pos_r = torch.arange(q0, T, device="cuda")
    pos_n = torch.arange(q0, q0 + len(NEUTRAL), device="cuda")
    for l in LAYERS:
        mean, mx = row_scores(qreal[l], K[l], pos_r, q0)
        bm, bx = blocks(mean, q0), blocks(mx, q0)                               # [T, nb]
        for w in WINS:
            out[f"L{l}_mean_w{w}"] = bm[-w:].sum(0)
            out[f"L{l}_max_w{w}"] = bx[-w:].sum(0)
        nmean, _ = row_scores(qneu[l], K[l], pos_n, q0)
        out[f"L{l}_neutral"] = blocks(nmean, q0).sum(0)
    acc = None
    for l in RH_LAYERS:                                                         # retrieval heads, their own layers
        m, _ = row_scores(qreal[l], K[l], pos_r, q0, heads=RH[l])
        b = blocks(m, q0)
        acc = b if acc is None else acc + b
    for w in WINS:
        out[f"retr_w{w}"] = acc[-w:].sum(0)
    return {k: v.float().cpu().numpy() for k, v in out.items()}, q0


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    tasks = set(a.tasks.split(",")) if a.tasks else None
    lens = set(a.lengths.split(",")) if a.lengths else None
    n = 0
    for line in open(a.prompts):
        r = json.loads(line)
        if (tasks and r["task"].split("@")[0] not in tasks) or (lens and r["task"].split("@")[-1] not in lens):
            continue
        n += 1
        if a.limit and n > a.limit:
            break
        f = os.path.join(a.out, r["id"] + ".npz")
        if os.path.exists(f):
            continue
        sc, q0 = probe(r["prompt_ids"])
        np.savez_compressed(f, q0=q0, block=a.block, **sc)
        print(r["id"], q0, len(sc), flush=True)
