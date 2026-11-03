"""GemFilter with LayerSieve's selection: run the prompt dense through layers 0..L-1 only, score every token at layer L
exactly as LayerSieve does (inmodel_prune/vllm_patch.py `_importance`: softmax attention received from the last
n_obs = 128 prompt tokens, max over heads, summed over those queries), drop the lowest-scoring whole units
(inmodel_prune/units.py, unit_max 128) until --keep of the prunable tokens is left (the first n_sink = 128 and the
last n_obs tokens are never dropped), then gather the kept tokens in prompt order, give them fresh positions 0..k-1
and run all layers on them (GemFilter's second pass) before greedy decoding. Dense Qwen2.5 forward from
scripts/selfextend_eval.py.

    python scripts/gem_ls.py --layer 17 --keep 0.25 --prompts data/gem_subset.jsonl --out results/selfext/gemls_l17_k25.jsonl
"""
import argparse
import json
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--layer", type=int, default=17, help="selection layer (LayerSieve start_layer)")
ap.add_argument("--keep", type=float, default=0.25, help="fraction of the prunable tokens kept (LayerSieve min_keep)")
ap.add_argument("--n-obs", type=int, default=128)
ap.add_argument("--n-sink", type=int, default=128)
ap.add_argument("--unit-max", type=int, default=128)
ap.add_argument("--nbr", type=int, default=0, help="keep every picked unit together with its NBR neighbouring units on "
                "each side (same token budget): units are taken in score order with their neighbours until --keep of the "
                "prunable tokens is reached; 0 = LayerSieve's plain unit selection")
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null"]
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import selfextend_eval as se  # noqa: E402
from inmodel_prune import units  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
BOUNDARY = units.boundary_token_ids(tok).cuda()
LAYERS = se.model.model.layers


@torch.no_grad()
def hidden_at(ids, L):
    """Input hidden states of layer L for every prompt token (layers 0..L-1 only, chunked prefill)."""
    T = len(ids)
    K = [torch.empty(T, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(k) for k in K]
    out = torch.empty(T, se.model.config.hidden_size, dtype=torch.bfloat16, device="cuda")
    for s in range(0, T, a.chunk):
        e = min(s + a.chunk, T)
        pos = torch.arange(s, e, device="cuda")
        x = se.model.model.embed_tokens(torch.tensor(ids[s:e], device="cuda"))
        for li in range(L):
            Ly = LAYERS[li]
            at = Ly.self_attn
            h = Ly.input_layernorm(x)
            K[li][s:e] = at.k_proj(h).view(-1, se.Hk, se.D)
            V[li][s:e] = at.v_proj(h).view(-1, se.Hk, se.D)
            o = se.attend(at.q_proj(h).view(-1, se.H, se.D), K[li][:e], V[li][:e], pos, False)
            x = x + at.o_proj(o.reshape(-1, se.H * se.D))
            x = x + Ly.mlp(Ly.post_attention_layernorm(x))
        out[s:e] = x
    return out


@torch.no_grad()
def importance(x, L, n_obs, obs_chunk=32):
    """LayerSieve `_importance` at layer L: attention every token receives from the last n_obs tokens."""
    at = LAYERS[L].self_attn
    T = x.shape[0]
    k = torch.empty(T, se.Hk, se.D, dtype=torch.bfloat16, device="cuda")
    for s in range(0, T, a.chunk):
        k[s:s + a.chunk] = at.k_proj(LAYERS[L].input_layernorm(x[s:s + a.chunk])).view(-1, se.Hk, se.D)
    q = at.q_proj(LAYERS[L].input_layernorm(x[-n_obs:])).view(-1, se.H, se.D)
    k = se.rope(k, torch.arange(T, device="cuda")).transpose(0, 1).float()                 # [Hk, T, D]
    q = se.rope(q, torch.arange(T - n_obs, T, device="cuda")).transpose(0, 1).float()     # [Hq, n_obs, D]
    G = se.H // se.Hk
    idx = torch.arange(T, device="cuda")
    imp = torch.zeros(T, device="cuda")
    for r0 in range(0, n_obs, obs_chunk):
        r1 = min(r0 + obs_chunk, n_obs)
        sc = torch.bmm(q[:, r0:r1].reshape(se.Hk, G * (r1 - r0), se.D), k.transpose(1, 2)) * se.D ** -0.5
        sc = sc.view(se.H, r1 - r0, T)
        rows = T - n_obs + torch.arange(r0, r1, device="cuda")
        sc.masked_fill_(idx[None, None] > rows[None, :, None], float("-inf"))
        imp += torch.softmax(sc, -1).amax(0).sum(0)
        del sc
    return imp


@torch.no_grad()
def generate(r):
    ids, max_new = r["prompt_ids"], r["max_new_tokens"]
    T = len(ids)
    x = hidden_at(ids, a.layer)
    imp = importance(x, a.layer, a.n_obs)
    del x
    t_ids = torch.tensor(ids, device="cuda")
    unit = units.unit_ids(t_ids, BOUNDARY, a.unit_max)
    idx = torch.arange(T, device="cuda")
    prunable = (idx >= a.n_sink) & (idx < T - a.n_obs)
    n_drop = int(int(prunable.sum()) * (1 - a.keep))
    keep = torch.ones(T, dtype=torch.bool, device="cuda")
    if n_drop > 0 and a.nbr:                                                     # units with their neighbours
        uid, inv = torch.unique(unit, return_inverse=True)
        U = len(uid)
        score = torch.full((U,), float("-inf"), device="cuda").scatter_reduce(0, inv, imp, reduce="amax")
        fixed = torch.zeros(U, dtype=torch.bool, device="cuda")
        fixed[inv[~prunable]] = True                                            # units holding sink / question tokens
        size = torch.bincount(inv, minlength=U).tolist()
        take = fixed.clone()
        budget, used = int(prunable.sum()) - n_drop, 0
        for u in torch.argsort(score.masked_fill(fixed, float("-inf")), descending=True).tolist():
            if used >= budget:
                break
            for v in range(max(u - a.nbr, 0), min(u + a.nbr, U - 1) + 1):
                if not take[v]:
                    take[v] = True
                    used += size[v]
        keep = take[inv]
    elif n_drop > 0:
        keep[units.drop_units(imp, unit, prunable, n_drop)] = False
    new = t_ids[keep].tolist()                                                  # prompt order, fresh positions
    K = [torch.empty(len(new) + max_new, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in LAYERS]
    V = [torch.empty_like(k) for k in K]
    for s in range(0, len(new), a.chunk):
        logits = se.forward(new[s:s + a.chunk], s, K, V)
    out, p = [], len(new)
    for _ in range(max_new):
        t = int(logits.argmax())
        if t in se.STOP:
            break
        out.append(t)
        logits = se.forward([t], p, K, V)
        p += 1
    return out, len(new), tok.decode(new)


if __name__ == "__main__":
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for line in open(a.prompts):
        r = json.loads(line)
        if r["id"] in done:
            continue
        out, kept, text = generate(r)
        pred = tok.decode(out, skip_special_tokens=True).strip()
        ans = r.get("answers") or r.get("outputs") or []
        rec = {"id": r["id"], "task": r["task"], "pred": pred, "len": len(r["prompt_ids"]), "kept": kept,
               "layer": a.layer, "keep": a.keep, "nbr": a.nbr, "answer_in_kept": [str(x) in text for x in ans]}
        print(rec["id"], kept, repr(pred[:80]), rec["answer_in_kept"], flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
