"""Grouped positions with a neighbour window (SelfExtend, Jin et al., 2024) on Qwen2.5, dense, in a plain HF forward.

A query at position i attends a key at position j with
    the exact distance i - j                               if i - j < w   (neighbour attention)
    floor(i/g) + w - floor(w/g) - floor(j/g)               otherwise      (grouped attention)
so g neighbouring tokens far from the query share one position and the largest distance shrinks by about g. Keys are
cached unrotated and rotated for both parts at every step; the two flash-attention parts are merged by their
log-sum-exp. g = 1 with w > 0 runs the same split and must equal plain attention (w = 0).

    python scripts/selfextend_eval.py --group 8 --window 2048 --prompts data/infbench.jsonl \
        --tasks longdialogue_qa_eng,kv_retrieval --out results/selfext/ib_g8_w2048.jsonl
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM
from vllm.vllm_flash_attn import flash_attn_varlen_func

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("SE_MODEL", "Qwen/Qwen2.5-7B-Instruct-1M"),
                help="HF id; env SE_MODEL sets it for the scripts that import this module (Qwen2.5 or Qwen3)")
ap.add_argument("--group", type=int, default=1)
ap.add_argument("--window", type=int, default=0, help="neighbour window w; 0 = plain attention")
ap.add_argument("--mode", default="floor", choices=["floor", "linear", "freq", "fastoff"],
                help="floor: SelfExtend groups; linear: far distance d -> w + (d - w)/g, no shared positions (STRING-like); "
                "freq: as linear but only in the rotary dims whose wavelength exceeds w, the faster dims keep d; "
                "fastoff: true distances, but for keys farther than w the rotary dims faster than w count --lam times")
ap.add_argument("--lam", type=float, default=0.0, help="fastoff: weight of the fast dims for far keys")
ap.add_argument("--from-layer", type=int, default=0, help="remap only in layers >= this one; plain attention below")
ap.add_argument("--to-layer", type=int, default=1000, help="remap only in layers < this one; plain attention above")
ap.add_argument("--tail", type=int, default=0, help="remap only for the last TAIL prompt tokens and the generated ones "
                "(the question); the context itself is encoded with plain attention. 0 = every query")
ap.add_argument("--qscale", type=float, default=1.0, help="multiply the attention logits of the remapped queries "
                "(the --tail ones, in layers --from-layer..--to-layer) by this: a sharper softmax over the long context")
ap.add_argument("--chunk", type=int, default=8192, help="prefill chunk")
ap.add_argument("--prompts", required=True)
ap.add_argument("--tasks", default="")
ap.add_argument("--lengths", default="", help="RULER only: comma list of lengths (task suffix @<len>)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
cfg = model.config
H, Hk = cfg.num_attention_heads, cfg.num_key_value_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H                # Qwen3: head_dim != hidden / H
INV = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, device="cuda").float() / D))
STOP = {151645, 151643}
G, W = a.group, a.window
LOW = 2 * torch.pi / INV > max(W, 1)                                         # rotary dims slower than the window
MODE, FROM, TO = a.mode, a.from_layer, a.to_layer
QSTART = 0                                                                   # first query position that remaps (--tail)


def rope(x, pos, pos_low=None):
    """Neox rotation of x [n, h, D] at positions pos [n]; with pos_low, the slow dims (LOW) rotate at pos_low instead."""
    ang = pos.double()[:, None] * INV.double()[None]
    if pos_low is not None:
        ang = torch.where(LOW[None], pos_low.double()[:, None] * INV.double()[None], ang)
    ang = ang.float()
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    x1, x2 = x[..., :D // 2].float(), x[..., D // 2:].float()
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).to(x.dtype)


def fa(q, k, v, window=(-1, -1)):
    """Causal flash attention, bottom-right aligned (query r sees keys <= r + len(k) - len(q)); returns out, lse [h, nq]."""
    cq = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda")
    ck = torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cuda")
    out, lse, _ = flash_attn_varlen_func(q, k, v, cq, ck, q.shape[0], k.shape[0], softmax_scale=D ** -0.5, causal=True,
                                         window_size=window, return_attn_probs=True)
    return out, lse


def attend(q, k, v, qpos, remap=True):
    """q [nq, H, D] at contiguous positions qpos; k, v [n, Hk, D] for positions 0..n-1 (unrotated keys)."""
    c = QSTART - int(qpos[0])
    if remap and c > 0:                                                      # queries before the tail: plain attention
        if c >= q.shape[0]:
            remap = False
        else:
            o1 = attend(q[:c], k[:int(qpos[c - 1]) + 1], v[:int(qpos[c - 1]) + 1], qpos[:c], False)
            return torch.cat((o1, attend(q[c:], k, v, qpos[c:], True)))
    if remap and a.qscale != 1.0:
        q = q * a.qscale
    kpos = torch.arange(k.shape[0], device="cuda")
    qn, kn = rope(q, qpos), rope(k, kpos)
    if W == 0 or not remap:
        return fa(qn, kn, v)[0]
    on, ln = fa(qn, kn, v, (W - 1, 0))                                       # distance < w, exact
    i0 = max(0, W - int(qpos[0]))                                            # first query that has keys at distance >= w
    if i0 >= q.shape[0]:
        return on
    qp = qpos[i0:]
    nk = int(qp[-1]) - W + 1                                                 # last query sees keys 0 .. qpos - w
    if MODE == "fastoff":                                                    # damp the fast dims' far-key term
        damp = torch.where(LOW, 1.0, a.lam).repeat(2).to(q.dtype)
        og, lg = fa(qn[i0:] * damp, kn[:nk], v[:nk])
    else:
        if MODE == "floor":
            qg, kg = qp // G + W - W // G, kpos[:nk] // G
        else:                                                                # (i - j)/g + w - w/g = w + (d - w)/g
            qg, kg = qp.double() / G + W - W / G, kpos[:nk].double() / G
        if MODE == "freq":                                                   # fast dims keep the true distance
            og, lg = fa(rope(q[i0:], qp, qg), rope(k[:nk], kpos[:nk], kg), v[:nk])
        else:
            og, lg = fa(rope(q[i0:], qg), rope(k[:nk], kg), v[:nk])
    l1, l2 = ln[:, i0:].T[..., None], lg.T[..., None]                        # [nq, H, 1]
    m = torch.maximum(l1, l2)
    e1, e2 = (l1 - m).exp(), (l2 - m).exp()
    out = on.clone()
    out[i0:] = ((on[i0:].float() * e1 + og.float() * e2) / (e1 + e2)).to(on.dtype)
    return out


def qkv(at, h):
    """Unrotated q [n, H, D], k, v [n, Hk, D] of attention module at; Qwen3's per-head q_norm / k_norm applied."""
    q, k = at.q_proj(h).view(-1, H, D), at.k_proj(h).view(-1, Hk, D)
    if hasattr(at, "q_norm"):
        q, k = at.q_norm(q), at.k_norm(k)
    return q, k, at.v_proj(h).view(-1, Hk, D)


@torch.no_grad()
def forward(ids, start, K, V):
    """Run tokens ids at positions start.. through the model, appending to the caches; returns last-token logits."""
    pos = torch.arange(start, start + len(ids), device="cuda")
    x = model.model.embed_tokens(torch.tensor(ids, device="cuda"))
    end = start + len(ids)
    for li, L in enumerate(model.model.layers):
        at = L.self_attn
        h = L.input_layernorm(x)
        q, K[li][start:end], V[li][start:end] = qkv(at, h)
        o = attend(q, K[li][:end], V[li][:end], pos, FROM <= li < TO)
        x = x + at.o_proj(o.reshape(-1, H * D))
        x = x + L.mlp(L.post_attention_layernorm(x))
    return model.lm_head(model.model.norm(x[-1:]))[0]


def generate(ids, max_new):
    global QSTART
    QSTART = max(len(ids) - a.tail, 0) if a.tail else 0
    n = len(ids) + max_new
    K = [torch.empty(n, Hk, D, dtype=torch.bfloat16, device="cuda") for _ in model.model.layers]
    V = [torch.empty_like(k) for k in K]
    for s in range(0, len(ids), a.chunk):
        logits = forward(ids[s:s + a.chunk], s, K, V)
    out, p = [], len(ids)
    for _ in range(max_new):
        t = int(logits.argmax())
        if t in STOP:
            break
        out.append(t)
        logits = forward([t], p, K, V)
        p += 1
    return out


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    tasks = set(a.tasks.split(",")) if a.tasks else None
    lens = set(a.lengths.split(",")) if a.lengths else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = 0
    for line in open(a.prompts):
        r = json.loads(line)
        if (tasks and r["task"].split("@")[0] not in tasks) or (lens and r["task"].split("@")[-1] not in lens):
            continue
        n += 1
        if a.limit and n > a.limit:
            break
        if r["id"] in done:
            continue
        pred = tok.decode(generate(r["prompt_ids"], r["max_new_tokens"]), skip_special_tokens=True)
        rec = {"id": r["id"], "task": r["task"], "pred": pred.strip(), "len": len(r["prompt_ids"]), "group": G, "window": W, "mode": MODE, "from_layer": FROM, "to_layer": TO, "tail": a.tail, "qscale": a.qscale, "lam": a.lam}
        print(rec["id"], repr(rec["pred"][:80]), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
