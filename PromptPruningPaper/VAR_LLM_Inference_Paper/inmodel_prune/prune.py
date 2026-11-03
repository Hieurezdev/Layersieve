"""In-model progressive prompt pruning (no draft model).

The target model itself scores token importance: layers [0, start_layer) run densely on the
full prompt; from `start_layer` on, before each layer we drop `drop_ratio` of the remaining
prunable tokens, ranked by the attention they receive from the last `n_obs` prompt tokens at
that layer (max over heads, mean over observation queries, smoothed with an average pool, as
in SpecPrefill). Dropped tokens leave the computation entirely (as queries and as keys, so
attention and MLP shrink); kept tokens keep their original RoPE positions. The attention-sink
prefix and the observation window are never dropped.

Decoding uses a per-layer KV cache: layers before `start_layer` hold every prompt token,
later layers hold only the tokens that survived up to that layer.

Supports HF decoder-only models exposing model.model.{embed_tokens, layers, norm, rotary_emb}
(Llama, Qwen2, Mistral) with SDPA attention, batch size 1.
"""
import inspect
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import DynamicCache


@dataclass
class InModelPruneConfig:
    start_layer: int = 17          # first layer that runs on a pruned token set
    drop_ratio: float = 0.12       # fraction of remaining prunable tokens dropped before each layer >= start_layer
    n_obs: int = 128               # observation window: last n_obs prompt tokens (always kept)
    n_sink: int = 128              # attention-sink prefix (always kept)
    pool_kernel: int = 13          # average-pool smoothing of the importance (0/1 = off)
    obs_chunk: int = 32            # observation queries per scoring chunk (memory)
    mlp_chunk: int = 16384         # token chunk for the MLP during long prefill (memory only, exact)


@dataclass
class PruneStats:
    ttft: float = 0.0
    prompt_len: int = 0
    kept_per_layer: List[int] = field(default_factory=list)


def patch_mlp_chunked(model, chunk: int):
    """Run each MLP in token chunks during long prefill. MLPs are per-token, so this is exact."""
    if not chunk:
        return model

    def fwd(self, x):
        if x.shape[1] <= chunk:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return torch.cat([self.down_proj(self.act_fn(self.gate_proj(xi)) * self.up_proj(xi))
                          for xi in x.split(chunk, dim=1)], dim=1)

    for layer in model.model.layers:
        layer.mlp.forward = fwd.__get__(layer.mlp)
    return model


def _rope(x, cos, sin):
    # x [1, H, T, D]; cos/sin [1, T, D]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


@torch.no_grad()
def token_importance(layer, h, cos, sin, n_obs: int, obs_chunk: int = 32):
    """Importance of every current token at this layer: attention it receives from the last
    n_obs tokens (causal softmax over the current tokens), max over heads, summed over queries."""
    attn = layer.self_attn
    T = h.shape[1]
    D = attn.head_dim
    x = layer.input_layernorm(h)
    k = attn.k_proj(x).view(1, T, -1, D).transpose(1, 2)                      # [1, Hk, T, D]
    k = _rope(k, cos, sin)[0].float()                                          # [Hk, T, D]
    q = attn.q_proj(x[:, -n_obs:]).view(1, n_obs, -1, D).transpose(1, 2)        # [1, Hq, n_obs, D]
    q = _rope(q, cos[:, -n_obs:], sin[:, -n_obs:])[0].float()                  # [Hq, n_obs, D]
    Hq, Hk = q.shape[0], k.shape[0]
    G = Hq // Hk
    key = torch.arange(T, device=h.device)
    imp = torch.zeros(T, device=h.device)
    for r0 in range(0, n_obs, obs_chunk):
        r1 = min(r0 + obs_chunk, n_obs)
        qq = q[:, r0:r1].reshape(Hk, G * (r1 - r0), D)                         # group queries by kv head
        s = torch.bmm(qq, k.transpose(1, 2)) * attn.scaling                    # [Hk, G*R, T]
        s = s.view(Hq, r1 - r0, T)
        rows = T - n_obs + torch.arange(r0, r1, device=h.device)
        s.masked_fill_(key[None, None] > rows[None, :, None], float("-inf"))
        imp += torch.softmax(s, -1).amax(0).sum(0)
        del s
    return imp


@torch.no_grad()
def select_tokens(layer, h, cos, sin, pos, cfg: InModelPruneConfig):
    """Indices (into the current token set, ascending) of the tokens kept for this layer."""
    T = h.shape[1]
    imp = token_importance(layer, h, cos, sin, cfg.n_obs, cfg.obs_chunk)
    if cfg.pool_kernel and cfg.pool_kernel > 1:
        imp = F.avg_pool1d(imp[None, None], cfg.pool_kernel, 1, cfg.pool_kernel // 2, count_include_pad=False)[0, 0]
    idx = torch.arange(T, device=h.device)
    prunable = (pos >= cfg.n_sink) & (idx < T - cfg.n_obs)
    n_drop = int(int(prunable.sum()) * cfg.drop_ratio)
    keep = torch.ones(T, dtype=torch.bool, device=h.device)
    if n_drop > 0:
        keep[torch.topk(imp.masked_fill(~prunable, float("inf")), n_drop, largest=False).indices] = False
    return idx[keep]


def _layer_call(layer):
    """Call a decoder layer across transformers versions (past_key_value vs past_key_values, tuple vs tensor)."""
    name = "past_key_values" if "past_key_values" in inspect.signature(layer.forward).parameters else "past_key_value"

    def call(h, cache, cos, sin):
        out = layer(h, attention_mask=None, use_cache=True, position_embeddings=(cos, sin), **{name: cache})
        return out[0] if isinstance(out, tuple) else out
    return call


@torch.no_grad()
def prefill_and_generate(model, input_ids: torch.Tensor, cfg: Optional[InModelPruneConfig] = None,
                         max_new_tokens: int = 128, eos_token_ids=(), prune: bool = True):
    """Greedy generation with in-model progressive pruning during prefill.
    prune=False runs the same code densely (baseline). Returns (generated token ids, PruneStats)."""
    cfg = cfg or InModelPruneConfig()
    assert input_ids.shape[0] == 1, "batch size 1 only"
    mm = model.model
    dev = input_ids.device
    T = input_ids.shape[1]
    calls = [_layer_call(l) for l in mm.layers]
    stats = PruneStats(prompt_len=T)
    eos = set(int(e) for e in eos_token_ids)

    torch.cuda.synchronize(dev)
    t0 = time.time()
    h = mm.embed_tokens(input_ids)
    pos = torch.arange(T, device=dev)
    cos, sin = mm.rotary_emb(h, pos[None])
    cache = DynamicCache()
    can_prune = prune and T > cfg.n_sink + cfg.n_obs
    for li, layer in enumerate(mm.layers):
        if can_prune and li >= cfg.start_layer:
            keep = select_tokens(layer, h, cos, sin, pos, cfg)
            h, cos, sin, pos = h[:, keep], cos[:, keep], sin[:, keep], pos[keep]
        stats.kept_per_layer.append(h.shape[1])
        h = calls[li](h, cache, cos, sin)
    tok = model.lm_head(mm.norm(h[:, -1:])).argmax(-1)                          # [1, 1]
    torch.cuda.synchronize(dev)
    stats.ttft = time.time() - t0

    out = [int(tok)]
    for step in range(1, max_new_tokens):
        if out[-1] in eos:
            break
        p = torch.tensor([[T + step - 1]], device=dev)                          # true position of the new token
        x = mm.embed_tokens(tok)
        c1, s1 = mm.rotary_emb(x, p)
        for call in calls:
            x = call(x, cache, c1, s1)
        tok = model.lm_head(mm.norm(x)).argmax(-1)
        out.append(int(tok))
    return out, stats


def load_model(name: str, cfg: Optional[InModelPruneConfig] = None, dtype=torch.bfloat16):
    """Load an HF causal LM ready for in-model pruning (SDPA, sliding window off, chunked MLP)."""
    from transformers import AutoModelForCausalLM
    cfg = cfg or InModelPruneConfig()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()
    if hasattr(model.config, "use_sliding_window"):
        model.config.use_sliding_window = False
        model.config.sliding_window = None
    for l in model.model.layers:
        if hasattr(l.self_attn, "sliding_window"):
            l.self_attn.sliding_window = None
    return patch_mlp_chunked(model, cfg.mlp_chunk)
