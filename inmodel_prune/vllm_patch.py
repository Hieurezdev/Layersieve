"""In-model progressive prompt pruning for vLLM 0.6.3 (FlashAttention backend), single request at a time.

Same algorithm as inmodel_prune/prune.py (HF reference): layers [0, start_layer) run on the full
prompt; before each layer >= start_layer, drop `drop_ratio` of the remaining prunable tokens, ranked by the
attention they receive from the last `n_obs` prompt tokens (max over heads, sum over queries, avg-pool smoothing).

KV cache layout: every layer owns its own cache tensor but all layers share one block table. A layer that keeps
n tokens writes their KV compactly into logical slots 0..n-1 of that block table (the blocks are already allocated
for the full prompt), and during decode it reads n + (#generated) slots and appends the new token at slot n + g - 1.
Layers before `start_layer` behave exactly as in stock vLLM. Memory is still allocated for the full prompt.

Requirements: enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False, one request per
step (max_num_seqs=1), Qwen2 models. Call enable_inmodel_prune(...) before creating the LLM.
"""
import dataclasses
import math
from typing import List, Optional

import torch
import torch.nn.functional as F

_CFG = {}
_ORIG = {}
_STATE = {"T": None, "kept": None, "idx": None}  # prompt length, tokens kept per layer (index = layer id), positions kept at the last layer


def _importance(layer, hidden, residual, positions, n_obs, obs_chunk=32):
    """Attention received by every current token from the last n_obs tokens at this layer (max over heads, summed)."""
    x = hidden if residual is None else hidden + residual
    x = layer.input_layernorm(x)
    attn = layer.self_attn
    qkv, _ = attn.qkv_proj(x)
    q, k, _ = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q, k = attn.rotary_emb(positions, q, k)
    T, D = k.shape[0], attn.head_dim
    Hq, Hk = attn.num_heads, attn.num_kv_heads
    G = Hq // Hk
    k = k.view(T, Hk, D).transpose(0, 1).float()                                  # [Hk, T, D]
    q = q[-n_obs:].view(n_obs, Hq, D).transpose(0, 1).float()                     # [Hq, n_obs, D]
    idx = torch.arange(T, device=hidden.device)
    imp = torch.zeros(T, device=hidden.device)
    for r0 in range(0, n_obs, obs_chunk):
        r1 = min(r0 + obs_chunk, n_obs)
        s = torch.bmm(q[:, r0:r1].reshape(Hk, G * (r1 - r0), D), k.transpose(1, 2)) * attn.scaling
        s = s.view(Hq, r1 - r0, T)
        rows = T - n_obs + torch.arange(r0, r1, device=hidden.device)
        s.masked_fill_(idx[None, None] > rows[None, :, None], float("-inf"))
        imp += torch.softmax(s, -1).amax(0).sum(0)
        del s
    return imp


def _select(layer, hidden, residual, positions):
    c = _CFG
    T = hidden.shape[0]
    imp = _importance(layer, hidden, residual, positions, c["n_obs"])
    if c["pool_kernel"] > 1:
        imp = F.avg_pool1d(imp[None, None], c["pool_kernel"], 1, c["pool_kernel"] // 2, count_include_pad=False)[0, 0]
    idx = torch.arange(T, device=hidden.device)
    prunable = (positions >= c["n_sink"]) & (idx < T - c["n_obs"])
    n_drop = int(int(prunable.sum()) * c["drop_ratio"])
    keep = torch.ones(T, dtype=torch.bool, device=hidden.device)
    if n_drop > 0:
        keep[torch.topk(imp.masked_fill(~prunable, float("inf")), n_drop, largest=False).indices] = False
    return idx[keep]


def _slim_select(layer, hidden, residual, positions, keep_tokens):
    """SlimInfer (Long et al., AAAI 2026) prefill block selection, re-implemented from their public code:
    representative keys = mean key of each 8-token unit (positions // unit), query = last `window` query rows,
    unit score = q.k / sqrt(d) averaged over heads and window rows, block (64 tokens) score = max over its units,
    keep the top ceil(keep_tokens / 64) blocks; the first (sink) and last (recent) active blocks are always kept.
    Returns indices (into the current tokens) of the tokens kept for the NEXT layers."""
    c = _CFG
    x = hidden if residual is None else hidden + residual
    x = layer.input_layernorm(x)
    attn = layer.self_attn
    qkv, _ = attn.qkv_proj(x)
    q, k, _ = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q, k = attn.rotary_emb(positions, q, k)
    T, D = k.shape[0], attn.head_dim
    Hq, Hk = attn.num_heads, attn.num_kv_heads
    unit = positions // c["slim_unit"]
    uid, inv = torch.unique(unit, return_inverse=True)                                  # sorted unit ids
    kk = k.view(T, Hk, D).float()
    ksum = torch.zeros(len(uid), Hk, D, device=k.device).index_add_(0, inv, kk)
    cnt = torch.zeros(len(uid), device=k.device).index_add_(0, inv, torch.ones(T, device=k.device))
    krep = (ksum / cnt[:, None, None]).repeat_interleave(Hq // Hk, 1)                  # [U, Hq, D]
    qw = q[-c["slim_window"]:].view(-1, Hq, D).float()                                  # [w, Hq, D]
    us = torch.einsum("whd,uhd->u", qw, krep) / (D ** 0.5) / (qw.shape[0] * Hq)         # mean over rows and heads
    ublock = uid // (c["slim_block"] // c["slim_unit"])
    bid, binv = torch.unique(ublock, return_inverse=True)
    bs = torch.full((len(bid),), float("-inf"), device=k.device).scatter_reduce(0, binv, us, reduce="amax")
    n_keep = min(math.ceil(keep_tokens / c["slim_block"]), len(bid))
    bs[0] = bs[-1] = float("inf")                                                       # sink and recent blocks
    kept_blocks = bid[torch.topk(bs, n_keep).indices]
    tok_block = positions // c["slim_block"]
    return torch.nonzero(torch.isin(tok_block, kept_blocks)).squeeze(1)


def _prefill_meta(meta, n):
    """Metadata for a prefill over n (compacted) tokens: KV written to logical slots 0..n-1."""
    m = dataclasses.replace(
        meta, num_prefill_tokens=n, slot_mapping=meta.slot_mapping[:n],
        seq_lens=[n], seq_lens_tensor=meta.seq_lens_tensor.new_tensor([n]), max_query_len=n, max_prefill_seq_len=n,
        query_start_loc=meta.query_start_loc.new_tensor([0, n]), seq_start_loc=meta.seq_start_loc.new_tensor([0, n]))
    m._cached_prefill_metadata = None
    m._cached_decode_metadata = None
    return m


def _decode_meta(meta, n_kept, T, block_size):
    """Metadata for one decode token of a layer that kept n_kept prompt tokens."""
    seq_len = meta.seq_lens_tensor[0] - T + n_kept                                 # cache length incl. new token
    logical = seq_len - 1
    bt = meta.block_tables[0]
    slot = bt[logical // block_size].long() * block_size + logical % block_size
    m = dataclasses.replace(meta, slot_mapping=slot.view(1).to(meta.slot_mapping.dtype),
                            seq_lens_tensor=seq_len.view(1).to(meta.seq_lens_tensor.dtype),
                            max_decode_seq_len=int(seq_len))
    m._cached_prefill_metadata = None
    m._cached_decode_metadata = None
    return m


def _patched_forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None,
                     inputs_embeds=None):
    from vllm.forward_context import set_forward_context
    c = _CFG
    n_seq = attn_metadata.num_prefills + attn_metadata.num_decode_tokens
    if n_seq != 1 or (attn_metadata.num_prefills == 0 and _STATE["kept"] is None):
        # memory-profiling dummy batch, or anything that is not a single request: stock behaviour
        return _ORIG["forward"](self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds)
    hidden = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
    residual = None
    is_prefill = attn_metadata.num_prefills > 0
    if is_prefill:
        T = hidden.shape[0]
        _STATE["T"], _STATE["kept"] = T, []
        do_prune = T > c["n_sink"] + c["n_obs"]
        orig_idx = torch.arange(T, device=hidden.device)
        meta = attn_metadata
        slim = c["method"] == "slim"
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            keep = None
            if do_prune and not slim and i >= c["start_layer"]:                        # ours: select BEFORE layer i
                keep = _select(layer, hidden, residual, positions)
            elif do_prune and slim and i in c["slim_layers"]:                          # SlimInfer: decided at layer i,
                kt = c["slim_layers"][i]                                                  # applied after it
                kt = kt * T if kt <= 1 else kt * (T / c["slim_ref_len"] if c["slim_ref_len"] else 1)
                pend = _slim_select(layer, hidden, residual, positions, kt)
            if keep is not None:
                hidden, positions, orig_idx = hidden[keep], positions[keep], orig_idx[keep]
                residual = residual[keep] if residual is not None else None
                meta = _prefill_meta(attn_metadata, hidden.shape[0])
            _STATE["kept"].append(hidden.shape[0])
            with set_forward_context(meta):
                hidden, residual = layer(positions, hidden, kv_caches[i - self.start_layer], meta, residual)
            if do_prune and slim and i in c["slim_layers"]:
                hidden, positions, orig_idx, residual = hidden[pend], positions[pend], orig_idx[pend], residual[pend]
                meta = _prefill_meta(attn_metadata, hidden.shape[0])
        _STATE["idx"] = orig_idx.cpu()
        hidden, _ = self.norm(hidden, residual)
        if hidden.shape[0] != T:                                                     # sampler indexes the full prompt
            full = hidden.new_zeros(T, hidden.shape[1])
            full[orig_idx] = hidden
            hidden = full
        return hidden

    # decode: layers that pruned use their own cache length and append slot
    T, kept = _STATE["T"], _STATE["kept"]
    bs = c["block_size"]
    for i in range(self.start_layer, self.end_layer):
        layer = self.layers[i]
        meta = attn_metadata if kept is None or kept[i] == T else _decode_meta(attn_metadata, kept[i], T, bs)
        with set_forward_context(meta):
            hidden, residual = layer(positions, hidden, kv_caches[i - self.start_layer], meta, residual)
    hidden, _ = self.norm(hidden, residual)
    return hidden


def enable_inmodel_prune(start_layer: int = 17, drop_ratio: float = 0.12, n_obs: int = 128, n_sink: int = 128,
                         pool_kernel: int = 13, block_size: int = 16, method: str = "ours",
                         slim_layers=None, slim_ref_len: int = 0, slim_block: int = 64, slim_unit: int = 8,
                         slim_window: int = 4):
    """Monkey-patch vLLM's Qwen2Model.forward. Must match the LLM's block_size (vLLM default 16).
    method="ours": progressive pruning (start_layer, drop_ratio, ...).
    method="slim": SlimInfer prefill pruning; slim_layers = {layer: tokens kept for later layers}, default = their
      Qwen2.5-7B config {8: 12k, 17: 6k, 25: 4k} (values <= 1 = fraction of the prompt); slim_ref_len > 0 scales the kept counts by prompt_len / slim_ref_len.
      Decode is static here (SlimInfer's decode-time block re-selection / CPU offload is not reproduced)."""
    _CFG.update(start_layer=start_layer, drop_ratio=drop_ratio, n_obs=n_obs, n_sink=n_sink,
                pool_kernel=pool_kernel, block_size=block_size, method=method,
                slim_layers=slim_layers or {8: 12 * 1024, 17: 6 * 1024, 25: 4 * 1024}, slim_ref_len=slim_ref_len,
                slim_block=slim_block, slim_unit=slim_unit, slim_window=slim_window)
    from vllm.model_executor.models import qwen2
    _ORIG.setdefault("forward", qwen2.Qwen2Model.forward)
    qwen2.Qwen2Model.forward = _patched_forward


def last_kept_positions() -> Optional[List[int]]:
    """Prompt positions that reached the last layer in the most recent prefill."""
    return _STATE["idx"].tolist() if _STATE["idx"] is not None else None


def last_kept() -> Optional[List[int]]:
    """Tokens kept per layer for the most recent prefill."""
    return list(_STATE["kept"]) if _STATE["kept"] is not None else None
