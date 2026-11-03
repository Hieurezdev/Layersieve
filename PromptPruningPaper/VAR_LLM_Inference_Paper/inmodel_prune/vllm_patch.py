"""In-model progressive prompt pruning for vLLM 0.6.3 (FlashAttention backend), single request at a time.

Same algorithm as inmodel_prune/prune.py (HF reference): layers [0, start_layer) run on the full
prompt; before each layer >= start_layer, drop `drop_ratio` of the remaining prunable tokens, ranked by the
attention they receive from the last `n_obs` prompt tokens (max over heads, sum over queries, avg-pool smoothing).

KV cache layout: every layer owns its own cache tensor but all layers share one block table. A layer that keeps
n tokens writes their KV compactly into logical slots 0..n-1 of that block table (the blocks are already allocated
for the full prompt), and during decode it reads n + (#generated) slots and appends the new token at slot n + g - 1.
Layers before `start_layer` behave exactly as in stock vLLM. Memory is still allocated for the full prompt.

Decode can run in a CUDA graph (enforce_eager=False, max_seq_len_to_capture >= max_model_len); prefill is always eager.
Requirements: enable_chunked_prefill=False, enable_prefix_caching=False, one request per
step (max_num_seqs=1), Qwen2 models. Call enable_inmodel_prune(...) before creating the LLM.
"""
import dataclasses
import math
from typing import List, Optional

import torch
import torch.nn.functional as F

from inmodel_prune import units

_CFG = {}
_ORIG = {}
_STATE = {"T": None, "kept": None, "idx": None,   # prompt length, tokens kept per layer (index = layer id), positions kept at the last layer
          "probe": None,                          # probe=True: importance of every current token before every layer
          "step": 0, "prof": None, "prof_n": 0}   # decode steps since prefill; CUDA profiler of the last steps


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


def _select(layer, hidden, residual, positions, aux, layer_idx):
    """aux: per-token tensors of the current token set (MOE variants): "unit" ids, "protect" mask (BM25),
    "acc" importance summed over the layers scored so far (updated in place), "lex" BM25 score (score="bm25")."""
    c = _CFG
    T = hidden.shape[0]
    if c["score"] == "bm25":
        imp = aux["lex"]
    else:
        imp = _importance(layer, hidden, residual, positions, c["n_score"] or c["n_obs"])
        if "acc" in aux:
            aux["acc"] += imp
            imp = aux["acc"]
    idx = torch.arange(T, device=hidden.device)
    prunable = (positions >= c["n_sink"]) & (idx < T - c["n_obs"])
    if "protect" in aux:
        prunable &= ~aux["protect"]
    n_prunable = int(prunable.sum())
    if layer_idx == c["start_layer"]:
        _STATE["n_prunable0"] = n_prunable
    if c["oneshot_keep"] is None:                                                # progressive: drop_ratio per layer
        n_drop = int(n_prunable * c["drop_ratio"])
        if c["min_keep"]:                                                        # floor: never below min_keep of start
            n_drop = max(0, min(n_drop, n_prunable - int(_STATE["n_prunable0"] * c["min_keep"])))
    elif layer_idx == c["start_layer"]:                                          # ablation: one cut to oneshot_keep
        n_drop = int(int(prunable.sum()) * (1 - c["oneshot_keep"]))
    else:
        n_drop = 0
    keep = torch.ones(T, dtype=torch.bool, device=hidden.device)
    if n_drop > 0 and c["unit"]:
        keep[units.drop_units(imp, aux["unit"], prunable, n_drop)] = False
    elif n_drop > 0:
        if c["pool_kernel"] > 1:
            imp = F.avg_pool1d(imp[None, None], c["pool_kernel"], 1, c["pool_kernel"] // 2, count_include_pad=False)[0, 0]
        keep[torch.topk(imp.masked_fill(~prunable, float("inf")), n_drop, largest=False).indices] = False
    return idx[keep]


def _aux_init(input_ids):
    """Per-token state for the MOE variants (empty dict for plain LayerSieve)."""
    c = _CFG
    aux = {}
    if c["unit"] or c["bm25_frac"] > 0 or c["score"] == "bm25":
        aux["unit"] = units.unit_ids(input_ids, c["boundary"].to(input_ids.device), c["unit_max"])
    if c["bm25_frac"] > 0 or c["score"] == "bm25":
        T = input_ids.shape[0]
        s = units.bm25_scores(input_ids, aux["unit"], input_ids[-c["n_obs"]:])
        if c["bm25_frac"] > 0:
            obs = torch.arange(T, device=input_ids.device) >= T - c["n_obs"]
            aux["protect"] = units.bm25_protect(aux["unit"], s, int(c["bm25_frac"] * T), obs)
        if c["score"] == "bm25":
            aux["lex"] = s[aux["unit"]]
    if c["accum"]:
        aux["acc"] = torch.zeros(input_ids.shape[0], device=input_ids.device)
    return aux


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


def _compact_decode_kv(kv_caches, sets, ref, slot_mapping):
    """Decode-time KV sieving: every layer i < ref keeps only the KV of the tokens that entered layer ref (sets[ref]).
    sets[i] = prompt positions whose KV layer i wrote, in order, at logical slots 0..len-1. The kept entries are moved
    to logical slots 0..n-1 (physical slot of logical j = slot_mapping[j]); returns n."""
    target = sets[ref]
    n = target.shape[0]
    dst = slot_mapping[:n].long()
    for i in range(ref):
        if sets[i].shape[0] == n:
            continue
        src = slot_mapping[torch.searchsorted(sets[i], target)].long()
        kv = kv_caches[i].view(2, -1, *kv_caches[i].shape[-2:])                      # [2, slots, Hk, D]
        kv[:, dst] = kv[:, src].clone()
    return n


def _rerotate(rope, kv_cache, slots, old, new, heads=None):
    """Move the cached keys at physical slots from rotary position old to new (RoPE composes: R(new) = R(new-old) R(old)).
    Plain or YaRN RotaryEmbedding; the angle difference is computed in fp32 from the inverse frequencies (YaRN's
    magnitude scale is already in the cached key, so the extra rotation is a pure one). heads: KV heads to move
    (default all)."""
    name = type(rope).__name__
    assert name in ("RotaryEmbedding", "YaRNScalingRotaryEmbedding"), name
    inv = (rope._compute_inv_freq(rope.scaling_factor) if name == "YaRNScalingRotaryEmbedding"
           else rope._compute_inv_freq(rope.base)).to(old.device)
    ang = (new - old).float()[:, None] * inv[None]                                  # [n, rd/2]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]                                # [n, 1, rd/2]
    k = kv_cache.view(2, -1, *kv_cache.shape[-2:])[0]                              # [slots, Hk, D]
    rd = rope.rotary_dim
    hs = torch.arange(k.shape[1], device=k.device) if heads is None else torch.tensor(heads, device=k.device)
    idx = (slots[:, None], hs[None, :])                                              # [n, h] gather / scatter
    x = k[idx][..., :rd].float()                                                     # [n, h, rd]
    if rope.is_neox_style:
        x1, x2 = x[..., :rd // 2], x[..., rd // 2:]
        x = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)
    else:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).flatten(-2)
    y = k[idx]
    y[..., :rd] = x.to(k.dtype)
    k[idx] = y


def _gapcap(pos, gap):
    """Positions for a sorted set of prompt positions: every distance between neighbours is capped at gap
    (gap=1: consecutive 0..n-1 when pos starts at 0)."""
    d = (pos[1:] - pos[:-1]).clamp_max(gap)
    return torch.cat((pos[:1], pos[:1] + torch.cumsum(d, 0)))


def _prefill_meta(meta, n):
    """Metadata for a prefill over n (compacted) tokens: KV written to logical slots 0..n-1."""
    m = dataclasses.replace(
        meta, num_prefill_tokens=n, slot_mapping=meta.slot_mapping[:n],
        seq_lens=[n], seq_lens_tensor=meta.seq_lens_tensor.new_tensor([n]), max_query_len=n, max_prefill_seq_len=n,
        query_start_loc=meta.query_start_loc.new_tensor([0, n]), seq_start_loc=meta.seq_start_loc.new_tensor([0, n]))
    m._cached_prefill_metadata = None
    m._cached_decode_metadata = None
    return m


def _decode_meta(meta, seq_len, slot):
    """Metadata for one decode token of a layer whose cache holds seq_len tokens (incl. the new one), written at slot.
    seq_len / slot are 1-element GPU tensors, so the step can be captured in a CUDA graph."""
    m = dataclasses.replace(meta, slot_mapping=slot, seq_lens_tensor=seq_len)
    m._cached_prefill_metadata = None
    m._cached_decode_metadata = None
    return m


def _patched_forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None,
                     inputs_embeds=None):
    from vllm.forward_context import set_forward_context
    c = _CFG
    n_seq = attn_metadata.num_prefills + attn_metadata.num_decode_tokens
    if n_seq != 1:
        # anything that is not a single request: stock behaviour
        return _ORIG["forward"][type(self)](self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors,
                                              inputs_embeds)
    hidden = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
    residual = None
    is_prefill = attn_metadata.num_prefills > 0
    if is_prefill:
        T = hidden.shape[0]
        _STATE["T"], _STATE["kept"], _STATE["step"] = T, [], 0
        do_prune = T > c["n_sink"] + c["n_obs"]
        orig_idx = torch.arange(T, device=hidden.device)
        meta = attn_metadata
        slim = c["method"] == "slim"
        aux = _aux_init(input_ids) if do_prune and not slim else {}
        sets = []                                                                    # positions whose KV layer i writes
        wpos = []                                                                    # rotary positions it wrote them at
        _STATE["probe"] = [] if c["probe"] else None
        ev = [] if c["layer_times"] else None                                        # CUDA events around every layer
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if ev is not None:
                ev.append(torch.cuda.Event(enable_timing=True))
                ev[-1].record()
            keep = None
            if c["probe"]:
                _STATE["probe"].append(_importance(layer, hidden, residual, positions, c["n_obs"]).cpu())
            if (do_prune and not slim and i >= c["start_layer"]                      # ours: select BEFORE layer i
                    and (c["oneshot_keep"] is None or i == c["start_layer"])):        # one-shot: score once only
                keep = _select(layer, hidden, residual, positions, aux, i)
                aux = {k: v[keep] for k, v in aux.items()}
            elif do_prune and slim and i in c["slim_layers"]:                          # SlimInfer: decided at layer i,
                kt = c["slim_layers"][i]                                                  # applied after it
                kt = kt * T if kt <= 1 else kt * (T / c["slim_ref_len"] if c["slim_ref_len"] else 1)
                pend = _slim_select(layer, hidden, residual, positions, kt)
            if keep is not None:
                hidden, positions, orig_idx = hidden[keep], positions[keep], orig_idx[keep]
                residual = residual[keep] if residual is not None else None
                meta = _prefill_meta(attn_metadata, hidden.shape[0])
                if c["repos_gap"] and c["repos_prefill"]:                            # survivors re-indexed in prefill
                    positions = _gapcap(orig_idx, c["repos_gap"]).to(positions.dtype)
            _STATE["kept"].append(hidden.shape[0])
            sets.append(orig_idx)
            wpos.append(positions)
            with set_forward_context(meta):
                hidden, residual = layer(positions, hidden, kv_caches[i - self.start_layer], meta, residual)
            if ev is not None and i == self.end_layer - 1:
                ev.append(torch.cuda.Event(enable_timing=True))
                ev[-1].record()
            if do_prune and slim and i in c["slim_layers"]:
                hidden, positions, orig_idx, residual = hidden[pend], positions[pend], orig_idx[pend], residual[pend]
                meta = _prefill_meta(attn_metadata, hidden.shape[0])
        _STATE["idx"] = orig_idx.cpu()
        if ev is not None:                                                           # per layer, scoring included
            torch.cuda.synchronize()
            _STATE["layer_ms"] = [a.elapsed_time(b) for a, b in zip(ev, ev[1:])]
        ref = c["decode_ref"]
        if (do_prune and not slim and ref is not None and self.start_layer == 0 and ref < len(sets)
                and kv_caches[0].numel() > 0):                                     # no cache yet while profiling
            n = _compact_decode_kv(kv_caches, sets, ref, attn_metadata.slot_mapping)
            _STATE["kept"] = [min(k, n) for k in _STATE["kept"]]
            wpos = [wpos[i][torch.searchsorted(sets[i], sets[ref])] if i < ref else w for i, w in enumerate(wpos)]
            sets = [sets[ref] if i < ref else s for i, s in enumerate(sets)]
        if do_prune and c["repos_gap"] and kv_caches[0].numel() > 0:               # decode at gap-capped positions
            for i, (s, w) in enumerate(zip(sets, wpos)):                            # (keys shifted so the last kept
                if s.shape[0] < T:                                                   # token stays at T-1: decode
                    new = _gapcap(s, c["repos_gap"])                                 # positions need no change)
                    new = new + (T - 1 - new[-1])
                    heads = None
                    if c["repos_heads"] is not None:                                 # only the non-retrieval heads
                        heads = [h for h in range(kv_caches[i - self.start_layer].shape[-2])
                                 if h not in c["repos_heads"].get(self.start_layer + i, ())]
                    if not torch.equal(new, w) and heads != []:
                        _rerotate(self.layers[i].self_attn.rotary_emb, kv_caches[i - self.start_layer],
                                  attn_metadata.slot_mapping[:s.shape[0]].long(), w, new, heads)
        hidden, _ = self.norm(hidden, residual)
        _drop_buf(hidden.device, len(self.layers)).copy_(
            torch.tensor([0] * self.start_layer + [T - k for k in _STATE["kept"]], dtype=torch.int32))
        if hidden.shape[0] != T:                                                     # sampler indexes the full prompt
            full = hidden.new_zeros(T, hidden.shape[1])
            full[orig_idx] = hidden
            hidden = full
        return hidden

    # decode: layer i holds T - drop[i] prompt tokens, so its cache length is seq_len - drop[i] and the new token goes to
    # logical slot seq_len - drop[i] - 1. All on the GPU from vLLM's own buffers: no sync, CUDA-graph capturable (the
    # graph is captured once; each replay reads the current seq_lens / block_tables and drop, updated in place).
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing:
        _STATE["step"] += 1
        if c["prof_from"] and _STATE["step"] == c["prof_from"] and _STATE["prof"] is None:
            _STATE["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            _STATE["prof"].start()
        if _STATE["prof"] is not None:
            _STATE["prof_n"] += 1
    bs = c["block_size"]
    drop = _drop_buf(hidden.device, len(self.layers))
    seq = attn_metadata.seq_lens_tensor[:1] - drop                                                  # [L]
    logical = (seq - 1).long()
    slot = (attn_metadata.block_tables[0, logical // bs].long() * bs + logical % bs).to(attn_metadata.slot_mapping.dtype)
    kept = _STATE["kept"]
    metas = {}
    for i in range(self.start_layer, self.end_layer):
        layer = self.layers[i]
        key = kept[i] if kept is not None and not capturing else i                 # eager: share equal lengths
        if key not in metas:
            metas[key] = _decode_meta(attn_metadata, seq[i:i + 1], slot[i:i + 1])
        meta = metas[key]
        with set_forward_context(meta):
            hidden, residual = layer(positions, hidden, kv_caches[i - self.start_layer], meta, residual)
    hidden, _ = self.norm(hidden, residual)
    return hidden


def _drop_buf(device, n_layers):
    """Persistent per-layer count of prompt tokens a layer does not hold (int32 [n_layers], 0 = full prompt)."""
    if _STATE.get("drop") is None:
        _STATE["drop"] = torch.zeros(n_layers, dtype=torch.int32, device=device)
    return _STATE["drop"]


def _mlp_chunked(self, x):
    """Qwen2MLP / LlamaMLP over token chunks: exact (the MLP is per token), only lowers the peak memory of a long
    prefill."""
    n, orig = _CFG["mlp_chunk"], _ORIG["mlp"][type(self)]
    if not n or x.shape[0] <= n:
        return orig(self, x)
    return torch.cat([orig(self, xi) for xi in x.split(n)])


def enable_inmodel_prune(start_layer: int = 17, drop_ratio: float = 0.12, n_obs: int = 128, n_sink: int = 128,
                         n_score: Optional[int] = None,
                         pool_kernel: int = 13, block_size: int = 16, method: str = "ours",
                         slim_layers=None, slim_ref_len: int = 0, slim_block: int = 64, slim_unit: int = 8,
                         slim_window: int = 4, unit: bool = False, unit_max: int = 128, accum: bool = False,
                         bm25_frac: float = 0.0, score: str = "attn", tokenizer: str = "Qwen/Qwen2.5-7B-Instruct-1M",
                         mlp_chunk: int = 0, decode_ref: Optional[int] = None, probe: bool = False,
                         prof_from: int = 0, oneshot_keep: Optional[float] = None,
                         min_keep: float = 0.0, repos_gap: int = 0, repos_prefill: bool = False,
                         repos_heads=None, layer_times: bool = False):
    """Monkey-patch vLLM's Qwen2Model.forward and LlamaModel.forward (same layer interface). Must match the LLM's block_size (vLLM default 16).
    method="ours": progressive pruning (start_layer, drop_ratio, ...). Variants borrowed from the top-1 team's
      SpecPrefill (inmodel_prune/units.py): unit=True drops whole units (<= unit_max tokens, score = max importance)
      instead of avg-pooled tokens; accum=True ranks by importance summed over all layers scored so far;
      bm25_frac > 0 never drops the units that best match the last n_obs tokens by BM25 (up to that fraction of the
      prompt); score="bm25" replaces attention importance by the BM25 score of the token's unit (ablation).
    mlp_chunk > 0: run every MLP in chunks of that many tokens (exact; lets a ~170k prefill fit on a 48 GB GPU).
    decode_ref = r: after prefill, every layer < r keeps only the KV of the tokens that reached layer r, so decode reads
      less KV (shallow layers no longer hold the whole prompt). None = off (layers keep what they computed).
    method="slim": SlimInfer prefill pruning; slim_layers = {layer: tokens kept for later layers}, default = their
      Qwen2.5-7B config {8: 12k, 17: 6k, 25: 4k} (values <= 1 = fraction of the prompt); slim_ref_len > 0 scales the kept counts by prompt_len / slim_ref_len.
      Decode is static here (SlimInfer's decode-time block re-selection / CPU offload is not reproduced).
    probe=True: record the importance of the current tokens before every layer (last_probe(); use with no pruning).
    n_score = s (ablation): score with the attention of the last s prompt tokens instead of the last n_obs (the last
      n_obs tokens are still never pruned).
    oneshot_keep = f (ablation): instead of dropping drop_ratio before every layer >= start_layer, cut once before
      start_layer to a fraction f of the prunable tokens and keep that set to the last layer (FastKV-style one-shot
      selection, with LayerSieve's scoring / units).
    min_keep = f > 0: progressive dropping stops once a fraction f of the prunable tokens of start_layer is left
      (aggregation tasks such as RULER cwe / fwe need many tokens, so a large drop_ratio starves them otherwise).
    repos_gap = G > 0: decode at re-indexed positions. After prefill every layer's cached keys are re-rotated so that the
      distance between neighbouring kept tokens is capped at G (G=1: consecutive 0..len-1), and decode tokens continue
      right after the last kept token. repos_prefill=True also re-indexes the survivors in prefill (layers >= start_layer
      compute at the capped positions; only the dense prefix is re-rotated afterwards). 0 = original prompt positions.
      repos_heads = {layer: [kv heads]}: these KV heads (e.g. retrieval heads) keep the original positions.
    layer_times=True: time every prefill layer with CUDA events (scoring included); read with last_layer_ms().
    prof_from = k > 0: from the k-th decode step on, profile GPU kernels (CUDA activity only); read with
      decode_kernel_time() after generate."""
    _CFG.update(start_layer=start_layer, drop_ratio=drop_ratio, n_obs=n_obs, n_sink=n_sink, n_score=n_score,
                pool_kernel=pool_kernel, block_size=block_size, method=method,
                slim_layers=slim_layers or {8: 12 * 1024, 17: 6 * 1024, 25: 4 * 1024}, slim_ref_len=slim_ref_len,
                slim_block=slim_block, slim_unit=slim_unit, slim_window=slim_window, unit=unit, unit_max=unit_max,
                accum=accum, bm25_frac=bm25_frac, score=score, mlp_chunk=mlp_chunk,
                decode_ref=decode_ref, probe=probe, prof_from=prof_from, oneshot_keep=oneshot_keep,
                min_keep=min_keep, repos_gap=repos_gap, repos_prefill=repos_prefill,
                repos_heads=repos_heads, layer_times=layer_times)
    if unit or bm25_frac > 0 or score == "bm25":
        from transformers import AutoTokenizer
        _CFG["boundary"] = units.boundary_token_ids(AutoTokenizer.from_pretrained(tokenizer))
    from vllm.model_executor.models import llama, qwen2
    for model_cls, mlp_cls in ((qwen2.Qwen2Model, qwen2.Qwen2MLP), (llama.LlamaModel, llama.LlamaMLP)):
        _ORIG.setdefault("forward", {}).setdefault(model_cls, model_cls.forward)
        model_cls.forward = _patched_forward
        _ORIG.setdefault("mlp", {}).setdefault(mlp_cls, mlp_cls.forward)
        mlp_cls.forward = _mlp_chunked


def last_kept_positions() -> Optional[List[int]]:
    """Prompt positions that reached the last layer in the most recent prefill."""
    return _STATE["idx"].tolist() if _STATE["idx"] is not None else None


def last_kept() -> Optional[List[int]]:
    """Tokens kept per layer for the most recent prefill."""
    return list(_STATE["kept"]) if _STATE["kept"] is not None else None


def last_layer_ms() -> Optional[List[float]]:
    """layer_times=True: prefill time of every layer (ms, its scoring and pruning included) in the most recent prefill."""
    return _STATE.get("layer_ms")


def last_probe() -> Optional[List[torch.Tensor]]:
    """probe=True: per layer, the importance of every token entering it in the most recent prefill (CPU tensors)."""
    return _STATE["probe"]


def decode_kernel_time() -> Optional[float]:
    """prof_from > 0: stop the profiler and return the summed GPU kernel time per profiled decode step (seconds), i.e.
    the step time without CPU launch overhead (what CUDA graphs would approach). None if nothing was profiled."""
    prof, n = _STATE["prof"], _STATE["prof_n"]
    _STATE["prof"], _STATE["prof_n"] = None, 0
    if prof is None or n == 0:
        return None
    torch.cuda.synchronize()
    prof.stop()
    us = sum(e.self_device_time_total for e in prof.key_averages())
    return us / 1e6 / n
