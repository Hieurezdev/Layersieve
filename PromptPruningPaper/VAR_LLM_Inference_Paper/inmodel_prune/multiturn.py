"""Multi-turn reuse of LayerSieve over one shared document (HF reference, batch size 1).

The document precedes the question, and attention is causal, so the dense layers [0, start_layer) of the document do not
depend on the question. After the first turn we keep (i) the dense KV of the document in every layer and (ii) the hidden
state entering layer start_layer for every document token. A new question then needs:
  dense      (mode B)  the question tokens over the full dense prefix cache: the answer of dense with a prefix cache
  resieve    (mode A)  the question tokens through layers [0, start_layer) over the dense prefix KV, then the sieve of
                       layers [start_layer, L) re-run from the stored hidden states with the new observation window:
                       the answer of LayerSieve run from scratch on (document, new question)
  naive               the pruned (and KV-sieved) cache of the first question reused as is for the new question
  partial  (mode C)    recover only what the new question adds: its tokens through layers [0, start_layer), one unit
                       selection with its attention at start_layer (floor budget), then layers [start_layer, L) computed
                       only for the selected units the first question had dropped and for the question tokens, over the
                       first question's (stale) pruned KV of the units it kept
  fill<p>  (mode B, progressive completion) the background completion has reached a fraction p of the document: the
                       pruned layers of the document tokens [0, pT) are dense (complete, exact); the new question runs
                       LayerSieve over the rest only, with that prefix protected and its dense KV read as is.
                       fill0 = resieve, fill1 = dense (the prefix's dense deep KV is sliced from the document's dense
                       prefill: under causal attention it does not depend on the tokens after it)
  fresh               LayerSieve from scratch on (document, question), to check that resieve reproduces it
Conversations (load_document(..., history=True), then extend() after every turn): the prefix grows by each turn's
new tokens (previous answer + next question) through every dense layer; tokens after the document are never pruned;
naive keeps extending its pruned cache and partial its set of computed tokens across turns.
The LayerSieve here is the paper's: unit-level removal, keep floor, decode-time KV sieving at a reference layer (same
selection rule as inmodel_prune/vllm_patch.py with unit=True, min_keep, decode_ref).
"""
import inspect
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from inmodel_prune import units
from inmodel_prune.prune import _rope as rope_heads, token_importance


@dataclass
class SieveConfig:
    start_layer: int = 17
    drop_ratio: float = 0.4
    min_keep: float = 0.25          # floor, fraction of the prunable tokens entering start_layer
    decode_ref: int = 22            # KV sieving: layers < decode_ref keep only the KV of the tokens that reached it
    n_obs: int = 128
    n_sink: int = 128
    unit_max: int = 128


def _call(layer):
    name = "past_key_values" if "past_key_values" in inspect.signature(layer.forward).parameters else "past_key_value"

    def call(h, cache, cos, sin, mask=None):
        out = layer(h, attention_mask=mask, use_cache=True, position_embeddings=(cos, sin), **{name: cache})
        return out[0] if isinstance(out, tuple) else out
    return call


def _cache(keys, values):
    c = DynamicCache()
    c.key_cache, c.value_cache = list(keys), list(values)
    return c


def _sync():
    torch.cuda.synchronize()
    return time.time()


class MultiTurn:
    def __init__(self, model, tokenizer, cfg: SieveConfig = None):
        self.m, self.mm, self.cfg = model, model.model, cfg or SieveConfig()
        self.calls = [_call(l) for l in self.mm.layers]
        self.L = len(self.mm.layers)
        self.boundary = units.boundary_token_ids(tokenizer).cuda()
        self.history = False

    # ---- building blocks -------------------------------------------------------------------------------------------
    def _rope(self, h, pos):
        return self.mm.rotary_emb(h, pos[None])

    @torch.no_grad()
    def _append(self, ids, pos0, cache, layers):
        """Run new tokens (ids, at positions pos0..) through `layers` on top of `cache` (any per-layer length): every
        new token sees the whole cached part and the new tokens before it. Returns the hidden states after them."""
        s = ids.shape[0]
        h = self.mm.embed_tokens(ids[None])
        cos, sin = self._rope(h, torch.arange(pos0, pos0 + s, device=ids.device))
        tri = torch.ones(s, s, dtype=torch.bool, device=ids.device).tril()
        for li in layers:
            n = cache.key_cache[li].shape[-2] if li < len(cache.key_cache) and torch.is_tensor(cache.key_cache[li]) else 0
            mask = torch.cat([torch.ones(s, n, dtype=torch.bool, device=ids.device), tri], 1)[None, None]
            h = self.calls[li](h, cache, cos, sin, mask)
        return h

    @torch.no_grad()
    def _sieve(self, h, ids, cache):
        """LayerSieve layers [start_layer, L) on hidden states h [1, T, d] of the whole prompt (positions 0..T-1);
        cache holds layers [0, start_layer) for all T tokens. Writes the pruned layers into cache, applies KV sieving,
        returns (last hidden [1, 1, d], per-layer kept original indices)."""
        c, T = self.cfg, h.shape[1]
        pos = torch.arange(T, device=h.device)
        cos, sin = self._rope(h, pos)
        unit = units.unit_ids(ids, self.boundary, c.unit_max)
        sets = [pos] * c.start_layer
        n0 = None
        for li in range(c.start_layer, self.L):
            layer = self.mm.layers[li]
            imp = token_importance(layer, h, cos, sin, c.n_obs)
            idx = torch.arange(h.shape[1], device=h.device)
            prunable = (pos >= c.n_sink) & (idx < h.shape[1] - c.n_obs)
            if self.history:
                prunable &= pos < self.Tdoc
            npr = int(prunable.sum())
            n0 = npr if n0 is None else n0
            n_drop = max(0, min(int(npr * c.drop_ratio), npr - int(n0 * c.min_keep)))
            if n_drop > 0:
                keep = torch.ones(h.shape[1], dtype=torch.bool, device=h.device)
                keep[units.drop_units(imp, unit, prunable, n_drop)] = False
                h, cos, sin, pos, unit = h[:, keep], cos[:, keep], sin[:, keep], pos[keep], unit[keep]
            sets.append(pos)
            h = self.calls[li](h, cache, cos, sin)
        # pruned layers before KV sieving (what the partial mode reuses): positions, keys, values per layer >= start_layer
        self.deep = (sets[c.start_layer:], cache.key_cache[c.start_layer:], cache.value_cache[c.start_layer:])
        r = c.decode_ref
        if r is not None and r < self.L:
            for li in range(r):
                sel = torch.searchsorted(sets[li], sets[r])
                cache.key_cache[li] = cache.key_cache[li][:, :, sel]
                cache.value_cache[li] = cache.value_cache[li][:, :, sel]
        return h[:, -1:], sets

    @torch.no_grad()
    def _importance_over(self, layer, h, cos, sin, k_pre, n_obs, obs_chunk=32):
        """token_importance for tokens h that follow a fixed prefix whose keys k_pre [1, Hk, P, D] (rotated) are cached:
        the softmax of the last n_obs tokens runs over prefix + tokens; returns the importance of the tokens only."""
        attn = layer.self_attn
        n, D, P = h.shape[1], attn.head_dim, k_pre.shape[2]
        x = layer.input_layernorm(h)
        k = rope_heads(attn.k_proj(x).view(1, n, -1, D).transpose(1, 2), cos, sin)
        q = rope_heads(attn.q_proj(x[:, -n_obs:]).view(1, n_obs, -1, D).transpose(1, 2), cos[:, -n_obs:], sin[:, -n_obs:])
        k, q = torch.cat([k_pre, k], 2)[0].float(), q[0].float()
        Hq, Hk = q.shape[0], k.shape[0]
        G = Hq // Hk
        key = torch.arange(P + n, device=h.device)
        imp = torch.zeros(P + n, device=h.device)
        for r0 in range(0, n_obs, obs_chunk):
            r1 = min(r0 + obs_chunk, n_obs)
            sc = torch.bmm(q[:, r0:r1].reshape(Hk, G * (r1 - r0), D), k.transpose(1, 2)) * attn.scaling
            sc = sc.view(Hq, r1 - r0, P + n)
            rows = P + n - n_obs + torch.arange(r0, r1, device=h.device)
            sc.masked_fill_(key[None, None] > rows[None, :, None], float("-inf"))
            imp += torch.softmax(sc, -1).amax(0).sum(0)
            del sc
        return imp[P:]

    @torch.no_grad()
    def _sieve_fixed(self, h, ids, cache, P):
        """_sieve with the document tokens [0, P) fixed: kept in every pruned layer with their dense KV (sliced from the
        document's dense prefill), never scored or computed; h [1, T - P, d] holds the tokens [P, T) entering start_layer."""
        c, S, L = self.cfg, self.cfg.start_layer, self.L
        T = P + h.shape[1]
        pos = torch.arange(P, T, device=h.device)
        ppos = torch.arange(P, device=h.device)
        cos, sin = self._rope(h, pos)
        unit = units.unit_ids(ids, self.boundary, c.unit_max)[P:]
        sets = [torch.arange(T, device=h.device)] * S
        n0 = None
        for li in range(S, L):
            kf, vf = self.dense.key_cache[li][:, :, :P], self.dense.value_cache[li][:, :, :P]
            idx = torch.arange(h.shape[1], device=h.device)
            prunable = (pos >= c.n_sink) & (idx < h.shape[1] - c.n_obs) & (pos < self.Tdoc)
            npr = int(prunable.sum())
            n0 = npr if n0 is None else n0
            n_drop = max(0, min(int(npr * c.drop_ratio), npr - int(n0 * c.min_keep)))
            if n_drop > 0:
                imp = self._importance_over(self.mm.layers[li], h, cos, sin, kf, c.n_obs)
                keep = torch.ones(h.shape[1], dtype=torch.bool, device=h.device)
                keep[units.drop_units(imp, unit, prunable, n_drop)] = False
                h, cos, sin, pos, unit = h[:, keep], cos[:, keep], sin[:, keep], pos[keep], unit[keep]
            sets.append(torch.cat([ppos, pos]))
            h, k, v = self._layer_over(li, h, cos, sin, pos, kf, vf, ppos)
            cache.key_cache.append(torch.cat([kf, k], 2))
            cache.value_cache.append(torch.cat([vf, v], 2))
        r = c.decode_ref
        if r is not None and r < L:
            for li in range(r):
                sel = torch.searchsorted(sets[li], sets[r])
                cache.key_cache[li] = cache.key_cache[li][:, :, sel]
                cache.value_cache[li] = cache.value_cache[li][:, :, sel]
        return h[:, -1:], sets

    @torch.no_grad()
    def _layer_over(self, li, h, cos, sin, pos, k_old, v_old, pos_old, chunk=2048):
        """Decoder layer li for tokens h at positions pos, attending causally (by position) to the cached keys/values
        k_old/v_old at positions pos_old and to themselves. Returns (hidden, new keys, new values)."""
        layer = self.mm.layers[li]
        attn = layer.self_attn
        n, D = h.shape[1], attn.head_dim
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(1, n, -1, D).transpose(1, 2)
        k = attn.k_proj(x).view(1, n, -1, D).transpose(1, 2)
        v = attn.v_proj(x).view(1, n, -1, D).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        G = q.shape[1] // k.shape[1]
        K = torch.cat([k_old, k], 2).repeat_interleave(G, 1)
        V = torch.cat([v_old, v], 2).repeat_interleave(G, 1)
        pk = torch.cat([pos_old, pos])
        out = torch.empty_like(q)
        for c0 in range(0, n, chunk):
            mask = pk[None] <= pos[c0:c0 + chunk, None]
            out[:, :, c0:c0 + chunk] = F.scaled_dot_product_attention(q[:, :, c0:c0 + chunk], K, V,
                                                                       attn_mask=mask[None, None], scale=attn.scaling)
        h = h + attn.o_proj(out.transpose(1, 2).reshape(1, n, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, k, v

    @torch.no_grad()
    def _decode(self, h_last, cache, next_pos, max_new, eos):
        tok = self.m.lm_head(self.mm.norm(h_last)).argmax(-1)
        out = [int(tok)]
        for step in range(1, max_new):
            if out[-1] in eos:
                break
            x = self.mm.embed_tokens(tok)
            cos, sin = self._rope(x, torch.tensor([next_pos + step - 1], device=x.device))
            for call in self.calls:
                x = call(x, cache, cos, sin)
            tok = self.m.lm_head(self.mm.norm(x)).argmax(-1)
            out.append(int(tok))
        return out

    # ---- the document (first turn's reusable part) -----------------------------------------------------------------
    @torch.no_grad()
    def load_document(self, prefix_ids, history=False):
        """Dense prefill of the document: full KV of every layer + hidden states entering start_layer."""
        t0 = _sync()
        self.P, self.Tdoc, self.history, self.turn1 = prefix_ids, prefix_ids.shape[0], history, None
        h = self.mm.embed_tokens(prefix_ids[None])
        cos, sin = self._rope(h, torch.arange(prefix_ids.shape[0], device=h.device))
        cache = DynamicCache()
        t = [t0]
        for li, call in enumerate(self.calls):
            if li == self.cfg.start_layer:
                self.h_ckpt = h
            h = call(h, cache, cos, sin)
            t.append(_sync())
        self.dense = cache
        self.layer_time = [b - a for a, b in zip(t, t[1:])]
        self.naive = None
        return t[-1] - t0

    @torch.no_grad()
    def extend(self, delta_ids):
        """Conversation: append this turn's new tokens to the dense prefix (every layer) and to the stored hidden states."""
        Tp, S = self.P.shape[0], self.cfg.start_layer
        h = self.mm.embed_tokens(delta_ids[None])
        cos, sin = self._rope(h, torch.arange(Tp, Tp + delta_ids.shape[0], device=h.device))
        tri = torch.ones(delta_ids.shape[0], delta_ids.shape[0], dtype=torch.bool, device=h.device).tril()
        mask = torch.cat([torch.ones(delta_ids.shape[0], Tp, dtype=torch.bool, device=h.device), tri], 1)[None, None]
        for li, call in enumerate(self.calls):
            if li == S:
                self.h_ckpt = torch.cat([self.h_ckpt, h], 1)
            h = call(h, self.dense, cos, sin, mask)
        self.P = torch.cat([self.P, delta_ids])

    # ---- one question ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def answer(self, mode, suffix_ids, max_new, eos=()):
        """Returns (generated ids, prefill seconds, info)."""
        Tp, s, L, c = self.P.shape[0], suffix_ids.shape[0], self.L, self.cfg
        eos = set(eos)
        t0 = _sync()
        info = {}
        if mode == "dense":
            cache = _cache(self.dense.key_cache, self.dense.value_cache)
            h = self._append(suffix_ids, Tp, cache, range(L))[:, -1:]
        elif mode == "resieve":
            S = c.start_layer
            cache = _cache(self.dense.key_cache[:S], self.dense.value_cache[:S])
            hs = self._append(suffix_ids, Tp, cache, range(S))
            h, sets = self._sieve(torch.cat([self.h_ckpt, hs], 1), torch.cat([self.P, suffix_ids]), cache)
            info["kept"] = sets[-1]
            info["deep_tokens"] = sum(int(x.shape[0]) for x in sets[c.start_layer:])
            # what the naive mode reuses: this question's pruned cache, restricted to the document tokens (independent
            # questions) or whole (conversation: the question stays in the history)
            lim = Tp + s if self.history else Tp
            held = [sets[c.decode_ref]] * c.decode_ref + sets[c.decode_ref:]
            self.last_pruned = ([k[:, :, :int((st < lim).sum())] for k, st in zip(cache.key_cache, held)],
                                [v[:, :, :int((st < lim).sum())] for v, st in zip(cache.value_cache, held)])
        elif mode == "partial":
            S, r = c.start_layer, c.decode_ref
            cache = _cache(self.dense.key_cache[:S], self.dense.value_cache[:S])
            hs = self._append(suffix_ids, Tp, cache, range(S))
            ids = torch.cat([self.P, suffix_ids])
            h_all = torch.cat([self.h_ckpt, hs], 1)
            T = h_all.shape[1]
            pos_all = torch.arange(T, device=h_all.device)
            cos, sin = self._rope(h_all, pos_all)
            # one unit selection with the new question at layer S, keeping the floor fraction
            imp = token_importance(self.mm.layers[S], h_all, cos, sin, c.n_obs)
            unit = units.unit_ids(ids, self.boundary, c.unit_max)
            prunable = (pos_all >= c.n_sink) & (pos_all < T - c.n_obs)
            if self.history:
                prunable &= pos_all < self.Tdoc
            npr = int(prunable.sum())
            keep = torch.ones(T, dtype=torch.bool, device=h_all.device)
            keep[units.drop_units(imp, unit, prunable, npr - int(npr * c.min_keep))] = False
            sets1, k1, v1 = self.turn1                                  # tokens with deep KV so far, per deep layer
            new = pos_all[keep & ~torch.isin(pos_all, sets1[-1])]       # selected now, not through all deep layers yet
            h, cn, sn = h_all[:, new], cos[:, new], sin[:, new]
            pos_l = [pos_all] * S                                       # positions held by each layer
            for j, li in enumerate(range(S, L)):
                old = ~torch.isin(sets1[j], new)                        # a token dropped part-way before is redone
                h, k, v = self._layer_over(li, h, cn, sn, new, k1[j][:, :, old], v1[j][:, :, old], sets1[j][old])
                cache.key_cache.append(torch.cat([k1[j][:, :, old], k], 2))
                cache.value_cache.append(torch.cat([v1[j][:, :, old], v], 2))
                pos_l.append(torch.cat([sets1[j][old], new]))
            if self.history:                                            # computed tokens carry over to the next turn
                self.turn1 = (pos_l[S:], cache.key_cache[S:], cache.value_cache[S:])
            final_r = torch.cat([sets1[r - S], new])                    # KV sieving to the tokens reaching layer r
            for li in range(r):
                sel = torch.nonzero(torch.isin(pos_l[li], final_r)).squeeze(1)
                cache.key_cache[li] = cache.key_cache[li][:, :, sel]
                cache.value_cache[li] = cache.value_cache[li][:, :, sel]
            h = h[:, -1:]
            info["kept"] = torch.unique(torch.cat([sets1[-1], new]))
            info["new"] = int((new < Tp).sum())
            info["deep_tokens"] = int(new.shape[0]) * (L - S)
        elif mode.startswith("fill"):
            S, P = c.start_layer, int(float(mode[4:]) * self.Tdoc)
            cache = _cache(self.dense.key_cache[:S], self.dense.value_cache[:S])
            hs = self._append(suffix_ids, Tp, cache, range(S))
            h, sets = self._sieve_fixed(torch.cat([self.h_ckpt[:, P:], hs], 1), torch.cat([self.P, suffix_ids]), cache, P)
            info["kept"] = sets[-1]
            info["deep_tokens"] = sum(int(x.shape[0]) - P for x in sets[S:])      # computed now (the prefix was done)
            info["fill_tokens"] = P * (L - S)                                    # spent in the background before
        elif mode == "naive":
            cache = _cache(*self.naive)
            h = self._append(suffix_ids, Tp, cache, range(L))[:, -1:]
        elif mode in ("fresh", "dense_fresh"):                # dense_fresh: floating-point noise floor of "dense"
            ids = torch.cat([self.P, suffix_ids])
            h = self.mm.embed_tokens(ids[None])
            cos, sin = self._rope(h, torch.arange(ids.shape[0], device=h.device))
            cache = DynamicCache()
            for li in range(c.start_layer if mode == "fresh" else L):
                h = self.calls[li](h, cache, cos, sin)
            if mode == "fresh":
                h, sets = self._sieve(h, ids, cache)
                info["kept"] = sets[-1]
            else:
                h = h[:, -1:]
        else:
            raise ValueError(mode)
        info["logits"] = self.m.lm_head(self.mm.norm(h)).float()[0, -1]
        ttft = _sync() - t0
        held = [k.shape[-2] for k in cache.key_cache]
        out = self._decode(h, cache, Tp + s, max_new, eos)
        if mode == "naive" and self.history:                       # the conversation goes on over the pruned cache
            self.naive = ([k[:, :, :n] for k, n in zip(cache.key_cache, held)],
                          [v[:, :, :n] for v, n in zip(cache.value_cache, held)])
        del cache
        return out, ttft, info

    def keep_for_naive(self):
        """Freeze the pruned cache of the last resieve answer as the cache the naive and partial modes reuse."""
        self.naive = self.last_pruned
        lim = float("inf") if self.history else self.P.shape[0]
        sets, ks, vs = self.deep                                        # document entries (+ question in conversations)
        n = [int((st < lim).sum()) for st in sets]
        self.turn1 = ([st[:m] for st, m in zip(sets, n)], [k[:, :, :m] for k, m in zip(ks, n)],
                      [v[:, :, :m] for v, m in zip(vs, n)])
