"""Bring far evidence close to the question without touching RoPE: attention-guided excerpt recall.

The last --qlen prompt tokens are taken as the question, with no task-specific parsing. Pass 1 prefills the context,
then runs the question tokens once to read their attention at a few middle layers (--layers, as LayerSieve's scoring layer), sums it
into 64-token blocks, and picks the best blocks (each widened by --pad tokens) up to --budget tokens. Pass 2 rewrites
the cache from the question on: a copy of the picked spans, in their original order, is placed right before the
question, so the evidence also sits at a short, ordinary distance. Dense Qwen2.5 forward from scripts/selfextend_eval.py
(plain attention, true positions everywhere).

    python scripts/recall_eval.py --prompts data/infbench_kv86.jsonl --limit 20 --out results/selfext/kv20_recall.jsonl

The answers are only used afterwards, to log whether they fall inside the excerpts.
"""
import argparse
import json
import math
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--tasks", default="")
ap.add_argument("--lengths", default="", help="RULER only: comma list of lengths (task suffix @<len>)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--layers", default="17", help="layers whose question attention scores the context")
ap.add_argument("--agg", default="mean", choices=["mean", "max", "retr"], help="how heads are combined into the context "
                "score: mean over heads, max over heads (per question token), or mean over the retrieval heads of --heads")
ap.add_argument("--heads", default="results/repos/heads_7b1m.json", help="--agg retr: retrieval-head file")
ap.add_argument("--retr-thr", type=float, default=0.3, help="--agg retr: query heads scoring above this are used")
ap.add_argument("--budget", type=int, default=2048, help="tokens of excerpts")
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--pad", type=int, default=96, help="context kept on each side of a picked block")
ap.add_argument("--snap", action="store_true", help="widen every picked span to whole lines (start after a line break, "
                "end with one), as the oracle windows of scripts/ld_move_probe.py")
ap.add_argument("--snap-max", type=int, default=1024, help="--snap: an edge moves at most this many tokens")
ap.add_argument("--spans-file", default="", help="diagnostic: JSON {id: [[s0, s1], ...]} replaces the picked spans "
                "(oracle evidence); --snap still applies")
ap.add_argument("--vgap", type=int, default=0, help="--vpos pack: positions left empty between two moved spans "
                "(default len(SEP)); tests whether adjacency of independently encoded spans matters")
ap.add_argument("--vscale", type=float, default=0.25, help="--vpos scale: factor on the span-to-question distances")
ap.add_argument("--vorder", default="text", choices=["text", "score"], help="--move: order of the moved spans; score = "
                "the span the question attends most (max block score of the selector) ends nearest the question")
ap.add_argument("--vpos", default="pack", choices=["pack", "perm", "scale", "scaleperm"], help="--move: scale = every span keeps its order and its distance to the question "
                "times --vscale (span ends at qs0 - f*(qs0 - s1); overlaps pushed back); scaleperm = the same targets turned into a "
                "permutation of 0..qs0-1 (slots ranked by target), so no moved token shares a position with an unmoved one; pack = spans share positions with "
                "the last context tokens; perm = a permutation (unmoved context in order, then the spans, then the "
                "question), so no two slots share a position")
ap.add_argument("--sink", type=int, default=64, help="leading tokens never picked")
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--qlen", type=int, default=16, help="the last QLEN prompt tokens score the context")
ap.add_argument("--ins", type=int, default=128, help="the excerpts go before the last INS prompt tokens (>= qlen)")
ap.add_argument("--gem", action="store_true", help="GemFilter-style: keep only the sink, the picked spans (text order) "
                "and the last INS tokens, re-encoded from position 0 through all layers; the rest of the context is dropped")
ap.add_argument("--gem-sep", action="store_true", help="--gem: put SEP ('\\n...\\n') between two picked spans so cut "
                "pieces do not fuse")
ap.add_argument("--reorder", action="store_true", help="move the picked spans (not copies) to the end of the context, "
                "right before the last INS tokens, keep everything else in its order, and re-encode from the first moved "
                "token on: every token appears once, nothing is added")
ap.add_argument("--fuse", action="store_true", help="decode the plain and the recalled prompt in lockstep (two caches "
                "sharing the context prefill) and pick each token by the sum of the two log-probabilities")
ap.add_argument("--draft", action="store_true", help="decode the plain answer first and let its tokens score the "
                "context too, with the last QLEN prompt tokens (what the model reads while answering)")
ap.add_argument("--hide-from", type=int, default=None, help="copy: the question and the answer cannot see the "
                "excerpts in layers hide_from <= l < hide_to (the excerpts themselves are encoded normally)")
ap.add_argument("--hide-to", type=int, default=1000)
ap.add_argument("--select", action="store_true", help="also decode the plain answer (from the pass-1 cache) and keep the "
                "one with the higher mean token log-probability")
ap.add_argument("--move", action="store_true", help="instead of copying, give the picked spans new RoPE positions right "
                "before the question (their cached K/V are reused, their far positions are left empty): no token is "
                "duplicated and no text is added")
ap.add_argument("--bias", type=float, default=0.0, help="instead of copying or moving, add BIAS * max(0, log(block "
                "score / mean block score)) to the attention logit of every context key, for the question and the answer "
                "(true positions, nothing selected or duplicated)")
ap.add_argument("--focus", default="", help="retrieval-head file (scripts/retrieval_heads.py): add log(context / "
                "picked tokens) to the logits of the picked spans, only in the query heads scoring above --head-thr, only "
                "for the question and the answer (true positions, nothing copied)")
ap.add_argument("--calib", default="", help="per-head calibration file (scripts/attn_calib.py): for the question and "
                "the answer, queries are scaled by tau_h and keys farther than d_ref get the RoPE prior correction c_h(d); "
                "nothing is picked or copied")
ap.add_argument("--calib-mode", default="both", choices=["tau", "corr", "both"])
ap.add_argument("--head-thr", type=float, default=0.1)
ap.add_argument("--bias-cap", type=float, default=4.0, help="--bias: upper bound of the added logit")
ap.add_argument("--sel", default="budget", choices=["budget", "rel"], help="budget: the best blocks up to --budget "
                "(scores of --layers / --qlen); rel: at --sel-layer, per block the larger of (max-over-heads attention of the "
                "last 16 tokens) and (head-mean attention of the last 128 tokens minus that of a neutral request), each over "
                "its own top block; blocks >= --rel-thr are picked and widened by --sel-pad (scripts/selector_probe.py study)")
ap.add_argument("--sel-layer", type=int, default=14)
ap.add_argument("--rel-thr", type=float, default=0.3)
ap.add_argument("--sel-pad", type=int, default=32)
ap.add_argument("--dual", action="store_true", help="for the question and the answer, in every layer, each picked key "
                "is scored twice: at its true position and, through the same K/V rotated next to the question, at a near "
                "one; nothing is added to the prompt or the cache, nothing is moved")
ap.add_argument("--move-from", type=int, default=0, help="--move: layers below this one keep the true positions")
ap.add_argument("--move-to", type=int, default=1000, help="--move: layers from this one on keep the true positions")
ap.add_argument("--random", action="store_true", help="control: pick random blocks (same budget) instead of the best")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
LAYERS = [int(x) for x in a.layers.split(",")]
HEAD = tok.encode("\n\n[Excerpts from the text above that may be relevant]\n")
SEP = tok.encode("\n...\n")
NL = {i for i in range(len(tok)) if "\n" in tok.decode([i])}                 # --snap: line-break tokens
SPANS = json.load(open(a.spans_file)) if a.spans_file else {}                       # --spans-file: oracle spans
TAILTXT = tok.encode("\n[End of excerpts]\n\n")


def scores(qs, K, qs_pos, n):
    """Attention that each context key 0..n-1 receives from the question tokens, per --agg: mean over heads and LAYERS,
    max over the heads of LAYERS (per question token), or mean over the retrieval heads (--heads) in their own layers."""
    s = torch.zeros(n, device="cuda")
    rep = se.H // se.Hk
    kpos = torch.arange(n, device="cuda")
    if a.agg == "retr":
        rh = torch.tensor(json.load(open(a.heads))["query_head"]) > a.retr_thr          # [L, H]
        layers = sorted({int(l) for l in torch.nonzero(rh)[:, 0]})
    else:
        layers = LAYERS
    for l in layers:
        q = se.rope(qs[l], qs_pos).float()                                   # [T, H, D]
        k = se.rope(K[l][:n], kpos).float()                                  # [n, Hk, D]
        mx = torch.zeros(q.shape[0], n, device="cuda")
        for g in range(se.Hk):
            hs = list(range(g * rep, (g + 1) * rep))
            if a.agg == "retr":
                hs = [h for h in hs if rh[l, h]]
                if not hs:
                    continue
            p = torch.softmax(torch.einsum("trd,nd->trn", q[:, hs], k[:, g]) * se.D ** -0.5, -1)
            if a.agg == "max":
                mx = torch.maximum(mx, p.amax(1))
            else:
                s += p.sum(1).mean(0)
            del p
        if a.agg == "max":
            s += mx.mean(0)
    return s / s.sum()


NEUTRAL = None                                                               # --sel rel: neutral request token ids


def block_attn(q, k, qpos, n, agg):
    """Per query row, attention over context keys 0..n-1 aggregated over heads (agg "max" or "mean"), summed into
    blocks of --block tokens -> [T, nb]."""
    rep = se.H // se.Hk
    qr = se.rope(q, qpos).float()
    kr = se.rope(k[:n], torch.arange(n, device="cuda")).float()
    out = torch.zeros(q.shape[0], n, device="cuda")
    for g in range(se.Hk):
        p = torch.softmax(torch.einsum("trd,nd->rtn", qr[:, g * rep:(g + 1) * rep], kr[:, g]) * se.D ** -0.5, -1)
        out = torch.maximum(out, p.amax(0)) if agg == "max" else out + p.sum(0) / se.H
        del p
    nb = n // a.block
    return out[:, :nb * a.block].view(out.shape[0], nb, a.block).sum(-1)


BSC = [None]                                                                 # --vorder score: block scores


def pick_rel(qreal, qneu, k, q0, n):
    """Spans of the blocks whose combined relative score reaches --rel-thr (see --sel)."""
    A = block_attn(qreal[-16:], k, torch.arange(q0 - 16, q0, device="cuda"), n, "max").sum(0)
    C = block_attn(qreal, k, torch.arange(q0 - qreal.shape[0], q0, device="cuda"), n, "mean").sum(0)
    N = block_attn(qneu, k, torch.arange(n, n + qneu.shape[0], device="cuda"), n, "mean").sum(0)
    C = C / C.sum() - N / N.sum()
    sink = (a.sink + a.block - 1) // a.block
    A[:sink] = -float("inf"); C[:sink] = -float("inf")
    sc = torch.maximum(A / A.max(), C / C.max())
    BSC[0] = sc
    spans = []
    for b in torch.nonzero(sc >= a.rel_thr).squeeze(1).tolist():
        lo, hi = max(b * a.block - a.sel_pad, a.sink), min((b + 1) * a.block + a.sel_pad, n)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return spans


def pick(s, n):
    """Spans [(start, end)] covering the best blocks, widened by pad, merged, within the budget, in text order."""
    nb = n // a.block
    b = s[:nb * a.block].view(nb, a.block).sum(1)
    b[: (a.sink + a.block - 1) // a.block] = -1
    if a.random:                                                             # control: random order, sink still excluded
        g = torch.Generator(device=b.device).manual_seed(SEED)
        b = torch.where(b < 0, b, torch.rand(nb, device=b.device, generator=g))
    if a.pad == 0 and not a.random:                                          # no widening: top blocks, consecutive runs
        top = torch.topk(b, min(a.budget // a.block, nb)).indices
        top = torch.sort(top[b[top] >= 0]).values.tolist()
        spans = []
        for i in top:
            if spans and spans[-1][1] == i * a.block:
                spans[-1] = (spans[-1][0], (i + 1) * a.block)
            else:
                spans.append((i * a.block, (i + 1) * a.block))
        return spans
    spans, used = [], 0
    for i in torch.argsort(b, descending=True).tolist():
        lo, hi = max(i * a.block - a.pad, a.sink), min((i + 1) * a.block + a.pad, n)
        merged = []
        for s0, s1 in spans:                                                 # merge with overlapping spans
            if s0 <= hi and lo <= s1:
                lo, hi = min(lo, s0), max(hi, s1)
            else:
                merged.append((s0, s1))
        cand = merged + [(lo, hi)]
        size = sum(e - s0 for s0, e in cand)
        if size > a.budget:
            if used:
                break
            continue
        spans, used = cand, size
    return sorted(spans)


STATS = {}
VPOS = None                                                                  # --move: position of every cache slot


def attend_move(q, k, v, qpos, remap=True):
    """Plain causal attention (cache order), RoPE at the virtual positions VPOS in the layers where remap is set."""
    if not remap:
        return se.fa(se.rope(q, qpos), se.rope(k, torch.arange(k.shape[0], device="cuda")), v)[0]
    return se.fa(se.rope(q, VPOS[qpos]), se.rope(k, VPOS[:k.shape[0]]), v)[0]


KBIAS = None                                                                 # --bias: logit added to every cache slot


HEADS = None                                                                 # --focus: [layers, H] 0/1 mask
CALL = [0]                                                                   # attend calls, to know the layer


def attend_bias(q, k, v, qpos, remap=True):
    """Causal attention with a per-key additive logit KBIAS, folded in as X = 32 extra head dims (q gets 1, k gets bias/scale)."""
    qn, kn = se.rope(q, qpos), se.rope(k, torch.arange(k.shape[0], device="cuda"))
    if not remap:
        return se.fa(qn, kn, v)[0]
    nq, nk, X = q.shape[0], k.shape[0], 32
    qx = torch.zeros(nq, q.shape[1], X, dtype=q.dtype, device="cuda")
    layer = CALL[0] % len(se.model.model.layers)
    CALL[0] += 1
    qx[..., 0] = 1 if HEADS is None else HEADS[layer].to(q.dtype)[None]
    kx = torch.zeros(nk, k.shape[1], X, dtype=k.dtype, device="cuda")
    scale = (se.D + X) ** -0.5                                         # flash attention needs head dims in multiples of 32
    kx[..., 0] = (KBIAS[:nk] / scale).to(k.dtype)[:, None]             # scale * 1 * kx = bias
    qd = torch.cat((qn * (se.D ** -0.5 / scale), qx), -1)              # keep q.k at the original D^-0.5 scale
    kd = torch.cat((kn, kx), -1)
    vd = torch.cat((v, torch.zeros(nk, v.shape[1], X, dtype=v.dtype, device="cuda")), -1)
    cq = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
    ck = torch.tensor([0, nk], dtype=torch.int32, device="cuda")
    out = se.flash_attn_varlen_func(qd, kd, vd, cq, ck, nq, nk, softmax_scale=scale, causal=True)
    return out[..., :se.D].contiguous()


CAL = None                                                                   # --calib: tensors on the GPU


def corr_of(b, dist, dgrid, dref):
    """c(d) = b(min(d, dref)) - b(d) at distances dist [n] from the prior b [H, len(dgrid)] (log-linear interp)."""
    x = torch.log(dgrid.float())
    t = torch.log(dist.clamp(min=1).float())
    i = torch.searchsorted(x, t).clamp(1, len(x) - 1)
    w = ((t - x[i - 1]) / (x[i] - x[i - 1])).clamp(0, 1)
    bd = b[:, i - 1] * (1 - w) + b[:, i] * w
    j = int(torch.searchsorted(dgrid, torch.tensor(dref, device="cuda")))
    return torch.where(dist[None] > dref, b[:, j:j + 1] - bd, torch.zeros_like(bd))


def attend_calib(q, k, v, qpos, remap=True):
    """Plain causal attention with per-head temperature on q and the per-head distance correction on far keys,
    folded into 32 extra head dims: channel h % REP of the key carries head h's correction, the query selects it."""
    layer = CALL[0] % len(se.model.model.layers)
    CALL[0] += 1
    nq, nk, X, rep = q.shape[0], k.shape[0], 32, se.H // se.Hk
    if CAL["tau"] is not None:
        q = q * CAL["tau"][layer].to(q.dtype)[None, :, None]
    qn, kn = se.rope(q, qpos), se.rope(k, torch.arange(nk, device="cuda"))
    if CAL["prior"] is None:
        return se.fa(qn, kn, v)[0]
    dist = int(qpos[-1]) - torch.arange(nk, device="cuda")
    c = corr_of(CAL["prior"][layer], dist, CAL["dgrid"], CAL["dref"])     # [H, nk]
    c[:, :CAL["sink"]] = 0                                                   # the attention sink keeps its logit
    scale = (se.D + X) ** -0.5
    kx = torch.zeros(nk, se.Hk, X, dtype=k.dtype, device="cuda")
    kx[..., :rep] = (c.view(se.Hk, rep, nk).permute(2, 0, 1) / scale).to(k.dtype)
    qx = torch.zeros(nq, se.H, X, dtype=q.dtype, device="cuda")
    qx[:, torch.arange(se.H), torch.arange(se.H) % rep] = 1
    qd = torch.cat((qn * (se.D ** -0.5 / scale), qx), -1)
    kd = torch.cat((kn, kx), -1)
    vd = torch.cat((v, torch.zeros(nk, v.shape[1], X, dtype=v.dtype, device="cuda")), -1)
    cq = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
    ck = torch.tensor([0, nk], dtype=torch.int32, device="cuda")
    return se.flash_attn_varlen_func(qd, kd, vd, cq, ck, nq, nk, softmax_scale=scale, causal=True)[..., :se.D].contiguous()


SEL = None                                                                   # --dual: (cache indices, near positions)


def attend_dual(q, k, v, qpos, remap=True):
    """Plain causal attention plus, in front of the keys, the picked keys rotated at their near positions (same V):
    an evidence token's weight becomes exp(s_far) + exp(s_near); the original causal visibility is unchanged."""
    qn, kn = se.rope(q, qpos), se.rope(k, torch.arange(k.shape[0], device="cuda"))
    if not remap or SEL is None or SEL[0].numel() == 0:
        return se.fa(qn, kn, v)[0]
    idx, pos = SEL
    ks = se.rope(k[idx], pos)
    return se.fa(qn, torch.cat((ks, kn)), torch.cat((v[idx], v)))[0]


def virtual_positions(spans, qs0, cap):
    """Spans packed in text order (gap of len(SEP) between them) so that the last one ends right before qs0. Every other
    slot keeps its true position: the question and its neighbourhood are untouched, the spans share positions with the
    last context tokens (RoPE only sees distances)."""
    vpos = torch.arange(cap, device="cuda")
    if a.vpos == "perm":
        moved = torch.zeros(qs0, dtype=torch.bool, device="cuda")
        for s0, s1 in spans:
            moved[s0:s1] = True
        n_keep = int((~moved).sum())
        vpos[:qs0][~moved] = torch.arange(n_keep, device="cuda")
        vpos[:qs0][moved] = torch.arange(n_keep, qs0, device="cuda")
        return vpos
    if a.vpos == "scaleperm":                                                # scale targets, ranked -> no collisions
        tgt = torch.arange(qs0, device="cuda", dtype=torch.float64)
        nxt = qs0 + len(SEP)
        for s0, s1 in sorted(spans, reverse=True):
            end = min(qs0 - int(a.vscale * (qs0 - s1)), nxt - len(SEP))
            tgt[s0:s1] = end - (s1 - s0) + 0.5 + 1e-7 * torch.arange(s1 - s0, device="cuda")  # one block
            nxt = end - (s1 - s0)
        vpos[:qs0][torch.argsort(tgt, stable=True)] = torch.arange(qs0, device="cuda")
        return vpos
    if a.vpos == "scale":                                                    # geometry kept, distances shrunk
        nxt = qs0 + len(SEP)
        for s0, s1 in sorted(spans, reverse=True):
            end = min(qs0 - int(a.vscale * (qs0 - s1)), nxt - len(SEP))
            vpos[s0:s1] = end - (s1 - s0) + torch.arange(s1 - s0, device="cuda")
            nxt = end - (s1 - s0)
        return vpos
    if a.vorder == "score":                                                  # weakest first, strongest last
        spans = sorted(spans, key=SPAN_KEY)
    gap = a.vgap or len(SEP)
    total = sum(s1 - s0 for s0, s1 in spans) + gap * max(len(spans) - 1, 0)
    off = qs0 - total
    for s0, s1 in spans:
        vpos[s0:s1] = off + torch.arange(s1 - s0, device="cuda")
        off += s1 - s0 + gap
    return vpos


def decode(logits, p, K, V, max_new):
    """Greedy decode from logits at cache position p; returns tokens and their mean log-probability."""
    out, lp = [], []
    for _ in range(max_new):
        t = int(logits.argmax())
        if t in se.STOP:
            break
        out.append(t)
        lp.append(float(torch.log_softmax(logits.float(), -1)[t]))
        logits = se.forward([t], p, K, V)
        p += 1
    return out, sum(lp) / max(len(lp), 1)


def forced_lp(logits, p, K, V, toks):
    """Mean log-probability of toks fed after logits at cache position p (teacher forcing, one token at a time)."""
    lp = []
    for t in toks:
        lp.append(float(torch.log_softmax(logits.float(), -1)[t]))
        logits = se.forward([t], p, K, V)
        p += 1
    return sum(lp) / max(len(lp), 1)


@torch.no_grad()
def generate(r):
    global SEED, VPOS, KBIAS
    SEED = sum(map(ord, r["id"]))
    ids, max_new = r["prompt_ids"], r["max_new_tokens"]
    qs0 = len(ids) - a.ins                                                   # excerpts go before the last INS tokens
    ctx, question = ids[:qs0], ids[qs0:]
    cap = len(ids) + a.budget + 64 * len(SEP) + 2 * a.pad * (a.budget // a.block + 1) + 256 + 2 * max_new
    K = [torch.empty(cap, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in se.model.model.layers]
    V = [torch.empty_like(k) for k in K]
    for s in range(0, len(ctx), a.chunk):
        se.forward(ctx[s:s + a.chunk], s, K, V)
    captured, orig = [], se.attend

    def grab(q, k, v, qpos, remap=True):
        captured.append(q.clone())
        return orig(q, k, v, qpos, remap)
    se.attend = grab
    logits = se.forward(question, qs0, K, V)                                 # pass 1: the question reads the context
    se.attend = orig
    q0 = len(ids) - a.qlen                                                   # the last QLEN tokens score the context
    L = len(se.model.model.layers)
    qs, qpos = [c[-a.qlen:] for c in captured], torch.arange(q0, len(ids), device="cuda")
    if a.draft:                                                              # plain answer, its queries kept
        captured.clear()
        se.attend = grab
        try:
            dense, lp_d = decode(logits, len(ids), K, V, max_new)
        finally:
            se.attend = orig
        nd = len(captured) // L
        if nd:
            qs = [torch.cat([qs[l]] + captured[l::L][:nd]) for l in range(L)]
            qpos = torch.cat((qpos, torch.arange(len(ids), len(ids) + nd, device="cuda")))
    sc = scores(qs, K, qpos, qs0)
    spans = pick(sc, qs0)
    if a.sel == "rel":                                                       # neutral request at the same positions
        global NEUTRAL
        if NEUTRAL is None:
            NEUTRAL = tok.encode("<|im_end|>\n<|im_start|>user\nPlease answer the question about the text above."
                                 "<|im_end|>\n<|im_start|>assistant\n")
        neu = []

        def grab_n(q, k, v, qpos, remap=True):
            neu.append(q.clone())
            return orig(q, k, v, qpos, remap)
        se.attend = grab_n
        se.forward(NEUTRAL, qs0, K, V)
        se.attend = orig
        logits = se.forward(question, qs0, K, V)                             # restore the question's cache
        l = a.sel_layer
        spans = pick_rel(captured[l], neu[l], K[l], len(ids), qs0)
    if a.spans_file:
        spans = [tuple(x) for x in SPANS[r["id"]]]
    if a.snap:                                                               # whole lines only, overlaps merged
        merged = []
        for s0, s1 in spans:                                                 # an edge with no line break within
            lo, hi = s0, s1                                                  # --snap-max tokens stays where it was
            while lo > a.sink and s0 - lo < a.snap_max and ctx[lo - 1] not in NL:
                lo -= 1
            while hi < qs0 and hi - s1 < a.snap_max and ctx[hi - 1] not in NL:
                hi += 1
            s0 = lo if lo == a.sink or ctx[lo - 1] in NL else s0
            s1 = hi if hi == qs0 or ctx[hi - 1] in NL else s1
            if merged and s0 <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], s1))
            else:
                merged.append((s0, s1))
        spans = merged
    global SPAN_KEY
    if BSC[0] is not None and a.sel == "rel":
        bs = BSC[0]
        def SPAN_KEY(sp):
            lo = min(sp[0] // a.block, len(bs) - 1)
            return float(bs[lo:max((sp[1] + a.block - 1) // a.block, lo + 1)].max())
    else:
        SPAN_KEY = lambda sp: float(sc[sp[0]:sp[1]].sum())
    STATS.clear()
    STATS["share"] = float(sum(sc[s0:s1].sum() for s0, s1 in spans) / sc[a.sink:].sum()) if spans else 0.0
    STATS["frac"] = sum(s1 - s0 for s0, s1 in spans) / max(qs0 - a.sink, 1)  # share expected if attention were flat
    nb = qs0 // a.block                                                      # how spread the question's reading is
    bm = sc[:nb * a.block].view(nb, a.block).sum(1)[(a.sink + a.block - 1) // a.block:]
    pm = bm / bm.sum()
    STATS["entropy"] = float(-(pm * pm.clamp(min=1e-12).log()).sum() / math.log(len(pm)))   # 1 = flat
    cs = pm.sort(descending=True).values.cumsum(0)
    STATS["blocks50"] = int((cs < 0.5).sum()) + 1                              # blocks holding half the mass
    STATS["blocks90"] = int((cs < 0.9).sum()) + 1
    STATS["nblocks"] = len(pm)
    if a.fuse:                                                               # product of the two next-token distributions
        Kd = [k[:len(ids) + max_new + 8].clone() for k in K]                 # plain branch keeps the pass-1 cache
        Vd = [v[:len(ids) + max_new + 8].clone() for v in V]
        ex = list(HEAD)
        for i, (s0, s1) in enumerate(spans):
            ex += (SEP if i else []) + ctx[s0:s1]
        ex += TAILTXT
        new = ex + question
        for s in range(0, len(new), a.chunk):
            lc = se.forward(new[s:s + a.chunk], qs0 + s, K, V)
        ld_, pd, pc, out, agree = logits, len(ids), qs0 + len(new), [], 0
        for _ in range(max_new):
            a_d, a_c = torch.log_softmax(ld_.float(), -1), torch.log_softmax(lc.float(), -1)
            t = int((a_d + a_c).argmax())
            agree += int(a_d.argmax()) == int(a_c.argmax())
            if t in se.STOP:
                break
            out.append(t)
            ld_ = se.forward([t], pd, Kd, Vd)
            lc = se.forward([t], pc, K, V)
            pd += 1
            pc += 1
        STATS["agree_steps"], STATS["steps"] = agree, len(out) + 1
        del Kd, Vd
        return out, spans, tok.decode(ex)
    if a.select:                                                             # plain answer first, from the pass-1 cache
        if not a.draft:
            dense, lp_d = decode(logits, len(ids), K, V, max_new)
        STATS["lp_dense"] = lp_d
        STATS["pred_dense"] = tok.decode(dense, skip_special_tokens=True).strip()
    extext = tok.decode([t for s0, s1 in spans for t in ctx[s0:s1]])
    if a.bias or a.move or a.focus or a.calib or a.dual:                     # pass 2: question again, other attention
        if a.dual:
            global SEL
            idx = torch.tensor([t for s0, s1 in spans for t in range(s0, s1)], dtype=torch.long, device="cuda")
            SEL = (idx, virtual_positions(spans, qs0, cap)[idx])
            se.attend = attend_dual
        elif a.calib:
            global CAL
            if CAL is None:
                cf = json.load(open(a.calib))
                CAL = {"dgrid": torch.tensor(cf["d_grid"], device="cuda"), "dref": cf["d_ref"], "sink": cf.get("sink", 64),
                       "prior": torch.tensor(cf["prior"], device="cuda") if a.calib_mode != "tau" else None,
                       "tau": torch.tensor(cf["tau_with_corr" if a.calib_mode == "both" else "tau"], device="cuda")
                       if a.calib_mode != "corr" else None}
            CALL[0] = 0
            se.attend = attend_calib
        elif a.focus:
            global HEADS
            if HEADS is None:
                h = torch.tensor(json.load(open(a.focus))["query_head"], device="cuda")
                HEADS = (h > a.head_thr).float()
            KBIAS = torch.zeros(cap, device="cuda")
            n_sel = sum(s1 - s0 for s0, s1 in spans)
            for s0, s1 in spans:
                KBIAS[s0:s1] = math.log(qs0 / max(n_sel, 1))
            STATS["focus_bias"], STATS["focus_heads"] = math.log(qs0 / max(n_sel, 1)), int(HEADS.sum())
            CALL[0] = 0
            se.attend = attend_bias
        elif a.bias:
            nb = qs0 // a.block
            bl = sc[:nb * a.block].view(nb, a.block).sum(1)
            bl[: (a.sink + a.block - 1) // a.block] = 0
            rel = bl / bl[bl > 0].mean()
            KBIAS = torch.zeros(cap, device="cuda")
            KBIAS[:nb * a.block] = (a.bias * rel.clamp(min=1e-9).log()).clamp(0, a.bias_cap).repeat_interleave(a.block)
            se.attend = attend_bias
        else:
            VPOS = virtual_positions(spans, qs0, cap)
            se.attend, se.FROM, se.TO = attend_move, a.move_from, a.move_to
        try:
            out, lp = decode(se.forward(question, qs0, K, V), len(ids), K, V, max_new)
        finally:
            se.attend, se.FROM, se.TO = orig, 0, 1000
    elif a.gem:                                                              # pass 2: a short prompt, fresh positions
        body = [t for k, (s0, s1) in enumerate(spans) for t in (SEP if a.gem_sep and k else []) + ctx[s0:s1]]
        new = ctx[:a.sink] + body + question
        for s in range(0, len(new), a.chunk):
            logits = se.forward(new[s:s + a.chunk], s, K, V)
        out, lp = decode(logits, len(new), K, V, max_new)
        STATS["gem_len"] = len(new)
        extext = tok.decode(body)
    elif a.reorder:                                                          # pass 2: the context, reordered
        keep, moved, prev = [], [], 0                                        # every seam gets SEP, so no two cut
        for i, (s0, s1) in enumerate(spans):                                 # pieces fuse into a new item
            keep += ctx[prev:s0] + (SEP if s0 > prev else [])
            moved += (SEP if i else []) + ctx[s0:s1]
            prev = s1
        keep += ctx[prev:]
        new = keep + SEP + moved + SEP + question
        p0 = spans[0][0] if spans else qs0                                   # the cache before it is unchanged
        for s in range(p0, len(new), a.chunk):
            logits = se.forward(new[s:s + a.chunk], s, K, V)
        out, lp = decode(logits, len(new), K, V, max_new)
        STATS["reencoded"] = len(new) - p0
        extext = tok.decode(moved)
    else:
        ex = list(HEAD)
        for i, (s0, s1) in enumerate(spans):
            ex += (SEP if i else []) + ctx[s0:s1]
        ex += TAILTXT
        new = ex + question                                                  # pass 2: overwrite the cache from qs0 on
        if a.hide_from is not None:                                          # excerpts plain, question blind to them
            # in layers hide_from .. hide_to - 1 (KBIAS is global, see the top of generate)
            for s in range(0, len(ex), a.chunk):
                se.forward(ex[s:s + a.chunk], qs0 + s, K, V)
            KBIAS = torch.zeros(cap, device="cuda")
            KBIAS[qs0:qs0 + len(ex)] = -1e4
            se.attend, se.FROM, se.TO = attend_bias, a.hide_from, a.hide_to
            CALL[0] = 0
            try:
                logits = se.forward(question, qs0 + len(ex), K, V)
                out, lp = decode(logits, qs0 + len(new), K, V, max_new)
            finally:
                se.attend, se.FROM, se.TO = orig, 0, 1000
        else:
            for s in range(0, len(new), a.chunk):
                logits = se.forward(new[s:s + a.chunk], qs0 + s, K, V)
        if a.hide_from is None:
            out, lp = decode(logits, qs0 + len(new), K, V, max_new)
        extext = tok.decode(ex)
        if a.select:                                                         # the plain answer, read with the excerpts
            STATS["lp_dense_in_recall"] = forced_lp(logits, qs0 + len(new), K, V, dense)
    STATS["lp_recall"] = lp
    if a.select and not (a.bias or a.move or a.focus or a.calib or a.reorder or a.gem or a.dual):                              # the recalled answer, read without them
        lg = se.forward(question, qs0, K, V)                                 # restores the plain cache from qs0 on
        STATS["lp_recall_in_dense"] = forced_lp(lg, len(ids), K, V, out)
    if a.select:
        STATS["pred_recall"] = tok.decode(out, skip_special_tokens=True).strip()
        STATS["chose"] = "recall" if lp > STATS["lp_dense"] else "dense"
        if STATS["chose"] == "dense":
            out = dense
    return out, spans, extext


if __name__ == "__main__":
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
        out, spans, extext = generate(r)
        pred = tok.decode(out, skip_special_tokens=True)
        ans = r.get("answers") or r.get("outputs") or []
        rec = {"id": r["id"], "task": r["task"], "pred": pred.strip(), "len": len(r["prompt_ids"]), "spans": spans,
               "answer_in_excerpts": [str(x) in extext for x in ans] if isinstance(ans, list) else None,
               "mask_in_excerpts": "$$MASK$$" in extext, "budget": a.budget, "layers": a.layers, "agg": a.agg, "snap": a.snap,
               "qlen": a.qlen, "ins": a.ins, "random": a.random, "move": a.move,
               "move_layers": [a.move_from, a.move_to], "bias": a.bias, "fuse": a.fuse, "focus": a.focus, "head_thr": a.head_thr, "draft": a.draft, "calib": a.calib, "calib_mode": a.calib_mode, "reorder": a.reorder, "dual": a.dual, "sel": a.sel, "rel_thr": a.rel_thr, "hide": [a.hide_from, a.hide_to], "gem": a.gem, "block": a.block, "pad": a.pad, "gem_sep": a.gem_sep, "vpos": a.vpos, "vorder": a.vorder, "vscale": a.vscale, "vgap": a.vgap, "spans_file": a.spans_file, **STATS}
        print(rec["id"], repr(rec["pred"][:80]), rec["answer_in_excerpts"], rec["mask_in_excerpts"], flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
