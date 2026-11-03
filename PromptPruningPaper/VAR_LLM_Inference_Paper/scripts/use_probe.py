"""Where does the model lose evidence it attends to? A probe at the decision token of LongDialogue.

For each prompt the dense answer is teacher-forced up to the first character name it commits to (the decision
position P, the token where the name starts). At P, in every layer:
  - logit lens: log-prob of the gold name's first token and of the best non-gold token (final norm + lm_head on the
    residual stream after the layer),
  - direct logit attribution of every head and every MLP to  logit(gold) - logit(rival)  (final-norm scale taken from
    the last residual),
  - attention mass of every head on: the tokens of gold-name mentions near a $$MASK$$ (within --near tokens), gold-name
    mentions elsewhere, mentions of the (wrong) name the dense answer gave, and the $$MASK$$ tokens of the context.
The rival is the highest-logit first token at P that is not a gold variant, so the same quantities exist whether the
dense answer is right or wrong. Gold variants: every alias as given, Title-case and UPPER, with the leading space the
prefix implies.

    python scripts/use_probe.py --prompts data/infbench_ld200.jsonl --dense results/selfext/ld200_g1w0.jsonl \
        --out results/selfext/useprobe/ld200
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--dense", required=True, help="jsonl with the dense answers (id, pred)")
ap.add_argument("--near", type=int, default=2048, help="a gold mention this close to a $$MASK$$ counts as near")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--moved", default="", help="jsonl of a recall_eval --move run (spans, pred_recall): probe THAT answer "
                "with the spans' keys at the packed virtual positions of --move (question + answer only, all layers)")
ap.add_argument("--rival-from", default="", help="jsonl whose answer names the 'rival' mention set (e.g. the dense "
                "answers, so a moved run is measured on the name dense gave)")
ap.add_argument("--ins", type=int, default=128, help="--moved: the question region (as recall_eval --ins)")
ap.add_argument("--short-w", type=int, default=0, help="probe the SHORT prompt of ld_move_probe --mode short (instruction "
                "+ the +-W window around the MASK snapped to lines + question) instead of the full one")
ap.add_argument("--ids", default="", help="file with one prompt id per line: probe only these")
ap.add_argument("--reverse", action="store_true", help="last prompt first (a second worker on the same list)")
ap.add_argument("--out", required=True, help="directory, one npz per prompt")
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
sys.path.insert(0, "scripts/analysis")
from tokmap import char_index  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
NL_TOK = {i for i in range(len(tok)) if "\n" in tok.decode([i])}
os.makedirs(a.out, exist_ok=True)
L = len(se.model.model.layers)
WU = se.model.lm_head.weight                                                 # [vocab, hidden]
NORM_W = se.model.model.norm.weight
EPS = se.model.model.norm.variance_epsilon


def name_start(pred):
    """Char offset where the first committed name starts (as scripts/analysis/ld_first.py reads it), or None."""
    m = re.search(r"\*\*([^*]+)\*\*", pred)
    if m:
        return m.start(1)
    m = re.search(r"(?:is likely|is probably|appears to be|is)\s*:?\s*([A-Za-z][A-Za-z .'-]{0,40})", pred)
    return m.start(1) if m else None


def mentions(text, ends, names):
    """Token indices of every case-insensitive whole-word mention of any of names."""
    idx = set()
    for n in names:
        for m in re.finditer(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", text, re.I):
            t0 = int(np.searchsorted(ends, m.start(), side="right"))
            t1 = int(np.searchsorted(ends, m.end(), side="left")) + 1
            idx.update(range(t0, t1))
    return sorted(idx)


def lens(x):
    """Final norm + lm_head of residual rows x [n, hidden] -> log-probs [n, vocab] (float)."""
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * NORM_W.float()
    return torch.log_softmax(xn @ WU.float().T, -1)


def short_ids(ids):
    """ld_move_probe --mode short: instruction line + window around the script's MASK (snapped to lines) + question."""
    NLs = NL_TOK
    text, ends = char_index(tok, ids)
    hits = [int(np.searchsorted(ends, m.start(), side="right")) for m in re.finditer(re.escape("$$MASK$$"), text)]
    m = [h for h in hits if 300 < h < len(ids) - 300][0]
    q0 = len(ids) - 128
    lo, hi = max(m - a.short_w, 200), min(m + a.short_w, q0)
    while lo > 0 and ids[lo - 1] not in NLs:
        lo -= 1
    while hi < len(ids) and ids[hi - 1] not in NLs:
        hi += 1
    txt = tok.decode(ids[:400]); c = txt.find("guess who that character is.") + len("guess who that character is.")
    h = next(k for k in range(1, 400) if len(tok.decode(ids[:k])) >= c)
    return ids[:h] + tok.encode("\n\n") + ids[lo:hi] + ids[q0:]


SEP = tok.encode("\n...\n")


def vpositions(spans, qs0, cap):
    """recall_eval.virtual_positions (pack): spans in text order, gap len(SEP), the last ending right before qs0."""
    vpos = torch.arange(cap, device="cuda")
    off = qs0 - sum(s1 - s0 for s0, s1 in spans) - len(SEP) * max(len(spans) - 1, 0)
    for s0, s1 in spans:
        vpos[s0:s1] = off + torch.arange(s1 - s0, device="cuda")
        off += s1 - s0 + len(SEP)
    return vpos


@torch.no_grad()
def probe(r, pred, spans=None):
    ids = r["prompt_ids"]
    if a.short_w:
        ids = short_ids(ids)
    off = name_start(pred)
    if off is None:
        return None
    prefix = pred[:off]
    lead = " " if prefix.endswith(" ") else ""
    pre_ids = tok.encode(prefix.rstrip(" ")) if lead else tok.encode(prefix)
    names = [n for n in r["answers"]]
    variants = {v for n in names for v in (n, n.title(), n.upper())}
    gold_toks = sorted({tok.encode(lead + v)[0] for v in variants})
    n_ctx = len(ids)
    cap = n_ctx + len(pre_ids) + 8
    K = [torch.empty(cap, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(k) for k in K]
    qs0 = n_ctx - a.ins
    for s in range(0, qs0, 8192):
        se.forward(ids[s:min(s + 8192, qs0)], s, K, V)
    vpos = vpositions(spans, qs0, cap) if spans else torch.arange(cap, device="cuda")
    text, ends = char_index(tok, ids)                                        # exact (tokmap.py)
    mask_tok = [i for i in mentions(text, ends, ["$$MASK$$"]) if i < n_ctx - 256]   # not the question's own
    gold_all = mentions(text, ends, names)
    near = [i for i in gold_all if any(abs(i - m) <= a.near for m in mask_tok)]
    far = sorted(set(gold_all) - set(near))
    # the forced answer prefix, recording the last row
    seq = ids[qs0:] + pre_ids                                                # question + forced answer prefix
    start = qs0
    P = start + len(seq) - 1
    pos = torch.arange(start, start + len(seq), device="cuda")
    x = se.model.model.embed_tokens(torch.tensor(seq, device="cuda"))
    resid, head_out, mlp_out, attn_q = [], [], [], []
    end = start + len(seq)
    for li, Ly in enumerate(se.model.model.layers):
        at = Ly.self_attn
        h = Ly.input_layernorm(x)
        q, K[li][start:end], V[li][start:end] = se.qkv(at, h)
        o = se.fa(se.rope(q, vpos[pos]), se.rope(K[li][:end], vpos[:end]), V[li][:end])[0]
        attn_q.append(se.rope(q[-1:], vpos[pos[-1:]])[0])                   # [H, D] at P
        W = at.o_proj.weight.view(-1, se.H, se.D)                            # [hidden, H, D]
        head_out.append(torch.einsum("khd,hd->hk", W.float(), o[-1].float()))  # [H, hidden], no bias
        x = x + at.o_proj(o.reshape(-1, se.H * se.D))
        m = Ly.mlp(Ly.post_attention_layernorm(x))
        mlp_out.append(m[-1].float())
        x = x + m
        resid.append(x[-1].float())
    final = lens(x[-1:])[0]
    gmask = torch.zeros_like(final, dtype=torch.bool)
    gmask[gold_toks] = True
    rival = int(final.masked_fill(gmask, -float("inf")).argmax())
    gold_lp = float(torch.logsumexp(final[gold_toks], 0))
    LZ = lens(torch.stack(resid))                                            # [L, vocab]
    lens_gold = torch.logsumexp(LZ[:, gold_toks], 1).cpu().numpy()
    lens_rival = LZ[:, rival].cpu().numpy()
    best_gold = gold_toks[int(final[gold_toks].argmax())]
    u = (WU[best_gold].float() - WU[rival].float()) * NORM_W.float()
    scale = float(torch.rsqrt(resid[-1].pow(2).mean() + EPS))
    dla_head = (torch.stack(head_out) @ u * scale).cpu().numpy()             # [L, H]
    dla_mlp = (torch.stack(mlp_out) @ u * scale).cpu().numpy()               # [L]
    # attention mass at P on the token sets
    rep = se.H // se.Hk
    sets = {"near": near, "far": far, "rival": [], "mask": mask_tok, "rival_near": [], "rival_far": []}
    rp = RIVAL.get(r["id"], pred)
    ro = name_start(rp) or 0
    said = re.sub(r"[^A-Za-z ]", " ", rp[ro:ro + 40]).split()                 # the name that answer gave
    if said and not any(w.upper() in (n.upper() for n in names) for w in said[:2]):
        sets["rival"] = [i for i in mentions(text, ends, [said[0]]) if i < n_ctx - 256]
        sets["rival_near"] = [i for i in sets["rival"] if any(abs(i - m) <= a.near for m in mask_tok)]
        sets["rival_far"] = sorted(set(sets["rival"]) - set(sets["rival_near"]))
    mass = {k: np.zeros((L, se.H), np.float32) for k in sets}
    for li in range(L):
        kr = se.rope(K[li][:end], vpos[:end])
        for g in range(se.Hk):
            p = torch.softmax(attn_q[li][g * rep:(g + 1) * rep].float() @ kr[:, g].float().T * se.D ** -0.5, -1)
            for k, s in sets.items():
                if s:
                    mass[k][li, g * rep:(g + 1) * rep] = p[:, s].sum(1).cpu().numpy()
        del kr
    return dict(lens_gold=lens_gold, lens_rival=lens_rival, dla_head=dla_head, dla_mlp=dla_mlp,
                gold_lp=gold_lp, rival_lp=float(final[rival]), rival=rival, gold_toks=np.array(gold_toks),
                n_near=len(near), n_far=len(far), n_rival=len(sets["rival"]), n_mask=len(mask_tok),
                P=P, **{"mass_" + k: v for k, v in mass.items()})


dense = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(a.dense)}
RIVAL = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(a.rival_from)} if a.rival_from else {}
MOVED = {}
for f in filter(None, a.moved.split(",")):
    for l in open(f):
        x = json.loads(l); MOVED[x["id"]] = x
if MOVED:
    dense = {i: MOVED[i]["pred_recall"] for i in dense if i in MOVED}
rows = [json.loads(l) for l in open(a.prompts)]
rows = [r for r in rows if r["id"] in dense][: a.limit or None]
if a.ids:
    keep = set(open(a.ids).read().split())
    rows = [r for r in rows if r["id"] in keep]
rows = rows[::-1] if a.reverse else rows
for r in rows:
    f = f"{a.out}/{r['id']}.npz"
    if os.path.exists(f):
        continue
    res = probe(r, dense[r["id"]], [tuple(x) for x in MOVED[r["id"]]["spans"]] if MOVED else None)
    if res is None:
        print(r["id"], "no name found", flush=True)
        continue
    np.savez(f, **res)
    print(r["id"], f"gold {res['gold_lp']:.2f} rival {tok.decode([res['rival']])!r} {res['rival_lp']:.2f}",
          f"near {res['n_near']} far {res['n_far']} mask {res['n_mask']}", flush=True)
    torch.cuda.empty_cache()
