"""Causal test of distance on LongDialogue (an oracle diagnostic, not a method): the masked occurrence of $$MASK$$ is
located in the script, a contiguous window of about +-W tokens around it (snapped to line breaks) is cut out and put
right before the question; nothing else changes (no token added or dropped). Control: the same move for a window of
the same length at a random place away from the mask. Dense Qwen2.5 forward from scripts/selfextend_eval.py.
Other modes, on the same window:
  copy_near    a copy of the window is put before the question at the next positions (the original stays);
  copy_orig    the same copy, but its tokens keep their ORIGINAL (far) RoPE positions: duplication without nearness;
  copy_far     copy_near, but for the question and the answer, in layers >= --from-layer, the copy's keys are rotated
               back to the window's original (far) positions: re-encoded evidence read from far;
  copy_prefix  copy_near, but while the copy is encoded it sees only what preceded the window in the prompt (keys
               [0, lo)) and itself, as when the window was first encoded; the question then sees everything;
  short        a SHORT prompt: the instruction line, the window and the question only (the rest of the script is
               dropped): can the model answer at all when the evidence is all it reads?
  rerot        nothing moves or is copied: for the question and the answer only, in layers >= --from-layer, the keys of
               the window are rotated as if the window sat right before the question; rerot_random: same for a random
               window.

    python scripts/ld_move_probe.py --mode mask --prompts data/infbench_ld50.jsonl --out results/selfext/ld50_move_mask.jsonl
"""
import argparse
import json
import os
import random
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--mode", required=True, choices=["mask", "random", "none", "copy_near", "copy_orig", "rerot", "rerot_random", "copy_far", "copy_prefix", "short"])
ap.add_argument("--depth", type=int, default=0, help="mask/random: the moved window ends this many tokens before the "
                "question (0 = right before it)")
ap.add_argument("--from-layer", type=int, default=17, help="rerot: re-rotate only in layers >= this one")
ap.add_argument("--w", type=int, default=1024, help="tokens kept on each side of the mask before snapping to lines")
ap.add_argument("--qlen", type=int, default=128, help="the question: the window goes before the last QLEN tokens")
ap.add_argument("--nosnap", action="store_true", help="keep the window at exactly mask +-W (no widening to line breaks)")
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
sys.path.insert(0, "scripts/analysis")
from tokmap import char_index  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
NL = {i for i in range(len(tok)) if "\n" in tok.decode([i])}


def mask_index(ids):
    """Token index of the $$MASK$$ inside the script (not the instruction at the start, not the question at the end)."""
    text, ends = char_index(tok, ids)                                        # exact (scripts/analysis/tokmap.py)
    ends = torch.tensor(ends)
    hits, p = [], text.find("$$MASK$$")
    while p >= 0:
        hits.append(int(torch.searchsorted(ends, p, right=True)))
        p = text.find("$$MASK$$", p + 1)
    inner = [h for h in hits if 300 < h < len(ids) - 300]
    return inner[0] if inner else None


def snap(ids, lo, hi):
    """Widen [lo, hi) to start right after a line break and end right after one."""
    if a.nosnap:
        return lo, hi
    while lo > 0 and ids[lo - 1] not in NL:
        lo -= 1
    while hi < len(ids) and ids[hi - 1] not in NL:
        hi += 1
    return lo, hi


VPOS = None


def attend_vpos(q, k, v, qpos, remap=True):
    """Plain causal attention in cache order; RoPE at the positions VPOS (by cache slot) where remap is set."""
    ar = torch.arange(k.shape[0], device="cuda")
    if not remap:
        return se.fa(se.rope(q, qpos), se.rope(k, ar), v)[0]
    return se.fa(se.rope(q, VPOS[qpos]), se.rope(k, VPOS[:k.shape[0]]), v)[0]


SKIP = None                                                                      # copy_prefix: (hi, q0) hidden keys


def attend_skip(q, k, v, qpos, remap=True):
    """Causal attention that leaves out the keys in [SKIP[0], SKIP[1]) (queries come after them in cache order)."""
    ar = torch.arange(k.shape[0], device="cuda")
    qn, kn = se.rope(q, qpos), se.rope(k, ar)
    h0, h1 = SKIP
    nq = q.shape[0]
    cq = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
    ca = torch.tensor([0, h0], dtype=torch.int32, device="cuda")
    oa, la, _ = se.flash_attn_varlen_func(qn, kn[:h0], v[:h0], cq, ca, nq, h0, softmax_scale=se.D ** -0.5,
                                          causal=False, return_attn_probs=True)
    ob, lb = se.fa(qn, kn[h1:], v[h1:])                                         # the copy itself, causal
    l1, l2 = la.T[..., None], lb.T[..., None]
    m = torch.maximum(l1, l2)
    e1, e2 = (l1 - m).exp(), (l2 - m).exp()
    return ((oa.float() * e1 + ob.float() * e2) / (e1 + e2)).to(oa.dtype)


def decode(logits, p, K, V, max_new):
    out = []
    for _ in range(max_new):
        t = int(logits.argmax())
        if t in se.STOP:
            break
        out.append(t)
        logits = se.forward([t], p, K, V)
        p += 1
    return out


@torch.no_grad()
def special(ids, max_new, lo, hi, q0, mode):
    """copy_near / copy_orig / rerot on the window [lo, hi)."""
    global VPOS
    n = hi - lo
    cap = len(ids) + n + max_new + 8
    K = [torch.empty(cap, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in se.model.model.layers]
    V = [torch.empty_like(k) for k in K]
    orig = se.attend
    VPOS = torch.arange(cap, device="cuda")
    if mode == "copy_prefix":                                                    # copy encoded with its old prefix
        global SKIP
        for s in range(0, q0, a.chunk):
            se.forward(ids[s:min(s + a.chunk, q0)], s, K, V)
        SKIP = (lo, q0)
        se.attend = attend_skip
        try:
            for s in range(0, n, a.chunk):
                se.forward(ids[lo + s:min(lo + s + a.chunk, hi)], q0 + s, K, V)
        finally:
            se.attend = orig
        logits = se.forward(ids[q0:], q0 + n, K, V)
        return decode(logits, len(ids) + n, K, V, max_new)
    if mode == "copy_far":                                                       # copy encoded near, read from far
        new = ids[:q0] + ids[lo:hi]
        for s in range(0, len(new), a.chunk):
            se.forward(new[s:s + a.chunk], s, K, V)
        VPOS[q0:q0 + n] = torch.arange(lo, hi, device="cuda")
        se.attend, se.FROM, se.TO = attend_vpos, a.from_layer, 1000
        try:
            logits = se.forward(ids[q0:], q0 + n, K, V)
            return decode(logits, len(ids) + n, K, V, max_new)
        finally:
            se.attend, se.FROM, se.TO = orig, 0, 1000
    if mode.startswith("copy"):
        new = ids[:q0] + ids[lo:hi] + ids[q0:]
        if mode == "copy_orig":                                                 # copy keeps the far positions,
            VPOS[q0:q0 + n] = torch.arange(lo, hi, device="cuda")              # question and answer keep theirs
            VPOS[q0 + n:] = torch.arange(q0, cap - n, device="cuda")
            se.attend, se.FROM, se.TO = attend_vpos, 0, 1000
        try:
            for s in range(0, len(new), a.chunk):
                logits = se.forward(new[s:s + a.chunk], s, K, V)
            return decode(logits, len(new), K, V, max_new)
        finally:
            se.attend, se.FROM, se.TO = orig, 0, 1000
    for s in range(0, q0, a.chunk):                                             # rerot: context as in dense
        se.forward(ids[s:min(s + a.chunk, q0)], s, K, V)
    VPOS[lo:hi] = torch.arange(q0 - n, q0, device="cuda")                       # window just before the question
    se.attend, se.FROM, se.TO = attend_vpos, a.from_layer, 1000
    try:
        logits = se.forward(ids[q0:], q0, K, V)
        return decode(logits, len(ids), K, V, max_new)
    finally:
        se.attend, se.FROM, se.TO = orig, 0, 1000


@torch.no_grad()
def generate(ids, max_new):
    K = [torch.empty(len(ids) + max_new, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in se.model.model.layers]
    V = [torch.empty_like(k) for k in K]
    for s in range(0, len(ids), a.chunk):
        logits = se.forward(ids[s:s + a.chunk], s, K, V)
    out, p = [], len(ids)
    for _ in range(max_new):
        t = int(logits.argmax())
        if t in se.STOP:
            break
        out.append(t)
        logits = se.forward([t], p, K, V)
        p += 1
    return out


if __name__ == "__main__":
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    for line in open(a.prompts):
        r = json.loads(line)
        if r["id"] in done:
            continue
        ids = r["prompt_ids"]
        T, q0 = len(ids), len(ids) - a.qlen
        m = mask_index(ids)
        rec = {"id": r["id"], "task": r["task"], "len": T, "mode": a.mode, "mask_at": m, "from_layer": a.from_layer}
        if m is None or a.mode == "none":
            new = ids
        else:
            lo, hi = snap(ids, max(m - a.w, 200), min(m + a.w, q0))
            if a.mode in ("random", "rerot_random"):                             # same length, away from the mask
                n = hi - lo
                rng = random.Random(sum(map(ord, r["id"])))
                for _ in range(100):
                    s0 = rng.randrange(200, q0 - n)
                    if s0 + n < lo - a.w or s0 > hi + a.w:
                        break
                lo, hi = snap(ids, s0, s0 + n)
            rest = ids[:lo] + ids[hi:q0]
            at = max(len(rest) - a.depth, lo)                                    # where the window goes
            new = rest[:at] + ids[lo:hi] + rest[at:] + ids[q0:]
            rec.update(window=[lo, hi], moved_from_distance=q0 - hi, depth=len(rest) - at)
        if m is not None and a.mode == "short":
            txt = tok.decode(ids[:400]); c = txt.find("guess who that character is.") + len("guess who that character is.")
            h = next(k for k in range(1, 400) if len(tok.decode(ids[:k])) >= c)
            new = ids[:h] + tok.encode("\n\n") + ids[lo:hi] + ids[q0:]
            rec.update(short_len=len(new))
            out = generate(new, r["max_new_tokens"])
        elif m is not None and a.mode not in ("mask", "random", "none"):
            out = special(ids, r["max_new_tokens"], lo, hi, q0, a.mode.replace("_random", ""))
        else:
            assert len(new) == T and sorted(new) == sorted(ids)
            out = generate(new, r["max_new_tokens"])
        rec["pred"] = tok.decode(out, skip_special_tokens=True).strip()
        print(r["id"], rec.get("window"), rec.get("moved_from_distance"), repr(rec["pred"][:70]), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
