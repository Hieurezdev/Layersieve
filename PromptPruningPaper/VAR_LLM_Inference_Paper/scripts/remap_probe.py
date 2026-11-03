"""Where does the last prompt token look under different distance remaps? Fast screen for rotary edits.

One dense forward (scripts/selfextend_eval.py, g = 1, w = 0) per prompt; then, for the last prompt token only, the
attention over all keys is recomputed with the key distances remapped as in selfextend_eval (--mode freq: only the
rotary dims slower than D0; floor/all: every dim) and the mass is measured on
  KV retrieval:  the gold "key": "value" pair, the pair a dense run copied instead (if wrong), and the rank of the gold
                 pair among all pairs by total mass;
  LongDialogue:  +-128 tokens around each $$MASK$$ in the script;
plus the sink (first 64 tokens) and the last 4096 tokens. Only the last token changes, so this is the query-only
version of the remap (the context is encoded as in the dense model).

    python scripts/remap_probe.py --task kv --n 20 --out results/selfext/remap_probe_kv.json
"""
import argparse
import json
import re
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["kv", "ld"], required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--dense-preds", default="results/bench/infbench/dense.jsonl")
ap.add_argument("--variants", default="", help="comma list: run only these (dense always included)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null", "--chunk", "4096"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
UU = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
L = len(se.model.model.layers)
# name -> (D0, g, slow dims only[, logit scale]) or a bare logit scale (sharper softmax, no remap)
VARIANTS = {"dense": None, "freq4k_g8": (4096, 8, True), "freq32k_g4": (32768, 4, True), "freq16k_g2": (16384, 2, True),
            "freq4k_clamp": (4096, 1e6, True), "freq32k_clamp": (32768, 1e6, True), "all4k_g8": (4096, 8, False),
            "all4k_clamp": (4096, 1e6, False), "scale1.15": 1.15, "scale1.3": 1.3, "scale1.5": 1.5,
            "freq4k_g8_scale1.3": (4096, 8, True, 1.3),
            "fastoff4k": ("fast", 4096, 0.0), "fasthalf4k": ("fast", 4096, 0.5), "fastoff32k": ("fast", 32768, 0.0),
            "fastoff1k": ("fast", 1024, 0.0)}
# ("fast", D0, lam): keys farther than D0 keep only lam times the rotary dims faster than D0 (their contribution there
# is a pseudo-random phase with mean ~0, scripts/rope_logit_profile.py)
ONLY = None
dense_pred = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(a.dense_preds)}


def offsets(ids, text):
    ends = torch.tensor([len(tok.decode([t])) for t in ids]).cumsum(0).tolist()
    return list(zip([0] + ends[:-1], ends)) if ends[-1] == len(text) else None


def tokmask(off, c0, c1, n):
    m = torch.zeros(n, dtype=torch.bool)
    for t, (s, e) in enumerate(off):
        if s < c1 and e > c0:
            m[t] = True
    return m


def targets(r):
    """dict name -> bool mask over prompt tokens; for kv also the per-pair token index list."""
    ids = r["prompt_ids"]; n = len(ids); text = tok.decode(ids); off = offsets(ids, text)
    if off is None:
        return None
    starts = torch.tensor([s for s, _ in off])
    def span(c0, c1):
        lo = int(torch.searchsorted(starts, c0, right=True)) - 1
        hi = int(torch.searchsorted(starts, c1))
        m = torch.zeros(n, dtype=torch.bool); m[max(lo, 0):hi] = True
        return m
    T = {}
    if a.task == "kv":
        pairs = [(m.start(), m.end(), m.group(2)) for m in re.finditer(r'"(%s)": "(%s)"' % (UU, UU), text)]
        gi = next((i for i, p in enumerate(pairs) if p[2] == r["answers"][0]), None)
        if gi is None:
            return None
        pid = torch.full((n,), -1, dtype=torch.long)
        for i, (c0, c1, _) in enumerate(pairs):
            lo = int(torch.searchsorted(starts, c0, right=True)) - 1; hi = int(torch.searchsorted(starts, c1))
            pid[lo:hi] = i
        T["gold"] = pid == gi
        hit = [u for u in re.findall(UU, dense_pred.get(r["id"], "")) if u != r["answers"][0]]
        wi = next((i for i, p in enumerate(pairs) if hit and p[2] == hit[0]), None)
        if wi is not None:
            T["dense_wrong"] = pid == wi
        T["_pid"], T["_gi"], T["_np"] = pid, gi, len(pairs)
    else:
        body_end = text.rfind("The name that has been replaced")
        ms = [m.start() for m in re.finditer(re.escape("$$MASK$$"), text) if m.start() < body_end and m.start() > 300]
        if not ms:
            return None
        m = torch.zeros(n, dtype=torch.bool)
        for c in ms:
            t = int(torch.searchsorted(starts, c, right=True)) - 1
            m[max(t - 128, 0):t + 128] = True
        T["mask_win"] = m
        T["_dist"] = n - 1 - int(torch.searchsorted(starts, ms[-1], right=True))
    T["sink"] = torch.zeros(n, dtype=torch.bool); T["sink"][:64] = True
    T["near4k"] = torch.zeros(n, dtype=torch.bool); T["near4k"][-4096:] = True
    return T


captured = {}
orig = se.attend


def attend(q, k, v, qpos, remap=True):
    captured.setdefault("q", []).append(q[-1:].clone())
    return orig(q, k, v, qpos, remap)


task = "kv_retrieval" if a.task == "kv" else "longdialogue_qa_eng"
src = "data/infbench_kv86.jsonl" if a.task == "kv" else "data/infbench_ld200.jsonl"
res = []
for line in open(src):
    if len(res) >= a.n:
        break
    r = json.loads(line)
    if r["task"] != task:
        continue
    T = targets(r)
    if T is None:
        continue
    ids = r["prompt_ids"]; n = len(ids)
    K = [torch.empty(n, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(x) for x in K]
    last = ((n - 1) // 4096) * 4096
    rec = {"id": r["id"], "n": n, "dense_right": r["answers"][0].lower() in dense_pred.get(r["id"], "").lower(),
           "variants": {}}
    if "_dist" in T:
        rec["mask_dist"] = T["_dist"]
    if "_gi" in T:
        rec["gold_pairs_from_end"] = T["_np"] - T["_gi"]
    masks = {k: v.cuda() for k, v in T.items() if not k.startswith("_")}
    with torch.no_grad():
        for s in range(0, n, 4096):
            if s == last:
                se.attend, captured = attend, {}
            se.forward(ids[s:s + 4096], s, K, V)
            se.attend = orig
        pos = torch.arange(n, device="cuda")
        d = (n - 1 - pos).double()
        qpos = torch.tensor([n - 1], device="cuda")
        for name, var in VARIANTS.items():
            if a.variants and name != "dense" and name not in a.variants.split(","):
                continue
            out = {k: [] for k in masks}
            pairmass = torch.zeros(T.get("_np", 1), device="cuda", dtype=torch.float64)
            for l in range(L):
                if isinstance(var, tuple) and var[0] == "fast":
                    _, D0, lam = var
                    slow = (2 * torch.pi / se.INV > D0).repeat(2)                  # neox: pair i = dims i, i + D/2
                    kr = se.rope(K[l], pos).float().repeat_interleave(se.H // se.Hk, 1)
                    qr = se.rope(captured["q"][l], qpos)[0].float()
                    full = torch.einsum("hd,nhd->nh", qr, kr)
                    damp = torch.einsum("hd,nhd->nh", qr * torch.where(slow, 1.0, lam), kr)
                    lg = torch.where((d > D0)[:, None], damp, full) * se.D ** -0.5
                    p = torch.softmax(lg, 0)
                    for k, m in masks.items():
                        out[k].append(p[m].sum(0).mean().item())
                    if "_pid" in T:
                        pid = T["_pid"].cuda(); ok = pid >= 0
                        pairmass.index_add_(0, pid[ok], p[ok].sum(1).double())
                    continue
                beta = var if isinstance(var, float) else (var[3] if var and len(var) > 3 else 1.0)
                if var is None or isinstance(var, float):
                    kr = se.rope(K[l], pos)
                    qr = se.rope(captured["q"][l], qpos)
                else:
                    D0, g, slow = var[:3]
                    dd = torch.where(d < D0, d, D0 + (d - D0) / g)
                    if g >= 1e5:
                        dd = torch.clamp(d, max=D0)
                    kp = (n - 1) - dd                                           # key position giving distance dd
                    if slow:
                        se.LOW = 2 * torch.pi / se.INV > D0
                        kr = se.rope(K[l], pos, kp)
                    else:
                        kr = se.rope(K[l], kp)
                    qr = se.rope(captured["q"][l], qpos)
                lg = torch.einsum("hd,nhd->nh", qr[0].float(), kr.float().repeat_interleave(se.H // se.Hk, 1)) * se.D ** -0.5 * beta
                p = torch.softmax(lg, 0)                                        # [n, H]
                for k, m in masks.items():
                    out[k].append(p[m].sum(0).mean().item())                    # mean over heads of the mass
                if "_pid" in T:
                    pid = T["_pid"].cuda(); ok = pid >= 0
                    pairmass.index_add_(0, pid[ok], p[ok].sum(1).double())
            if "_pid" in T:
                out["gold_rank"] = int((pairmass > pairmass[T["_gi"]]).sum())
                out["gold_rank_ge14"] = None
            rec["variants"][name] = out
    res.append(rec)
    v = rec["variants"]
    key = "gold" if a.task == "kv" else "mask_win"
    print(r["id"], {nm: round(sum(x[key][14:]) / 14, 4) for nm, x in v.items()},
          {nm: x.get("gold_rank") for nm, x in v.items()} if a.task == "kv" else rec.get("mask_dist"), flush=True)
    json.dump(res, open(a.out, "w"))
    del K, V
    torch.cuda.empty_cache()
