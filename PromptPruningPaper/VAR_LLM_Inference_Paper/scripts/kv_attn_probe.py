"""Why does the model copy the value of a wrong key in KV retrieval? Score the gold pair against the wrong pair.

Dense forward (scripts/selfextend_eval.py, g = 1, w = 0) over an InfiniteBench kv_retrieval prompt; for the last prompt
token (the one that emits the first value token) and every layer/head, take the attention logits over the tokens of
the gold "key": "value" pair and of the pair whose value the model wrote instead, split into the slow rotary dims
(wavelength > --wavelength) and the fast ones. Also recompute the slow part as if every key beyond --wavelength sat at
distance --wavelength (the clamp of scripts/run_remap7.sh), to see how much of the margin the distance penalty makes.

    python scripts/kv_attn_probe.py --preds results/bench/infbench/dense.jsonl --out results/selfext/kv_probe.json
"""
import argparse
import json
import re
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--wavelength", type=int, default=4096)
ap.add_argument("--n-right", type=int, default=6, help="also probe this many correctly answered prompts (vs a near pair)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null", "--chunk", "4096"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
LOW = 2 * torch.pi / se.INV > a.wavelength
UU = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
data = {json.loads(l)["id"]: json.loads(l) for l in open("data/infbench_kv86.jsonl")}
L = len(se.model.model.layers)


def spans(r, pred):
    """Token ranges of the gold pair and of the pair whose value was predicted (or a pair ~10 pairs nearer if right)."""
    text = tok.decode(r["prompt_ids"])
    ends = torch.tensor([len(tok.decode([t])) for t in r["prompt_ids"]]).cumsum(0).tolist()
    if ends[-1] != len(text):                                              # non-ASCII split across tokens
        return None
    off = list(zip([0] + ends[:-1], ends))
    pairs = [(m.start(), m.end(), m.group(1), m.group(2)) for m in re.finditer(r'"(%s)": "(%s)"' % (UU, UU), text)]
    gi = next(i for i, p in enumerate(pairs) if p[3] == r["answers"][0])
    hit = [u for u in re.findall(UU, pred) if any(p[3] == u for p in pairs)]
    wi = next((i for i, p in enumerate(pairs) if hit and p[3] == hit[0]), None)
    if wi is None or wi == gi:
        if r["answers"][0] not in pred:
            return None
        wi = min(gi + 10, len(pairs) - 1)                                  # correct answer: compare with a nearer pair

    def tokrange(c0, c1):
        return [t for t, (s, e) in enumerate(off) if s < c1 and e > c0]
    return tokrange(*pairs[gi][:2]), tokrange(*pairs[wi][:2]), len(pairs) - gi, len(pairs) - wi


captured = {}
orig = se.attend


def attend(q, k, v, qpos, remap=True):
    captured.setdefault("q", []).append(q[-1:].clone())
    return orig(q, k, v, qpos, remap)


res, nright = [], 0
for line in open(a.preds):
    p = json.loads(line)
    if p["id"] not in data:
        continue
    r = data[p["id"]]
    right = r["answers"][0] in p["pred"]
    if right and nright >= a.n_right:
        continue
    sp = spans(r, p["pred"])
    if sp is None:
        continue
    nright += right
    gold, wrong, gd, wd = sp
    ids = r["prompt_ids"]; n = len(ids)
    K = [torch.empty(n, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(x) for x in K]
    last = ((n - 1) // 4096) * 4096
    with torch.no_grad():
        for s in range(0, n, 4096):
            if s == last:
                se.attend, captured = attend, {}
            se.forward(ids[s:s + 4096], s, K, V)
            se.attend = orig
        pos = torch.arange(n, device="cuda")
        clamp = torch.clamp(pos, min=n - 1 - a.wavelength).double()            # slow dims: far keys at distance w
        rec = {"id": r["id"], "right": right, "gold_pairs_from_end": gd, "other_pairs_from_end": wd, "layers": []}
        for l in range(L):
            q = captured["q"][l]                                                # [1, H, D] unrotated
            qr = se.rope(q, torch.tensor([n - 1], device="cuda"))[0].float()
            out = {}
            for name, kp_low in (("dense", None), ("clamp", clamp)):
                k = se.rope(K[l], pos, kp_low).float().repeat_interleave(se.H // se.Hk, 1)  # [n, H, D]
                prod = (qr[None] * k) * se.D ** -0.5
                pair = prod[..., :se.D // 2] + prod[..., se.D // 2:]
                lo, hi = pair[..., LOW].sum(-1), pair[..., ~LOW].sum(-1)        # [n, H]
                lg = lo + hi
                lse = torch.logsumexp(lg, 0)
                out[name] = {
                    "mass_gold": torch.exp(lg[gold] - lse).sum(0).tolist(),     # per head
                    "mass_other": torch.exp(lg[wrong] - lse).sum(0).tolist(),
                    "max_gold": lg[gold].max(0).values.tolist(), "max_other": lg[wrong].max(0).values.tolist(),
                    "low_gold": lo[gold].mean(0).tolist(), "low_other": lo[wrong].mean(0).tolist(),
                    "high_gold": hi[gold].mean(0).tolist(), "high_other": hi[wrong].mean(0).tolist()}
            rec["layers"].append(out)
    res.append(rec)
    print(r["id"], "right" if right else "wrong", gd, wd, flush=True)
    del K, V
    torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"))
