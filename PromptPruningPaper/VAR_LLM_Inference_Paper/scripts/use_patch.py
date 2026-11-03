"""Where is the "who is masked" decision formed? Activation patching between the dense run and the --move run.

Both runs share the context cache (--move only changes the RoPE positions seen by the question and answer rows), and
both are fed the same tokens: question + the DENSE answer forced up to the name. At the decision token (the last row)
we read  margin = log p(gold first token) - log p(dense's first token).  Then, for every layer l, the dense run is
repeated with the residual stream after layer l replaced by the --move run's, either
  - "last": only at the decision row, or
  - "all":  at every question + answer row,
and the margin is read again. The first layer from which "all" no longer recovers the --move margin is where the
question has finished reading what --move changed.

    python scripts/use_patch.py --moved results/selfext/ld50_move_one_retr_snap.jsonl,... --ids-from cells \
        --out results/selfext/useprobe/patch_ld200.jsonl
"""
import argparse
import json
import re
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="data/infbench_ld200.jsonl")
ap.add_argument("--dense", default="results/selfext/ld200_g1w0.jsonl")
ap.add_argument("--moved", required=True, help="comma list of recall_eval --move jsonl (spans, pred_recall)")
ap.add_argument("--which", default="rescued", choices=["rescued", "both_right", "all"])
ap.add_argument("--ins", type=int, default=128)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(se.a.model)
lf = {"__file__": "scripts/analysis/ld_first.py"}
exec(open("scripts/analysis/ld_first.py").read().split("names = sys.argv")[0], lf)
L = len(se.model.model.layers)
SEP = tok.encode("\n...\n")


def name_start(pred):
    m = re.search(r"\*\*([^*]+)\*\*", pred)
    if m:
        return m.start(1)
    m = re.search(r"(?:is likely|is probably|appears to be|is)\s*:?\s*([A-Za-z][A-Za-z .'-]{0,40})", pred)
    return m.start(1) if m else None


def vpositions(spans, qs0, cap):
    vpos = torch.arange(cap, device="cuda")
    off = qs0 - sum(s1 - s0 for s0, s1 in spans) - len(SEP) * max(len(spans) - 1, 0)
    for s0, s1 in spans:
        vpos[s0:s1] = off + torch.arange(s1 - s0, device="cuda")
        off += s1 - s0 + len(SEP)
    return vpos


@torch.no_grad()
def run(seq, start, K, V, vpos, patch=None, rows=None):
    """Question + answer rows through all layers; returns (residual after each layer [L, n, hidden], last logits).
    patch = (layer, saved residuals [L, n, hidden]) replaces the residual after that layer on `rows`."""
    pos = torch.arange(start, start + len(seq), device="cuda")
    x = se.model.model.embed_tokens(torch.tensor(seq, device="cuda"))
    end = start + len(seq)
    res = []
    for li, Ly in enumerate(se.model.model.layers):
        at = Ly.self_attn
        h = Ly.input_layernorm(x)
        q, K[li][start:end], V[li][start:end] = se.qkv(at, h)
        o = se.fa(se.rope(q, vpos[pos]), se.rope(K[li][:end], vpos[:end]), V[li][:end])[0]
        x = x + at.o_proj(o.reshape(-1, se.H * se.D))
        x = x + Ly.mlp(Ly.post_attention_layernorm(x))
        if patch is not None and patch[0] == li:
            x = x.clone()
            x[rows] = patch[1][li][rows]
        res.append(x)
    logits = se.model.lm_head(se.model.model.norm(x[-1:]))[0].float()
    return torch.stack(res), torch.log_softmax(logits, -1)


dense = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(a.dense)}
moved = {}
for f in a.moved.split(","):
    for l in open(f):
        x = json.loads(l); moved[x["id"]] = x
rows_in = [json.loads(l) for l in open(a.prompts)]
done = set()
try:
    done = {json.loads(l)["id"] for l in open(a.out)}
except FileNotFoundError:
    pass
fo = open(a.out, "a")
for r in rows_in:
    i = r["id"]
    if i in done or i not in moved:
        continue
    ok_d = lf["sc_first"](dense[i], r["answers"]) > 0
    ok_m = lf["sc_first"](moved[i]["pred_recall"], r["answers"]) > 0
    if a.which == "rescued" and not (ok_m and not ok_d):
        continue
    if a.which == "both_right" and not (ok_m and ok_d):
        continue
    pred = dense[i]; off = name_start(pred)
    if off is None:
        continue
    prefix = pred[:off]; lead = " " if prefix.endswith(" ") else ""
    pre_ids = tok.encode(prefix.rstrip(" ")) if lead else tok.encode(prefix)
    gold = sorted({tok.encode(lead + v)[0] for n in r["answers"] for v in (n, n.title(), n.upper())})
    said = re.sub(r"[^A-Za-z ]", " ", pred[off:off + 40]).split()
    rival = tok.encode(lead + said[0])[0] if said else None
    ids = r["prompt_ids"]; qs0 = len(ids) - a.ins
    seq = ids[qs0:] + pre_ids
    cap = len(ids) + len(pre_ids) + 8
    K = [torch.empty(cap, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(k) for k in K]
    for s in range(0, qs0, 8192):
        se.forward(ids[s:min(s + 8192, qs0)], s, K, V)
    ident = torch.arange(cap, device="cuda")
    vmov = vpositions([tuple(x) for x in moved[i]["spans"]], qs0, cap)

    def margin(lp):
        g = float(torch.logsumexp(lp[gold], 0))
        return g - float(lp[rival]) if rival is not None and rival not in gold else g
    rd, lpd = run(seq, qs0, K, V, ident)
    rm, lpm = run(seq, qs0, K, V, vmov)
    out = {"id": i, "dense_right": ok_d, "move_right": ok_m, "m_dense": margin(lpd), "m_move": margin(lpm),
           "last": [], "all": [], "rev_all": []}
    n = len(seq)
    for li in range(L):
        out["last"].append(margin(run(seq, qs0, K, V, ident, (li, rm), [n - 1])[1]))
        out["all"].append(margin(run(seq, qs0, K, V, ident, (li, rm), list(range(n)))[1]))
        out["rev_all"].append(margin(run(seq, qs0, K, V, vmov, (li, rd), list(range(n)))[1]))
    fo.write(json.dumps(out) + "\n"); fo.flush()
    print(i, f"dense {out['m_dense']:.2f} move {out['m_move']:.2f}",
          "all:", " ".join(f"{x:.1f}" for x in out["all"][::3]), flush=True)
    del K, V
    torch.cuda.empty_cache()
