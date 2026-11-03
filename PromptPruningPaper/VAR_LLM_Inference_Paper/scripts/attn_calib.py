"""Per-head attention calibration for long contexts, measured once per model on unlabeled long text.

For every layer and query head:
  * RoPE distance prior b_h(d) = E[q . R(d) k] over the head's own queries (last NQ tokens) and a sample of keys, as a
    function of the distance d alone (keys rotated at 0, queries at d), so content-distance correlation is left out.
    The correction c_h(d) = b_h(min(d, D_REF)) - b_h(d) makes every distance beyond D_REF look like D_REF on average and
    leaves nearer keys (and the local differences between neighbouring keys) untouched.
  * temperature tau_h >= 1 such that the head's attention entropy over the full long context (logits * tau, with or
    without the correction) equals its entropy over a short context made of the last SHORT tokens alone.
Books from data/infbench_sum20.jsonl are used as plain text (no question or answer is read).

    python scripts/attn_calib.py --prompts data/infbench_sum20.jsonl --n 4 --out results/selfext/calib_7b1m.json
"""
import argparse
import json
import math
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--min-len", type=int, default=120000)
ap.add_argument("--short", type=int, default=4096, help="length of the short reference context")
ap.add_argument("--nq", type=int, default=16, help="trailing tokens whose queries are measured")
ap.add_argument("--d-ref", type=int, default=4096, help="distances beyond this get the prior of this distance")
ap.add_argument("--nk", type=int, default=4096, help="sampled keys for the distance prior")
ap.add_argument("--sink", type=int, default=64, help="leading tokens (attention sink) never corrected nor sampled")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null", "--chunk", "8192"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402

L, H, Hk, D = len(se.model.model.layers), se.H, se.Hk, se.D
REP = H // Hk
DGRID = torch.unique(torch.logspace(0, math.log10(1 << 20), 96, device="cuda").round().long())   # 1 .. 1M
TAUS = torch.linspace(1.0, 3.0, 41, device="cuda")


def capture(ids):
    """Prefill all but the last NQ tokens, run those capturing their (unrotated) queries; returns q per layer, K."""
    n = len(ids)
    K = [torch.empty(n, Hk, D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(k) for k in K]
    q0 = n - a.nq
    for s in range(0, q0, 8192):
        se.forward(ids[s:min(s + 8192, q0)], s, K, V)
    got, orig = [], se.attend

    def grab(q, k, v, qpos, remap=True):
        got.append(q.clone())
        return orig(q, k, v, qpos, remap)
    se.attend = grab
    se.forward(ids[q0:], q0, K, V)
    se.attend = orig
    del V
    return got, K


def logits(q, k, qpos, n):
    """[nq, H, n] pre-softmax logits of the queries against keys 0..n-1 (true positions)."""
    qr = se.rope(q, qpos).float()
    kr = se.rope(k[:n], torch.arange(n, device="cuda")).float()
    return torch.einsum("qhd,khd->qhk", qr, kr.repeat_interleave(REP, 1)) * D ** -0.5


def entropy(lg):
    p = torch.softmax(lg, -1)
    return -(p * p.clamp(min=1e-30).log()).sum(-1)                          # [nq, H]


def prior(q, k):
    """b(d) for d in DGRID: mean over queries and sampled keys of q . R(d) k, per head -> [H, len(DGRID)]."""
    idx = a.sink + torch.randperm(k.shape[0] - a.sink, device="cuda")[:a.nk]
    kr = se.rope(k[idx], torch.zeros(len(idx), dtype=torch.long, device="cuda")).float().repeat_interleave(REP, 1)
    out = torch.empty(H, len(DGRID), device="cuda")
    for j, d in enumerate(DGRID.tolist()):
        qr = se.rope(q, torch.full((q.shape[0],), d, device="cuda")).float()
        out[:, j] = torch.einsum("qhd,khd->h", qr, kr) / (q.shape[0] * len(idx)) * D ** -0.5
    return out


def corr_of(b, dist):
    """Correction c(d) = b(min(d, D_REF)) - b(d) at distances dist [n], from b [H, len(DGRID)] (log-linear interp)."""
    x = torch.log(DGRID.float())
    t = torch.log(dist.clamp(min=1).float())
    i = torch.searchsorted(x, t).clamp(1, len(x) - 1)
    w = ((t - x[i - 1]) / (x[i] - x[i - 1])).clamp(0, 1)
    bd = b[:, i - 1] * (1 - w) + b[:, i] * w                                 # [H, n]
    j = int(torch.searchsorted(DGRID, torch.tensor(a.d_ref, device="cuda")))
    c = torch.where(dist[None] > a.d_ref, b[:, j:j + 1] - bd, torch.zeros_like(bd))
    c[:, :a.sink] = 0                                                         # the sink keeps its logit
    return c


acc = {"b": torch.zeros(L, H, len(DGRID), device="cuda"), "h_short": torch.zeros(L, H, device="cuda"),
       "h_long": torch.zeros(L, H, len(TAUS), device="cuda"), "h_long_c": torch.zeros(L, H, len(TAUS), device="cuda")}
used = []
with torch.no_grad():
    for line in open(a.prompts):
        r = json.loads(line)
        ids = r["prompt_ids"]
        if len(ids) < a.min_len or len(used) >= a.n:
            continue
        n = len(ids)
        qs, K = capture(ids)
        qpos = torch.arange(n - a.nq, n, device="cuda")
        dist = (n - 1) - torch.arange(n - a.nq, device="cuda")              # distance of each context key (approx.)
        for l in range(L):
            b = prior(qs[l], K[l][:n - a.nq])
            acc["b"][l] += b
            lg = logits(qs[l], K[l], qpos, n)[..., :n - a.nq]               # context keys only
            c = corr_of(b, dist)                                             # [H, n - nq]
            for t, tau in enumerate(TAUS.tolist()):
                acc["h_long"][l, :, t] += entropy(lg * tau).mean(0)
                acc["h_long_c"][l, :, t] += entropy(lg * tau + c[None]).mean(0)
            del lg
        del K
        short = ids[n - a.short:]                                            # the same text, alone
        qs2, K2 = capture(short)
        qpos2 = torch.arange(a.short - a.nq, a.short, device="cuda")
        for l in range(L):
            acc["h_short"][l] += entropy(logits(qs2[l], K2[l], qpos2, a.short)[..., :a.short - a.nq]).mean(0)
        del K2
        used.append(r["id"])
        print(r["id"], n, flush=True)

k = len(used)
for v in acc.values():
    v /= k


def match(hl):
    """Smallest tau in TAUS whose long-context entropy is <= the short-context one (TAUS[-1] if none)."""
    ok = hl <= acc["h_short"][..., None]
    first = torch.where(ok.any(-1), ok.float().argmax(-1), torch.full_like(ok[..., 0], len(TAUS) - 1, dtype=torch.long))
    return TAUS[first]


tau, tau_c = match(acc["h_long"]), match(acc["h_long_c"])
json.dump({"books": used, "d_grid": DGRID.tolist(), "d_ref": a.d_ref, "short": a.short, "nq": a.nq, "sink": a.sink,
           "prior": acc["b"].tolist(), "h_short": acc["h_short"].tolist(), "h_long_tau1": acc["h_long"][..., 0].tolist(),
           "h_long_c_tau1": acc["h_long_c"][..., 0].tolist(), "tau": tau.tolist(), "tau_with_corr": tau_c.tolist()},
          open(a.out, "w"))
print("entropy short / long (nats), mean over heads:", round(float(acc["h_short"].mean()), 2),
      round(float(acc["h_long"][..., 0].mean()), 2), "| with corr", round(float(acc["h_long_c"][..., 0].mean()), 2))
for name, t in (("tau", tau), ("tau_with_corr", tau_c)):
    print(name, "heads > 1.05:", int((t > 1.05).sum()), "of", t.numel(), "| median", round(float(t.median()), 2),
          "| max", round(float(t.max()), 2), "| per layer mean", [round(x, 2) for x in t.mean(1).tolist()])
