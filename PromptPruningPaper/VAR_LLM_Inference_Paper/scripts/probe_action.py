"""Action probe (direction A): at each pruning point, how much does a candidate cut hurt, given what is still alive,
and does an oracle that picks the cut size per layer from these labels beat simple fixed schedules at equal cost?
HF reference path (inmodel_prune/prune.py) with LayerSieve's unit dropping (inmodel_prune/units.py), batch size 1.

Reference answer y = the dense greedy answer (prompts are ones the dense model answers correctly). y is appended to the
prompt, so one causal forward gives the teacher-forced loss L_y = mean_t -log p(y_t | x, y_<t); prompt tokens never see
y, so pruning decisions (scored on the prompt part only) are those of the real pipeline.
State z_l = hidden states entering layer l (alive prompt tokens + y). From z_l:
  branch(D)  drop the units D before layer l, run layers l..L-1 with no further pruning, return L_y.
  L_keep(z_l) = branch({}) = the branch chosen at the previous decision point (computed once, never re-run).
Candidate actions at layer l: drop a fraction f in --actions of the current prunable tokens, as whole lowest-scoring
units (nested sets, same ranking as LayerSieve). Only the cut at l differs between branches: earlier layers keep the KV
they computed (decode_ref off), later layers do not prune.

Trajectories per prompt (all replayed with a KV cache for free greedy generation, so correctness is real):
  dense
  fixed  l0:rho  LayerSieve schedule (drop rho of the prunable tokens before every layer >= l0)
  local  tau     oracle: at each layer the largest f with L_cut - L_keep <= tau
  global eps     oracle: at each layer the largest f with L_cut - L_dense <= eps
Cost = prompt tokens entering each layer, summed over layers / (L * T) (token x layer fraction, tracks TTFT).
Outputs (--out dir): traj.jsonl (one row per prompt x trajectory), branches.jsonl (one row per evaluated branch with
pre-cut features: attention mass of D, rank stability, alive fraction, evidence in D).
Usage: python scripts/probe_action.py --per-task 5 --out results/actprobe/pilot
"""
import argparse
import json
import os
import re
import string
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--prompts", default=os.path.join(ROOT, "data/hard.jsonl"))
ap.add_argument("--ids", default=os.path.join(ROOT, "results/scaling/ids/7b1m.txt"))
ap.add_argument("--per-task", type=int, default=0, help="only the first k ids of every task (pilot)")
ap.add_argument("--tasks", default="")
ap.add_argument("--first-layer", type=int, default=8, help="first decision point (all fixed l0 must be >= this)")
ap.add_argument("--actions", default="0.05,0.1,0.2,0.35,0.5", help="candidate drop fractions (0 = keep is implicit)")
ap.add_argument("--local", default="0.003,0.01,0.03,0.1", help="oracle thresholds on L_cut - L_keep")
ap.add_argument("--global", dest="glob", default="0.01,0.03,0.1,0.3", help="oracle thresholds on L_cut - L_dense")
ap.add_argument("--fixed", default="8:0.05,8:0.1,8:0.2,8:0.35,11:0.1,11:0.2,11:0.35,14:0.12,14:0.25,14:0.5,"
                                   "17:0.12,17:0.25,17:0.5,20:0.25,20:0.5,8:0.25,8:0.5,11:0.5,11:1,14:1,17:1,20:1")
ap.add_argument("--label-fixed", default="17:0.12", help="also label every action along these fixed trajectories")
ap.add_argument("--policies", default="", help="closed-loop policies name:threshold (inmodel_prune/harm.py): at each "
                "layer the largest action whose predicted harm <= threshold; 'mass' = attention mass of D (no learning)")
ap.add_argument("--harm-model", default="", help="JSON from scripts/train_harm.py (cross-fitted by prompt)")
ap.add_argument("--only-policies", action="store_true", help="skip fixed / oracle trajectories (dense + policies)")
ap.add_argument("--no-gen", action="store_true", help="skip free generation of non-dense trajectories")
ap.add_argument("--n-obs", type=int, default=128)
ap.add_argument("--n-sink", type=int, default=128)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer, DynamicCache  # noqa: E402

from inmodel_prune import harm as harm_mod, units  # noqa: E402
from inmodel_prune.prune import _layer_call, _rope, load_model  # noqa: E402

MAX_TOKENS = {"kv": 100, "vt": 50}
ACTIONS = [float(x) for x in a.actions.split(",")]
FIXED = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in a.fixed.split(",") if s]
LABEL_FIXED = {(int(s.split(":")[0]), float(s.split(":")[1])) for s in a.label_fixed.split(",") if s}
if a.only_policies:
    FIXED, LABEL_FIXED, a.local, a.glob = [], set(), "", ""
assert all(l0 >= a.first_layer for l0, _ in FIXED)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def correct(task, pred, answers):
    if task in ("kv", "vt"):
        return any(x in pred for x in answers)
    return any(norm(x) in norm(pred) for x in answers)


tok = AutoTokenizer.from_pretrained(a.model)
model = load_model(a.model)
mm = model.model
L = len(mm.layers)
calls = [_layer_call(l) for l in mm.layers]
boundary = units.boundary_token_ids(tok).cuda()
harm = harm_mod.HarmModel(a.harm_model) if a.harm_model else harm_mod.HarmModel(None)
eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}


@torch.no_grad()
def importance(layer, h, cos, sin, n_p, obs_chunk=32):
    """LayerSieve importance of the n_p prompt tokens at the front of h (attention from the last n_obs prompt tokens,
    causal softmax over the alive prompt tokens): max over heads (the score LayerSieve uses) and mean over heads."""
    attn, n_obs = layer.self_attn, a.n_obs
    D = attn.head_dim
    x = layer.input_layernorm(h[:, :n_p])
    k = _rope(attn.k_proj(x).view(1, n_p, -1, D).transpose(1, 2), cos[:, :n_p], sin[:, :n_p])[0].float()
    q = attn.q_proj(x[:, -n_obs:]).view(1, n_obs, -1, D).transpose(1, 2)
    q = _rope(q, cos[:, n_p - n_obs:n_p], sin[:, n_p - n_obs:n_p])[0].float()
    Hq, Hk = q.shape[0], k.shape[0]
    G = Hq // Hk
    key = torch.arange(n_p, device=h.device)
    imax = torch.zeros(n_p, device=h.device)
    imean = torch.zeros(n_p, device=h.device)
    for r0 in range(0, n_obs, obs_chunk):
        r1 = min(r0 + obs_chunk, n_obs)
        s = torch.bmm(q[:, r0:r1].reshape(Hk, G * (r1 - r0), D), k.transpose(1, 2)) * attn.scaling
        s = s.view(Hq, r1 - r0, n_p)
        rows = n_p - n_obs + torch.arange(r0, r1, device=h.device)
        s.masked_fill_(key[None, None] > rows[None, :, None], float("-inf"))
        p = torch.softmax(s, -1)
        imax += p.amax(0).sum(0)
        imean += p.mean(0).sum(0)
        del s, p
    return imax, imean


def unit_pct(imp, inv, U, prun_u):
    """Per-unit score (max token importance) and its percentile among the prunable units (1 = top)."""
    sc = torch.full((U,), float("-inf"), device=imp.device).scatter_reduce(0, inv, imp, reduce="amax")
    pct = torch.full((U,), float("nan"), device=imp.device)
    pu = torch.nonzero(prun_u).squeeze(1)
    order = torch.argsort(sc[pu])
    pct[pu[order]] = torch.arange(len(pu), device=imp.device, dtype=torch.float) / max(len(pu) - 1, 1)
    return sc, pct


@torch.no_grad()
def loss_of(h, y):
    m = y.shape[0]
    logits = model.lm_head(mm.norm(h[:, -m - 1:-1])).float()[0]
    return float(F.cross_entropy(logits, y))


@torch.no_grad()
def branch(h, cos, sin, li, keep, y):
    h, cos, sin = h[:, keep], cos[:, keep], sin[:, keep]
    for j in range(li, L):
        h = calls[j](h, None, cos, sin)
    return loss_of(h, y)


@torch.no_grad()
def generate(ids, drops, max_new):
    """Greedy generation with a KV cache; drops[l] = prompt positions removed before layer l."""
    T = ids.shape[1]
    h = mm.embed_tokens(ids)
    pos = torch.arange(T, device=ids.device)
    cos, sin = mm.rotary_emb(h, pos[None])
    cache = DynamicCache()
    for li in range(L):
        if li in drops and len(drops[li]):
            keep = ~torch.isin(pos, drops[li])
            h, cos, sin, pos = h[:, keep], cos[:, keep], sin[:, keep], pos[keep]
        h = calls[li](h, cache, cos, sin)
    t = model.lm_head(mm.norm(h[:, -1:])).argmax(-1)
    out = [int(t)]
    for step in range(1, max_new):
        if out[-1] in eos:
            break
        p = torch.tensor([[T + step - 1]], device=ids.device)
        x = mm.embed_tokens(t)
        c1, s1 = mm.rotary_emb(x, p)
        for call in calls:
            x = call(x, cache, c1, s1)
        t = model.lm_head(mm.norm(x)).argmax(-1)
        out.append(int(t))
    if out and out[-1] in eos:
        out = out[:-1]
    return out


@torch.no_grad()
def run_prompt(r):
    ids = torch.tensor([r["prompt_ids"]], device="cuda")
    Tp = ids.shape[1]
    t0 = time.time()
    y_ids = generate(ids, {}, MAX_TOKENS.get(r["task"], 32))
    pred = tok.decode(y_ids, skip_special_tokens=True)
    base = {"id": r["id"], "task": r["task"], "len": Tp}
    if not correct(r["task"], pred, r["answers"]) or not y_ids:
        return [dict(base, traj="dense", kind="dense", ok=False, pred=pred, skipped=True)], []
    y = torch.tensor(y_ids, device="cuda")
    m = len(y_ids)
    x = torch.cat([ids, y[None]], 1)
    h = mm.embed_tokens(x)
    pos_all = torch.arange(Tp + m, device="cuda")
    cos_all, sin_all = mm.rotary_emb(h, pos_all[None])
    for li in range(a.first_layer):
        h = calls[li](h, None, cos_all, sin_all)
    state0 = h
    h_full = h
    for li in range(a.first_layer, L):
        h_full = calls[li](h_full, None, cos_all, sin_all)
    L_dense = loss_of(h_full, y)
    del h_full
    unit0 = units.unit_ids(ids[0], boundary)
    ev = torch.zeros(Tp, dtype=torch.bool, device="cuda")
    for s, e in r["evidence"]:
        ev[s:e] = True
    ev[:a.n_sink] = False
    ev[Tp - a.n_obs:] = False
    n_ev = int(ev.sum())
    memo = {}                                                                  # (hist, f) -> loss
    brows = []
    n_prun0 = Tp - a.n_sink - a.n_obs

    def walk(name, kind, param):
        h, pos, cos, sin = state0, pos_all, cos_all, sin_all
        hist, cur, prev_imp, n_eval = (), L_dense, None, 0
        drops, kept = {}, [Tp] * a.first_layer
        labelled = kind in ("local", "global") or (kind == "fixed" and param in LABEL_FIXED)
        for li in range(a.first_layer, L):
            n_p = h.shape[1] - m
            pp = pos[:n_p]
            imax, imean = importance(mm.layers[li], h, cos, sin, n_p)
            prunable = (pp >= a.n_sink) & (torch.arange(n_p, device="cuda") < n_p - a.n_obs)
            n_prun = int(prunable.sum())
            uid = unit0[pp]

            def drop_idx(f):
                nd = int(n_prun * f)
                return units.drop_units(imax, uid, prunable, nd) if nd > 0 else imax.new_zeros(0, dtype=torch.long)

            uinfo = {}

            def feats(d):
                """Pre-cut features of dropping d (indices into the current prompt tokens): everything a predictor
                can see online. Unit percentiles (this layer and the previous one) are computed once per layer."""
                if not uinfo:
                    _, inv = torch.unique(uid, return_inverse=True)
                    U = int(inv.max()) + 1
                    prun_u = torch.ones(U, dtype=torch.bool, device="cuda")
                    prun_u[inv[~prunable]] = False
                    sc, pct = unit_pct(imax, inv, U, prun_u)
                    ppct = unit_pct(prev_imp, inv, U, prun_u)[1] if prev_imp is not None else None
                    uinfo.update(inv=inv, prun_u=prun_u, sc=sc, pct=pct, ppct=ppct,
                                 med=sc[prun_u].median(), smax=imax[prunable].sum(), smean=imean[prunable].sum())
                u = uinfo
                du = torch.unique(u["inv"][d])
                row = dict(layer=li, n_tok=len(d), n_units=len(du), alive=n_prun / n_prun0,
                           mass_max=float(imax[d].sum() / u["smax"]), mass_mean=float(imean[d].sum() / u["smean"]),
                           dmax_pct=float(u["pct"][du].max()) if len(du) else None,
                           dmax_rel=float(u["sc"][du].max() / u["med"]) if len(du) else None)
                if u["ppct"] is not None and len(du):
                    row.update(prev_pct_mean=float(u["ppct"][du].mean()), prev_pct_max=float(u["ppct"][du].max()))
                return row

            def label(f):
                nonlocal n_eval
                key = (hist, f)
                if key not in memo:
                    d = drop_idx(f)
                    keep = torch.ones(h.shape[1], dtype=torch.bool, device="cuda")
                    keep[d] = False
                    memo[key] = branch(h, cos, sin, li, keep, y)
                    n_eval += 1
                    brows.append(dict(base, traj=name, hist=",".join(f"{v:g}" for v in hist), frac=f, **feats(d),
                                      loss_keep=cur, loss_cut=memo[key], loss_dense=L_dense,
                                      ev_alive=float(ev[pp].sum()) / n_ev if n_ev else None,
                                      ev_in_d=float(ev[pp[d]].sum()) / n_ev if n_ev else None))
                return memo[key]

            if kind == "dense":
                f = 0.0
            elif kind == "fixed":
                f = param[1] if li >= param[0] else 0.0
                if labelled and li >= param[0]:
                    for g in ACTIONS:
                        label(g)
            elif kind == "pred":                                               # learned / rule policy, no labels
                f = 0.0
                for g in sorted(ACTIONS):
                    d = drop_idx(g)
                    if len(d) and harm.prob(param[0], dict(frac=g, **feats(d)), r["id"]) <= param[1]:
                        f = g
            else:
                f = 0.0
                for g in sorted(ACTIONS):
                    lc = label(g)
                    if (lc - cur if kind == "local" else lc - L_dense) <= param:
                        f = g
            if f > 0:
                if labelled:
                    cur = label(f)
                d = drop_idx(f)
                if len(d):
                    drops[li] = pp[d]
                    keep = torch.ones(h.shape[1], dtype=torch.bool, device="cuda")
                    keep[d] = False
                    h, pos, cos, sin = h[:, keep], pos[keep], cos[:, keep], sin[:, keep]
                    imax = imax[keep[:n_p]]
            kept.append(h.shape[1] - m)
            hist = hist + (f,)
            prev_imp = imax
            h = calls[li](h, None, cos, sin)
        lf = loss_of(h, y)
        final = pos[:h.shape[1] - m]
        row = dict(base, traj=name, kind=kind, param=list(param) if isinstance(param, tuple) else param,
                   cost=sum(kept) / (L * Tp), final_keep=kept[-1] / Tp, kept=kept, loss=lf, loss_dense=L_dense,
                   dloss=lf - L_dense, n_eval=n_eval,
                   ev_kept=float(torch.isin(torch.nonzero(ev).squeeze(1), final).float().mean()) if n_ev else None)
        if kind == "dense":
            row.update(pred=pred, ok=True)
        elif not a.no_gen:
            g = tok.decode(generate(ids, drops, MAX_TOKENS.get(r["task"], 32)), skip_special_tokens=True)
            row.update(pred=g, ok=correct(r["task"], g, r["answers"]))
        return row

    rows = [walk("dense", "dense", None)]
    for l0, rho in FIXED:
        rows.append(walk(f"fixed_{l0}_{rho:g}", "fixed", (l0, rho)))
    for tau in [float(v) for v in a.local.split(",") if v]:
        rows.append(walk(f"local_{tau:g}", "local", tau))
    for eps in [float(v) for v in a.glob.split(",") if v]:
        rows.append(walk(f"global_{eps:g}", "global", eps))
    for pol in [v for v in a.policies.split(",") if v]:
        nm, th = pol.split(":")
        rows.append(walk(f"pred_{nm}_{th}", "pred", (nm, float(th))))
    rows[0]["time"] = time.time() - t0
    return rows, brows


os.makedirs(a.out, exist_ok=True)
tp, bp = os.path.join(a.out, "traj.jsonl"), os.path.join(a.out, "branches.jsonl")
done = {json.loads(l)["id"] for l in open(tp)} if os.path.exists(tp) else set()
only = [l.strip() for l in open(a.ids) if l.strip()] if a.ids else None
tasks = set(a.tasks.split(",")) if a.tasks else None
recs = {}
for line in open(a.prompts):
    r = json.loads(line)
    if (only is None or r["id"] in only) and (tasks is None or r["task"] in tasks):
        recs[r["id"]] = r
by_task = {}
for i in (only or list(recs)):
    if i in recs:
        by_task.setdefault(recs[i]["task"], []).append(i)
if a.per_task:
    by_task = {t: v[:a.per_task] for t, v in by_task.items()}
order = [v[k] for k in range(max(map(len, by_task.values()))) for v in by_task.values() if k < len(v)]  # interleave
for i in order:
    if i in done:
        continue
    for attempt in range(5):                                                   # the GPU may be shared: retry OOMs
        try:
            rows, brows = run_prompt(recs[i])
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(i, "OOM, retry in 120s", flush=True)
            time.sleep(120)
    else:
        continue
    with open(bp, "a") as f:
        f.writelines(json.dumps(b) + "\n" for b in brows)
    with open(tp, "a") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    print(i, f"{rows[0].get('time', 0):.0f}s", " ".join(
        f"{x['traj']}:{x.get('cost', 0):.2f}/{int(x.get('ok', 0))}" for x in rows), flush=True)
    torch.cuda.empty_cache()
