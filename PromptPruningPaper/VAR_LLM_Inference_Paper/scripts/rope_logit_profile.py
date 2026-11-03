"""How the query at the end of a long prompt scores keys as a function of distance, split by rotary frequency band.

For every layer and query head, the pre-softmax logit q . R(d) k of the last prompt token against every prompt key is
split into the part from the rotary dims slower than --wavelength (the "low" band) and the rest ("high"), then averaged
in distance bins. Dense forward, Qwen2.5 (scripts/selfextend_eval.py with g = 1, w = 0).

    python scripts/rope_logit_profile.py --prompts data/infbench_ld200.jsonl --n 4 --out results/selfext/profile_ld.json
"""
import argparse
import json
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--wavelength", type=int, default=32768)
ap.add_argument("--min-len", type=int, default=100000)
ap.add_argument("--skip-sink", type=int, default=128, help="leave the first tokens (attention sink) out of the bins")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = ["x", "--prompts", a.prompts, "--out", "/dev/null", "--chunk", "4096"]
sys.path.insert(0, "scripts")
import selfextend_eval as se  # noqa: E402

LOW = 2 * torch.pi / se.INV > a.wavelength
EDGES = [0, 64, 512, 4096, 16384, 32768, 49152, 65536, 98304, 1 << 20]
L = len(se.model.model.layers)
acc = {b: torch.zeros(L, se.H, len(EDGES) - 1, dtype=torch.float64) for b in ("low", "high", "all", "mass")}
cnt = torch.zeros(len(EDGES) - 1, dtype=torch.float64)
used = []
captured = {}
orig = se.attend


def attend(q, k, v, qpos, remap=True):
    captured.setdefault("q", []).append(q[-1:].clone())                       # layer order within the last chunk
    return orig(q, k, v, qpos, remap)


for line in open(a.prompts):
    r = json.loads(line)
    ids = r["prompt_ids"]
    if len(ids) < a.min_len or len(used) >= a.n:
        continue
    n = len(ids)
    K = [torch.empty(n, se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    V = [torch.empty_like(x) for x in K]
    last = ((n - 1) // 4096) * 4096
    with torch.no_grad():
        for s in range(0, n, 4096):
            if s == last:
                se.attend, captured = attend, {}
            se.forward(ids[s:s + 4096], s, K, V)
            se.attend = orig
        d = (n - 1) - torch.arange(n, device="cuda")                           # distance of every key to the query
        b = torch.bucketize(d, torch.tensor(EDGES[1:-1], device="cuda"), right=True)
        b[:a.skip_sink] = len(EDGES) - 1                                       # sink -> overflow bin, dropped below
        for l in range(L):
            q = se.rope(captured["q"][l], torch.tensor([n - 1], device="cuda"))[0].float()   # [H, D]
            k = se.rope(K[l], torch.arange(n, device="cuda")).float()                       # [n, Hk, D]
            k = k.repeat_interleave(se.H // se.Hk, 1)                                       # [n, H, D]
            half = se.D // 2
            prod = (q[None] * k) * se.D ** -0.5                                             # [n, H, D]
            pair = prod[..., :half] + prod[..., half:]                                      # per rotary pair
            parts = {"low": pair[..., LOW].sum(-1), "high": pair[..., ~LOW].sum(-1), "all": pair.sum(-1)}
            parts["mass"] = torch.softmax(parts["all"], 0)                                  # attention share
            for name, x in parts.items():                                                   # x [n, H]
                acc[name][l] += torch.zeros(len(EDGES), se.H, device="cuda", dtype=torch.float64).index_add_(
                    0, b, x.double())[:-1].T.cpu()
        cnt += torch.bincount(b, minlength=len(EDGES))[:-1].double().cpu()
    used.append(r["id"])
    print(r["id"], n, flush=True)
    del K, V

prof = {name: (x / (cnt[None, None] if name != "mass" else len(used))).tolist() for name, x in acc.items()}
json.dump({"prompts": used, "edges": EDGES, "low_pairs": int(LOW.sum()), "wavelength": a.wavelength, "mean_logit": prof},
          open(a.out, "w"))
for name in ("low", "high", "all", "mass"):
    m = torch.tensor(prof[name])
    print(name, "mean over layers/heads per distance bin:", [round(x, 2) for x in m.mean((0, 1)).tolist()])
    print(name, "layers>=17:", [round(x, 2) for x in m[17:].mean((0, 1)).tolist()])
