"""How many trailing prompt tokens should read the context to pick the excerpts? One prefill per prompt; the last 128
tokens are run once with their queries captured, then the picks of scripts/recall_eval.py are made from the last k of
them for every k in --ks. Logged per k: whether each answer (and LongDialogue's $$MASK$$) falls inside the picks. The
answers are used only for this after-the-fact check.

    python scripts/qlen_probe.py --prompts data/infbench_ld50.jsonl --out results/selfext/qlen_ld50.jsonl
"""
import argparse
import json
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--tasks", default="")
ap.add_argument("--lengths", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--ks", default="1,4,8,16,32,64,128")
ap.add_argument("--out", required=True)
p = ap.parse_args()
sys.argv = ["x", "--prompts", "none", "--out", "/dev/null"]
sys.path.insert(0, "scripts")
import recall_eval as re_  # noqa: E402
se, tok = re_.se, re_.tok
KS = [int(x) for x in p.ks.split(",")]
T = max(KS)

done = {json.loads(l)["id"] for l in open(p.out)} if os.path.exists(p.out) else set()
tasks = set(p.tasks.split(",")) if p.tasks else None
lens = set(p.lengths.split(",")) if p.lengths else None
n = 0
for line in open(p.prompts):
    r = json.loads(line)
    if (tasks and r["task"].split("@")[0] not in tasks) or (lens and r["task"].split("@")[-1] not in lens):
        continue
    n += 1
    if p.limit and n > p.limit:
        break
    if r["id"] in done:
        continue
    ids = r["prompt_ids"]
    qs0 = len(ids) - T
    K = [torch.empty(len(ids), se.Hk, se.D, dtype=torch.bfloat16, device="cuda") for _ in se.model.model.layers]
    V = [torch.empty_like(k) for k in K]
    with torch.no_grad():
        for s in range(0, qs0, re_.a.chunk):
            se.forward(ids[s:min(s + re_.a.chunk, qs0)], s, K, V)
        captured, orig = [], se.attend

        def grab(q, k, v, qpos, remap=True):
            captured.append(q.clone())
            return orig(q, k, v, qpos, remap)
        se.attend = grab
        se.forward(ids[qs0:], qs0, K, V)
        se.attend = orig
        rec = {"id": r["id"], "task": r["task"], "len": len(ids)}
        ans = [str(x) for x in (r.get("answers") or r.get("outputs") or [])]
        for k in KS:
            lo = len(ids) - k                                                    # context = everything before the k
            sc = re_.scores([c[T - k:] for c in captured], K, torch.arange(lo, len(ids), device="cuda"), lo)
            spans = re_.pick(sc, lo)
            text = tok.decode([t for s0, s1 in spans for t in ids[s0:s1]])
            rec[f"k{k}"] = {"ans_in": [x in text for x in ans], "mask_in": "$$MASK$$" in text}
    print(r["id"], {k: (v["ans_in"], v["mask_in"]) for k, v in rec.items() if k.startswith("k")}, flush=True)
    with open(p.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    del K, V
