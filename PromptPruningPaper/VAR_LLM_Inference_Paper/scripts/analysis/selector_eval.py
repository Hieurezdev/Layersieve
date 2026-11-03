"""Compare the block scores of scripts/selector_probe.py with the known evidence (analysis only).

Evidence units (token spans) per task, located in the prompt text:
  kv_retrieval     gold: the pair "<key>": "<value>" of the asked key;   distractors: none listed (all other pairs)
  niah_multikey_3  gold: the needle holding the answer;                  distractors: the other "One of the special
                                                                          magic uuids for ..." needles
  niah_multivalue  gold: the needles holding each answer value
  vt               gold: the "VAR X = ..." lines of the answer chain;     distractors: every other "VAR" line
  longdialogue     gold: +-128 tokens around the masked name in the script
Per variant: rank of the best gold block (0 = top), recall@8 / @32 of the gold units, and the number of distractor
blocks ranked above the best gold block. Contrastive variants: block share under the real question minus the share
under the neutral request, same layer and head aggregation.

    python scripts/analysis/selector_eval.py results/selfext/selprobe/vt20 [more dirs]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
PROMPTS = {}
for f in ["data/ruler_eval.jsonl", "data/infbench_kv86.jsonl", "data/infbench_ld200.jsonl"]:
    for l in open(f):
        r = json.loads(l); PROMPTS[r["id"]] = r


def char_index(ids):
    """Exact (scripts/analysis/tokmap.py); decoding tokens one by one drifted on multi-byte characters."""
    sys.path.insert(0, "scripts/analysis")
    from tokmap import char_index as _ci
    return _ci(tok, ids)


def tok_span(ends, c0, c1):
    return int(np.searchsorted(ends, c0, side="right")), int(np.searchsorted(ends, c1, side="left")) + 1


def evidence(r):
    """(gold spans, distractor spans) in token indices."""
    ids, task = r["prompt_ids"], r["task"].split("@")[0]
    text, ends = char_index(ids)
    ans = [str(x) for x in (r.get("answers") or r.get("outputs"))]
    gold, dis = [], []
    if task == "kv_retrieval":
        for m in re.finditer(r'"([0-9a-f-]{36})": "([0-9a-f-]{36})"', text):
            if m.group(2) in ans:
                gold.append(tok_span(ends, m.start(), m.end()))
    elif task in ("niah_multikey_3", "niah_multivalue"):
        for m in re.finditer(r"One of the special magic [^\n]*?(?:\n|$)", text):
            (gold if any(x in m.group(0) for x in ans) else dis).append(tok_span(ends, m.start(), m.end()))
    elif task == "vt":
        for m in re.finditer(r"VAR ([A-Z]+) = [^\n]*", text):
            (gold if m.group(1) in ans else dis).append(tok_span(ends, m.start(), m.end()))
    elif task == "longdialogue_qa_eng":
        hits = [m.start() for m in re.finditer(r"\$\$MASK\$\$", text)]
        inner = [h for h in hits if 2000 < h < len(text) - 2000]
        if inner:
            a0, _ = tok_span(ends, inner[0], inner[0] + 8)
            gold.append((max(a0 - 128, 0), a0 + 128))
    return gold, dis


def blocks_of(spans, B, nb):
    return [sorted({min(i // B, nb - 1) for i in range(s0, s1)}) for s0, s1 in spans if s0 < nb * B]


def evaluate(scores, gold_b, dis_b):
    order = np.argsort(-scores)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    g = [min(rank[b] for b in u) for u in gold_b if u]
    if not g:
        return None
    best = min(g)
    dr = [min(rank[b] for b in u) for u in dis_b if u]
    dis_above = sum(x < best for x in dr)
    return best, np.mean([x < 8 for x in g]), np.mean([x < 32 for x in g]), dis_above, sum(x < 8 for x in dr), sum(x < 32 for x in dr)


rows = defaultdict(list)
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        pid = os.path.basename(f)[:-4]
        z = np.load(f)
        r = PROMPTS[pid]; task = r["task"].split("@")[0]
        B, q0 = int(z["block"]), int(z["q0"])
        sc = {k: z[k] for k in z.files if k not in ("block", "q0")}
        nb = len(next(iter(sc.values())))
        for k in list(sc):                                                       # contrastive variants
            m = re.match(r"(L\d+)_mean_w(\d+)$", k)
            if m and f"{m.group(1)}_neutral" in sc:
                qn = sc[k] / sc[k].sum(); nn = sc[f"{m.group(1)}_neutral"] / sc[f"{m.group(1)}_neutral"].sum()
                sc[f"{m.group(1)}_contrast_w{m.group(2)}"] = qn - nn
        for k in sc:                                                             # the sink block never counts
            sc[k] = sc[k].copy(); sc[k][0] = -np.inf
        gold, dis = evidence(r)
        gb, db = blocks_of(gold, B, nb), blocks_of(dis, B, nb)
        for k, s in sc.items():
            e = evaluate(s, gb, db)
            if e:
                rows[(task, k)].append(e)

tasks = sorted({t for t, _ in rows})
for t in tasks:
    print(f"\n=== {t}  (n={len(rows[(t, 'L17_mean_w16')])})   best-gold-rank median | recall@8 | recall@32 | distractors above | distr. in top8 | top32")
    res = []
    for (tt, k), v in rows.items():
        if tt != t or "neutral" in k:
            continue
        v = np.array(v, dtype=float)
        res.append((np.median(v[:, 0]), -v[:, 1].mean(), k, v))
    for med, _, k, v in sorted(res)[:14]:
        print(f"  {k:22s} {med:7.0f} | {v[:, 1].mean():.2f} | {v[:, 2].mean():.2f} | {v[:, 3].mean():6.1f} | {v[:, 4].mean():5.1f} | {v[:, 5].mean():5.1f}")
    for k in ("L17_mean_w16", "L17_max_w128"):
        v = np.array(rows[(t, k)], dtype=float)
        print(f"  [{k} = {'recall_eval' if 'mean' in k else 'LayerSieve'}] {np.median(v[:, 0]):7.0f} | {v[:, 1].mean():.2f} | "
              f"{v[:, 2].mean():.2f} | {v[:, 3].mean():6.1f} | {v[:, 4].mean():5.1f} | {v[:, 5].mean():5.1f}")
