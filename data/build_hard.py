"""Build tokenized prompts (<= ~30k tokens, so 32k-context Qwen2.5 drafts fit) for the "hard cases" experiment:
draft-small-fails / target-succeeds cases for SpecPrefill.
  kv   synthetic InfiniteBench-style kv_retrieval (uuid JSON), 1 evidence span (the queried pair)
  vt   RULER-style variable tracking, asked forward ("value of VAR Q?") so finding hop k+1 needs hop k solved;
       evidence = every assignment of the queried chain; distractor chains with other values
  qa   LongBench hotpotqa + 2wikimqa (2-hop QA), no evidence spans
Output lines: {"id", "task", "len", "prompt_ids", "answers", "evidence": [[tok_start, tok_end), ...]}
    python data/build_hard.py --out data/hard.jsonl
"""
import argparse
import json
import random
import string
import uuid
import zipfile

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--lengths", default="8000,16000,28000", help="target context tokens for kv / vt")
ap.add_argument("--n-kv", type=int, default=40, help="per length")
ap.add_argument("--n-vt", type=int, default=40, help="per length")
ap.add_argument("--vt-hops", type=int, default=4)
ap.add_argument("--vt-chains", type=int, default=4, help="1 queried + distractors")
ap.add_argument("--n-qa", type=int, default=200, help="per LongBench dataset")
ap.add_argument("--max-len", type=int, default=30000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = random.Random(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)


def chat(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def record(rid, task, prompt, answers, ev_strings=()):
    """Tokenize; map each evidence substring (unique in the prompt) to its token span."""
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    off = enc.offset_mapping
    ev = []
    for s in ev_strings:
        c0 = prompt.index(s)
        c1 = c0 + len(s)
        t = [i for i, (x, y) in enumerate(off) if y > c0 and x < c1]
        ev.append([t[0], t[-1] + 1])
    return {"id": rid, "task": task, "len": len(enc.input_ids), "prompt_ids": enc.input_ids, "answers": answers,
            "evidence": ev}


def ntok(s):
    return len(tok(s).input_ids)


def build_kv(target, k):
    kv_tok = ntok(json.dumps({str(uuid.UUID(int=1)): str(uuid.UUID(int=2))})) - 2
    pairs = [(str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128))))
             for _ in range(max(2, target // kv_tok))]
    key, val = pairs[rng.randrange(len(pairs))]
    ctx = "JSON data:\n" + json.dumps(dict(pairs), indent=0)
    q = f'Key: "{key}"\nThe value associated with the specified Key is: '
    user = f"Extract the value corresponding to the specified key in the JSON object below.\n\n{ctx}\n\n{q}"
    return record(f"kv_{target}_{k}", "kv", chat(user), [val], [f'"{key}": "{val}"'])


NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def build_vt(target, k):
    names = set()
    while len(names) < a.vt_chains * (a.vt_hops + 1):
        names.add("".join(rng.choices(string.ascii_uppercase, k=5)))
    names = sorted(names)                                            # set order depends on PYTHONHASHSEED
    vals = rng.sample(range(10000, 99999), a.vt_chains)
    chains = []
    for c in range(a.vt_chains):
        v = names[c * (a.vt_hops + 1):(c + 1) * (a.vt_hops + 1)]
        chains.append([f"VAR {v[0]} = {vals[c]}."] + [f"VAR {v[i]} = VAR {v[i - 1]}." for i in range(1, len(v))])
    head = "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    qvar = names[a.vt_hops]                                          # last variable of chain 0
    tail = (f"\n\nQuestion: Following the chain(s) of variable assignment in the text above, what is the numerical "
            f"value of VAR {qvar}? Answer with only the number.")
    n_noise = max(1, (target - ntok(head + tail) - 15 * len(chains) * (a.vt_hops + 1)) // ntok(NOISE))
    sents = [NOISE] * n_noise
    # each chain's statements in order, at random sentence slots; interleave chains
    slots = sorted(rng.sample(range(n_noise + 1), len(chains) * (a.vt_hops + 1)))
    # random interleaving of chains, each chain's statements kept in order
    seq = [c for c in range(len(chains)) for _ in range(a.vt_hops + 1)]
    rng.shuffle(seq)
    nxt = [0] * len(chains)
    placed = []
    for slot, c in zip(slots, seq):
        placed.append((slot, chains[c][nxt[c]]))
        nxt[c] += 1
    out = []
    pi = 0
    for i in range(n_noise + 1):
        while pi < len(placed) and placed[pi][0] == i:
            out.append(placed[pi][1] + " ")
            pi += 1
        if i < n_noise:
            out.append(sents[i])
    user = head + "".join(out) + tail
    return record(f"vt_{target}_{k}", "vt", chat(user), [str(vals[0])], chains[0])


def build_qa():
    z = zipfile.ZipFile(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
    d2p = json.load(open("baselines/SpecPrefill/eval/long_bench/configs/dataset2prompt.json"))
    out = []
    for ds in ["hotpotqa", "2wikimqa"]:
        rows = [json.loads(l) for l in z.open(f"data/{ds}.jsonl")]
        for r in rows[:a.n_qa]:
            ids = tok(d2p[ds].format(**r), add_special_tokens=False).input_ids
            if len(ids) > a.max_len:                                     # LongBench-style middle truncation
                h = a.max_len // 2
                ids = ids[:h] + ids[-h:]
            out.append(record(f"{ds}_{r['_id']}", ds, chat(tok.decode(ids)), r["answers"]))
    return out


recs = []
for L in map(int, a.lengths.split(",")):
    recs += [build_kv(L, k) for k in range(a.n_kv)]
    recs += [build_vt(L, k) for k in range(a.n_vt)]
recs += build_qa()
assert max(r["len"] for r in recs) <= a.max_len + 300, max(r["len"] for r in recs)
with open(a.out, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print({t: sum(r["task"] == t for r in recs) for t in sorted({r["task"] for r in recs})},
      "max len", max(r["len"] for r in recs))
