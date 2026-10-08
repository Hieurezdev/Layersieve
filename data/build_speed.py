"""Synthetic kv_retrieval prompts (uuid JSON, as in data/build_hard.py) at fixed lengths, for speed benchmarks only
(scripts/bench_speed.py). Output lines: {"id", "len", "prompt_ids", "answers"}; each prompt is within ~1% of its target.
    python data/build_speed.py --lengths 32768,65536,131072,262144,524288,1000000 --n 2 --out data/speed.jsonl
"""
import argparse
import json
import random
import uuid

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--lengths", default="32768,65536,131072,262144,524288,1000000")
ap.add_argument("--n", type=int, default=2, help="prompts per length")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = random.Random(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)


def build(target, k):
    pair_tok = len(tok(json.dumps({str(uuid.UUID(int=1)): str(uuid.UUID(int=2))}), add_special_tokens=False).input_ids) - 2
    n = target // pair_tok
    while True:
        pairs = [(str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128)))) for _ in range(n)]
        key, val = pairs[rng.randrange(n)]
        user = ("Extract the value corresponding to the specified key in the JSON object below.\n\nJSON data:\n"
                + json.dumps(dict(pairs), indent=0)
                + f'\n\nKey: "{key}"\nThe value associated with the specified Key is: ')
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=True).input_ids
        if 0.99 * target <= len(ids) <= target:
            return {"id": f"speed_{target}_{k}", "len": len(ids), "prompt_ids": ids, "answers": [val]}
        n = int(n * target / len(ids) * 0.997)


with open(a.out, "w") as f:
    for L in map(int, a.lengths.split(",")):
        for k in range(a.n):
            r = build(L, k)
            print(r["id"], r["len"], flush=True)
            f.write(json.dumps(r) + "\n")
