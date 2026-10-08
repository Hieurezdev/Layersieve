"""RULER eval set from data/build_ruler.sh output: per length and task, 100 of the 500 generated samples (random, seed
42), tokenized (input + answer_prefix, as RULER's pred/call_api.py sends them). Same schema as data/build_infbench.py:
{"id", "task", "idx", "len", "prompt_ids", "answers", "max_new_tokens", "metric"}; metric "all" / "part" =
RULER's string_match_all / string_match_part.
    python data/build_ruler_eval.py --out data/ruler_eval.jsonl
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler"))
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--lengths", default="32768,65536,131072")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", required=True)
a = ap.parse_args()

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
GEN = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}                # RULER synthetic/constants.py
tok = AutoTokenizer.from_pretrained(a.model)
with open(a.out, "w") as f:
    for L in a.lengths.split(","):
        for t in TASKS:
            rows = [json.loads(l) for l in open(f"{a.src}/{L}/{t}/validation.jsonl")]
            assert len(rows) == 500, (L, t, len(rows))
            for i in sorted(random.Random(a.seed).sample(range(len(rows)), a.n)):
                r = rows[i]
                ids = tok(r["input"] + r.get("answer_prefix", ""), add_special_tokens=False).input_ids
                # i = row of validation.jsonl; RULER's own "index" is not a sample id for NIAH (it is the character
                # offset of the answer, so it repeats across samples)
                f.write(json.dumps({"id": f"ruler{L}_{t}_{i}", "task": f"{t}@{L}", "idx": i,
                                    "len": len(ids), "prompt_ids": ids, "answers": r["outputs"],
                                    "max_new_tokens": GEN[t.split("_")[0]],
                                    "metric": "part" if t.startswith("qa") else "all"}) + "\n")
            print(L, t, flush=True)
