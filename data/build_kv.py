"""Build tokenized InfiniteBench kv_retrieval prompts (official InfiniteBench prompt, Qwen chat template).
Default = kv50: ids 0, 10, ..., 490. Output lines: {"id", "prompt_ids", "answers"}.
    python data/build_kv.py --out data/kv50.jsonl
"""
import argparse
import json

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

TEMPLATE = "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--src", default="", help="local kv_retrieval.jsonl (default: download from xinrongzhang2022/InfiniteBench)")
ap.add_argument("--ids", default="0:500:10", help="start:stop:step")
ap.add_argument("--out", required=True)
a = ap.parse_args()

src = a.src or hf_hub_download("xinrongzhang2022/InfiniteBench", "kv_retrieval.jsonl", repo_type="dataset")
data = [json.loads(l, strict=False) for l in open(src)]
tok = AutoTokenizer.from_pretrained(a.model)
with open(a.out, "w") as f:
    for i in range(*map(int, a.ids.split(":"))):
        eg = data[i]
        prompt = tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(**eg)}], tokenize=False,
                                         add_generation_prompt=True)
        ans = eg["answer"]
        f.write(json.dumps({"id": i, "prompt_ids": tok(prompt, add_special_tokens=True).input_ids,
                            "answers": ans if isinstance(ans, list) else [ans]}) + "\n")
