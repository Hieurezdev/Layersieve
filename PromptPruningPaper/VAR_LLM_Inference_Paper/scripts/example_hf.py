"""Minimal example: in-model progressive prompt pruning vs dense on one long prompt."""
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inmodel_prune import InModelPruneConfig, load_model, prefill_and_generate  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"

cfg = InModelPruneConfig(start_layer=17, drop_ratio=0.12)
tok = AutoTokenizer.from_pretrained(MODEL)
model = load_model(MODEL, cfg)

doc = "\n".join(f"Record {i}: the code for item {i} is {i * 7919 % 100000:05d}." for i in range(20000))
msg = [{"role": "user", "content": f"{doc}\n\nWhat is the code for item 12345? Answer with the code only."}]
ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").cuda()
eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

for prune in (False, True):
    out, st = prefill_and_generate(model, ids, cfg, max_new_tokens=16, eos_token_ids=eos, prune=prune)
    print(f"{'pruned' if prune else 'dense '}: ttft {st.ttft:.2f}s, tokens kept at last layer "
          f"{st.kept_per_layer[-1]}/{st.prompt_len} -> {tok.decode(out, skip_special_tokens=True)!r}")
