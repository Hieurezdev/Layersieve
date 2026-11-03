"""Retrieval-head scores (copy-paste criterion of Wu et al., 2024) for a HF model on key-value retrieval prompts.

For every answer token, teacher-forced after a dense prefill, a query head scores a hit when its strongest attention
over the prompt falls inside the span where the answer occurs in the prompt and on the very token being produced.
A head's score is its hit rate over all answer tokens. Scores are reported per query head and per KV head (mean over
its query group), since a KV head is the unit at which cached keys can be re-rotated.

    python scripts/retrieval_heads.py --prompts data/hard.jsonl --task kv --n 30 --out results/repos/heads_7b1m.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--prompts", required=True)
ap.add_argument("--task", default="kv")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--max-len", type=int, default=9000, help="skip longer prompts (eager attention memory)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
cfg = model.config
L, Hq, Hk = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads


def find(seq, sub):
    """Start of the first occurrence of sub in seq, or -1."""
    n = len(sub)
    for i in range(len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return -1


hits = torch.zeros(L, Hq, dtype=torch.float64)
n_tok, used = 0, []
for line in open(a.prompts):
    r = json.loads(line)
    if r["task"] != a.task or len(used) >= a.n or len(r["prompt_ids"]) > a.max_len:
        continue
    ids = r["prompt_ids"]
    ans = r["answers"][0] if isinstance(r["answers"], list) else r["answers"]
    span = None
    for cand in (tok.encode(ans, add_special_tokens=False), tok.encode(" " + ans, add_special_tokens=False)):
        s = find(ids, cand)
        if s >= 0:
            span, a_ids = (s, s + len(cand)), cand
            break
    if span is None:
        continue
    with torch.no_grad():
        out = model(torch.tensor([ids[:-1]], device="cuda"), use_cache=True)
        past = out.past_key_values
        prompt = torch.tensor(ids, device="cuda")
        inp = [ids[-1]] + a_ids[:-1]
        for j, (x, y) in enumerate(zip(inp, a_ids)):
            out = model(torch.tensor([[x]], device="cuda"), past_key_values=past, use_cache=True, output_attentions=True)
            past = out.past_key_values
            for l, att in enumerate(out.attentions):                                # [1, Hq, 1, S]
                top = att[0, :, 0, :len(ids)].argmax(-1)                           # strongest prompt position per head
                ok = (top >= span[0]) & (top < span[1]) & (prompt[top] == y)
                hits[l] += ok.cpu().double()
            n_tok += 1
    used.append(r["id"])
    print(r["id"], len(ids), "answer tokens", len(a_ids), flush=True)

q = (hits / max(n_tok, 1))
kv = q.view(L, Hk, Hq // Hk).mean(-1)
json.dump({"model": a.model, "prompts": used, "answer_tokens": n_tok, "query_head": q.tolist(), "kv_head": kv.tolist()},
          open(a.out, "w"))
top = sorted(((kv[l, h].item(), l, h) for l in range(L) for h in range(Hk)), reverse=True)[:16]
print("top KV heads (score, layer, head):", [(round(s, 3), l, h) for s, l, h in top])
