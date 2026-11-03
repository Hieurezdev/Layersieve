#!/bin/bash
# Tai lap bang kv50 (results/kv50): dense, cat dan, SlimInfer (vLLM, cai lai) o nhieu budget, SlimInfer goc (HF).
# Can: .venv (vllm 0.6.3.post1) cho vLLM; baselines/SlimInfer/.venv (transformers 4.53, flash-attn) cho ban goc.
set -e
cd "$(dirname "$0")/.."
P=data/kv50.jsonl; R=results/kv50
[ -f $P ] || .venv/bin/python data/build_kv.py --out $P
run() { .venv/bin/python scripts/eval_kv_vllm.py --prompts $P "$@"; }
run --dense --out $R/kv50_dense.jsonl
run --out $R/kv50_catdan.jsonl                                    # L17, bo 12%/layer
run --method slim --out $R/kv50_slim.jsonl                         # cau hinh goc 12k/6k/4k
run --method slim --slim-ref-len 32768 --out $R/kv50_slim_scaled.jsonl
for f in 0.8,0.6,0.4 0.9,0.75,0.5 0.5,0.25,0.15; do
  run --method slim --slim-frac $f --out $R/kv50_slim_f${f//,/_}.jsonl
done
(cd baselines/SlimInfer && .venv/bin/python run_kv.py --prompts ../../$P --n 10 --out ../../$R/kv50_slim_hf_official.jsonl)
