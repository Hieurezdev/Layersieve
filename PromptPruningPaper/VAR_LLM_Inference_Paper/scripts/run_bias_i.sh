#!/bin/bash
# Lane I: soft attention bias (every key gets max(0, log(block score / mean)) added),
# question = last 16 tokens. Tasks where copying helped and where it hurt.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec kv20_bias1 --bias 1 --prompts data/infbench_kv86.jsonl --limit 20
for t in niah_multivalue cwe vt; do rec ${t}_128k_bias1 --bias 1 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_bias1 --bias 1 --prompts data/infbench_ld50.jsonl
for t in qa_2 qa_1 fwe; do rec ${t}_128k_bias1 --bias 1 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec mk3_128k_bias1 --bias 1 --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
log bias_i het
