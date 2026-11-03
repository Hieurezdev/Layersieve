#!/bin/bash
# Bias controls for excerpt recall: random excerpts (same budget), and a task-agnostic split (last 128 tokens).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec kv20_recall_rand --random --layers 17 --prompts data/infbench_kv86.jsonl --limit 20
rec kv20_recall_q128 --qlen 128 --layers 17 --prompts data/infbench_kv86.jsonl --limit 20
rec ld50_recall_rand --random --layers 17 --prompts data/infbench_ld50.jsonl
rec ld50_recall_q128 --qlen 128 --layers 17 --prompts data/infbench_ld50.jsonl
for t in math_find code_debug; do run ${t}20_g1w0 --tasks $t --prompts data/infbench.jsonl --limit 20; done
log bias_a het
