#!/bin/bash
# Excerpt recall on non-retrieval tasks (task-agnostic split, layer 17), 20 prompts each; then HF dense on RULER ones.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
for t in qa_1 qa_2 vt cwe fwe; do rec ${t}_128k_recall --qlen 128 --layers 17 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
for t in math_find code_debug; do rec ${t}20_recall --qlen 128 --layers 17 --tasks $t --prompts data/infbench.jsonl --limit 20; done
for t in qa_1 qa_2 vt cwe fwe; do run ${t}_128k_g1w0 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
log bias_b het
