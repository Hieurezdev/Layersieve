#!/bin/bash
# Copy + select with cross-context log-probs (verify fields), part 2.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec ld50_v --select --prompts data/infbench_ld50.jsonl
for t in qa_1 fwe; do rec ${t}_128k_v --select --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
for t in math_find code_debug; do rec ${t}20_v --select --prompts data/infbench.jsonl --tasks $t --limit 20; done
rec ld200_v --select --prompts data/infbench_ld200.jsonl
log sel_v2 het
