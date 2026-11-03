#!/bin/bash
# Copy + select with cross-context log-probs (verify fields), part 1.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec kv20_v --select --prompts data/infbench_kv86.jsonl --limit 20
for t in niah_multivalue cwe vt qa_2; do rec ${t}_128k_v --select --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec mk3_128k_v --select --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
rec kv86_v --select --prompts data/infbench_kv86.jsonl
log sel_v1 het
