#!/bin/bash
# Reorder (picked spans moved before the question, nothing copied) + paired plain answer. Retries on OOM.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  until $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    grep -q OutOfMemoryError $R/$n.log || { log $n "loi khac OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
F="--select --reorder"
rec kv20_reorder $F --prompts data/infbench_kv86.jsonl --limit 20
rec math_find20_reorder $F --prompts data/infbench.jsonl --tasks math_find --limit 20
for t in cwe vt; do rec ${t}_128k_reorder $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
log reorder1 het
