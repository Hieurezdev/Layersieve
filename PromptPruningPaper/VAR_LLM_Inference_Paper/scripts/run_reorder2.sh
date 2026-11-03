#!/bin/bash
# Reorder, part 2.
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
rec ld50_reorder $F --prompts data/infbench_ld50.jsonl
for t in qa_2 niah_multivalue; do rec ${t}_128k_reorder $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
log reorder2 het
