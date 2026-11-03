#!/bin/bash
# More prompts on the tasks copy hurts, plus helpful ones, with attention-spread stats for a gate. Retries on OOM.
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
F="--select"
rec math_find50_g $F --prompts data/infbench.jsonl --tasks math_find --limit 50
for t in qa_2 cwe; do rec ${t}_128k_g $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 50; done
rec kv40_g $F --prompts data/infbench_kv86.jsonl --limit 40
rec ld50_g $F --prompts data/infbench_ld50.jsonl
log night14 het
