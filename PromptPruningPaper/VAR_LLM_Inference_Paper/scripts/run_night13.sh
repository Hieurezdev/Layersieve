#!/bin/bash
# After run_night10.sh (PID $1): fused decoding on multikey-3, then the missing 16-head focus tasks. Retries on OOM.
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
while kill -0 $1 2>/dev/null; do sleep 30; done
rec mk3_128k_fuse --fuse --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
F="--select --focus results/repos/heads_7b1m.json --head-thr 0.3"
for t in niah_multivalue qa_1 fwe; do rec ${t}_128k_focus3 $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
log night13 het
