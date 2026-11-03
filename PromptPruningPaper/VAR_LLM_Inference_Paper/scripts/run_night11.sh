#!/bin/bash
# Retry of the fused-decoding runs that hit OOM (shared GPU), after run_night9.sh (PID $1); runs resume by id.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
for t in qa_2 niah_multivalue vt; do rec ${t}_128k_fuse --fuse --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
log night11 het
