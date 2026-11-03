#!/bin/bash
# Lane A after run_recall_a2.sh (PID $1): multikey-3 at 128k with excerpt recall (retrieval guard).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 $1 2>/dev/null; do sleep 30; done
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec mk3_128k_recall --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
log recall_a3 het
