#!/bin/bash
# Lane F after run_bias_e.sh (PID $1): math_find with copy and move (the first copy run crashed in logging).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
rec math_find20_recall --prompts data/infbench.jsonl --tasks math_find --limit 20
rec math_find20_move --move --prompts data/infbench.jsonl --tasks math_find --limit 20
rec code_debug20_move --move --prompts data/infbench.jsonl --tasks code_debug --limit 20
log bias_f het
