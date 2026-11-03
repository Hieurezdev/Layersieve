#!/bin/bash
# Lane E after run_bias_d2.sh (PID $1): how many trailing tokens should score the context (qlen_probe.py).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
pr() { local n=$1; shift; log $n "bat dau"
  $PY scripts/qlen_probe.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
pr qlen_kv10 --prompts data/infbench_kv86.jsonl --limit 10
pr qlen_ld30 --prompts data/infbench_ld50.jsonl --limit 30
for t in qa_1 qa_2 niah_multikey_3; do pr qlen_${t} --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 10; done
log bias_e het
