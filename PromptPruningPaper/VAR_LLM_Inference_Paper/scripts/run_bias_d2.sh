#!/bin/bash
# Lane D after run_bias_b.sh (PID $1) and run_bias_a.sh (PID $2): the main results again with the task-agnostic
# method only (question = last 128 tokens, layer 17), seeded with the matching control runs.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null || kill -0 $2 2>/dev/null; do sleep 30; done
[ -f $R/kv86_q128.jsonl ] || cp $R/kv20_recall_q128.jsonl $R/kv86_q128.jsonl
[ -f $R/ld200_q128.jsonl ] || cp $R/ld50_recall_q128.jsonl $R/ld200_q128.jsonl
rec kv86_q128 --prompts data/infbench_kv86.jsonl
rec ld200_q128 --prompts data/infbench_ld200.jsonl
rec mk3_128k_q128 --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
log bias_d2 het
