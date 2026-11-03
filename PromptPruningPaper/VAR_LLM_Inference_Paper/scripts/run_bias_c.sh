#!/bin/bash
# Lane C after run_bias_a.sh (PID $1): frame-only control (no excerpts), short contexts where dense is already strong,
# multi-needle tasks (more answers than one excerpt budget may hold).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
rec kv20_recall_b0 --budget 0 --layers 17 --prompts data/infbench_kv86.jsonl --limit 20
rec ld50_recall_b0 --budget 0 --layers 17 --prompts data/infbench_ld50.jsonl
rec mk3_32k_recall --qlen 128 --layers 17 --tasks niah_multikey_3 --lengths 32768 --prompts data/ruler_eval.jsonl --limit 40
for t in qa_1 qa_2; do rec ${t}_32k_recall --qlen 128 --layers 17 --tasks $t --lengths 32768 --prompts data/ruler_eval.jsonl --limit 20; done
for t in niah_multivalue niah_multiquery; do
  rec ${t}_128k_recall --qlen 128 --layers 17 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
  run ${t}_128k_g1w0 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
for t in qa_1 qa_2; do run ${t}_32k_g1w0 --tasks $t --lengths 32768 --prompts data/ruler_eval.jsonl --limit 20; done
log bias_c het
