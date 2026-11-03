#!/bin/bash
# Lane H after run_bias_c2.sh (PID $1): move mode (no copy) with the last 16 tokens as the question.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
rec kv20_move --move --qlen 16 --prompts data/infbench_kv86.jsonl --limit 20
for t in cwe vt; do rec ${t}_128k_move --move --qlen 16 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_move --move --qlen 16 --prompts data/infbench_ld50.jsonl
for t in qa_2 fwe qa_1; do rec ${t}_128k_move --move --qlen 16 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec mk3_128k_move --move --qlen 16 --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
for t in math_find code_debug; do rec ${t}20_move --move --qlen 16 --prompts data/infbench.jsonl --tasks $t --limit 20; done
log bias_h het
