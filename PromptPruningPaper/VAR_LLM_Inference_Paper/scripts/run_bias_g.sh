#!/bin/bash
# Lane G (after run_bias_i.sh, PID $1): copy with the shared question length (last 16 tokens, chosen with qlen_probe.py on LD 1-30, KV 1-10, RULER 1-10).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
rec ld200_q16 --qlen 16 --prompts data/infbench_ld200.jsonl
for t in vt cwe qa_2; do rec ${t}_128k_q16 --qlen 16 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
log bias_g het
