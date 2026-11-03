#!/bin/bash
# Night lane after run_focus.sh (PID $1): copy whose excerpts are scored by the last 16 prompt tokens AND the plain
# answer's tokens (--draft), with --select for the paired plain answer.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
F="--select --draft"
for t in qa_2 vt cwe; do rec ${t}_128k_draft $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec math_find20_draft $F --prompts data/infbench.jsonl --tasks math_find --limit 20
rec kv20_draft $F --prompts data/infbench_kv86.jsonl --limit 20
for t in niah_multivalue qa_1; do rec ${t}_128k_draft $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_draft $F --prompts data/infbench_ld50.jsonl
log night10 het
