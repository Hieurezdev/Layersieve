#!/bin/bash
# Night lane after run_sel_v2.sh (PID $1): focus with the 16 strongest retrieval heads (score > 0.3), then the fused
# decoding runs that were stopped (cwe done; the rest resume).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
F="--select --focus results/repos/heads_7b1m.json --head-thr 0.3"
rec kv20_focus3 $F --prompts data/infbench_kv86.jsonl --limit 20
rec math_find20_focus3 $F --prompts data/infbench.jsonl --tasks math_find --limit 20
for t in cwe qa_2 vt; do rec ${t}_128k_focus3 $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_focus3 $F --prompts data/infbench_ld50.jsonl
for t in qa_2 niah_multivalue vt qa_1 fwe; do rec ${t}_128k_fuse --fuse --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_fuse --fuse --prompts data/infbench_ld50.jsonl
log night9 het
