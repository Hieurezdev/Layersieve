#!/bin/bash
# Token-level fusion (sum of plain and recalled log-probs) on the 11 tasks. Takes over from run_sel_v1.sh (PID $1)
# once its mk3 run is done (its KV86 run is left for later).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
until grep -q "\[mk3_128k_v\] xong" $R/run.log; do sleep 30; done
C=$(pgrep -P $1); kill $1; sleep 1; [ -n "$C" ] && kill $C; log sel_v1 "dung sau mk3 (kv86_v de sau)"
rec kv20_fuse --fuse --prompts data/infbench_kv86.jsonl --limit 20
for t in math_find code_debug; do rec ${t}20_fuse --fuse --prompts data/infbench.jsonl --tasks $t --limit 20; done
for t in cwe qa_2 niah_multivalue vt qa_1 fwe; do rec ${t}_128k_fuse --fuse --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_fuse --fuse --prompts data/infbench_ld50.jsonl
rec mk3_128k_fuse --fuse --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
log fuse het
