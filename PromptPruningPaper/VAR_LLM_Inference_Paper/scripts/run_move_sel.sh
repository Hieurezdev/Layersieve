#!/bin/bash
# Selective re-rotation with the real selector (no oracle): --move = picked spans keep their KV, their keys get
# positions right before the last 128 tokens in every layer, for the question and the answer. LD50, then KV20, math20.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|CUDA error" $R/$1.log; }
job() { local n=$1; shift; log $n "bat dau"
  until "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
job ld50_move_sel $PY scripts/recall_eval.py --select --move --prompts data/infbench_ld50.jsonl
job math_find20_move_sel $PY scripts/recall_eval.py --select --move --prompts data/infbench.jsonl --tasks math_find --limit 20
job kv20_move_sel $PY scripts/recall_eval.py --select --move --prompts data/infbench_kv86.jsonl --limit 20
log move_sel het
