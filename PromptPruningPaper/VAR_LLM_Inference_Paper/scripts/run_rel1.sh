#!/bin/bash
# --move with the relative-threshold selector (--sel rel: layer 14, max/16 + contrast/128, threshold 0.3, pad 32),
# paired plain answer (--select). Retries on CUDA OOM.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
job() { local n=$1; shift; log $n "bat dau"
  until $PY scripts/recall_eval.py --select --move --sel rel "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    grep -qiE "out of memory|CUDA error" $R/$n.log || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
job ld50_move_rel --prompts data/infbench_ld50.jsonl
job vt_128k_move_rel --tasks vt --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
job math_find20_move_rel --prompts data/infbench.jsonl --tasks math_find --limit 20
log rel1 het
