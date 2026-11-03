#!/bin/bash
# After ld50b_rerot17 (stops run_ld50b.sh, PID $1, before its copy_prefix replication): selective re-rotation of
# ONE contiguous span (the best block widened by 992 on each side = 2048 tokens), real selector, LD50.
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
until grep -qE "\[ld50b_rerot17\] (xong|loi)" $R/run.log; do sleep 30; done
C=$(pgrep -P $1); kill $1; sleep 1; [ -n "$C" ] && kill $C; log ld50b "dung truoc copy_prefix"
job ld50_move_sel_one $PY scripts/recall_eval.py --select --move --pad 992 --prompts data/infbench_ld50.jsonl
log move_one het
