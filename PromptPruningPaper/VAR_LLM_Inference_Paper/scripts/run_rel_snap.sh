#!/bin/bash
# --move with --sel rel (all blocks near the top score; real question minus a neutral request) + --snap, no --select
# needed for scoring (pred_recall is kept). Target: prompts where the model attends to the evidence but answers wrong
# (mv: 4 needles all in top-8, one-span move carried 1); vt: the top block is the demo, which rel should drop.
# Starts after the LD200 lane PID $2 ends.   bash scripts/run_rel_snap.sh A PID   |   bash scripts/run_rel_snap.sh B PID
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|CUDA error" $R/$1.log; }
job() { local n=$1; shift; [ -s $R/$n.jsonl ] && grep -qE "\[$n\] xong" $R/run.log && return; log $n "bat dau"
  until $PY scripts/recall_eval.py --select --move --sel rel --snap "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
[ -n "$2" ] && while ps -o stat= -p $2 2>/dev/null | grep -qv Z; do sleep 30; done
RU="--lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
if [ "$1" = A ]; then
  job vt_128k_move_rel_snap --tasks vt $RU
  job mk3_128k_move_rel_snap --tasks niah_multikey_3 $RU
else
  job mv_128k_move_rel_snap --tasks niah_multivalue $RU
  job kv20_move_rel_snap --prompts data/infbench_kv86.jsonl --limit 20
fi
log rel_snap_$1 het
