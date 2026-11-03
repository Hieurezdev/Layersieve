#!/bin/bash
# Why does --move with the retr span (overlap 0.73-0.99 with the oracle window) get LD50 26 vs oracle rerot0 34
# (oracle-only correct 4, move-only 0)? The one remaining difference is the window edges: the oracle snaps to line
# breaks, --move cuts at 64-token blocks. Both directions: A = --move with --snap, B = oracle rerot0 without snapping.
#   bash scripts/run_snap.sh A   |   bash scripts/run_snap.sh B
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|CUDA error" $R/$1.log; }
job() { local n=$1; shift; [ -s $R/$n.jsonl ] && grep -qE "\[$n\] xong" $R/run.log && return; log $n "bat dau"
  until $PY "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
LD="--prompts data/infbench_ld50.jsonl"
if [ "$1" = A ]; then
  job ld50_move_one_retr_snap scripts/recall_eval.py --select --move --pad 992 --agg retr --snap $LD
else
  job ld50_rerot0_nosnap scripts/ld_move_probe.py --mode rerot --from-layer 0 --nosnap $LD
fi
log snap_$1 het
