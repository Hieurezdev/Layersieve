#!/bin/bash
# Takes run_cause.sh's slot (PID $1) once its random-window control is done (its rerot0 / rerot22 are dropped:
# rerot17 was null), then: copy_prefix = the copy near the question but encoded with only its original prefix:
# is the gain the nearness of the (re-)encoding, or the extra context the copy sees? One HF job (~24GB).
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
until grep -qE "\[ld50_rerot17_random\] (xong|loi)" $R/run.log; do sleep 30; done
C=$(pgrep -P $1); kill $1; sleep 1; [ -n "$C" ] && kill $C; log cause "dung sau rerot17_random (bo rerot0, rerot22)"
P="$PY scripts/ld_move_probe.py"
job ld50_copy_prefix $P --mode copy_prefix --prompts data/infbench_ld50.jsonl
log cause3 het
