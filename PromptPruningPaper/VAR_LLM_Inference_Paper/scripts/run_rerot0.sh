#!/bin/bash
# After run_depth_b.sh (PID $1): rerot in ALL layers (original encoding, every layer reads the window as if near).
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
while kill -0 $1 2>/dev/null; do sleep 30; done
job ld50_rerot0 $PY scripts/ld_move_probe.py --mode rerot --from-layer 0 --prompts data/infbench_ld50.jsonl
log rerot0 het
