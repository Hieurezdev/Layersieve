#!/bin/bash
# LD only, after run_gemls_ld.sh (PID $1): oracle distance test. The +-1024-token window around the masked name is
# moved right before the question (nothing added or dropped); control: a random window of the same length.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
while kill -0 $1 2>/dev/null; do sleep 30; done
retry() { grep -qiE "out of memory|must be on CUDA" $R/$1.log; }
job() { local n=$1; shift; log $n "bat dau"
  until "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
job ld50_move_mask $PY scripts/ld_move_probe.py --mode mask --prompts data/infbench_ld50.jsonl
job ld50_move_random $PY scripts/ld_move_probe.py --mode random --prompts data/infbench_ld50.jsonl
log ld_move het
