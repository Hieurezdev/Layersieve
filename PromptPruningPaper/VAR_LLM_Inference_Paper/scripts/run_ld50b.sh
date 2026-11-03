#!/bin/bash
# LD prompts 51-100: the two key conditions again (masked window moved right before the question; rerot17).
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
P="$PY scripts/ld_move_probe.py"
job ld50b_move_mask $P --mode mask --prompts data/infbench_ld50b.jsonl
job ld50b_rerot17 $P --mode rerot --from-layer 17 --prompts data/infbench_ld50b.jsonl
job ld50b_copy_prefix $P --mode copy_prefix --prompts data/infbench_ld50b.jsonl
log ld50b het
