#!/bin/bash
# After the vLLM lane (PID $1) frees its 45GB: is the gain the re-encoded evidence or its nearness?
# copy_far (copy re-encoded near the question but read from its far positions in layers >= 17) and the masked window
# moved to 16k / 48k tokens before the question. One HF job (~24GB) next to run_cause.sh's.
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
P="$PY scripts/ld_move_probe.py"
job ld50_copy_far17 $P --mode copy_far --from-layer 17 --prompts data/infbench_ld50.jsonl
job ld50_move_mask_d16k $P --mode mask --depth 16384 --prompts data/infbench_ld50.jsonl
job ld50_move_mask_d48k $P --mode mask --depth 49152 --prompts data/infbench_ld50.jsonl
log cause2 het
