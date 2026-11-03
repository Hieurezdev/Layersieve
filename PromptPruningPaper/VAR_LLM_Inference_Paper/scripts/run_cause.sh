#!/bin/bash
# Night 21-22/10: what in copy/move makes LD better. Same oracle window (+-1024 tokens around the masked name) for all:
# copy_near (duplicated + re-encoded + near), rerot from layer 17 / 0 / 22 (only near: the window's keys rotated as if
# right before the question, for the question and the answer only), rerot17 on a random window (control).
# HF jobs ~24GB each, one at a time (the 70GB cap leaves room for one vLLM job at 0.3).
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
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
job ld2_rerot_off $P --mode rerot --from-layer 1000 --prompts results/selfext/tmp_focus_test/ld2.jsonl
job ld50_rerot17 $P --mode rerot --from-layer 17 --prompts data/infbench_ld50.jsonl
job ld50_copy_near $P --mode copy_near --prompts data/infbench_ld50.jsonl
job ld50_rerot17_random $P --mode rerot_random --from-layer 17 --prompts data/infbench_ld50.jsonl
job ld50_rerot0 $P --mode rerot --from-layer 0 --prompts data/infbench_ld50.jsonl
job ld50_rerot22 $P --mode rerot --from-layer 22 --prompts data/infbench_ld50.jsonl
log cause het
