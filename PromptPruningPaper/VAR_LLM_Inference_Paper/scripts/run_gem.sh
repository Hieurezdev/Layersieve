#!/bin/bash
# Official GemFilter (ASL port): filter at layer 14, then 17; last-token query, top-2048 tokens, full re-run on them.
# Same prompts as the *_v copy runs (data/gem_subset.jsonl). One lane; retries on any CUDA out-of-memory error.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python
for LYR in 14 17; do
  n=gem_l${LYR}_k2048; log $n "bat dau"
  until $HF baselines/asl_driver.py --method gemfilter --select-layer $LYR --select-k 2048 --prompts data/gem_subset.jsonl \
      --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    grep -qi "out of memory" $R/$n.log || { log $n "loi khac OOM"; break; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n xong
done
log gem het
