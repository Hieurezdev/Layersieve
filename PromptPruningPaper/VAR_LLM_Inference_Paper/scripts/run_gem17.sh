#!/bin/bash
# Official GemFilter (ASL port): filter at layer 17, last-token query, top-2048 tokens, full model re-run on them.
# Same prompts as the *_v copy runs (data/gem_subset.jsonl). Retries on OOM; the driver resumes by id.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python; n=gem_l17_k2048; log $n "bat dau"
until $HF baselines/asl_driver.py --method gemfilter --select-layer 17 --select-k 2048 --prompts data/gem_subset.jsonl \
    --out $R/$n.jsonl >> $R/$n.log 2>&1; do
  grep -q OutOfMemoryError $R/$n.log || { log $n "loi khac OOM"; exit 1; }
  log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
log $n xong
