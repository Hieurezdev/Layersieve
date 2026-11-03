#!/bin/bash
# GemFilter with LayerSieve's own selection (scripts/gem_ls.py: n_obs 128, max over heads, units, keep 25%), at layer
# 17 then 14, on the prompts of the *_v copy runs; then the rest of the official layer-14 GemFilter. Retries on CUDA OOM.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA" $R/$1.log; }
job() { local n=$1; shift; log $n "bat dau"
  until "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
job gemls_l17_k25 $PY scripts/gem_ls.py --layer 17 --keep 0.25 --prompts data/gem_subset.jsonl
job gemls_l14_k25 $PY scripts/gem_ls.py --layer 14 --keep 0.25 --prompts data/gem_subset.jsonl
job gem_l14_k2048 /root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python baselines/asl_driver.py --method gemfilter --select-layer 14 --select-k 2048 --prompts data/gem_subset.jsonl
log gemls het
