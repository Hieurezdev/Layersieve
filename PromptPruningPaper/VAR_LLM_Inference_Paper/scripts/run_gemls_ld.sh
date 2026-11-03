#!/bin/bash
# LD only: GemFilter with LayerSieve's selection (layer 17, keep 25%) where every picked unit also keeps its NBR
# neighbouring units on each side, re-encoded at short distances. Tests whether LD is low because the evidence is far.
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
for W in 4 16; do job ld50_gemls_l17_k25_n$W $PY scripts/gem_ls.py --layer 17 --keep 0.25 --nbr $W --prompts data/infbench_ld50.jsonl; done
log gemls_ld het
