#!/bin/bash
# Dual-position reading, part 2: KV, vt.
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
F="--select --dual"
job kv20_dual $PY scripts/recall_eval.py $F --prompts data/infbench_kv86.jsonl --limit 20
job vt_128k_dual $PY scripts/recall_eval.py $F --tasks vt --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
log dual2 het
