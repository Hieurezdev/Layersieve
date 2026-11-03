#!/bin/bash
# Selector precision probe: block scores under many scoring variants (scripts/selector_probe.py); evidence positions are
# compared offline (scripts/analysis/selector_eval.py). One HF job (~25GB) at a time. Retries on CUDA OOM.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
job() { local n=$1; shift; log $n "bat dau"
  until $PY scripts/selector_probe.py "$@" --out $R/selprobe/$n >> $R/$n.log 2>&1; do
    grep -qiE "out of memory|CUDA error" $R/$n.log || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
job vt20 --tasks vt --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
job kv20 --prompts data/infbench_kv86.jsonl --limit 20
job mk3_20 --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
job ld30 --prompts data/infbench_ld50.jsonl --limit 30
job mv20 --tasks niah_multivalue --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
log selprobe het
