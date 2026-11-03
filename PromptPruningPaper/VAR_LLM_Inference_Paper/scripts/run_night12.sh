#!/bin/bash
# Free slot (night9 ended early on OOM). The shared GPU is near full, so a job that fails to start (OOM) is retried
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  until $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    grep -qi "out of memory" $R/$n.log || { log $n "loi khac OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
rec mk3_128k_draft --select --draft --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
rec kv86_draft --select --draft --prompts data/infbench_kv86.jsonl
rec ld50_fuse --fuse --prompts data/infbench_ld50.jsonl
rec fwe_128k_fuse --fuse --tasks fwe --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
log night12 het
