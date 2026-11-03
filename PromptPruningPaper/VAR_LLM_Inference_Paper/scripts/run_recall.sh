#!/bin/bash
# Lane B: attention-guided excerpt recall (scripts/recall_eval.py), then the logit-scale run of run_remap9.sh.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec kv20_recall --prompts data/infbench_kv86.jsonl --limit 20
rec ld100_recall --prompts data/infbench_ld200.jsonl --limit 100
run kv20_tail_scale1.3 --tail 512 --qscale 1.3 --prompts data/infbench_kv86.jsonl --limit 20
rec mk3_128k_recall --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
log recall het
