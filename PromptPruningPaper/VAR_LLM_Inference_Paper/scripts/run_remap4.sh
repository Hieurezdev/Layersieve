#!/bin/bash
# HF dense reference for multikey-3 at 128k (same code path as the remap runs), after run_remap3.sh.
cd "$(dirname "$0")/.."
for p in ${WAIT_PID:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
PY=.venv/bin/python; R=results/selfext
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
waitmem 45000; log mk3_128k_g1w0 "bat dau"
$PY scripts/selfextend_eval.py --group 1 --window 0 --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl \
    --out $R/mk3_128k_g1w0.jsonl > $R/mk3_128k_g1w0.log 2>&1; log mk3_128k_g1w0 "xong ($?)"
log remap4 het
