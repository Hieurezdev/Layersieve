#!/bin/bash
# Why did the far-distance remap break multikey-3 at 128k (garbled UUIDs)? On the same test, all layers:
#   freq   = remap only the rotary dims slower than D0 (hypothesis: fractional phases in fast dims break copying)
#   linear = remap every dim (if it still garbles with all layers, layer mixing was not the cause)
#   floor  = SelfExtend g8 w2048 (integer positions) as reference
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; mkdir -p $R
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
M3="--tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl"
run mk3_128k_freq32k_g4 45000 --mode freq --group 4 --window 32768 $M3
run mk3_128k_lin32k_g4 45000 --mode linear --group 4 --window 32768 $M3
run mk3_128k_g8w2048 45000 --mode floor --group 8 --window 2048 $M3
log remap2 het
