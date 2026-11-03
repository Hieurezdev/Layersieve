#!/bin/bash
# Remove the RoPE distance penalty instead of shrinking it: in the slow rotary dims (wavelength > 4k) every key farther
# than 4k is seen at distance 4k (group -> infinity), fast dims exact. Only layers < 17, where the profile puts the
# penalty. Starts after run_remap5.sh (PID passed as $1).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
while kill -0 $1 2>/dev/null; do sleep 60; done
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
M3="--tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl"
LD="--prompts data/infbench_ld200.jsonl"
C="--mode freq --group 1000000 --window 4096 --to-layer 17"
run mk3_128k_clamp4k_E17 $C $M3 --limit 40
run ld200_clamp4k_E17 $C $LD
log remap6 het
