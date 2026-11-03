#!/bin/bash
# Clamping the slow dims (no distance penalty at all) broke KV retrieval (3/13 vs dense 11/13), so screen milder
# frequency-aware shrinks on the first 20 answerable KV prompts: far distance d -> D0 + (d - D0)/g in the dims with
# wavelength > D0, all layers.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
KV="--prompts data/infbench_kv86.jsonl --limit 20"
run kv20_g1w0 --group 1 --window 0 $KV
run kv20_freq32k_g4 --mode freq --group 4 --window 32768 $KV
run kv20_freq4k_g8 --mode freq --group 8 --window 4096 $KV
run kv20_freq16k_g2 --mode freq --group 2 --window 16384 $KV
run kv20_freq64k_g2 --mode freq --group 2 --window 65536 $KV
log remap8 het
