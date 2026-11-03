#!/bin/bash
# Remove the RoPE distance penalty instead of shrinking it. Keys within w = 4096 keep the exact distance; farther keys
# are seen at distance 4096 (group -> infinity):
#   slow   = only the rotary dims with wavelength > 4096 are clamped, fast dims keep the true distance (freq mode)
#   rerope = every dim clamped (floor mode, as ReRoPE with no leak)
#   E17    = only layers < 17, where the logit profile puts the penalty
# KV retrieval on the 86 InfiniteBench prompts whose answer survives truncation; multikey-3 128k (40) as the copying
# guard; LongDialogue first 100 (same ids as the earlier 100-prompt runs).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
KV="--prompts data/infbench_kv86.jsonl"
M3="--tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40"
LD="--prompts data/infbench_ld200.jsonl --limit 100"
SLOW="--mode freq --group 1000000 --window 4096"
RR="--mode floor --group 1000000 --window 4096"
run kv86_slow4k $SLOW $KV
run mk3_128k_slow4k $SLOW $M3
run kv86_rerope4k $RR $KV
run ld100_slow4k $SLOW $LD
run mk3_128k_rerope4k $RR $M3
run kv86_slow4k_E17 $SLOW --to-layer 17 $KV
run ld100_rerope4k $RR $LD
run kv86_g1w0 --group 1 --window 0 $KV
run ld100_slow4k_E17 $SLOW --to-layer 17 $LD
log remap7 het
