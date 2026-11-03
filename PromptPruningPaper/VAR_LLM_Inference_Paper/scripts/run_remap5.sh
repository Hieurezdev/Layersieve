#!/bin/bash
# Stronger frequency-aware remap (D0 = 4k, g = 8: 128k -> ~19.5k in the slow dims, about what SelfExtend g8 w2k gives),
# all layers: multikey-3 128k (40) as the retrieval guard, LongDialogue (200); then the same only in layers < 17, and
# the HF dense reference for multikey-3 128k.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; mkdir -p $R
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
M3="--tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl"
LD="--prompts data/infbench_ld200.jsonl"
F="--mode freq --group 8 --window 4096"
run mk3_128k_freq4k_g8 45000 $F $M3 --limit 40
run ld200_freq4k_g8 45000 $F $LD
run mk3_128k_g1w0 45000 --group 1 --window 0 $M3 --limit 40
run ld200_freq4k_g8_E17 45000 $F --to-layer 17 $LD
run mk3_128k_freq4k_g8_E17 45000 $F --to-layer 17 $M3 --limit 40
log remap5 het
