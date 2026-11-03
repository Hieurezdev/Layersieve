#!/bin/bash
# Frequency-aware far-distance remap (only rotary dims slower than D0). After the running multikey-3 128k check:
# LongDialogue (200) with the remap in all layers and only in layers < 17 (the far-distance penalty of the slow dims
# sits in the early layers, rope_logit_profile.py), multikey-3 128k (40) for the latter; a stronger D0 = 4k, g = 8;
# then 40-prompt diagnostics (linear on all dims, all layers; SelfExtend floor) for the earlier failure.
cd "$(dirname "$0")/.."
for p in ${WAIT_PID:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
PY=.venv/bin/python; R=results/selfext; mkdir -p $R
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
M3="--tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl"
LD="--prompts data/infbench_ld200.jsonl"
log mk3_128k_freq32k_g4 "xong (hang doi cu da dung, job chay het)"
run ld200_freq32k_g4 45000 --mode freq --group 4 --window 32768 $LD
run ld200_freq32k_g4_E17 45000 --mode freq --group 4 --window 32768 --to-layer 17 $LD
run mk3_128k_freq32k_g4_E17 45000 --mode freq --group 4 --window 32768 --to-layer 17 $M3 --limit 40
run mk3_128k_freq4k_g8 45000 --mode freq --group 8 --window 4096 $M3 --limit 40
run ld200_freq4k_g8 45000 --mode freq --group 8 --window 4096 $LD
run mk3_128k_lin32k_g4 45000 --mode linear --group 4 --window 32768 $M3 --limit 40
run mk3_128k_g8w2048 45000 --mode floor --group 8 --window 2048 $M3 --limit 40
log remap3 het
