#!/bin/bash
# Far-distance remap f(d) = d (d <= D0), D0 + (d - D0)/g beyond (linear, no shared positions), D0 = 32k, g = 4, dense HF:
# all layers vs layers >= 17 only. E1: LongDialogue (200); E2: multikey-3 at 128k (retrieval harm), HF dense reference.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; mkdir -p $R
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
LIN="--mode linear --group 4 --window 32768"
LD="--prompts data/infbench_ld200.jsonl"
M3="--tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl"
run ld200_lin32k_g4_L17 45000 $LIN --from-layer 17 $LD
run mk3_128k_lin32k_g4_L17 45000 $LIN --from-layer 17 $M3
run ld200_lin32k_g4 45000 $LIN $LD
run mk3_128k_lin32k_g4 45000 $LIN $M3
run mk3_128k_g1w0 45000 --group 1 --window 0 $M3
log remap het
