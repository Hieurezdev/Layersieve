#!/bin/bash
# LongDialogue on all 200 InfiniteBench examples: HF dense (g=1, w=0) and SelfExtend g=8/4, w=2048 (the first 100
# SelfExtend answers are reused from run_selfext.sh, same prompts).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; mkdir -p $R
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
LD="--prompts data/infbench_ld200.jsonl"
run ld200_g8w2048 45000 --group 8 --window 2048 $LD
run ld200_g1w0 45000 --group 1 --window 0 $LD
run ld200_g4w2048 45000 --group 4 --window 2048 $LD
log selfext2 het
