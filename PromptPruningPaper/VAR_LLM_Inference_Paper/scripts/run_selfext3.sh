#!/bin/bash
# SelfExtend g=8, w=2048 vs HF dense on the other two InfiniteBench tasks (math_find, code_debug): side effects.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; mkdir -p $R
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
O="--tasks math_find,code_debug --prompts data/infbench.jsonl"
run other_g1w0 45000 --group 1 --window 0 $O
run other_g8w2048 45000 --group 8 --window 2048 $O
log selfext3 het
