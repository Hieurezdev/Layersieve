#!/bin/bash
# Grouped positions with a neighbour window (SelfExtend) on dense Qwen2.5-7B-Instruct-1M: multikey-3 32k first (fast),
# then InfiniteBench long-dialogue and kv retrieval.
cd "$(dirname "$0")/.."
for p in ${WAIT_PID:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
PY=.venv/bin/python; R=results/selfext; mkdir -p $R
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
M3="--tasks niah_multikey_3 --lengths 32768 --prompts data/ruler_eval.jsonl"
IB="--tasks longdialogue_qa_eng,kv_retrieval --prompts data/infbench.jsonl"
run mk3_32k_g1w0 30000 --group 1 --window 0 $M3
for gw in "8 2048" "4 2048" "16 2048" "8 512"; do set -- $gw; run mk3_32k_g$1w$2 30000 --group $1 --window $2 $M3; done
for gw in "8 2048" "4 2048"; do set -- $gw; run ib_g$1w$2 45000 --group $1 --window $2 $IB; done
log selfext het
