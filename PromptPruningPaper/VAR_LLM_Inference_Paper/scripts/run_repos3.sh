#!/bin/bash
# Reproduce the long-dialogue gain of position compression on InfiniteBench (Qwen2.5-7B-Instruct-1M, LayerSieve default).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/repos; mkdir -p $R
LS="--method layersieve --unit --drop-ratio 0.4 --min-keep 0.25 --decode-ref 22 --gpu-util 0.4"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 60000 ]; do sleep 60; done; }
run() { local n=$1; shift; waitmem; echo "$(date '+%F %T') [$n] bat dau" >> $R/run.log
  $PY scripts/eval_bench.py $LS "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; echo "$(date '+%F %T') [$n] xong ($?)" >> $R/run.log; }
IB="--tasks longdialogue_qa_eng,kv_retrieval --prompts data/infbench.jsonl"
run ib_g1 --repos-gap 1 $IB
run ib_g16 --repos-gap 16 $IB
echo "$(date '+%F %T') het repos3" >> $R/run.log
