#!/bin/bash
# Idea 3: repos on the 32k-native Qwen2.5-7B-Instruct at 64k (YaRN x4), after run_repos.sh (WAIT_PID).
cd "$(dirname "$0")/.."
for p in ${WAIT_PID:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
PY=.venv/bin/python; R=results/repos; mkdir -p $R
LS="--model Qwen/Qwen2.5-7B-Instruct --yarn 4 --method layersieve --unit --drop-ratio 0.4 --min-keep 0.25 --decode-ref 22 --gpu-util 0.3"
# shared GPU: wait until 46 GB are free before loading (gpu-util 0.3 = 43 GB), else vLLM OOMs at cache allocation
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 47000 ]; do sleep 60; done; }
run() { local n=$1; shift; waitmem; echo "$(date '+%F %T') [$n] bat dau" >> $R/run.log
  $PY scripts/eval_bench.py $LS "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; echo "$(date '+%F %T') [$n] xong ($?)" >> $R/run.log; }
T64="--max-model-len 73728 --tasks niah_multikey_3,qa_1 --lengths 65536 --prompts data/ruler_eval.jsonl"
run q7b_64k_g0 $T64
run q7b_64k_g1 --repos-gap 1 $T64
run q7b_64k_pre_g1 --repos-gap 1 --repos-prefill $T64
echo "$(date '+%F %T') het repos2" >> $R/run.log
