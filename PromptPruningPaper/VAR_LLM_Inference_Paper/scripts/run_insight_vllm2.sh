#!/bin/bash
# vLLM dense on LD200 at gpu-util 0.3 (~45GB), started once run_insight.sh (PID $1, the 30GB budget-32k jobs) ends.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "memory|CUDA error" $R/$1.log; }
job() { local n=$1; shift; log $n "bat dau"
  until "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
while kill -0 $1 2>/dev/null; do sleep 30; done
job ld200_vllm_dense $PY scripts/eval_bench.py --gpu-util 0.3 --prompts data/infbench_ld200.jsonl --method dense
log insight_vllm2 het
