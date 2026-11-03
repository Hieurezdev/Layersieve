#!/bin/bash
# Three tests of the insights (21/10), HF runs first (they fit the free memory), vLLM last: (1) LayerSieve scored by the last 16 instead of 128 tokens on LD200 (vLLM);
# (2) copy visible to the question only in layers < 17, or only in layers >= 17 (LD50); (3) copy budget 8k / 32k on
# math_find and cwe. Every job retries while the shared GPU has no room.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|memory|CUDA error" $R/$1.log; }
job() { local n=$1; shift; log $n "bat dau"
  until "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
job ld50_copy_only_below17 $PY scripts/recall_eval.py --hide-from 17 --hide-to 1000 --prompts data/infbench_ld50.jsonl
job ld50_copy_only_from17 $PY scripts/recall_eval.py --hide-from 0 --hide-to 17 --prompts data/infbench_ld50.jsonl
for B in 8192 32768; do
  job math_find20_copy_b$B $PY scripts/recall_eval.py --select --budget $B --prompts data/infbench.jsonl --tasks math_find --limit 20
  job cwe_128k_copy_b$B $PY scripts/recall_eval.py --select --budget $B --tasks cwe --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
done
V="$PY scripts/eval_bench.py --gpu-util 0.25 --prompts data/infbench_ld200.jsonl"
LS="--method layersieve --unit --drop-ratio 0.4 --min-keep 0.25 --decode-ref 22"
job ld200_vllm_dense $V --method dense
job ld200_vllm_ls $V $LS
job ld200_vllm_ls_s16 $V $LS --n-score 16
log insight het
