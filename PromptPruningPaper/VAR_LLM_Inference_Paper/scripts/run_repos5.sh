#!/bin/bash
# Per-head repos (rerun of repos4 after the head scorer failed): score retrieval heads, then compress only the other heads.
cd "$(dirname "$0")/.."
for p in ${WAIT_PID:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
PY=.venv/bin/python; R=results/repos; mkdir -p $R
S=/tmp/claude-0/-root-PromptPruningPaper-VAR-LLM-Inference-Paper/f208dd15-a98d-4c59-8b1c-dd4d0380d9ce/scratchpad
LS="--method layersieve --unit --drop-ratio 0.4 --min-keep 0.25 --decode-ref 22"
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 60; done; }
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1 m=$2; shift 2; waitmem $m; log $n "bat dau"
  $PY scripts/eval_bench.py $LS "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
M3="--max-model-len 40960 --gpu-util 0.3 --tasks niah_multikey_3 --lengths 32768 --prompts data/ruler_eval.jsonl"
IB="--gpu-util 0.4 --tasks longdialogue_qa_eng,kv_retrieval --prompts data/infbench.jsonl"

waitmem 40000; log heads "bat dau"
$PY scripts/retrieval_heads.py --prompts data/hard.jsonl --task kv --n 30 --out $R/heads_7b1m.json > $R/heads.log 2>&1; rc=$?; log heads "xong ($rc)"; [ $rc -eq 0 ] || exit 1
for k in 8 16 32; do run mk3_32k_g1_keep$k 47000 --repos-gap 1 --repos-heads $R/heads_7b1m.json --repos-topk $k $M3; done
for k in 16 32; do run ib_g1_keep$k 60000 --repos-gap 1 --repos-heads $R/heads_7b1m.json --repos-topk $k $IB; done
log repos5 het
