#!/bin/bash
# Decode-time RoPE re-indexing (repos) on RULER niah_multikey_3 @32k, LayerSieve default config, then 7B (32k-native).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/repos; mkdir -p $R
S=/tmp/claude-0/-root-PromptPruningPaper-VAR-LLM-Inference-Paper/f208dd15-a98d-4c59-8b1c-dd4d0380d9ce/scratchpad
LS="--method layersieve --unit --drop-ratio 0.4 --min-keep 0.25 --decode-ref 22 --gpu-util 0.3"
run() { local n=$1; shift; echo "$(date '+%F %T') [$n] bat dau" >> $R/run.log
  $PY scripts/eval_bench.py $LS "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; echo "$(date '+%F %T') [$n] xong ($?)" >> $R/run.log; }
M3="--max-model-len 40960 --tasks niah_multikey_3 --lengths 32768 --prompts data/ruler_eval.jsonl"
run smoke_g1e9 --repos-gap 1000000000 --max-model-len 40960 --prompts $S/smoke.jsonl
for g in 1 16 128 1024; do run mk3_32k_g$g --repos-gap $g $M3; done
for g in 1 128; do run mk3_32k_pre_g$g --repos-gap $g --repos-prefill $M3; done
echo "$(date '+%F %T') het repos" >> $R/run.log
