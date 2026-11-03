#!/bin/bash
# GemFilter-style re-run with our selector (last 16 tokens, layer 17, 2048 tokens kept, rest dropped): token level
# (block 1, no pad) and block level (64 + pad 96); then the rest of the official layer-14 GemFilter. Retries on CUDA
# OOM and on device_map offload ("must be on CUDA"), both caused by the shared GPU being full.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA" $R/$1.log; }
rec() { local n=$1; shift; log $n "bat dau"
  until $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
F="--select --gem"
rec kv20_gemtok $F --block 1 --pad 0 --prompts data/infbench_kv86.jsonl --limit 20
rec kv20_gemblk $F --prompts data/infbench_kv86.jsonl --limit 20
rec math_find20_gemtok $F --block 1 --pad 0 --prompts data/infbench.jsonl --tasks math_find --limit 20
rec math_find20_gemblk $F --prompts data/infbench.jsonl --tasks math_find --limit 20
rec ld50_gemtok $F --block 1 --pad 0 --prompts data/infbench_ld50.jsonl
rec ld50_gemblk $F --prompts data/infbench_ld50.jsonl
for t in cwe qa_2 niah_multivalue vt; do
  rec ${t}_128k_gemblk $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20
  rec ${t}_128k_gemtok $F --block 1 --pad 0 --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python; n=gem_l14_k2048; log $n "tiep tuc"
until $HF baselines/asl_driver.py --method gemfilter --select-layer 14 --select-k 2048 --prompts data/gem_subset.jsonl \
    --out $R/$n.jsonl >> $R/$n.log 2>&1; do
  retry $n || { log $n "loi khong phai OOM"; break; }
  log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
log gem3 het
