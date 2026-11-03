#!/bin/bash
# Per-head calibrated attention (scripts/attn_calib.py), mode corr, for the question and the answer; --select keeps the
# paired plain answer. Retries on OOM.
# every 2 minutes until it gets memory; finished prompts are skipped on each retry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  until $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    grep -q OutOfMemoryError $R/$n.log || { log $n "loi khac OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
[ -n "$1" ] && while kill -0 $1 2>/dev/null; do sleep 30; done
F="--select --calib results/selfext/calib_7b1m.json --calib-mode corr"
rec kv20_cal_corr $F --prompts data/infbench_kv86.jsonl --limit 20
rec math_find20_cal_corr $F --prompts data/infbench.jsonl --tasks math_find --limit 20
rec ld50_cal_corr $F --prompts data/infbench_ld50.jsonl
for t in cwe qa_2 niah_multivalue vt; do rec ${t}_128k_cal_corr $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
log calib_corr het
