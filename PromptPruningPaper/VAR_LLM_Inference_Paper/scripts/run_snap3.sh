#!/bin/bash
# LD200 with --move --snap (retr span, whole lines): prompts 51-200 in two lanes (the first 50 are
# ld50_move_one_retr_snap). Prompt splits: lines 51-125 / 126-200 of data/infbench_ld200.jsonl.
#   bash scripts/run_snap3.sh B   |   bash scripts/run_snap3.sh C
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|CUDA error" $R/$1.log; }
job() { local n=$1; shift; [ -s $R/$n.jsonl ] && grep -qE "\[$n\] xong" $R/run.log && return; log $n "bat dau"
  until $PY scripts/recall_eval.py --select --move --pad 992 --agg retr --snap "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
L=$(echo $1 | tr BC bc)
job ld200${L}_move_one_retr_snap --prompts /tmp/claude-0/-root-team-1-QuangNM-PromptPruningPaper-VAR-LLM-Inference-Paper/e4addc12-9e78-485b-9a55-d7b2f2d1d952/scratchpad/ld200_${L}.jsonl
log snap3_$1 het
