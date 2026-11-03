#!/bin/bash
# --move with better block selectors (selector probe 22/10: mean-over-heads puts the LD evidence at rank 1 in 0/30
# prompts, max-over-heads 28/30, retrieval heads 30/30). One contiguous span (best block +-992) and the 2048 budget.
# Two independent lanes, one HF job (~25GB) each. Retries on CUDA OOM.
#   bash scripts/run_move_agg.sh A [PID]   |   bash scripts/run_move_agg.sh B [PID]   (PID: a running job to wait for)
# 22/10 10:30: the retr span overlaps the oracle LD window by 0.81-0.99 (oracle 10/29 vs 7/29), so the other LD
# selector variants were dropped for harm checks on vt / cwe / mk3 / mv.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|CUDA error" $R/$1.log; }
job() { local n=$1; shift; [ -s $R/$n.jsonl ] && grep -qE "\[$n\] xong" $R/run.log && return; log $n "bat dau"
  until $PY scripts/recall_eval.py --select --move "$@" --out $R/$n.jsonl >> $R/$n.log 2>&1; do
    retry $n || { log $n "loi khong phai OOM"; return; }
    log $n "OOM, thu lai sau 2 phut"; : > $R/$n.log; sleep 120; done
  log $n "xong"; }
LD="--prompts data/infbench_ld50.jsonl"
VT="--tasks vt --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
if [ -n "$2" ]; then while ps -o stat= -p $2 2>/dev/null | grep -qv Z; do sleep 30; done; fi
RU="--lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
if [ "$1" = A ]; then
  job ld50_move_one_retr --pad 992 --agg retr $LD
  job vt_128k_move_one_retr --pad 992 --agg retr $VT
  job mk3_128k_move_one_retr --pad 992 --agg retr --tasks niah_multikey_3 $RU
else
  job kv20_move_one_retr --pad 992 --agg retr --prompts data/infbench_kv86.jsonl --limit 20
  job math_find20_move_one_retr --pad 992 --agg retr --prompts data/infbench.jsonl --tasks math_find --limit 20
  job cwe_128k_move_one_retr --pad 992 --agg retr --tasks cwe $RU
  job mv_128k_move_one_retr --pad 992 --agg retr --tasks niah_multivalue $RU
fi
log move_agg_$1 het
