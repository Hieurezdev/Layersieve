#!/bin/bash
# --move + --snap (retr span, whole lines) on the other tasks: LD50 rose 26 -> 32 with snapping (oracle 34).
# Does snapping change the gains (KV, mk3) or the harm (vt, cwe, mv)? Two lanes, one HF job each.
#   bash scripts/run_snap2.sh A   |   bash scripts/run_snap2.sh B
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
RU="--lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
if [ "$1" = A ]; then
  job vt_128k_move_one_retr_snap --tasks vt $RU
  job mk3_128k_move_one_retr_snap --tasks niah_multikey_3 $RU
  job kv20_move_one_retr_snap --prompts data/infbench_kv86.jsonl --limit 20
else
  job cwe_128k_move_one_retr_snap --tasks cwe $RU
  job mv_128k_move_one_retr_snap --tasks niah_multivalue $RU
fi
log snap2_$1 het
