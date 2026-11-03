#!/bin/bash
# Lane A after run_remap8.sh (PID $1): excerpt recall on multikey-3 128k (copying guard), the second half of
# LongDialogue 200 (ld100_recall covers the first), and the rest of the 86 KV prompts (kv20_recall seeds the file).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 $1 2>/dev/null; do sleep 30; done
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec mk3_128k_recall --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
rec ld200b_recall --prompts data/infbench_ld200_b.jsonl
[ -f $R/kv86_recall.jsonl ] || cp $R/kv20_recall.jsonl $R/kv86_recall.jsonl
rec kv86_recall --prompts data/infbench_kv86.jsonl
log recall_a het
