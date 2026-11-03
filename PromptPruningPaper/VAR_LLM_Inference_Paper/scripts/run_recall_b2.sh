#!/bin/bash
# Lane B after run_recall.sh (PID $1): excerpt-recall ablations. Scoring from the LayerSieve layer alone (17), and
# smaller / larger excerpt budgets, on KV (20) and LongDialogue (first 100).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 $1 2>/dev/null; do sleep 30; done
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
KV="--prompts data/infbench_kv86.jsonl --limit 20"
LD="--prompts data/infbench_ld200.jsonl --limit 100"
rec kv20_recall_L17 --layers 17 $KV
rec kv20_recall_b512 --budget 512 $KV
rec ld100_recall_L17 --layers 17 $LD
rec ld100_recall_b512 --budget 512 $LD
rec ld100_recall_b4096 --budget 4096 $LD
log recall_b2 het
