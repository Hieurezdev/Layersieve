#!/bin/bash
# Head-targeted focus: picked spans get +log(context / picked) only in the retrieval heads (score > 0.1, 45 of 784),
# only for the question and the answer; --select also decodes the plain answer for a paired comparison.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
F="--select --focus results/repos/heads_7b1m.json --head-thr 0.1"
rec kv20_focus $F --prompts data/infbench_kv86.jsonl --limit 20
for t in math_find code_debug; do rec ${t}20_focus $F --prompts data/infbench.jsonl --tasks $t --limit 20; done
for t in cwe qa_2 niah_multivalue vt qa_1 fwe; do rec ${t}_128k_focus $F --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_focus $F --prompts data/infbench_ld50.jsonl
rec mk3_128k_focus $F --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
log focus het
