#!/bin/bash
# overlay move + confidence select: plain and recalled answers are both decoded, the higher mean log-prob is kept.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
rec() { local n=$1; shift; log $n "bat dau"
  $PY scripts/recall_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
rec kv20_sel_k --move --select --prompts data/infbench_kv86.jsonl --limit 20
for t in niah_multivalue cwe vt; do rec ${t}_128k_sel_k --move --select --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec ld50_sel_k --move --select --prompts data/infbench_ld50.jsonl
for t in qa_2 qa_1 fwe; do rec ${t}_128k_sel_k --move --select --tasks $t --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20; done
rec mk3_128k_sel_k --move --select --tasks niah_multikey_3 --lengths 131072 --prompts data/ruler_eval.jsonl --limit 40
for t in math_find code_debug; do rec ${t}20_sel_k --move --select --prompts data/infbench.jsonl --tasks $t --limit 20; done
log sel_k het
