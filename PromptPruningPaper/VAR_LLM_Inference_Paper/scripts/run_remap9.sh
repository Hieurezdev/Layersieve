#!/bin/bash
# Query-only edits (--tail 512: the context is encoded with plain attention, only the question and the answer tokens
# change). The remap probe showed that shrinking or clamping distances for the question lowers the gold mass 4-16x,
# while a sharper softmax (logits x 1.3) raises it, so: logit scale on KV and LongDialogue, plus the LongDialogue
# probe. Second lane next to run_remap8.sh.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
run() { local n=$1; shift; log $n "bat dau"
  $PY scripts/selfextend_eval.py "$@" --out $R/$n.jsonl > $R/$n.log 2>&1; log $n "xong ($?)"; }
KV="--prompts data/infbench_kv86.jsonl --limit 20"
LD="--prompts data/infbench_ld200.jsonl --limit 100"
T="--tail 512"
run kv20_tail_scale1.3 $T --qscale 1.3 $KV
log remap_probe_ld "bat dau"
$PY scripts/remap_probe.py --task ld --n 30 --variants freq4k_g8,freq32k_g4,scale1.15,scale1.3,scale1.5,fastoff4k,fasthalf4k \
    --out $R/remap_probe_ld.json > $R/remap_probe_ld.log 2>&1
log remap_probe_ld "xong ($?)"
run ld100_tail_scale1.3 $T --qscale 1.3 $LD
log remap9 het
