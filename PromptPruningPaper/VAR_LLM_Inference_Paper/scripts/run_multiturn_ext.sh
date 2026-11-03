#!/bin/bash
# Multi-turn extensions, one after the other (after the run with PID $1, if given, has ended):
#   progressive completion (fill<p>) on the multi-question sets, then SCBench at 128k with conversation history on Qwen2.5-7B-1M (smoke session first), then the second model
#   Llama-3.1-8B-UltraLong-1M (L = 32; paper Eq. 4: l0 = 0.61L = 20, rho = 1 - 0.6^(28/L) = 0.361, decode_ref = 0.79L = 25)
#   on the multi-question RULER / InfiniteBench sets and on SCBench. Progress in results/multiturn/ext.log.
#   setsid nohup bash scripts/run_multiturn_ext.sh [pid] > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; R=results/multiturn
log() { echo "$(date '+%F %T') $*" >> $R/ext.log; }
step() { local name=$1; shift; log "bat dau $name"; "$@" > $R/$name.log 2>&1 && log "xong $name" || { log "loi $name"; return 1; }; }
[ -n "$1" ] && { log "cho PID $1"; while kill -0 $1 2>/dev/null; do sleep 60; done; }

# progressive completion (fill<p>): check fill0 = resieve and fill1 = dense on 2 documents per task, then p = 1/4, 1/2, 3/4
step fill_check $PY scripts/multiturn_hf.py --data data/multiq.jsonl --n 2 --verify 0 \
  --modes dense,resieve,fill0,fill1 --out $R/hf_fill_check.jsonl &&
  for d in multiq multiq2; do
    step fill_$d $PY scripts/multiturn_hf.py --data data/$d.jsonl --tasks kv,mk3,mk2,qa1 --verify 0 \
      --modes dense,resieve,fill0.25,fill0.5,fill0.75 --out $R/hf_fill_$d.jsonl
  done
LLAMA=nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
LS_LLAMA="--start-layer 20 --drop-ratio 0.361 --decode-ref 25"
step scbench_smoke $PY scripts/session_hf.py --tasks kv --n 1 --verify 1 --out $R/scbench_smoke.jsonl &&
  step scbench_qwen $PY scripts/session_hf.py --data data/scbench128k.jsonl --verify 2 --out $R/scbench_qwen.jsonl
step build_llama_multiq $PY data/build_multiq.py --model $LLAMA --kv data/infbench_llama.jsonl \
  --ruler data/ruler_eval_llama.jsonl --tasks kv,mk3,mk1,mk2,mq,qa1 --n 50 --out data/multiq_llama.jsonl &&
  step multiq_llama $PY scripts/multiturn_hf.py --model $LLAMA --data data/multiq_llama.jsonl --verify 2 \
    --modes dense,resieve,naive,partial $LS_LLAMA --out $R/hf_llama.jsonl
step build_llama_scbench $PY data/build_scbench.py --model $LLAMA --n 20 --out data/scbench128k_llama.jsonl &&
  step scbench_llama $PY scripts/session_hf.py --model $LLAMA --data data/scbench128k_llama.jsonl --verify 2 \
    $LS_LLAMA --out $R/scbench_llama.jsonl
log het
