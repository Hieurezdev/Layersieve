#!/bin/bash
# Resume the steps of scripts/run_multiturn_ext.sh that stopped on CUDA out-of-memory (other jobs filled the GPU): the
# drivers skip finished sessions, so a step is simply re-run, up to 20 times, 10 minutes apart, while its log shows OOM.
#   setsid nohup bash scripts/run_multiturn_resume.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; R=results/multiturn
log() { echo "$(date '+%F %T') $*" >> $R/ext.log; }
step() { local name=$1; shift
  for t in $(seq 20); do log "bat dau $name (lan $t)"
    "$@" > $R/$name.log 2>&1 && { log "xong $name"; return 0; }
    grep -q "OutOfMemoryError" $R/$name.log || { log "loi $name"; return 1; }
    log "OOM $name, thu lai sau 10 phut"; sleep 600; done; log "loi $name (OOM 20 lan)"; return 1; }
LLAMA=nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
LS_LLAMA="--start-layer 20 --drop-ratio 0.361 --decode-ref 25"
step scbench_qwen $PY scripts/session_hf.py --data data/scbench128k.jsonl --verify 2 --out $R/scbench_qwen.jsonl
step multiq_llama $PY scripts/multiturn_hf.py --model $LLAMA --data data/multiq_llama.jsonl --verify 2 \
  --modes dense,resieve,naive,partial $LS_LLAMA --out $R/hf_llama.jsonl
step scbench_llama $PY scripts/session_hf.py --model $LLAMA --data data/scbench128k_llama.jsonl --verify 2 \
  $LS_LLAMA --out $R/scbench_llama.jsonl
log het
