#!/bin/bash
# Llama-3.1-8B-UltraLong steps of the multi-turn study (peak ~62 GB on the HF reference). Before every attempt waits
# until the GPU has >= NEED MiB free for 2 checks in a row; on CUDA OOM waits again. The drivers skip finished documents,
# so every attempt resumes. Gives up after MAXWAIT seconds (default 48 h). Progress in results/multiturn/ext.log.
#   setsid nohup bash scripts/run_multiturn_llama.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; R=results/multiturn; NEED=${NEED:-70000}; t0=$(date +%s)
log() { echo "$(date '+%F %T') $*" >> $R/ext.log; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_room() { local ok=0; until [ $ok -ge 2 ]; do
    [ $(( $(date +%s) - t0 )) -gt ${MAXWAIT:-172800} ] && { log "bo cuoc (het ${MAXWAIT:-172800} s)"; exit 1; }
    [ "$(free_mib)" -ge $NEED ] && ok=$((ok + 1)) || ok=0; sleep 60; done; }
step() { local name=$1; shift
  while true; do wait_room; log "bat dau $name"
    "$@" >> $R/$name.log 2>&1 && { log "xong $name"; return 0; }
    tail -50 $R/$name.log | grep -q "OutOfMemoryError" || { log "loi $name"; return 1; }
    log "OOM $name, cho GPU"; done; }
LLAMA=nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
LS_LLAMA="--start-layer 20 --drop-ratio 0.361 --decode-ref 25"
step multiq_llama $PY scripts/multiturn_hf.py --model $LLAMA --data data/multiq_llama.jsonl --verify 2 \
  --modes dense,resieve,naive,partial $LS_LLAMA --out $R/hf_llama.jsonl
step scbench_llama $PY scripts/session_hf.py --model $LLAMA --data data/scbench128k_llama.jsonl --verify 2 \
  $LS_LLAMA --out $R/scbench_llama.jsonl
log het llama
