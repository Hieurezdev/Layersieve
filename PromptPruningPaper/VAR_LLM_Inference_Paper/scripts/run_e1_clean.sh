#!/bin/bash
# E1 (scripts/bench_multiturn.py) on an idle GPU: waits until no process at all has used the GPU for 3 minutes in a
# row (gives up after MAXWAIT seconds, default 24 h), then runs layers + prefix at 128k and 256k, twice each, into results/multiturn/clean/.
# A vLLM start that fails because another process changed GPU memory during profiling is retried.
#   setsid nohup bash scripts/run_e1_clean.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
R=results/multiturn/clean; mkdir -p $R
log() { echo "$(date '+%F %T') $*" >> $R/run.log; }
idle=0; t0=$(date +%s)
log "cho GPU trong"
until [ $idle -ge 3 ]; do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && idle=$((idle + 1)) || idle=0
  [ $(( $(date +%s) - t0 )) -gt ${MAXWAIT:-86400} ] && { log "bo cuoc sau ${MAXWAIT:-86400} s"; exit 1; }
  sleep 60
done
log "GPU trong, bat dau"
run() { local log=$1; shift
  for t in 1 2 3 4; do .venv/bin/python scripts/bench_multiturn.py "$@" >> $log 2>&1 && return
    grep -q "memory profiling" $log || { log "loi: $*"; return; }; sleep 60; done; }
for rep in 1 2; do
  for L in 131072 262144; do U=$([ $L = 131072 ] && echo 0.4 || echo 0.49)
    run $R/layers.log --mode layers --lengths $L --gpu-util $U --out $R/layers.jsonl
    run $R/prefix.log --mode prefix --lengths $L --gpu-util $U --out $R/prefix.jsonl
  done
done
log "xong; GPU khac luc ket thuc: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
