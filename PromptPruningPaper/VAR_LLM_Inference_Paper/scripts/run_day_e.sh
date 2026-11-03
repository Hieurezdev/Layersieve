#!/bin/bash
# Day lane E (Qwen2.5-7B-1M): is the gem harm on aggregation tasks only budget? top retrieval-head blocks, re-encoded,
# budget 4k / 16k on fwe and cwe (1k/2k already: cwe 1k 6.5, fwe 2k 71.7; dense 24.5 / 80).
#   bash scripts/run_day_e.sh   (team_1 <= 70GB gate)
#   bash scripts/run_gemretr.sh   (team_1 <= 70GB gate)
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; CAP=70000; NEED=26000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|CUDA error" $R/$1.log; }
team_mem() {                                   # MiB used by GPU processes started from /root/team_1
  local s=0
  while IFS=, read -r p m; do
    p=${p// /}; m=${m// MiB/}; m=${m// /}
    { readlink /proc/$p/cwd; tr '\0' ' ' < /proc/$p/cmdline; } 2>/dev/null | grep -q /root/team_1 && s=$((s + m))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
  echo $s; }
gate() {                                       # wait for room, start "$@" in the background, hold the lock 3 min
  exec 9> $R/.team1_gpu.lock; flock 9
  local w=0
  until [ $(team_mem) -le $((CAP - NEED)) ]; do
    [ $w -eq 0 ] && log $N "cho VRAM team_1 ($(team_mem) MiB)"; w=1; sleep 60; done
  "$@" 9>&- & local pid=$!; sleep 180; flock -u 9; exec 9>&-
  wait $pid; }
job() { N=$1; shift; [ -s $R/$N.jsonl ] && grep -qE "\[$N\] xong" $R/run.log && return; log $N "bat dau"
  until gate $PY scripts/recall_eval.py --select "$@" --out $R/$N.jsonl >> $R/$N.log 2>&1; do
    retry $N || { log $N "loi khong phai OOM"; return; }
    log $N "OOM, thu lai sau 2 phut"; : > $R/$N.log; sleep 120; done
  log $N "xong"; }
RU="--prompts data/ruler_eval.jsonl --lengths 131072 --limit 20"
for b in 4096 16384; do
  k=$((b / 1024))k
  for t in fwe cwe; do job ${t}_128k_top${k}_gem --tasks $t $RU --agg retr --budget $b --pad 0 --gem --gem-sep; done
done
log day_e het
