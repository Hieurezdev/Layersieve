#!/bin/bash
# Retrieval heads cover the evidence (top 1k tokens: vt 99%, mv 96% (rest is inside the question), KV 91%, mk3 92%),
# the --sel rel threshold is what dropped vt chain lines. Fixed budget of the top retrieval-head blocks (--agg retr,
# multi-span pick of whole top blocks: --pad 0, budget 1024 / 2048 = 16 / 32 blocks), then (a) --gem --gem-sep or (b) --move --vgap 16384.  (team_1 <= 70GB gate)
#   bash scripts/run_topk.sh A | B
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
RU="--lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
KV="--prompts data/infbench_kv86.jsonl --limit 20"
for b in 1024 2048; do
  T="--agg retr --budget $b --pad 0"; k=$((b / 1024))k
  if [ "$1" = A ]; then
    job vt_128k_top${k}_gem --tasks vt $RU $T --gem --gem-sep
    job mv_128k_top${k}_gem --tasks niah_multivalue $RU $T --gem --gem-sep
    job kv20_top${k}_gem $KV $T --gem --gem-sep
    job mk3_128k_top${k}_gem --tasks niah_multikey_3 $RU $T --gem --gem-sep
  else
    job vt_128k_top${k}_gap16k --tasks vt $RU $T --move --vgap 16384
    job mv_128k_top${k}_gap16k --tasks niah_multivalue $RU $T --move --vgap 16384
    job kv20_top${k}_gap16k $KV $T --move --vgap 16384
    job mk3_128k_top${k}_gap16k --tasks niah_multikey_3 $RU $T --move --vgap 16384
  fi
done
log topk_$1 het
