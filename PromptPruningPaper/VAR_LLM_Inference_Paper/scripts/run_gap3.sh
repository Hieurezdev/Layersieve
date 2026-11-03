#!/bin/bash
# Real selector (--sel rel --snap) with --vgap 16384 (text order): vt 58->88, mv 80->86, cwe 19.5 (dense 24.5), math 85 =,
# but KV 75->30, mk3 100->80, LD50 22 (one span 32): with wide gaps a distractor span ends up nearer the question than
# the gold one. --vorder score puts the span with the highest selector score nearest the question. gap 16k and 4k.
# Team_1 may use 70GB in total (gate below).   bash scripts/run_gap3.sh A   |   bash scripts/run_gap3.sh B
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
for g in 16384 4096; do
  G="--move --sel rel --snap --vorder score --vgap $g"; k=$((g / 1024))k
  if [ "$1" = A ]; then
    job kv20_rel_gap${k}_ord --prompts data/infbench_kv86.jsonl --limit 20 $G
    job vt_128k_rel_gap${k}_ord --tasks vt $RU $G
    job ld50_rel_gap${k}_ord --prompts data/infbench_ld50.jsonl $G
  else
    job mk3_128k_rel_gap${k}_ord --tasks niah_multikey_3 $RU $G
    job mv_128k_rel_gap${k}_ord --tasks niah_multivalue $RU $G
  fi
done
log gap3_$1 het
