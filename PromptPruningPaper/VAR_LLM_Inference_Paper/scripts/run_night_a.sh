#!/bin/bash
# Overnight lane A (Qwen2.5-7B-1M): does the gem recipe (only retrieval-head-picked pieces, re-encoded) carry to real QA
# (RULER qa_1/qa_2), hurt a whole-context task (fwe), and hold at n=50 (KV, mv)? Plus the LD one-window size
# (1k / 4k around the top block; 2k = ld50_gem_retr).
#   bash scripts/run_night_a.sh   (team_1 <= 70GB gate)
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
RU="--prompts data/ruler_eval.jsonl --lengths 131072"
T2="--agg retr --budget 2048 --pad 0 --gem --gem-sep"
job qa_1_128k_top2k_gem --tasks qa_1 $RU --limit 20 $T2
job qa_2_128k_top2k_gem --tasks qa_2 $RU --limit 20 $T2
job ld50_gem_retr_w1k --prompts data/infbench_ld50.jsonl --gem --gem-sep --pad 480 --agg retr --snap
job ld50_gem_retr_w4k --prompts data/infbench_ld50.jsonl --gem --gem-sep --pad 2016 --agg retr --snap
job fwe_128k_top2k_gem --tasks fwe $RU --limit 20 $T2
job kv50_top2k_gem --prompts data/infbench_kv86.jsonl --limit 50 $T2
job mv50_128k_top2k_gem --tasks niah_multivalue $RU --limit 50 $T2
log night_a het
