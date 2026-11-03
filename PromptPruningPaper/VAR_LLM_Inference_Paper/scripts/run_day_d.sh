#!/bin/bash
# Day lane D (Qwen3-4B-Instruct-2507): one-window gem on real QA and on whole-context tasks, window size on LD50.
#   bash scripts/run_day_d.sh   (team_1 <= 70GB gate, ~32GB per job)
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; CAP=70000; NEED=32000
export PYTHONPATH=/root/team_1/hieurezdev/pylib_tf451 SE_MODEL=Qwen/Qwen3-4B-Instruct-2507; M=$SE_MODEL
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
job() { N=$1; S=$2; shift 2; [ -s $R/$N.jsonl ] && grep -qE "\[$N\] xong" $R/run.log && return; log $N "bat dau"
  until gate $PY scripts/$S "$@" --out $R/$N.jsonl >> $R/$N.log 2>&1; do
    retry $N || { log $N "loi khong phai OOM"; return; }
    log $N "OOM, thu lai sau 2 phut"; : > $R/$N.log; sleep 120; done
  log $N "xong"; }
RU="--lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
G="recall_eval.py --select --heads results/repos/heads_q3_4b.json --agg retr --gem --gem-sep"
for t in qa_1 qa_2; do job q3_${t}_128k_gem_retr $G --tasks $t $RU --pad 992 --snap; done
for t in qa_1 qa_2; do job q3_${t}_128k_top2k_gem $G --tasks $t $RU --budget 2048 --pad 0; done
job q3_math_find20_gem_retr $G --prompts data/infbench.jsonl --tasks math_find --limit 20 --pad 992 --snap
job q3_ld50_gem_w1k_b1k $G --prompts data/infbench_ld50.jsonl --pad 480 --budget 1024 --snap
job q3_ld50_gem_w4k_b4k $G --prompts data/infbench_ld50.jsonl --pad 2016 --budget 4096 --snap
log day_d het
