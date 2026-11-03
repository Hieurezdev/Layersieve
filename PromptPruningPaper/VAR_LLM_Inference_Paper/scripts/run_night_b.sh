#!/bin/bash
# Overnight lane B (Qwen3-4B-Instruct-2507): the gem recipe on the stronger model, with its own retrieval heads.
#   bash scripts/run_night_b.sh   (team_1 <= 70GB gate, ~32GB per job)
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
job q3_ld50_gem_retr $G --prompts data/infbench_ld50.jsonl --pad 992 --snap
job q3_ld50_top1k_gem $G --prompts data/infbench_ld50.jsonl --budget 1024 --pad 0
job q3_kv20_top2k_gem $G --prompts data/infbench_kv86.jsonl --limit 20 --budget 2048 --pad 0
for p in b c; do job q3_ld200${p}_gem_retr $G --prompts data/ld200_parts/ld200_$p.jsonl --pad 992 --snap; done
for p in b c; do job q3_ld200${p}_top1k_gem $G --prompts data/ld200_parts/ld200_$p.jsonl --budget 1024 --pad 0; done
log night_b het
