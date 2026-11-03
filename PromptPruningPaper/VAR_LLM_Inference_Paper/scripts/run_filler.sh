#!/bin/bash
# Length vs. interference control (scripts/build_ld_filler.py): the retrieval-head window kept at its original
# position, everything else replaced by essays / another screenplay, same total length; plain dense decoding.
# Qwen2.5-7B-1M first, then Qwen3-4B-2507 (its own env; ~32GB).
#   bash scripts/run_filler.sh   (team_1 <= 70GB gate)
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
job() { N=$1; S=$2; shift 2; [ -s $R/$N.jsonl ] && grep -qE "\[$N\] xong" $R/run.log && return; log $N "bat dau"
  until gate $PY scripts/$S "$@" --out $R/$N.jsonl >> $R/$N.log 2>&1; do
    retry $N || { log $N "loi khong phai OOM"; return; }
    log $N "OOM, thu lai sau 2 phut"; : > $R/$N.log; sleep 120; done
  log $N "xong"; }
for f in essay other; do job ld200_filler_$f ld_move_probe.py --mode none --prompts data/ld200_filler_$f.jsonl; done
export PYTHONPATH=/root/team_1/hieurezdev/pylib_tf451 SE_MODEL=Qwen/Qwen3-4B-Instruct-2507; NEED=32000
for f in essay other; do job q3_ld200_filler_$f ld_move_probe.py --mode none --prompts data/ld200_filler_$f.jsonl; done
log filler het
