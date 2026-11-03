#!/bin/bash
# Only the retrieved pieces, re-encoded from scratch (--gem: sink + picked spans in text order + question, fresh
# positions; --gem-sep puts "\n...\n" between pieces). Oracle spans = the ceiling of "evidence only"; --sel rel --snap =
# the real selector (with its junk spans); LD: the retr one-span scene.  (team_1 <= 70GB gate)
#   bash scripts/run_gem2.sh A   |   bash scripts/run_gem2.sh B
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
OR="--spans-file $R/oracle_spans.json"
G="--gem --gem-sep"
KV="--prompts data/infbench_kv86.jsonl --limit 20"
if [ "$1" = A ]; then
  job vt_128k_gem_oracle --tasks vt $RU $OR $G
  job vt_128k_gem_rel --tasks vt $RU $G --sel rel --snap
  job kv20_gem_rel $KV $G --sel rel --snap
  job ld50_gem_retr --prompts data/infbench_ld50.jsonl $G --pad 992 --agg retr --snap
else
  job mv_128k_gem_oracle --tasks niah_multivalue $RU $OR $G
  job mv_128k_gem_rel --tasks niah_multivalue $RU $G --sel rel --snap
  job kv20_gem_oracle $KV $OR $G
  job mk3_128k_gem_rel --tasks niah_multikey_3 $RU $G --sel rel --snap
fi
log gem2_$1 het
