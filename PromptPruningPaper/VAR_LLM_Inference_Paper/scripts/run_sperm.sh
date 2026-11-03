#!/bin/bash
# --vpos scale (order and proportions kept, distances x0.25): vt 80, mv 86, but KV 5 and mk3 70. Suspect: the moved
# spans land on the same positions as unmoved key-value pairs (collisions). Test: --vpos perm (packed, no collisions)
# and --vpos scaleperm (scale targets ranked into a permutation) on KV, mk3, vt, mv.  Team_1 <= 70GB (gate below).
#   bash scripts/run_sperm.sh
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
G="--move --sel rel --snap"
job kv20_rel_perm --prompts data/infbench_kv86.jsonl --limit 20 $G --vpos perm
job kv20_rel_sp25 --prompts data/infbench_kv86.jsonl --limit 20 $G --vpos scaleperm --vscale 0.25
job mk3_128k_rel_sp25 --tasks niah_multikey_3 $RU $G --vpos scaleperm --vscale 0.25
job vt_128k_rel_sp25 --tasks vt $RU $G --vpos scaleperm --vscale 0.25
job mv_128k_rel_sp25 --tasks niah_multivalue $RU $G --vpos scaleperm --vscale 0.25
log sperm het
