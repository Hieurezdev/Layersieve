#!/bin/bash
# (1) LongDialogue, one ~2k window around the top retrieval-head block (--pad 992 --snap), re-encoded alone (--gem):
#     LD50 50.0 first-name vs dense 18.0 -> run on LD200 b/c.
# (2) Harm check of the top-1k gem recipe on whole-context tasks (math_find, cwe).
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
for p in b c; do job ld200${p}_gem_retr --prompts data/ld200_parts/ld200_$p.jsonl --gem --gem-sep --pad 992 --agg retr --snap; done
job math_find20_top1k_gem --prompts data/infbench.jsonl --tasks math_find --limit 20 --agg retr --budget 1024 --pad 0 --gem --gem-sep
job cwe_128k_top1k_gem --tasks cwe $RU --agg retr --budget 1024 --pad 0 --gem --gem-sep
log gemretr het
