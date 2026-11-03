#!/bin/bash
# Oracle spans on vt/mv: packed --move breaks (vt 82, mv 68 vs dense 97/85), --vpos perm too (84/69); leaving empty
# positions between the moved spans fixes it: --vgap 4096 vt 91 / mv 86, --vgap 16384 vt 98 / mv 95 (rescues 9/11),
# oracle copy vt 100 / mv 84. Now the real selector (--sel rel --snap) with --vgap 16384 on every task, harm guards
# (cwe, math_find) and an oracle gap sweep on mv.  Team_1 may use 70GB in total (gate below).
#   bash scripts/run_gap2.sh A   |   bash scripts/run_gap2.sh B
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
G="--move --sel rel --snap --vgap 16384"
if [ "$1" = A ]; then
  job vt_128k_rel_gap16k --tasks vt $RU $G
  job kv20_rel_gap16k --prompts data/infbench_kv86.jsonl --limit 20 $G
  job ld50_rel_gap16k --prompts data/infbench_ld50.jsonl $G
  job math_find20_rel_gap16k --prompts data/infbench.jsonl --tasks math_find --limit 20 $G
  job mv_128k_oracle_gap8k --tasks niah_multivalue $RU $OR --move --vgap 8192
else
  job mv_128k_rel_gap16k --tasks niah_multivalue $RU $G
  job mk3_128k_rel_gap16k --tasks niah_multikey_3 $RU $G
  job cwe_128k_rel_gap16k --tasks cwe $RU $G
  job mv_128k_oracle_gap32k --tasks niah_multivalue $RU $OR --move --vgap 32768
fi
log gap2_$1 het
