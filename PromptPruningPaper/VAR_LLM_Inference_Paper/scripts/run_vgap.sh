#!/bin/bash
# Oracle spans (gold evidence statements) on vt/mv: --move packed (82/68 vs dense 97/85) and --vpos perm (84) both
# break the model, so neither leftover far evidence nor shared positions is the cause. Next: (1) --vgap = empty
# positions between moved spans (are adjacent, independently encoded spans the problem?), (2) oracle copy (the same
# spans re-encoded as text before the question; real-selector copy gave mv 94).
# Team_1 may use 70GB in total: a job starts only when team_1 use + NEED <= CAP.
#   bash scripts/run_vgap.sh A   |   bash scripts/run_vgap.sh B
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
if [ "$1" = A ]; then
  job vt_128k_oracle_gap4k --tasks vt $RU $OR --move --vgap 4096
  job vt_128k_oracle_copy --tasks vt $RU $OR
  job mv_128k_oracle_perm --tasks niah_multivalue $RU $OR --move --vpos perm
else
  job mv_128k_oracle_gap4k --tasks niah_multivalue $RU $OR --move --vgap 4096
  job mv_128k_oracle_copy --tasks niah_multivalue $RU $OR
  job vt_128k_oracle_gap16k --tasks vt $RU $OR --move --vgap 16384
  job mv_128k_oracle_gap16k --tasks niah_multivalue $RU $OR --move --vgap 16384
fi
log vgap_$1 het
