#!/bin/bash
# Why does --move break multi-span tasks (vt 17/17 dense-right prompts broken with --sel rel although 79% of the chain
# lines were moved)? Two suspects: (a) the packed spans share RoPE positions with the last context tokens and lose
# their order relative to the unmoved context (--vpos pack); (b) part of the evidence is left far away. Oracle spans
# (results/selfext/oracle_spans.json: the gold evidence statements) remove (b); --vpos perm (a permutation, no shared
# positions) removes (a). Then the real selector with perm.
# Team_1 may use 70GB in total: a job starts only when team_1 use + NEED <= CAP (lock so two lanes do not race).
#   bash scripts/run_vpos.sh A   |   bash scripts/run_vpos.sh B
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
  exec 9> $R/.vpos_gpu.lock; flock 9
  local w=0
  until [ $(team_mem) -le $((CAP - NEED)) ]; do
    [ $w -eq 0 ] && log $N "cho VRAM team_1 ($(team_mem) MiB)"; w=1; sleep 60; done
  "$@" 9>&- & local pid=$!; sleep 180; flock -u 9; exec 9>&-
  wait $pid; }
job() { N=$1; shift; [ -s $R/$N.jsonl ] && grep -qE "\[$N\] xong" $R/run.log && return; log $N "bat dau"
  until gate $PY scripts/recall_eval.py --select --move "$@" --out $R/$N.jsonl >> $R/$N.log 2>&1; do
    retry $N || { log $N "loi khong phai OOM"; return; }
    log $N "OOM, thu lai sau 2 phut"; : > $R/$N.log; sleep 120; done
  log $N "xong"; }
RU="--lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
OR="--spans-file $R/oracle_spans.json"
if [ "$1" = A ]; then
  job vt_128k_oracle_pack --tasks vt $RU $OR
  job vt_128k_oracle_perm --tasks vt $RU $OR --vpos perm
  job vt_128k_rel_snap_perm --tasks vt $RU --sel rel --snap --vpos perm
  job kv20_oracle_perm --prompts data/infbench_kv86.jsonl --limit 20 $OR --vpos perm
  job ld50_one_retr_snap_perm --prompts data/infbench_ld50.jsonl --pad 992 --agg retr --snap --vpos perm
else
  job mv_128k_oracle_pack --tasks niah_multivalue $RU $OR
  job mv_128k_oracle_perm --tasks niah_multivalue $RU $OR --vpos perm
  job mv_128k_rel_snap_perm --tasks niah_multivalue $RU --sel rel --snap --vpos perm
  job kv20_rel_snap_perm --prompts data/infbench_kv86.jsonl --limit 20 --sel rel --snap --vpos perm
  job mk3_128k_rel_snap_perm --tasks niah_multikey_3 $RU --sel rel --snap --vpos perm
fi
log vpos_$1 het
