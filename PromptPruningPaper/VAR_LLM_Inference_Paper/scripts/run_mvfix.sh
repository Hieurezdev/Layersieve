#!/bin/bash
# Re-run of the mv ORACLE experiments with exact oracle spans: the old char->token map (one-token decodes) drifted
# on multi-byte characters and cut some needles before the end of the value (oracle_spans_drift.json). vt / KV spans
# were unaffected.  (team_1 <= 70GB gate)   bash scripts/run_mvfix.sh
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
RU="--tasks niah_multivalue --lengths 131072 --prompts data/ruler_eval.jsonl --limit 20 --spans-file $R/oracle_spans.json"
job mv_128k_gem_oracle_v2 $RU --gem --gem-sep
job mv_128k_oracle_gap16k_v2 $RU --move --vgap 16384
job mv_128k_oracle_pack_v2 $RU --move
job mv_128k_oracle_copy_v2 $RU
job mv_128k_oracle_gap4k_v2 $RU --move --vgap 4096
log mvfix het
