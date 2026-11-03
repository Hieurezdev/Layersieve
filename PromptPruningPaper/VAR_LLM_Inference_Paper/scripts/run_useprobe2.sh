#!/bin/bash
# Decision-token probe (scripts/use_probe.py) on LD200 dense answers: logit lens, per-head / per-MLP logit attribution
# to gold-vs-rival, attention mass on gold mentions near the MASK. Team_1 <= 70GB (gate).   bash scripts/run_useprobe.sh
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; CAP=70000; NEED=26000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
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
# Same probe on the --move --snap one-span answers (LD200), rival = the dense answer's name.
#   bash scripts/run_useprobe2.sh [--reverse]
MV=$R/ld50_move_one_retr_snap.jsonl,$R/ld200b_move_one_retr_snap.jsonl,$R/ld200c_move_one_retr_snap.jsonl
N=useprobe_ld200_move$1; log $N "bat dau"
until gate $PY scripts/use_probe.py --prompts data/infbench_ld200.jsonl --dense $R/ld200_g1w0.jsonl --moved $MV \
    --rival-from $R/ld200_g1w0.jsonl --out $R/useprobe/ld200_move $1 >> $R/$N.log 2>&1; do
  grep -qiE "out of memory|CUDA error" $R/$N.log || { log $N "loi khong phai OOM"; break; }
  log $N "OOM, thu lai sau 2 phut"; sleep 120; done
log $N xong
