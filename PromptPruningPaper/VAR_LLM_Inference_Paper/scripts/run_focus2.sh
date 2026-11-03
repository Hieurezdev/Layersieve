#!/bin/bash
# Decision-token probe (LD_CAUSE.md s.13): the question locates the MASK scene, but at the decision token the
# retrieval heads drift to the wrong character's mentions in OTHER scenes (short prompt: gold 1.60 vs rival 0.22; long:
# gold-in-scene 0.07, rival elsewhere 0.37). Fix tested: --focus (+log(ctx/picked) on the picked span, retrieval heads
# thr 0.1, question + answer) with the scene picked by the retr one-span selector (+ --snap). No positions moved.
#   bash scripts/run_focus2.sh A   |   bash scripts/run_focus2.sh B   (team_1 <= 70GB gate)
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
F="--focus results/repos/heads_7b1m.json --head-thr 0.1 --pad 992 --agg retr --snap"
if [ "$1" = A ]; then
  job ld50_focus_retr $F --prompts data/infbench_ld50.jsonl
  job ld200c_focus_retr $F --prompts data/ld200_parts/ld200_c.jsonl
else
  job ld200b_focus_retr $F --prompts data/ld200_parts/ld200_b.jsonl
fi
log focus2_$1 het
