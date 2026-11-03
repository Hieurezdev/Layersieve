#!/bin/bash
# Same diagnostics on a stronger model: Qwen3-4B-Instruct-2507 (36 layers, 32 q / 8 kv heads, head_dim 128, 262k
# native). transformers 4.51 from a private dir (the shared .venv has 4.50, no qwen3); SE_MODEL picks the model for
# every script that imports selfextend_eval. ~32GB per job.  (team_1 <= 70GB gate)
#   bash scripts/run_q3.sh
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/selfext; CAP=70000; NEED=32000
export PYTHONPATH=/root/team_1/hieurezdev/pylib_tf451 SE_MODEL=Qwen/Qwen3-4B-Instruct-2507; M=$SE_MODEL
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
RU="--lengths 131072 --prompts data/ruler_eval.jsonl --limit 20"
HQ=results/repos/heads_q3_4b.json
[ -s $HQ ] || { log heads_q3 "bat dau"; gate $PY scripts/retrieval_heads.py --model $M --prompts data/hard.jsonl --task kv --n 30 --out $HQ >> $R/heads_q3.log 2>&1; log heads_q3 "xong ($?)"; }
job q3_ld200_dense ld_move_probe.py --mode none --prompts data/infbench_ld200.jsonl
job q3_ld200_short_w1024 ld_move_probe.py --mode short --w 1024 --prompts data/infbench_ld200.jsonl
for t in vt niah_multivalue niah_multikey_3; do job q3_${t}_128k_dense selfextend_eval.py --tasks $t $RU; done
job q3_kv20_dense selfextend_eval.py --prompts data/infbench_kv86.jsonl --limit 20
MV="recall_eval.py --select --move --pad 992 --agg retr --heads $HQ --snap"
job q3_ld50_move_one_retr_snap $MV --prompts data/infbench_ld50.jsonl
job q3_ld200b_move_one_retr_snap $MV --prompts data/ld200_parts/ld200_b.jsonl
job q3_ld200c_move_one_retr_snap $MV --prompts data/ld200_parts/ld200_c.jsonl
log q3 het
