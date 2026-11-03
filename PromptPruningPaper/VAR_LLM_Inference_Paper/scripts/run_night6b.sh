#!/bin/bash
# Chen 2 chan doan Llama IB vao giua night6, ngay sau frontier: tam dung bash cua night6 (SIGSTOP) khi job frontier
# cuoi bat dau, cho no xong, chay 2 job, roi SIGCONT. Tach "san qua thap" vs "cat mot lan tot hon":
#   prog l17 floor .375 (C ~ .72, cung chi phi LS)  |  mot lan cat L19 giu 25% (cung trang thai cuoi voi LS, C ~ .70)
#   N6=<pid bash night6> setsid nohup bash scripts/run_night6b.sh >> results/floor/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOGDIR=results/review/logs; B=results/bench
job() {
  local name=$1; shift
  [ -f "$LOGDIR/$name.done" ] && return 0
  echo "$(date '+%F %T') [$name] bat dau"
  if "$@" >> "$LOGDIR/$name.log" 2>&1; then
    touch "$LOGDIR/$name.done"; echo "$(date '+%F %T') [$name] xong"
  else
    echo "$(date '+%F %T') [$name] loi (xem $LOGDIR/$name.log)"
  fi
}
LAST="fr_c72_acc_l17_f0.25_r0.4"
until grep -q "\[$LAST\] bat dau" results/floor/run.log; do sleep 60; done
kill -STOP $N6
while ps --ppid $N6 -o stat= | grep -qv Z; do sleep 30; done
LM=nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
A="--model $LM --method layersieve --unit --max-model-len 140000 --prompts data/infbench_llama.jsonl"
job llama_ib_prog_l17_f0.375 $PY scripts/eval_bench.py $A --start-layer 17 --drop-ratio 0.36 --min-keep 0.375 \
  --out $B/llama/infbench/ls_unit_l17_d0.36_f0.375_noref.jsonl
job llama_ib_onecut_l19_k0.25 $PY scripts/eval_bench.py $A --start-layer 19 --oneshot-keep 0.25 \
  --out $B/llama/infbench/abl_oneshot_unit_l19_k0.25.jsonl
kill -CONT $N6
