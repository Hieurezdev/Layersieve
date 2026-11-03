#!/bin/bash
# P1f: chay du F4 co sang KV decode (r22) ngay sau SlimInfer, roi tiep hang doi P1e (job da xong deu bo qua).
#   WAIT_PID=<pid> setsid nohup bash scripts/run_bench_p1f.sh >> results/bench/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/bench; LOGDIR=$R/logs; IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl; L128=131072
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
vllm128() {
  local n=$1; shift
  job ib_$n $PY scripts/eval_bench.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru128_$n $PY scripts/eval_bench.py "$@" --lengths $L128 --prompts $RU --out $R/ruler/$n.jsonl
}
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
vllm128 m_slim_0.75_0.55_0.35 --method slim --slim-frac 0.75,0.55,0.35
vllm128 ls_unit_d0.4_f0.25_r22 --method layersieve --unit --drop-ratio 0.4 --min-keep 0.25 --decode-ref 22
bash scripts/run_bench_p1e.sh
