#!/bin/bash
# Chan doan fwe (RULER 128k): LS mat chu dung thu 3 (sinh muon nhat trong decode) nhieu hon ASL / dense, chu dung thu
# nhat (tu prefill) gan nhu khong mat -> nghi sang KV o decode (layer < r chi giu KV token song toi r). Chay F4 khong
# sang KV va voi r=27, so voi F4 r=22. Xong thi chay tiep hang doi P1d.
#   WAIT_PID=<pid> setsid nohup bash scripts/run_fwe_diag.sh >> results/bench/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/bench; LOGDIR=$R/logs; RU=data/ruler_eval.jsonl
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
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
F4="--method layersieve --unit --drop-ratio 0.4 --min-keep 0.25"
job diag_fwe_f4_noref $PY scripts/eval_bench.py $F4 --lengths 131072 --tasks fwe,cwe --prompts $RU \
  --out $R/ruler/ls_unit_d0.4_f0.25_noref.jsonl
job diag_fwe_f4_r27 $PY scripts/eval_bench.py $F4 --decode-ref 27 --lengths 131072 --tasks fwe,cwe --prompts $RU \
  --out $R/ruler/ls_unit_d0.4_f0.25_r27.jsonl
echo "$(date '+%F %T') chan doan fwe xong"
bash scripts/run_bench_p1d.sh
