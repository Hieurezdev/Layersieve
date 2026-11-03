#!/bin/bash
# Do toc do: moi lich sang + sang KV decode, cung dense,
# trong mot tien trinh tren mot GPU ranh -> results/sweep/speed.jsonl (summarize_sweep.py va summarize_decode.py doc).
# Giu khoa hang doi (scripts/gpu_queue.sh) ngay tu luc cho: khong job nao khac duoc khoi dong cho den khi co GPU ranh
# (util <= 10%) va lan do nay da chiem bo nho, de job do chinh xac khong chen vao GPU vua ranh.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOGDIR=results/sweep/logs; mkdir -p $LOGDIR
source scripts/gpu_queue.sh
MAX_UTIL=10; MARGIN_MIB=1024; POLL=20
exec {lk}>"$LOCK"; flock "$lk"
g=$(pick_gpu 0.68)
echo "$(date '+%F %T') [speed_all] GPU $g"
CUDA_VISIBLE_DEVICES=$g $PY scripts/bench_speed.py --gpu-util 0.68 --n 3 --out results/sweep/speed.jsonl \
  --configs 28:0,17:0.12,17:0.12:27,7:0.08,7:0.08:27,17:0.25,17:0.12:22,17:0.12:17,14:0.06,14:0.10,14:0.15,14:0.25,7:0.03,7:0.05,7:0.12,17:0.25:27,17:0.25:22 \
  >> $LOGDIR/speed_all.log 2>&1 {lk}>&- &
pid=$!
sleep 240; flock -u "$lk"; exec {lk}>&-
wait $pid && echo "$(date '+%F %T') [speed_all] xong" || echo "$(date '+%F %T') [speed_all] loi (xem $LOGDIR/speed_all.log)"
