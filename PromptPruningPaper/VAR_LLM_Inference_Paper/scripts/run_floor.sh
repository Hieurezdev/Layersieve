#!/bin/bash
# Muc san phi (ty le token con lai o cuoi) co phu thuoc co model / do dai khong?
# Voi l0 = 0.5L va rho = 0.4, san dat sau ~3-6 layer o moi co model -> layer cuoi luon giu dung phi * phan co the bo,
# nen cau hoi con lai la chon phi bao nhieu. Sweep phi in {.05,.10,.15,.25,.35}:
#   A) hard set (kv/vt/hotpotqa/2wiki, 1k-27k token, chi cau dense dung) cho 6 model -> results/floor/hard/
#   B) RULER 32k cwe/fwe/niah_multikey_3 (100 cau moi task) cho 3B, 7B-1M, 14B, 32B + dense -> results/floor/ruler/
#   C) 7B-1M, l0=17 (cau hinh chinh): cwe/fwe o 128k de xem phi la ty le hay so token tuyet doi
#      (32k da co trong A/B khi l0=14; phi=.25 l0=17 o 32k/128k da co trong results/bench/ruler/ls_unit_d0.4_f0.25_noref)
# Xong thi chay lai hang doi p2 (resume).
#   WAIT_PID=<pid eval dang chay> setsid nohup bash scripts/run_floor.sh >> results/floor/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/floor; LOGDIR=$R/logs; mkdir -p $R/hard $R/ruler $LOGDIR
declare -A M=([3b]=Qwen/Qwen2.5-3B-Instruct [7b]=Qwen/Qwen2.5-7B-Instruct [7b1m]=Qwen/Qwen2.5-7B-Instruct-1M
              [14b]=Qwen/Qwen2.5-14B-Instruct [14b1m]=Qwen/Qwen2.5-14B-Instruct-1M [32b]=Qwen/Qwen2.5-32B-Instruct)
declare -A L0=([3b]=18 [7b]=14 [7b1m]=14 [14b]=24 [14b1m]=24 [32b]=32)
FLOORS="0.05 0.10 0.15 0.25 0.35"

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

# A) hard set
for m in 7b1m 7b 3b 14b 14b1m 32b; do
  for f in $FLOORS; do
    job hard_${m}_f$f $PY scripts/eval_hard.py --model ${M[$m]} --method layersieve --unit --no-ttft --graph \
      --gpu-util 0.85 --ids results/scaling/ids/$m.txt --start-layer ${L0[$m]} --drop-ratio 0.4 --min-keep $f \
      --out $R/hard/${m}_f$f.jsonl
  done
done
# B) RULER 32k
T=cwe,fwe,niah_multikey_3
for m in 7b1m 3b 14b 32b; do
  job ru_${m}_dense $PY scripts/eval_bench.py --model ${M[$m]} --method dense --tasks $T --lengths 32768 \
    --max-model-len 32768 --prompts data/ruler_eval.jsonl --out $R/ruler/${m}_dense.jsonl
  for f in $FLOORS; do
    job ru_${m}_f$f $PY scripts/eval_bench.py --model ${M[$m]} --method layersieve --unit --start-layer ${L0[$m]} \
      --drop-ratio 0.4 --min-keep $f --tasks $T --lengths 32768 --max-model-len 32768 \
      --prompts data/ruler_eval.jsonl --out $R/ruler/${m}_f$f.jsonl
  done
done
# C) 7B-1M l0=17: 32k + 128k cwe/fwe
for f in 0.05 0.10 0.15 0.35; do
  job ru_7b1m17_f$f $PY scripts/eval_bench.py --method layersieve --unit --start-layer 17 --drop-ratio 0.4 \
    --min-keep $f --tasks cwe,fwe --lengths 32768,131072 --prompts data/ruler_eval.jsonl --out $R/ruler/7b1m17_f$f.jsonl
done
echo "$(date '+%F %T') het floor, chay lai p2"
setsid nohup bash scripts/run_bench_p2.sh >> results/bench/run.log 2>&1 < /dev/null &
