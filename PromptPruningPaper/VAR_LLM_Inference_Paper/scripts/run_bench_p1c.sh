#!/bin/bash
# P1c: LayerSieve nhanh hon ASL ca o prefill (ASL ~0.84-0.86 token x layer) ma giu do chinh xac: sang dan co san giu
# toi thieu (--min-keep) de rho lon / l0 som khong lam doi token o cwe / fwe. Muc tieu compute ~0.68-0.76.
#   F1 l0=17 rho=.25 san .30 (0.76) | F2 l0=14 rho=.25 san .30 (0.68) | F3 l0=14 rho=.35 san .35 (0.69)
#   F4 l0=17 rho=.40 san .25 (0.72)
# 1) toc do (TTFT/TPOT, 128k/256k) cac lich tren + lich cat mot lan o L17..L23 de tinh toc do ASL theo dung phan bo layer
#    no tu chon; 2) sang loc cwe/fwe/niah_multikey_3 128k; 3) chay du F1, F2; 4) phan con lai cua P1b, P2, P3.
#   WAIT_PID=<pid> setsid nohup bash scripts/run_bench_p1c.sh >> results/bench/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=results/bench; LOGDIR=$R/logs; mkdir -p $R/infbench $R/ruler $LOGDIR
IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl; L128=131072

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
screen() {
  local n=$1; shift
  job scr_$n $PY scripts/eval_bench.py "$@" --lengths $L128 --tasks cwe,fwe,niah_multikey_3 --prompts $RU \
    --out $R/ruler/$n.jsonl
}

if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
# job dang chay luc doi hang doi: chay lai qua job() de danh dau xong (chi con mau thieu, neu co)
vllm128 ls_unit_d0.12_r27 --method layersieve --unit --drop-ratio 0.12 --decode-ref 27

job speed_floor $PY scripts/bench_speed.py --graph --model Qwen/Qwen2.5-7B-Instruct-1M --prompts data/speed.jsonl \
  --lengths 131072,262144 --n 2 --decode 256 --gpu-util 0.93 \
  --configs "dense,17:0.25:22:0.3,14:0.25:22:0.3,14:0.35:22:0.35,17:0.4:22:0.25,17:0.08:27,17:0.12:27,o17:0.375,o18:0.375,o19:0.375,o20:0.375,o21:0.375,o22:0.375,o23:0.375" \
  --out $R/speed_vs.jsonl

F1="--method layersieve --unit --drop-ratio 0.25 --min-keep 0.3 --decode-ref 22"
F2="--method layersieve --unit --start-layer 14 --drop-ratio 0.25 --min-keep 0.3 --decode-ref 22"
F3="--method layersieve --unit --start-layer 14 --drop-ratio 0.35 --min-keep 0.35 --decode-ref 22"
F4="--method layersieve --unit --drop-ratio 0.4 --min-keep 0.25 --decode-ref 22"
screen ls_unit_d0.25_f0.3_r22 $F1
screen ls_unit_l14_d0.25_f0.3_r22 $F2
screen ls_unit_l14_d0.35_f0.35_r22 $F3
screen ls_unit_d0.4_f0.25_r22 $F4
echo "$(date '+%F %T') sang loc san xong"
vllm128 ls_unit_d0.25_f0.3_r22 $F1
vllm128 ls_unit_l14_d0.25_f0.3_r22 $F2

# ---- phan con lai cua P1b, P2, P3
vllm128 m_slim_0.75_0.55_0.35 --method slim --slim-frac 0.75,0.55,0.35
vllm128 ls_unit_d0.08_r27 --method layersieve --unit --drop-ratio 0.08 --decode-ref 27
vllm128 ls_tok_d0.12_r27 --method layersieve --drop-ratio 0.12 --decode-ref 27
vllm128 abl_oneshot_tok_l15_k0.375 --method layersieve --start-layer 15 --oneshot-keep 0.375
vllm128 abl_oneshot_unit_l15_k0.375 --method layersieve --unit --start-layer 15 --oneshot-keep 0.375
vllm128 abl_prog_tok_d0.25 --method layersieve --drop-ratio 0.25
vllm128 abl_prog_unit_d0.25 --method layersieve --unit --drop-ratio 0.25
bash scripts/run_bench_rest.sh
