#!/bin/bash
# P1b: bien the LayerSieve o compute <= ASL (ASL do duoc ~0.84 IB / 0.86 RULER 128k token x layer, chua tinh chi phi
# cham diem tung layer) + SlimInfer, roi quay lai P2 / P3.
#   ASL thang LS d0.25 o RULER cwe/fwe: ASL giu 37.5% tu L17-20, LS d0.25 con 4% o layer cuoi -> thu rho nho hon,
#   bat dau muon hon, va token-level.
#   Sang loc truoc tren cwe+fwe 128k (ghi thang vao file ket qua, job day du sau do chay tiep phan con thieu).
#   WAIT_PID=<pid job dang chay> setsid nohup bash scripts/run_bench_p1b.sh >> results/bench/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=results/bench; LOGDIR=$R/logs; mkdir -p $R/infbench $R/ruler $LOGDIR
IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl; L128=131072
SP=baselines/SpecPrefill/configs

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
  job scr_$n $PY scripts/eval_bench.py "$@" --lengths $L128 --tasks cwe,fwe --prompts $RU --out $R/ruler/$n.jsonl
}

if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
vllm128 m_specprefill_p50 --method specprefill --spec-config $SP/config_p5_full_lah8.yaml --spec-yarn 4

V_A="--method layersieve --unit --drop-ratio 0.12 --decode-ref 27"                    # 0.80
V_B="--method layersieve --drop-ratio 0.12 --decode-ref 27"                           # 0.80, token-level
V_C="--method layersieve --unit --start-layer 20 --drop-ratio 0.25 --decode-ref 24"   # 0.81
V_D="--method layersieve --unit --drop-ratio 0.08 --decode-ref 27"                    # 0.85
screen ls_unit_d0.12_r27 $V_A
screen ls_tok_d0.12_r27 $V_B
screen ls_unit_l20_d0.25_r24 $V_C
screen ls_unit_d0.08_r27 $V_D
echo "$(date '+%F %T') sang loc cwe/fwe xong"

vllm128 ls_unit_d0.12_r27 $V_A
vllm128 m_slim_0.75_0.55_0.35 --method slim --slim-frac 0.75,0.55,0.35
vllm128 ls_tok_d0.12_r27 $V_B
vllm128 ls_unit_l20_d0.25_r24 $V_C
vllm128 ls_unit_d0.08_r27 $V_D
echo "$(date '+%F %T') P1b het"

# ---- P2 ablation, P3 phan con lai
vllm128 abl_oneshot_tok_l15_k0.375 --method layersieve --start-layer 15 --oneshot-keep 0.375
vllm128 abl_oneshot_unit_l15_k0.375 --method layersieve --unit --start-layer 15 --oneshot-keep 0.375
vllm128 abl_prog_tok_d0.25 --method layersieve --drop-ratio 0.25
vllm128 abl_prog_unit_d0.25 --method layersieve --unit --drop-ratio 0.25
bash scripts/run_bench_rest.sh
