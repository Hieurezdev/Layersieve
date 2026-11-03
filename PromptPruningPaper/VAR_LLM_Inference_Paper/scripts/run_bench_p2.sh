#!/bin/bash
# P2: F4 (LayerSieve unit, l0=17 rho=0.4 san 25%) la cau hinh chinh; xep lai hang doi:
#   1) F4 (co / khong sang KV decode) RULER 32k + 64k
#   2) baseline cung chi phi (FastKV, ASL, SlimInfer, SpecPrefill) RULER 32k + 64k
#   3) ablation khop F4 (IB + RULER 128k, khong sang KV de tach phan prefill):
#      cat mot lan L15 giu 37.5% {token, unit}; sang dan co san {token}; sang dan khong san {unit}
#      (sang dan co san unit = ls_unit_d0.4_f0.25_noref, da co)
#   4) muc mac dinh cua tung paper (d_*)
# Bo: RULER 32k/64k cua LS cu (rho .25 r22, rho .12 r27), LS rho .08 / token rho .12 (khong con dung).
#   setsid nohup bash scripts/run_bench_p2.sh >> results/bench/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python
R=results/bench; LOGDIR=$R/logs; mkdir -p $R/infbench $R/ruler $LOGDIR
IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl
SP=baselines/SpecPrefill/configs
bash scripts/run_night5.sh >> results/floor/run.log 2>&1   # sweep l0 (lich nhanh / cham) truoc ablation

job() {  # job <ten> <lenh...>: bo qua neu da xong; log o $LOGDIR/<ten>.log
  local name=$1; shift
  [ -f "$LOGDIR/$name.done" ] && return 0
  echo "$(date '+%F %T') [$name] bat dau"
  if "$@" >> "$LOGDIR/$name.log" 2>&1; then
    touch "$LOGDIR/$name.done"; echo "$(date '+%F %T') [$name] xong"
  else
    echo "$(date '+%F %T') [$name] loi (xem $LOGDIR/$name.log)"
  fi
}

vllm() {  # vllm <ten> <tham so eval_bench...>: InfiniteBench roi RULER
  local n=$1; shift
  job ib_$n $PY scripts/eval_bench.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru_$n $PY scripts/eval_bench.py "$@" --prompts $RU --out $R/ruler/$n.jsonl
}
spec() {  # spec <ten> <config>: draft native o 32k, YaRN x4 o 64k/128k va InfiniteBench (<= 128k)
  local n=$1 c=$2
  job ib_$n $PY scripts/eval_bench.py --method specprefill --spec-config $c --spec-yarn 4 --prompts $IB \
    --out $R/infbench/$n.jsonl
  job ru_${n}_32k $PY scripts/eval_bench.py --method specprefill --spec-config $c --lengths 32768 --max-model-len 32768 \
    --prompts $RU --out $R/ruler/$n.jsonl
  job ru_${n}_long $PY scripts/eval_bench.py --method specprefill --spec-config $c --spec-yarn 4 --lengths 65536,131072 \
    --prompts $RU --out $R/ruler/$n.jsonl
}
hf() {  # hf <ten> <tham so asl_driver...>
  local n=$1; shift
  job ib_$n $HF baselines/asl_driver.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru_$n $HF baselines/asl_driver.py "$@" --prompts $RU --out $R/ruler/$n.jsonl
}
vllm128() {
  local n=$1; shift
  job ib_$n $PY scripts/eval_bench.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru128_$n $PY scripts/eval_bench.py "$@" --lengths 131072 --prompts $RU --out $R/ruler/$n.jsonl
}
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
F4="--method layersieve --unit --drop-ratio 0.4 --min-keep 0.25"

# ---- 1) F4 RULER 32k/64k (job ru_ chay du 3 do dai, 128k da co nen chi con 32k/64k)
vllm ls_unit_d0.4_f0.25_r22 $F4 --decode-ref 22
vllm ls_unit_d0.4_f0.25_noref $F4

# ---- 2) baseline RULER 32k/64k; FastKV va ASL (HF) song song
hf m_fastkv_l15_f0.375 --method fastkv --select-layer 15 --select-k 0.375 &
hf m_asl_t0.3_f0.375 --method asl --select-layer 10 --tau 0.3 --select-k 0.375 --monitor-k 2048 &
wait
vllm m_slim_0.75_0.55_0.35 --method slim --slim-frac 0.75,0.55,0.35
spec m_specprefill_p50 $SP/config_p5_full_lah8.yaml
echo "$(date '+%F %T') bang chinh 32k/64k xong"

# ---- 3) ablation khop F4
vllm128 abl_oneshot_tok_l15_k0.375 --method layersieve --start-layer 15 --oneshot-keep 0.375
vllm128 abl_oneshot_unit_l15_k0.375 --method layersieve --unit --start-layer 15 --oneshot-keep 0.375
vllm128 abl_prog_tok_d0.4_f0.25 --method layersieve --drop-ratio 0.4 --min-keep 0.25
vllm128 abl_prog_unit_d0.4_nofloor --method layersieve --unit --drop-ratio 0.4
echo "$(date '+%F %T') ablation xong"

# ---- 4) muc mac dinh cua tung paper
hf d_fastkv_l15_k2048 --method fastkv --select-layer 15 --select-k 2048
hf d_asl_t0.3_k2048 --method asl --select-layer 10 --tau 0.3 --select-k 2048
spec d_specprefill_p10 $SP/config_p1_full_lah8.yaml
vllm d_slim_12k_6k_4k --method slim
hf d_snapkv_k2048 --method vanilla --cache-k 2048
echo "$(date '+%F %T') P2 het"
