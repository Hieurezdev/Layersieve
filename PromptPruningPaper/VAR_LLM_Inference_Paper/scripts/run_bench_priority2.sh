#!/bin/bash
# Nhu run_bench_priority.sh nhung FastKV va ASL (HF, de GPU ranh nhieu, ~30-40 GB moi cai) chay song song: chi do do
# chinh xac nen chay chung GPU khong anh huong ket qua. Cac job vLLM van chay mot minh (vLLM 0.6.3 khong tinh bo nho
# cua process khac khi cap KV).
#   WAIT_PID=<pid job dang chay> setsid nohup bash scripts/run_bench_priority2.sh >> results/bench/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python
R=results/bench; LOGDIR=$R/logs; mkdir -p $R/infbench $R/ruler $LOGDIR
IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl; L128=131072
SP=baselines/SpecPrefill/configs

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
vllm128() {
  local n=$1; shift
  job ib_$n $PY scripts/eval_bench.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru128_$n $PY scripts/eval_bench.py "$@" --lengths $L128 --prompts $RU --out $R/ruler/$n.jsonl
}
hf128() {
  local n=$1; shift
  job ib_$n $HF baselines/asl_driver.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru128_$n $HF baselines/asl_driver.py "$@" --lengths $L128 --prompts $RU --out $R/ruler/$n.jsonl
}

# job LayerSieve RULER 128k dang chay tu runner cu: doi no xong roi chay lai qua job() (chi con mau thieu, neu co)
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
vllm128 ls_unit_d0.25_r22 --method layersieve --unit --drop-ratio 0.25 --decode-ref 22

# ---- P1: FastKV va ASL song song
hf128 m_fastkv_l15_f0.375 --method fastkv --select-layer 15 --select-k 0.375 &
hf128 m_asl_t0.3_f0.375 --method asl --select-layer 10 --tau 0.3 --select-k 0.375 --monitor-k 2048 &
wait
vllm128 m_specprefill_p50 --method specprefill --spec-config $SP/config_p5_full_lah8.yaml --spec-yarn 4
vllm128 ls_unit_d0.12_r27 --method layersieve --unit --drop-ratio 0.12 --decode-ref 27

# ---- P2: ablation sang dan / cat mot lan x token / unit (cung ~0.71 token x layer)
vllm128 abl_oneshot_tok_l15_k0.375 --method layersieve --start-layer 15 --oneshot-keep 0.375
vllm128 abl_oneshot_unit_l15_k0.375 --method layersieve --unit --start-layer 15 --oneshot-keep 0.375
vllm128 abl_prog_tok_d0.25 --method layersieve --drop-ratio 0.25
vllm128 abl_prog_unit_d0.25 --method layersieve --unit --drop-ratio 0.25
echo "$(date '+%F %T') priority het"

# ---- P3: phan con lai
bash scripts/run_bench_rest.sh
