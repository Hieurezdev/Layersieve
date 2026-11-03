#!/bin/bash
# Thu tu uu tien cho so sanh baseline (thay cho thu tu cua scripts/run_bench.sh, cung file ket qua; job da xong / mau
# da chay deu duoc bo qua, nen chay xen voi run_bench.sh khong lap viec):
#   P1 quyet dinh: InfiniteBench + RULER 128k cho LayerSieve (2 setting), FastKV, ASL, SpecPrefill o muc cung chi phi
#   P2 ablation (cung chi phi ~0.71 token x layer, InfiniteBench + RULER 128k, khong sang KV de tach phan prefill):
#      {cat mot lan o L15 giu 37.5%, sang dan l0=17 rho=0.25} x {token, unit}
#   P3 phan con lai: scripts/run_bench_rest.sh = run_bench.sh bo GemFilter (RULER 32k/64k, SlimInfer, GemFilter, muc mac dinh cua paper)
#   setsid nohup bash scripts/run_bench_priority.sh >> results/bench/run.log 2>&1 < /dev/null &
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
vllm128() {  # vllm128 <ten> <tham so eval_bench...>: InfiniteBench (job ib_<ten>, nhu run_bench.sh) + RULER 128k
  local n=$1; shift
  job ib_$n $PY scripts/eval_bench.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru128_$n $PY scripts/eval_bench.py "$@" --lengths $L128 --prompts $RU --out $R/ruler/$n.jsonl
}
hf128() {
  local n=$1; shift
  job ib_$n $HF baselines/asl_driver.py "$@" --prompts $IB --out $R/infbench/$n.jsonl
  job ru128_$n $HF baselines/asl_driver.py "$@" --lengths $L128 --prompts $RU --out $R/ruler/$n.jsonl
}

# ---- P0: dense - chay bu cac mau RULER chua co (sau khi doi id RULER), lam moc cho moi bang
job ru_dense $PY scripts/eval_bench.py --method dense --prompts $RU --out $R/ruler/dense.jsonl

# ---- P1: ket qua quyet dinh
vllm128 ls_unit_d0.25_r22 --method layersieve --unit --drop-ratio 0.25 --decode-ref 22
hf128 m_fastkv_l15_f0.375 --method fastkv --select-layer 15 --select-k 0.375
hf128 m_asl_t0.3_f0.375 --method asl --select-layer 10 --tau 0.3 --select-k 0.375 --monitor-k 2048
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
