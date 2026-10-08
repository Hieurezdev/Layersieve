#!/bin/bash
# So sanh baseline prompt pruning tren Qwen2.5-7B-Instruct-1M, chi do chinh xac (toc do do rieng):
#   InfiniteBench 4 task x 100 (data/infbench.jsonl), RULER 13 task x 100 x {32k,64k,128k} (data/ruler_eval.jsonl)
# Hai muc ngan sach:
#   m (cung chi phi prefill voi LayerSieve unit rho=0.25 r=22, token x layer ~ 0.71) - bang chinh, chay truoc
#   d (mac dinh cua tung paper: giu 2048 token, SpecPrefill 10%, SlimInfer 12k/6k/4k, SnapKV 2048)
# vLLM (.venv): dense, LayerSieve, SlimInfer, SpecPrefill (draft Qwen2.5-1.5B, YaRN x4 cho draft o 64k/128k)
# HF (ext/ASL/.venv, code repo ASL): FastKV, ASL, SnapKV; GemFilter bo (khong ro noi dang)
#   setsid nohup bash scripts/run_bench.sh >> results/bench/run.log 2>&1 < /dev/null &
# Tong hop: python scripts/score_bench.py
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; HF=/root/PromptPruningPaper/ext/ASL/.venv/bin/python
R=results/bench; LOGDIR=$R/logs; mkdir -p $R/infbench $R/ruler $LOGDIR
IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl
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

# ---- muc m: cung chi phi prefill (~0.71 token x layer)
vllm dense --method dense
vllm ls_unit_d0.25_r22 --method layersieve --unit --drop-ratio 0.25 --decode-ref 22
vllm ls_unit_d0.12_r27 --method layersieve --unit --drop-ratio 0.12 --decode-ref 27
hf m_fastkv_l15_f0.375 --method fastkv --select-layer 15 --select-k 0.375
hf m_asl_t0.3_f0.375 --method asl --select-layer 10 --tau 0.3 --select-k 0.375
vllm m_slim_0.75_0.55_0.35 --method slim --slim-frac 0.75,0.55,0.35
spec m_specprefill_p50 $SP/config_p5_full_lah8.yaml

# ---- muc d: mac dinh cua tung paper
hf d_fastkv_l15_k2048 --method fastkv --select-layer 15 --select-k 2048
hf d_asl_t0.3_k2048 --method asl --select-layer 10 --tau 0.3 --select-k 2048
spec d_specprefill_p10 $SP/config_p1_full_lah8.yaml
vllm d_slim_12k_6k_4k --method slim
# PyramidInfer: bo - ban cai cua repo ASL dung attention eager (ma tran T x T), OOM tren H200 tu 32k (~98 GB) den 128k
hf d_snapkv_k2048 --method vanilla --cache-k 2048
echo "$(date '+%F %T') bench het"
