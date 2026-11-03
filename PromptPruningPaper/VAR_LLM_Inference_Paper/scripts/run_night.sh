#!/bin/bash
# Hang doi dem 11-12/10: (1) phan con lai cua sweep san phi (run_floor.sh, bo phi .15 o phan C),
# (2) ho model thu hai: Llama-3.1-8B-UltraLong-1M (model chinh cua paper ASL, 32 layer) - smoke test, InfiniteBench
#     va RULER 128k cho dense / LayerSieve / FastKV / ASL, toc do; (3) chay lai p2 (ablation, d_*).
# LayerSieve tren Llama: cung vi tri tuong doi voi 7B (l0 17/28 -> 19/32, r 22/28 -> 25/32), rho .4, phi .25.
# FastKV / ASL: mac dinh repo ASL cho chinh model nay (TSP 15; L_min 10, tau .3), giu 37.5%, monitor-k 2048 nhu Qwen.
#   WAIT_PID=<pid> setsid nohup bash scripts/run_night.sh >> results/floor/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python
LM=nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
B=results/bench; LOGDIR=$B/logs; FL=results/floor
mkdir -p $B/llama/infbench $B/llama/ruler $FL/logs
job() {  # job <logdir> <ten> <lenh...>
  local d=$1 name=$2; shift 2
  [ -f "$d/$name.done" ] && return 0
  echo "$(date '+%F %T') [$name] bat dau"
  if "$@" >> "$d/$name.log" 2>&1; then
    touch "$d/$name.done"; echo "$(date '+%F %T') [$name] xong"
  else
    echo "$(date '+%F %T') [$name] loi (xem $d/$name.log)"
  fi
}
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

# (1) sweep phi: 32B con lai (eval_bench bo qua cau da co) + phan C
declare -A FM=([32b]=Qwen/Qwen2.5-32B-Instruct)
for f in 0.25 0.35; do
  job $FL/logs ru_32b_f$f $PY scripts/eval_bench.py --model ${FM[32b]} --method layersieve --unit --start-layer 32 \
    --drop-ratio 0.4 --min-keep $f --tasks cwe,fwe,niah_multikey_3 --lengths 32768 --max-model-len 32768 \
    --prompts data/ruler_eval.jsonl --out $FL/ruler/32b_f$f.jsonl
done
# (2a) smoke test Llama: 4 cau kv InfiniteBench + 4 cau RULER 128k, dense va LayerSieve
head -n 400 data/infbench_llama.jsonl | grep '"kv_retrieval' | head -4 > $B/llama/smoke.jsonl
grep -m4 niah_multikey_3 data/ruler_eval_llama.jsonl >> $B/llama/smoke.jsonl
LS="--method layersieve --unit --start-layer 19 --drop-ratio 0.4 --min-keep 0.25"
job $LOGDIR llama_smoke_dense $PY scripts/eval_bench.py --model $LM --method dense --max-model-len 140000 \
  --prompts $B/llama/smoke.jsonl --out $B/llama/smoke_dense.jsonl
job $LOGDIR llama_smoke_ls $PY scripts/eval_bench.py --model $LM $LS --decode-ref 25 --max-model-len 140000 \
  --prompts $B/llama/smoke.jsonl --out $B/llama/smoke_ls.jsonl
# (1b) phan C cua sweep phi
for f in 0.05 0.10 0.35; do
  job $FL/logs ru_7b1m17_f$f $PY scripts/eval_bench.py --method layersieve --unit --start-layer 17 --drop-ratio 0.4 \
    --min-keep $f --tasks cwe,fwe --lengths 32768,131072 --prompts data/ruler_eval.jsonl --out $FL/ruler/7b1m17_f$f.jsonl
done
# (2b) Llama InfiniteBench: FastKV || ASL (HF, 2 tien trinh nhu Qwen), roi dense, LayerSieve (vLLM)
llama_ib_hf() {
  job $LOGDIR llama_ib_fastkv $HF baselines/asl_driver.py --model $LM --method fastkv --select-layer 15 --select-k 0.375 \
    --prompts data/infbench_llama.jsonl --out $B/llama/infbench/m_fastkv_l15_f0.375.jsonl &
  job $LOGDIR llama_ib_asl $HF baselines/asl_driver.py --model $LM --method asl --select-layer 10 --tau 0.3 \
    --select-k 0.375 --monitor-k 2048 --prompts data/infbench_llama.jsonl --out $B/llama/infbench/m_asl_t0.3_f0.375.jsonl
  wait
}
job $LOGDIR llama_ib_dense $PY scripts/eval_bench.py --model $LM --method dense --max-model-len 140000 \
  --prompts data/infbench_llama.jsonl --out $B/llama/infbench/dense.jsonl
job $LOGDIR llama_ib_ls $PY scripts/eval_bench.py --model $LM $LS --decode-ref 25 --max-model-len 140000 \
  --prompts data/infbench_llama.jsonl --out $B/llama/infbench/ls_unit_l19_d0.4_f0.25_r25.jsonl
llama_ib_hf
# (2c) toc do Llama: dense, LayerSieve (co / khong sang KV), lich FastKV, lich ASL o moi layer cat co the (10..31)
O=$(seq -s, 10 31 | sed 's/\([0-9]*\)/o\1:0.375/g')
job $LOGDIR llama_speed $PY scripts/bench_speed.py --graph --model $LM --prompts data/speed_llama.jsonl \
  --lengths 131072,262144 --n 2 --decode 256 --gpu-util 0.93 --configs "dense,19:0.4::0.25,19:0.4:25:0.25,$O" \
  --out $B/llama/speed.jsonl
# (2d) Llama RULER 128k
job $LOGDIR llama_ru_dense $PY scripts/eval_bench.py --model $LM --method dense --max-model-len 140000 \
  --prompts data/ruler_eval_llama.jsonl --out $B/llama/ruler/dense.jsonl
job $LOGDIR llama_ru_ls $PY scripts/eval_bench.py --model $LM $LS --decode-ref 25 --max-model-len 140000 \
  --prompts data/ruler_eval_llama.jsonl --out $B/llama/ruler/ls_unit_l19_d0.4_f0.25_r25.jsonl
job $LOGDIR llama_ru_fastkv $HF baselines/asl_driver.py --model $LM --method fastkv --select-layer 15 --select-k 0.375 \
  --prompts data/ruler_eval_llama.jsonl --out $B/llama/ruler/m_fastkv_l15_f0.375.jsonl &
job $LOGDIR llama_ru_asl $HF baselines/asl_driver.py --model $LM --method asl --select-layer 10 --tau 0.3 \
  --select-k 0.375 --monitor-k 2048 --prompts data/ruler_eval_llama.jsonl --out $B/llama/ruler/m_asl_t0.3_f0.375.jsonl
wait
echo "$(date '+%F %T') het night, chay lai p2"
setsid nohup bash scripts/run_bench_p2.sh >> results/bench/run.log 2>&1 < /dev/null &
