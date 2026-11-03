#!/bin/bash
# Do chinh xac cua cac setting sang KV decode (7B-1M, unit), tren H200, decode chay CUDA graph:
#   bo case kho (<= 27k): cung mau voi Q1 (results/scaling/ids/7b1m.txt, <= 30 cau dense dung moi task)
#     -> results/scaling/decode_acc/hard_<ten>.jsonl
#   kv50 (~170k token)  -> results/scaling/decode_acc/kv50_<ten>.jsonl
# Setting: unit rho 0.12 (khong sang KV, r=27), unit rho 0.25 (khong sang KV, r=22, r=27); kv50 them dense.
#   setsid nohup bash scripts/run_decode_acc.sh >> results/scaling/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/scaling; LOGDIR=$R/logs; D=$R/decode_acc
mkdir -p $D $R/dense $R/ids $LOGDIR
M=Qwen/Qwen2.5-7B-Instruct-1M

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

# mau cau hoi: dung chung voi Q1 (job dense_7b1m cua run_scaling.sh acc se duoc bo qua)
job dense_7b1m $PY scripts/eval_hard.py --model $M --method dense --no-ttft --graph --out $R/dense/7b1m.jsonl
[ -f $R/ids/7b1m.txt ] || $PY scripts/scaling_ids.py $R/dense/7b1m.jsonl $R/ids/7b1m.txt

for v in "d0.12|0.12|" "d0.12_r27|0.12|27" "d0.25|0.25|" "d0.25_r22|0.25|22" "d0.25_r27|0.25|27"; do
  IFS='|' read -r n d r <<< "$v"
  ref=(); [ -n "$r" ] && ref=(--decode-ref $r)
  job dacc_hard_$n $PY scripts/eval_hard.py --model $M --method layersieve --unit --no-ttft --graph \
    --ids $R/ids/7b1m.txt --start-layer 17 --drop-ratio $d "${ref[@]}" --out $D/hard_$n.jsonl
done

KV="--prompts data/kv50.jsonl --no-ttft --mlp-chunk 16384 --gpu-util 0.85 --max-model-len 170000"
job dacc_kv50_dense $PY scripts/eval_kv_vllm.py $KV --dense --out $D/kv50_dense.jsonl
for v in "d0.12|0.12|" "d0.12_r27|0.12|27" "d0.25|0.25|" "d0.25_r22|0.25|22" "d0.25_r27|0.25|27"; do
  IFS='|' read -r n d r <<< "$v"
  ref=(); [ -n "$r" ] && ref=(--decode-ref $r)
  job dacc_kv50_$n $PY scripts/eval_kv_vllm.py $KV --unit --start-layer 17 --drop-ratio $d "${ref[@]}" \
    --out $D/kv50_$n.jsonl
done
echo "$(date '+%F %T') decode_acc het"
