#!/bin/bash
# Thi nghiem scaling tren 1 GPU (H200), chay tuan tu. Hai che do:
#   bash scripts/run_scaling.sh speed  : Q2 + Q3 (do thoi gian), moi job doi GPU khong con process nao khac
#   bash scripts/run_scaling.sh acc    : Q1 (chi do do chinh xac), chay duoc khi GPU dung chung;
#                                        ACC_UTIL (mac dinh 0.85) = gpu_memory_utilization, MODELS = danh sach model,
#                                        HARD = file prompt (mac dinh data/hard.jsonl), R = thu muc ket qua
#   Q2 toc do prefill theo co model: 3B (36 layer), 7B-1M (28), 14B (48), 32B (64) o 32k/64k/128k (YaRN x4 cho model
#      context 32k), moi lich sang cua sweep Q1 + l0=17 rho=0.12 co dinh -> results/scaling/speed/<m>.jsonl
#   Q3 decode khi sang KV theo do dai context 32k..1M: 7B-1M (va 14B-1M den 256k), decode chay CUDA graph
#      (--graph; eager bi nghen o CPU launch tren H200) -> results/scaling/decode/<m>.jsonl
#   Q1 layer bat dau sang co co dinh khong: 3B, 7B, 7B-1M, 14B, 14B-1M, 32B tren data/hard.jsonl
#      dense -> chon <=30 cau dung moi task -> probe tung layer -> sweep unit, l0 = {.25,.33,.4,.5,.6,.7,.8}*L va 17,
#      rho chon de layer cuoi con 24.5% (= l0 17, rho 0.12 tren 28 layer) -> results/scaling/{dense,ids,probe,sweep}
# Chay nen: setsid nohup bash scripts/run_scaling.sh speed > results/scaling/run_speed.log 2>&1 < /dev/null &
#   (GPU dung chung, vd. con ~70GB): ACC_UTIL=0.4 MODELS="7b1m 7b 3b 14b 14b1m" bash scripts/run_scaling.sh acc
# Tong hop: python scripts/summarize_scaling.py
set -u
MODE=${1:?speed | acc}
ACC_UTIL=${ACC_UTIL:-0.85}
MODELS=${MODELS:-7b1m 7b 3b 14b 14b1m 32b}
HARD=${HARD:-data/hard.jsonl}
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=${R:-results/scaling}; LOGDIR=$R/logs
mkdir -p $R/{speed,decode,dense,ids,probe,sweep} $LOGDIR
declare -A M=([3b]=Qwen/Qwen2.5-3B-Instruct [7b]=Qwen/Qwen2.5-7B-Instruct [7b1m]=Qwen/Qwen2.5-7B-Instruct-1M
              [14b]=Qwen/Qwen2.5-14B-Instruct [14b1m]=Qwen/Qwen2.5-14B-Instruct-1M [32b]=Qwen/Qwen2.5-32B-Instruct)
declare -A NL=([3b]=36 [7b]=28 [7b1m]=28 [14b]=48 [14b1m]=48 [32b]=64)

wait_idle() {  # doi den khi GPU khong con compute process nao, 2 lan do lien tiep cach 30 s
  local n=0
  while [ $n -lt 2 ]; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then n=$((n + 1)); else n=0; fi
    [ $n -lt 2 ] && sleep 30
  done
}

job() {  # job <ten> <lenh...>: bo qua neu da xong; log o $LOGDIR/<ten>.log
  local name=$1; shift
  [ -f "$LOGDIR/$name.done" ] && return 0
  [ $MODE = speed ] && wait_idle
  echo "$(date '+%F %T') [$name] bat dau"
  if "$@" >> "$LOGDIR/$name.log" 2>&1; then
    touch "$LOGDIR/$name.done"; echo "$(date '+%F %T') [$name] xong"
  else
    echo "$(date '+%F %T') [$name] loi (xem $LOGDIR/$name.log)"
  fi
}

sched() {  # sched <L>: cac lich "l0:rho" cua sweep, layer cuoi giu 24.5% token co the bo
  python3 -c "
L = $1; s = sorted({round(L * x) for x in (.25, .33, .4, .5, .6, .7, .8)} | {17})
print(' '.join(f'{l}:{1 - 0.245 ** (1 / (L - l)):.4f}' for l in s))"
}

if [ $MODE = speed ]; then
# ---- Q2: toc do prefill theo co model
for m in 7b1m 3b 14b 32b; do
  c="dense,$(sched ${NL[$m]} | tr ' ' ',')"; [ ${NL[$m]} != 28 ] && c="$c,17:0.12"
  y=4; [ $m = 7b1m ] && y=0
  job speed_$m $PY scripts/bench_speed.py --model ${M[$m]} --yarn $y --prompts data/speed.jsonl \
    --lengths 32768,65536,131072 --n 2 --decode 64 --gpu-util 0.9 --configs "$c" --out $R/speed/$m.jsonl
done

# ---- Q3: decode khi sang KV, context den 1M
DEC=dense,17:0.12,17:0.12:22,17:0.12:27,17:0.25,17:0.25:22,17:0.25:27
job decode_7b1m $PY scripts/bench_speed.py --graph --model ${M[7b1m]} --prompts data/speed.jsonl \
  --lengths 32768,65536,131072,262144,524288,1000000 --n 2 --decode 512 --gpu-util 0.93 --configs $DEC \
  --out $R/decode/7b1m.jsonl
job decode_14b1m $PY scripts/bench_speed.py --graph --model ${M[14b1m]} --prompts data/speed.jsonl \
  --lengths 32768,65536,131072,262144 --n 2 --decode 512 --gpu-util 0.93 \
  --configs dense,29:0.0714,29:0.0714:38,29:0.0714:47,17:0.12,17:0.12:47 --out $R/decode/14b1m.jsonl

fi

# ---- Q1: layer bat dau sang
[ $MODE = acc ] && for m in $MODELS; do
  job dense_$m $PY scripts/eval_hard.py --model ${M[$m]} --method dense --no-ttft --graph --gpu-util $ACC_UTIL --prompts $HARD \
    --out $R/dense/$m.jsonl
  [ -f $LOGDIR/dense_$m.done ] || continue
  [ -f $R/ids/$m.txt ] || $PY scripts/scaling_ids.py $R/dense/$m.jsonl $R/ids/$m.txt || continue
  job probe_$m $PY scripts/probe_layers.py --model ${M[$m]} --ids $R/ids/$m.txt --gpu-util $ACC_UTIL --prompts $HARD \
    --out $R/probe/$m.jsonl
  for c in $(sched ${NL[$m]}) 17:0.12; do
    s=${c%:*}; d=${c#*:}
    [ ${NL[$m]} = 28 ] && [ $c = 17:0.12 ] && continue
    job sweep_${m}_s${s}_d${d} $PY scripts/eval_hard.py --model ${M[$m]} --method layersieve --unit --no-ttft --graph \
      --gpu-util $ACC_UTIL --prompts $HARD \
      --ids $R/ids/$m.txt --start-layer $s --drop-ratio $d --out $R/sweep/${m}_s${s}_d${d}.jsonl
  done
done
echo "$(date '+%F %T') het"
