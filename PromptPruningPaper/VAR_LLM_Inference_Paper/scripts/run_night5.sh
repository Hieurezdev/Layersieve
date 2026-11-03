#!/bin/bash
# Sweep l0 voi lich cat nhanh (rho_L = 1 - 0.6^(28/L), cham san sau ~0.11L, phi .25) cho 7B-1M, 14B, 32B:
# l0 = .55L va .68L (0.5L = *_scaled / 7b1m f0.25, 0.61L = *_rel da co), va lich cat cham tu 0.5L
# (rho = 1 - 0.245^(1/(L-l0)), layer cuoi con 24.5% nhu Fig. start) tren RULER 32k de ve accuracy theo C.
# Goi tu dau scripts/run_bench_p2.sh (chay sau Llama, truoc ablation).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; FL=results/floor
mkdir -p $FL/logs $FL/hard $FL/ruler
job() {  # job <ten> <lenh...>
  local name=$1; shift
  [ -f "$FL/logs/$name.done" ] && return 0
  echo "$(date '+%F %T') [$name] bat dau"
  if "$@" >> "$FL/logs/$name.log" 2>&1; then
    touch "$FL/logs/$name.done"; echo "$(date '+%F %T') [$name] xong"
  else
    echo "$(date '+%F %T') [$name] loi (xem $FL/logs/$name.log)"
  fi
}
# Decode den 1M cho cau hinh mac dinh (rho .4, phi .25, co / khong sang KV r22): bang decode trong paper
mkdir -p results/scaling/decode
job decode_default_7b1m $PY scripts/bench_speed.py --graph --model Qwen/Qwen2.5-7B-Instruct-1M --prompts data/speed.jsonl \
  --lengths 131072,262144,524288,1000000 --n 2 --decode 512 --gpu-util 0.93 \
  --configs dense,17:0.4::0.25,17:0.4:22:0.25 --out results/scaling/decode/7b1m_default.jsonl
job decode_rule_14b1m $PY scripts/bench_speed.py --graph --model Qwen/Qwen2.5-14B-Instruct-1M --prompts data/speed.jsonl \
  --lengths 131072,262144 --n 2 --decode 512 --gpu-util 0.93 \
  --configs dense,29:0.258::0.25,29:0.258:38:0.25 --out results/scaling/decode/14b1m_rule.jsonl
declare -A QM=([7b1m]=Qwen/Qwen2.5-7B-Instruct-1M [14b]=Qwen/Qwen2.5-14B-Instruct [32b]=Qwen/Qwen2.5-32B-Instruct)
declare -A QR=([7b1m]=0.4 [14b]=0.258 [32b]=0.200)
declare -A L055=([7b1m]=15 [14b]=26 [32b]=35) L068=([7b1m]=19 [14b]=33 [32b]=44)
declare -A SL0=([7b1m]=14 [14b]=24 [32b]=32) SR=([7b1m]=0.0956 [14b]=0.0569 [32b]=0.0430)
for m in 7b1m 14b 32b; do
  for tag in 055 068; do
    if [ $tag = 055 ]; then l0=${L055[$m]}; else l0=${L068[$m]}; fi
    job hard_${m}_l$tag $PY scripts/eval_hard.py --model ${QM[$m]} --method layersieve --unit --no-ttft --graph \
      --gpu-util 0.85 --ids results/scaling/ids/$m.txt --start-layer $l0 --drop-ratio ${QR[$m]} --min-keep 0.25 \
      --out $FL/hard/${m}_l${tag}_f0.25.jsonl
    job ru_${m}_l$tag $PY scripts/eval_bench.py --model ${QM[$m]} --method layersieve --unit --start-layer $l0 \
      --drop-ratio ${QR[$m]} --min-keep 0.25 --tasks cwe,fwe,niah_multikey_3 --lengths 32768 --max-model-len 32768 \
      --prompts data/ruler_eval.jsonl --out $FL/ruler/${m}_l${tag}_f0.25.jsonl
  done
  job ru_${m}_slow $PY scripts/eval_bench.py --model ${QM[$m]} --method layersieve --unit --start-layer ${SL0[$m]} \
    --drop-ratio ${SR[$m]} --min-keep 0 --tasks cwe,fwe,niah_multikey_3 --lengths 32768 --max-model-len 32768 \
    --prompts data/ruler_eval.jsonl --out $FL/ruler/${m}_slow.jsonl
done
# Toc do cua quy tac tuong doi (l0 .61L, rho_L, phi .25) o moi co, cung harness voi results/scaling/speed (YaRN x4
# cho 3B/14B/32B), de bang scaling dung lich mac dinh thay vi lich cham tu 0.5L
declare -A SM=([7b1m]=Qwen/Qwen2.5-7B-Instruct-1M [3b]=Qwen/Qwen2.5-3B-Instruct [14b]=Qwen/Qwen2.5-14B-Instruct
              [32b]=Qwen/Qwen2.5-32B-Instruct)
declare -A SC=([7b1m]=17:0.4::0.25 [3b]=22:0.328::0.25 [14b]=29:0.258::0.25 [32b]=39:0.2::0.25)
mkdir -p results/scaling/speed
for m in 7b1m 3b 14b 32b; do
  y=4; [ $m = 7b1m ] && y=0
  job speed_rule_$m $PY scripts/bench_speed.py --model ${SM[$m]} --yarn $y --prompts data/speed.jsonl \
    --lengths 131072 --n 2 --decode 64 --gpu-util 0.9 --configs "dense,${SC[$m]}" --out results/scaling/speed/rule_$m.jsonl
done
echo "$(date '+%F %T') het night5"
