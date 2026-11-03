#!/bin/bash
# Phan C cua night7, chia 2 lan song song (vLLM || HF) de GPU khong chay mot nua. vLLM dung --gpu-util 0.55 de chua cho job HF.
# Moi job giu ten va thu muc log cu nen job da xong (.done) duoc bo qua; eval_* tu chay tiep tu file out neu dang do.
#  A (vai gio, mo khoa paper): L18 RULER 128k -> hard set H200 -> toc do n=5 -> decode 1M mac dinh + 14B-1M ->
#    ablation con thieu -> TTFT quy tac -> IB 32k
#  B (HF, ~12 h/job, 2 job song song): FastKV co nen KV || SnapKV -> ASL tau .45 || .6 -> ASL co nen KV
#  C: mot lan cat L13 -> sweep l0 theo kich co -> cau hinh mac dinh cua tung paper (d_*)
#   HF_PID="<pid python ASL tau .3 RULER>" setsid nohup bash scripts/run_night8.sh >> results/floor/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct-1M
B=results/bench; RV=results/review; FL=results/floor
LR=$RV/logs; LB=$B/logs; LF=$FL/logs
mkdir -p $RV/hard $RV/infbench32k $RV/speed $LR $LB $LF results/scaling/decode results/scaling/speed
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
echo "$(date '+%F %T') night8 bat dau"
VU="--gpu-util 0.55"
oc() {  # oc <s> <k>: mot lan cat + units, IB + RULER 128k
  local n=abl_oneshot_unit_l$1_k$2
  job $LR ib_$n $PY scripts/eval_bench.py --method layersieve --unit --start-layer $1 --oneshot-keep $2 $VU \
    --prompts data/infbench.jsonl --out $B/infbench/$n.jsonl
  job $LR ru128_$n $PY scripts/eval_bench.py --method layersieve --unit --start-layer $1 --oneshot-keep $2 $VU \
    --prompts data/ruler_eval.jsonl --lengths 131072 --out $B/ruler/$n.jsonl
}
vllm128() {  # vllm128 <ten> <tham so>: IB + RULER 128k
  local n=$1; shift
  job $LB ib_$n $PY scripts/eval_bench.py "$@" --prompts data/infbench.jsonl --out $B/infbench/$n.jsonl
  job $LB ru128_$n $PY scripts/eval_bench.py "$@" --lengths 131072 --prompts data/ruler_eval.jsonl --out $B/ruler/$n.jsonl
}
hfj() {  # hfj <ten> <tham so asl_driver>: IB + RULER 128k
  local n=$1; shift
  job $LR ib_$n $HF baselines/asl_driver.py "$@" --prompts data/infbench.jsonl --out $B/infbench/$n.jsonl
  job $LR ru128_$n $HF baselines/asl_driver.py "$@" --prompts data/ruler_eval.jsonl --lengths 131072 --out $B/ruler/$n.jsonl
}

# ================= C
SP=baselines/SpecPrefill/configs
declare -A QM=([7b1m]=$M [14b]=Qwen/Qwen2.5-14B-Instruct [32b]=Qwen/Qwen2.5-32B-Instruct)
declare -A QR=([7b1m]=0.4 [14b]=0.258 [32b]=0.200)
declare -A L055=([7b1m]=15 [14b]=26 [32b]=35) L068=([7b1m]=19 [14b]=33 [32b]=44)
declare -A SL0=([7b1m]=14 [14b]=24 [32b]=32) SR=([7b1m]=0.0956 [14b]=0.0569 [32b]=0.0430)
# cau hinh mac dinh cua tung paper (W10), IB + RULER day du
IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl
hf() { local n=$1; shift
  job $LB ib_$n $HF baselines/asl_driver.py "$@" --prompts $IB --out $B/infbench/$n.jsonl
  job $LB ru_$n $HF baselines/asl_driver.py "$@" --prompts $RU --out $B/ruler/$n.jsonl; }
spec() { local n=$1 c=$2
  job $LB ib_$n $PY scripts/eval_bench.py --method specprefill --spec-config $c $VU --spec-yarn 4 --prompts $IB --out $B/infbench/$n.jsonl
  job $LB ru_${n}_32k $PY scripts/eval_bench.py --method specprefill --spec-config $c $VU --lengths 32768 --max-model-len 32768 \
    --prompts $RU --out $B/ruler/$n.jsonl
  job $LB ru_${n}_long $PY scripts/eval_bench.py --method specprefill --spec-config $c $VU --spec-yarn 4 --lengths 65536,131072 \
    --prompts $RU --out $B/ruler/$n.jsonl; }

ASLA="--method asl --select-layer 10 --select-k 0.375 --monitor-k 2048"
{ # lan HF: doi ASL tau .3 co nen KV (RULER 128k, dang chay tu night7) roi chay mac dinh FastKV, SnapKV
  for p in ${HF_PID:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
  [ "$(wc -l < $B/ruler/m_asl_t0.3_f0.375_kv0.375.jsonl)" -ge 1300 ] && touch $LR/ru128_m_asl_t0.3_f0.375_kv0.375.done
  hfj m_asl_t0.3_f0.375_kv0.375 $ASLA --tau 0.3 --cache-k 0.375
  hf d_fastkv_l15_k2048 --method fastkv --select-layer 15 --select-k 2048
  hf d_snapkv_k2048 --method vanilla --cache-k 2048; } &
{ # lan vLLM
  oc 13 0.5
for m in 7b1m 14b 32b; do
  for tag in 055 068; do
    if [ $tag = 055 ]; then l0=${L055[$m]}; else l0=${L068[$m]}; fi
    job $LF hard_${m}_l$tag $PY scripts/eval_hard.py --model ${QM[$m]} --method layersieve --unit --no-ttft --graph \
      $VU --ids results/scaling/ids/$m.txt --start-layer $l0 --drop-ratio ${QR[$m]} --min-keep 0.25 \
      --out $FL/hard/${m}_l${tag}_f0.25.jsonl
    job $LF ru_${m}_l$tag $PY scripts/eval_bench.py --model ${QM[$m]} --method layersieve --unit --start-layer $l0 $VU \
      --drop-ratio ${QR[$m]} --min-keep 0.25 --tasks cwe,fwe,niah_multikey_3 --lengths 32768 --max-model-len 32768 \
      --prompts data/ruler_eval.jsonl --out $FL/ruler/${m}_l${tag}_f0.25.jsonl
  done
  job $LF ru_${m}_slow $PY scripts/eval_bench.py --model ${QM[$m]} --method layersieve --unit --start-layer ${SL0[$m]} $VU \
    --drop-ratio ${SR[$m]} --min-keep 0 --tasks cwe,fwe,niah_multikey_3 --lengths 32768 --max-model-len 32768 \
    --prompts data/ruler_eval.jsonl --out $FL/ruler/${m}_slow.jsonl
done
  spec d_specprefill_p10 $SP/config_p1_full_lah8.yaml
  job $LB ib_d_slim_12k_6k_4k $PY scripts/eval_bench.py --method slim $VU --prompts $IB --out $B/infbench/d_slim_12k_6k_4k.jsonl
  job $LB ru_d_slim_12k_6k_4k $PY scripts/eval_bench.py --method slim $VU --prompts $RU --out $B/ruler/d_slim_12k_6k_4k.jsonl
  hf d_asl_t0.3_k2048 --method asl --select-layer 10 --tau 0.3 --select-k 2048; } &
wait
echo "$(date '+%F %T') night8 het (Llama con tam hoan: run_night4.sh)"
