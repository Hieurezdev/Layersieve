#!/bin/bash
# Hang doi sap lai theo muc can cho paper (thay phan con lai cua night6 / night4 / night5 / p2). Tam bo moi job Llama.
# Moi job giu ten va thu muc log cu nen job da xong (.done) duoc bo qua; eval_* tu chay tiep tu file out neu dang do.
#  A (vai gio, mo khoa paper): L18 RULER 128k -> hard set H200 -> toc do n=5 -> decode 1M mac dinh + 14B-1M ->
#    ablation con thieu -> TTFT quy tac -> IB 32k
#  B (HF, ~12 h/job, 2 job song song): FastKV co nen KV || SnapKV -> ASL tau .45 || .6 -> ASL co nen KV
#  C: mot lan cat L13 -> sweep l0 theo kich co -> cau hinh mac dinh cua tung paper (d_*)
#   WAIT_PIDS="<pid>" setsid nohup bash scripts/run_night7.sh >> results/floor/run.log 2>&1 < /dev/null &
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
for p in ${WAIT_PIDS:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') night7 bat dau"
oc() {  # oc <s> <k>: mot lan cat + units, IB + RULER 128k
  local n=abl_oneshot_unit_l$1_k$2
  job $LR ib_$n $PY scripts/eval_bench.py --method layersieve --unit --start-layer $1 --oneshot-keep $2 \
    --prompts data/infbench.jsonl --out $B/infbench/$n.jsonl
  job $LR ru128_$n $PY scripts/eval_bench.py --method layersieve --unit --start-layer $1 --oneshot-keep $2 \
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

# ================= A
# A1. doi chung review 2: mot lan cat L18 (job dang chay luc sap lai; tu chay tiep tu file out)
oc 18 0.22
# A2. hard set tren H200, bo 120 cau (Table II, Q3, Q6)
IDS="--ids results/scaling/ids/7b1m.txt --gpu-util 0.85 --no-ttft"
SP=baselines/SpecPrefill/configs
LSD="--method layersieve --start-layer 17 --drop-ratio 0.4 --min-keep 0.25"
job $LR hard_draft15 $PY scripts/eval_hard.py --model Qwen/Qwen2.5-1.5B-Instruct --method dense $IDS --out $RV/hard/draft15.jsonl
job $LR hard_draft05 $PY scripts/eval_hard.py --model Qwen/Qwen2.5-0.5B-Instruct --method dense $IDS --out $RV/hard/draft05.jsonl
for k in 9 5 3 1; do
  job $LR hard_spec_p$k $PY scripts/eval_hard.py --method specprefill --spec-config $SP/config_p${k}_full_lah8.yaml $IDS \
    --out $RV/hard/spec_p$k.jsonl
done
job $LR hard_ls_tok $PY scripts/eval_hard.py $LSD $IDS --out $RV/hard/ls_tok.jsonl
job $LR hard_ls_unit $PY scripts/eval_hard.py $LSD --unit $IDS --out $RV/hard/ls_unit.jsonl
job $LR hard_ls_unit_r22 $PY scripts/eval_hard.py $LSD --unit --decode-ref 22 $IDS --out $RV/hard/ls_unit_r22.jsonl
job $LR hard_onecut_unit $PY scripts/eval_hard.py --method layersieve --start-layer 15 --oneshot-keep 0.375 --unit $IDS \
  --out $RV/hard/onecut_unit_l15.jsonl
for o in 32 512; do
  job $LR hard_ls_obs$o $PY scripts/eval_hard.py $LSD --unit --n-obs $o $IDS --out $RV/hard/ls_unit_obs$o.jsonl
done
for z in 16 512; do
  job $LR hard_ls_sink$z $PY scripts/eval_hard.py $LSD --unit --n-sink $z $IDS --out $RV/hard/ls_unit_sink$z.jsonl
done
# A3. toc do n=5 (bang toc do, W3/W6)
CF="dense,17:0.4::0.25,17:0.4:22:0.25,o15:0.375,o15:0.375:15,slim:0.75/0.55/0.35,o0:0.5"
job $LR speed_review $PY scripts/bench_speed.py --graph --model $M --prompts data/speed.jsonl --lengths 131072,262144 \
  --n 5 --decode 256 --gpu-util 0.93 --configs "$CF" --out $RV/speed/7b1m.jsonl
job $LR speed_draft15 $PY scripts/bench_speed.py --graph --model Qwen/Qwen2.5-1.5B-Instruct --yarn 4 --prompts data/speed.jsonl \
  --lengths 131072 --n 5 --decode 8 --gpu-util 0.6 --configs dense --out $RV/speed/draft15.jsonl
# A4. decode den 1M voi cau hinh mac dinh (bang decode, abstract) + 14B-1M r = 38
job $LF decode_default_7b1m $PY scripts/bench_speed.py --graph --model $M --prompts data/speed.jsonl \
  --lengths 131072,262144,524288,1000000 --n 2 --decode 512 --gpu-util 0.93 \
  --configs dense,17:0.4::0.25,17:0.4:22:0.25 --out results/scaling/decode/7b1m_default.jsonl
job $LF decode_rule_14b1m $PY scripts/bench_speed.py --graph --model Qwen/Qwen2.5-14B-Instruct-1M --prompts data/speed.jsonl \
  --lengths 131072,262144 --n 2 --decode 512 --gpu-util 0.93 \
  --configs dense,29:0.258::0.25,29:0.258:38:0.25 --out results/scaling/decode/14b1m_rule.jsonl
# A5. o \pend cua bang ablation: progressive token RULER, khong san
vllm128 abl_prog_tok_d0.4_f0.25 --method layersieve --drop-ratio 0.4 --min-keep 0.25
vllm128 abl_prog_unit_d0.4_nofloor --method layersieve --unit --drop-ratio 0.4
# A6. TTFT 128k theo quy tac (tab:rule)
declare -A SM=([7b1m]=$M [3b]=Qwen/Qwen2.5-3B-Instruct [14b]=Qwen/Qwen2.5-14B-Instruct [32b]=Qwen/Qwen2.5-32B-Instruct)
declare -A SC=([7b1m]=17:0.4::0.25 [3b]=22:0.328::0.25 [14b]=29:0.258::0.25 [32b]=39:0.2::0.25)
for m in 7b1m 3b 14b 32b; do
  y=4; [ $m = 7b1m ] && y=0
  job $LF speed_rule_$m $PY scripts/bench_speed.py --model ${SM[$m]} --yarn $y --prompts data/speed.jsonl \
    --lengths 131072 --n 2 --decode 64 --gpu-util 0.9 --configs "dense,${SC[$m]}" --out results/scaling/speed/rule_$m.jsonl
done
# A7. IB cat 32k: SpecPrefill khong YaRN (W4)
I32="--prompts data/infbench_32k.jsonl --max-model-len 34000"
job $LR ib32_dense $PY scripts/eval_bench.py --method dense $I32 --out $RV/infbench32k/dense.jsonl
job $LR ib32_ls $PY scripts/eval_bench.py $LSD --unit --decode-ref 22 $I32 --out $RV/infbench32k/ls_unit_d0.4_f0.25_r22.jsonl
job $LR ib32_spec env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 $PY scripts/eval_bench.py --method specprefill --spec-config $SP/config_p5_full_lah8.yaml $I32 \
  --out $RV/infbench32k/m_specprefill_p50.jsonl
job $LR ib32_spec30 env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 $PY scripts/eval_bench.py --method specprefill --spec-config $SP/config_p3_full_lah8.yaml $I32 \
  --out $RV/infbench32k/m_specprefill_p30.jsonl
echo "$(date '+%F %T') night7 nhom A xong"

# ================= B (HF)
ASLA="--method asl --select-layer 10 --select-k 0.375 --monitor-k 2048"
# 2 lan song song, khong co rao chan giua cac cap, de GPU khong chay mot nua khi job ben kia xong som.
# SNAP_PID: python SnapKV RULER con chay tu lan khoi dong truoc; doi no xong roi danh dau .done.
{ hfj m_fastkv_l15_f0.375_kv0.375 --method fastkv --select-layer 15 --select-k 0.375 --cache-k 0.375
  hfj m_asl_t0.45_f0.375 $ASLA --tau 0.45
  hfj m_asl_t0.3_f0.375_kv0.375 $ASLA --tau 0.3 --cache-k 0.375; } &
{ for p in ${SNAP_PID:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
  [ -n "${SNAP_PID:-}" ] && [ "$(wc -l < $B/ruler/m_snapkv_kv0.25.jsonl)" -ge 1300 ] && touch $LR/ru128_m_snapkv_kv0.25.done
  hfj m_snapkv_kv0.25 --method vanilla --cache-k 0.25
  hfj m_asl_t0.6_f0.375 $ASLA --tau 0.6; } &
wait
echo "$(date '+%F %T') night7 nhom B xong"

# ================= C
oc 13 0.5
declare -A QM=([7b1m]=$M [14b]=Qwen/Qwen2.5-14B-Instruct [32b]=Qwen/Qwen2.5-32B-Instruct)
declare -A QR=([7b1m]=0.4 [14b]=0.258 [32b]=0.200)
declare -A L055=([7b1m]=15 [14b]=26 [32b]=35) L068=([7b1m]=19 [14b]=33 [32b]=44)
declare -A SL0=([7b1m]=14 [14b]=24 [32b]=32) SR=([7b1m]=0.0956 [14b]=0.0569 [32b]=0.0430)
for m in 7b1m 14b 32b; do
  for tag in 055 068; do
    if [ $tag = 055 ]; then l0=${L055[$m]}; else l0=${L068[$m]}; fi
    job $LF hard_${m}_l$tag $PY scripts/eval_hard.py --model ${QM[$m]} --method layersieve --unit --no-ttft --graph \
      --gpu-util 0.85 --ids results/scaling/ids/$m.txt --start-layer $l0 --drop-ratio ${QR[$m]} --min-keep 0.25 \
      --out $FL/hard/${m}_l${tag}_f0.25.jsonl
    job $LF ru_${m}_l$tag $PY scripts/eval_bench.py --model ${QM[$m]} --method layersieve --unit --start-layer $l0 \
      --drop-ratio ${QR[$m]} --min-keep 0.25 --tasks cwe,fwe,niah_multikey_3 --lengths 32768 --max-model-len 32768 \
      --prompts data/ruler_eval.jsonl --out $FL/ruler/${m}_l${tag}_f0.25.jsonl
  done
  job $LF ru_${m}_slow $PY scripts/eval_bench.py --model ${QM[$m]} --method layersieve --unit --start-layer ${SL0[$m]} \
    --drop-ratio ${SR[$m]} --min-keep 0 --tasks cwe,fwe,niah_multikey_3 --lengths 32768 --max-model-len 32768 \
    --prompts data/ruler_eval.jsonl --out $FL/ruler/${m}_slow.jsonl
done
# cau hinh mac dinh cua tung paper (W10), IB + RULER day du
IB=data/infbench.jsonl; RU=data/ruler_eval.jsonl
hf() { local n=$1; shift
  job $LB ib_$n $HF baselines/asl_driver.py "$@" --prompts $IB --out $B/infbench/$n.jsonl
  job $LB ru_$n $HF baselines/asl_driver.py "$@" --prompts $RU --out $B/ruler/$n.jsonl; }
spec() { local n=$1 c=$2
  job $LB ib_$n $PY scripts/eval_bench.py --method specprefill --spec-config $c --spec-yarn 4 --prompts $IB --out $B/infbench/$n.jsonl
  job $LB ru_${n}_32k $PY scripts/eval_bench.py --method specprefill --spec-config $c --lengths 32768 --max-model-len 32768 \
    --prompts $RU --out $B/ruler/$n.jsonl
  job $LB ru_${n}_long $PY scripts/eval_bench.py --method specprefill --spec-config $c --spec-yarn 4 --lengths 65536,131072 \
    --prompts $RU --out $B/ruler/$n.jsonl; }
spec d_specprefill_p10 $SP/config_p1_full_lah8.yaml
job $LB ib_d_slim_12k_6k_4k $PY scripts/eval_bench.py --method slim --prompts $IB --out $B/infbench/d_slim_12k_6k_4k.jsonl
job $LB ru_d_slim_12k_6k_4k $PY scripts/eval_bench.py --method slim --prompts $RU --out $B/ruler/d_slim_12k_6k_4k.jsonl
hf d_fastkv_l15_k2048 --method fastkv --select-layer 15 --select-k 2048 &
hf d_asl_t0.3_k2048 --method asl --select-layer 10 --tau 0.3 --select-k 2048
wait
hf d_snapkv_k2048 --method vanilla --cache-k 2048
echo "$(date '+%F %T') night7 het (Llama con tam hoan: run_night4.sh)"
