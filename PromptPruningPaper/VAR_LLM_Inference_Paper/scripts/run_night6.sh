#!/bin/bash
# Hang doi sau review (W1-W8). Thay cho phan con lai cua run_night4.sh (chay lai no o buoc cuoi: job da xong bi bo qua).
#  P0 (re): bang hard set lam lai tren H200 voi bo 120 cau cua nghien cuu scaling (Q6, minor H200/RTX), SpecPrefill
#      va LayerSieve cung C (W4a); InfiniteBench cat 32k, draft trong cua so goc, khong YaRN (W4b); toc do: FastKV co nen
#      KV truoc diem cat, SlimInfer, SpecPrefill (W3), n=5 de co do lech (W6); do nhay cua so quan sat / sink (Q3).
#  P1: bien Pareto mot lan cat + units cung C ~0.72 (W1/W2): (L13, 50%), (L17, 25%), (L19, 10%); L15 37.5% da co.
#  P2: phan Llama con lai + night5 + p2 (run_night4.sh).
#   WAIT_PIDS="<pid> <pid>" setsid nohup bash scripts/run_night6.sh >> results/floor/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct-1M
B=results/bench; RV=results/review; LOGDIR=$RV/logs
mkdir -p $RV/hard $RV/infbench32k $RV/speed $LOGDIR $B/infbench $B/ruler
job() {  # job <ten> <lenh...>
  local name=$1; shift
  [ -f "$LOGDIR/$name.done" ] && return 0
  echo "$(date '+%F %T') [$name] bat dau"
  if "$@" >> "$LOGDIR/$name.log" 2>&1; then
    touch "$LOGDIR/$name.done"; echo "$(date '+%F %T') [$name] xong"
  else
    echo "$(date '+%F %T') [$name] loi (xem $LOGDIR/$name.log)"
  fi
}
for p in ${WAIT_PIDS:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
# Llama IB HF da chay trong night4 (bi dung giua chung): danh dau xong neu file du 400 dong
for x in fastkv_l15 asl_t0.3; do
  f=$B/llama/infbench/m_${x}_f0.375.jsonl; n=llama_ib_${x%%_*}
  [ -f $f ] && [ $(wc -l < $f) -ge 400 ] && touch $B/logs/$n.done
done

IDS="--ids results/scaling/ids/7b1m.txt --gpu-util 0.85 --no-ttft"

# ---- P0-: Llama IB, LayerSieve thua dense 3 diem (26.3 -> 23.3): tach KV sieving (noref) va diem bat dau (l0 22 = .69L)
LM=nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
LL="--model $LM --method layersieve --unit --drop-ratio 0.36 --min-keep 0.25 --max-model-len 140000 --prompts data/infbench_llama.jsonl"
job llama_ib_ls_noref $PY scripts/eval_bench.py $LL --start-layer 19 --out $B/llama/infbench/ls_unit_l19_d0.36_f0.25_noref.jsonl
job llama_ib_ls_l22 $PY scripts/eval_bench.py $LL --start-layer 22 --out $B/llama/infbench/ls_unit_l22_d0.36_f0.25_noref.jsonl
job llama_ib_onecut_unit $PY scripts/eval_bench.py --model $LM --method layersieve --unit --start-layer 15 --oneshot-keep 0.375 \
  --max-model-len 140000 --prompts data/infbench_llama.jsonl --out $B/llama/infbench/abl_oneshot_unit_l15_k0.375.jsonl

# ---- F (uu tien 1): bien accuracy-chi phi, progressive vs mot lan cat (ca hai co units), C = .55 / .62 / .72.
#      Moi ben chon cau hinh tot nhat trong luoi. Sang loc tren hard set (120 cau) + RULER 32k 6 task kho; progressive
#      them bien the --accum (xep hang theo diem cong don qua cac layer da cham: dieu mot lan cat khong co).
FR=results/frontier; mkdir -p $FR/hard $FR/ruler
RT="--tasks cwe,fwe,niah_multikey_3,vt,niah_multivalue,qa_2 --lengths 32768 --max-model-len 32768 --prompts data/ruler_eval.jsonl"
fr() {  # fr <ten> <tham so...>
  local n=$1; shift
  job fh_$n $PY scripts/eval_hard.py --method layersieve --unit "$@" $IDS --out $FR/hard/$n.jsonl
  job fr_$n $PY scripts/eval_bench.py --method layersieve --unit "$@" $RT --out $FR/ruler/$n.jsonl
}
for x in 55:8:0.37 55:10:0.3 55:12:0.213 55:14:0.1 62:10:0.409 62:12:0.335 62:14:0.24 62:16:0.113 \
         72:12:0.51 72:14:0.44 72:16:0.347 72:17:0.287 72:18:0.216; do
  IFS=: read c s k <<< "$x"; fr c${c}_one_s${s}_k$k --start-layer $s --oneshot-keep $k
done
for x in 55:8:0.15:0.111 55:10:0.15:0.166 55:12:0.1:0.26 62:10:0.15:0.105 62:12:0.15:0.154 62:14:0.15:0.289 \
         72:12:0.15:0.085 72:14:0.15:0.119 72:17:0.25:0.4; do
  IFS=: read c l f r <<< "$x"
  fr c${c}_prog_l${l}_f${f}_r$r --start-layer $l --drop-ratio $r --min-keep $f
  fr c${c}_acc_l${l}_f${f}_r$r --start-layer $l --drop-ratio $r --min-keep $f --accum
done

# ---- P1 (review 2, uu tien cao nhat): mot lan cat + units ngay sau l0, cung C = 0.72 voi LayerSieve
#      (start-layer s = so layer dense): s17 k.293 = (17 + 11k)/28; s18 k.22 = (18 + 10k)/28; them s13 k.5 (C .73)
oc() {  # oc <s> <k>
  local n=abl_oneshot_unit_l$1_k$2
  job ib_$n $PY scripts/eval_bench.py --method layersieve --unit --start-layer $1 --oneshot-keep $2 \
    --prompts data/infbench.jsonl --out $B/infbench/$n.jsonl
  job ru128_$n $PY scripts/eval_bench.py --method layersieve --unit --start-layer $1 --oneshot-keep $2 \
    --prompts data/ruler_eval.jsonl --lengths 131072 --out $B/ruler/$n.jsonl
}
oc 17 0.293
oc 18 0.22

# ---- P0a: hard set, 120 cau dense-dung cua 7B-1M (results/scaling/ids/7b1m.txt)
IDS="--ids results/scaling/ids/7b1m.txt --gpu-util 0.85 --no-ttft"
SP=baselines/SpecPrefill/configs
job hard_draft15 $PY scripts/eval_hard.py --model Qwen/Qwen2.5-1.5B-Instruct --method dense $IDS --out $RV/hard/draft15.jsonl
job hard_draft05 $PY scripts/eval_hard.py --model Qwen/Qwen2.5-0.5B-Instruct --method dense $IDS --out $RV/hard/draft05.jsonl
for k in 9 5 3 1; do
  job hard_spec_p$k $PY scripts/eval_hard.py --method specprefill --spec-config $SP/config_p${k}_full_lah8.yaml $IDS \
    --out $RV/hard/spec_p$k.jsonl
done
LSD="--method layersieve --start-layer 17 --drop-ratio 0.4 --min-keep 0.25"
job hard_ls_tok $PY scripts/eval_hard.py $LSD $IDS --out $RV/hard/ls_tok.jsonl
job hard_ls_unit $PY scripts/eval_hard.py $LSD --unit $IDS --out $RV/hard/ls_unit.jsonl
job hard_ls_unit_r22 $PY scripts/eval_hard.py $LSD --unit --decode-ref 22 $IDS --out $RV/hard/ls_unit_r22.jsonl
job hard_onecut_unit $PY scripts/eval_hard.py --method layersieve --start-layer 15 --oneshot-keep 0.375 --unit $IDS \
  --out $RV/hard/onecut_unit_l15.jsonl
# Q3: do nhay cua so quan sat va sink
for o in 32 512; do
  job hard_ls_obs$o $PY scripts/eval_hard.py $LSD --unit --n-obs $o $IDS --out $RV/hard/ls_unit_obs$o.jsonl
done
for z in 16 512; do
  job hard_ls_sink$z $PY scripts/eval_hard.py $LSD --unit --n-sink $z $IDS --out $RV/hard/ls_unit_sink$z.jsonl
done

# ---- P0b: InfiniteBench cat 32k (draft 1.5B trong cua so 32k goc, khong YaRN)
I32="--prompts data/infbench_32k.jsonl --max-model-len 34000"
job ib32_dense $PY scripts/eval_bench.py --method dense $I32 --out $RV/infbench32k/dense.jsonl
job ib32_ls $PY scripts/eval_bench.py $LSD --unit --decode-ref 22 $I32 --out $RV/infbench32k/ls_unit_d0.4_f0.25_r22.jsonl
job ib32_spec $PY scripts/eval_bench.py --method specprefill --spec-config $SP/config_p5_full_lah8.yaml $I32 \
  --out $RV/infbench32k/m_specprefill_p50.jsonl
job ib32_spec30 $PY scripts/eval_bench.py --method specprefill --spec-config $SP/config_p3_full_lah8.yaml $I32 \
  --out $RV/infbench32k/m_specprefill_p30.jsonl

# ---- P0c: toc do, 5 prompt / do dai, CUDA graph
CF="dense,17:0.4::0.25,17:0.4:22:0.25,o15:0.375,o15:0.375:15,slim:0.75/0.55/0.35,o0:0.5"
job speed_review $PY scripts/bench_speed.py --graph --model $M --prompts data/speed.jsonl --lengths 131072,262144 \
  --n 5 --decode 256 --gpu-util 0.93 --configs "$CF" --out $RV/speed/7b1m.jsonl
job speed_draft15 $PY scripts/bench_speed.py --graph --model Qwen/Qwen2.5-1.5B-Instruct --yarn 4 --prompts data/speed.jsonl \
  --lengths 131072 --n 5 --decode 8 --gpu-util 0.6 --configs dense --out $RV/speed/draft15.jsonl

# ---- P2 (HF, 2 tien trinh song song): ASL theo tau (Pareto), FastKV va ASL co nen KV truoc diem cat (cung cau hinh cho
#      accuracy va toc do), SnapKV 25% sau prefill dense (gia tri cua KV sieving so voi eviction thuong)
HF=/root/team_1/hieurezdev/PromptPruningPaper/ext/ASL/.venv/bin/python
hfj() {  # hfj <ten> <tham so asl_driver...>: IB roi RULER 128k
  local n=$1; shift
  job ib_$n $HF baselines/asl_driver.py "$@" --prompts data/infbench.jsonl --out $B/infbench/$n.jsonl
  job ru128_$n $HF baselines/asl_driver.py "$@" --prompts data/ruler_eval.jsonl --lengths 131072 --out $B/ruler/$n.jsonl
}
ASLA="--method asl --select-layer 10 --select-k 0.375 --monitor-k 2048"
hfj m_asl_t0.45_f0.375 $ASLA --tau 0.45 &
hfj m_asl_t0.6_f0.375 $ASLA --tau 0.6
wait
hfj m_fastkv_l15_f0.375_kv0.375 --method fastkv --select-layer 15 --select-k 0.375 --cache-k 0.375 &
hfj m_snapkv_kv0.25 --method vanilla --cache-k 0.25
wait
hfj m_asl_t0.3_f0.375_kv0.375 $ASLA --tau 0.3 --cache-k 0.375
oc 13 0.5

echo "$(date '+%F %T') het night6 P0/P1, chay tiep night4 (Llama) -> p2"
bash scripts/run_night4.sh
