#!/bin/bash
# SpecPrefill tren du lieu case kho moi (prompt vt da doi, xem scripts/run_moe_vt.sh), cung tap mau voi ablation MOE
# (results/hard/ids_sub.txt + results/moe/hard_vt/ids.txt). Ket qua: results/moe/sp.
#   draft15_dense : draft 1.5B tu tra loi (de chia tap kho/de theo draft 1.5B)
#   sp15_p1, sp15_p3, sp05_p1 : cau hinh nhu bang trong README
#   sp15_p9 : kiem tra ban port Qwen (baselines/specprefill_qwen.py): giu 90% thi phai gan bang target dense
# Moi job doi GPU du bo nho trong (scripts/gpu_queue.sh). Chay nen:
#   setsid nohup bash scripts/run_sp_check.sh > results/moe/run_sp.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/moe; S=$R/sp; C=baselines/SpecPrefill/configs; LOGDIR=$R/logs; mkdir -p $S $LOGDIR
source scripts/gpu_queue.sh
cat results/hard/ids_sub.txt $R/hard_vt/ids.txt > $S/ids.txt

run sp_draft15_dense 0.15 $PY scripts/eval_hard.py --method dense --model Qwen/Qwen2.5-1.5B-Instruct --ids $S/ids.txt \
  --gpu-util 0.15 --out $S/draft15_dense.jsonl
# draft nam trong cung phan gpu-util voi target: 0.5 chi con KV cho ~18k token (< 32k); 0.55 du (~59k)
for cfg in "sp15_p9 1.5B p9" "sp15_p1 1.5B p1" "sp15_p3 1.5B p3" "sp05_p1 0.5B p1"; do
  set -- $cfg
  MARGIN_MIB=2048 run "$1" 0.55 $PY scripts/eval_hard.py --method specprefill --spec-model Qwen/Qwen2.5-$2-Instruct \
    --spec-config $C/config_$3_full_lah8.yaml --ids $S/ids.txt --gpu-util 0.55 --out $S/$1.jsonl
done
