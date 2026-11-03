#!/bin/bash
# Ablation cac cai tien cua doi top 1 (MOE, docs/MOE_VIETTEL_AI_RACE_2026_Technical_Report.pdf) ap vao LayerSieve:
#   unit  : bo ca don vi (dong/cau/menh de) thay vi tung token
#   accum : xep hang bang diem cong don qua cac layer da cham
#   bm25  : khong bao gio bo cac don vi khop cau hoi nhat theo BM25 (toi da 2% prompt)
#   bm25only : BM25 thay han diem attention (ablation: phan nao den tu khop tu khoa)
# Tap case kho (results/moe/hard, chi mau trong results/hard/ids_sub.txt) chay truoc, roi kv50 cho unit
# (results/moe/kv50; LayerSieve goc tren kv50 = results/kv50/kv50_catdan.jsonl, scripts/run_kv50.sh).
# Prompt vt da doi (build_hard.py truoc day khong tai lap duoc) nen chay lai target dense va draft 0.5B dense de chia kho/de.
# Moi job doi den khi co GPU bat ky du bo nho trong (+ du phong) moi chay; job loi thi doi va chay lai (cac script
# eval tu chay tiep tu mau da xong). Chay nen: nohup bash scripts/run_moe.sh > results/moe/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/moe; LOGDIR=$R/logs; mkdir -p $R/hard $R/kv50 $LOGDIR
source scripts/gpu_queue.sh
[ -f data/hard.jsonl ] || $PY data/build_hard.py --out data/hard.jsonl || exit 1
[ -f data/kv50.jsonl ] || $PY data/build_kv.py --out data/kv50.jsonl || exit 1

VARIANTS=(
  "base|"
  "unit|--unit"
  "accum|--accum"
  "bm25|--bm25 0.02"
  "all|--unit --accum --bm25 0.02"
  "bm25only|--unit --score bm25"
)

run hard_target_dense 0.5 $PY scripts/eval_hard.py --method dense --ids results/hard/ids_sub.txt --gpu-util 0.5 \
  --out $R/hard/target_dense.jsonl
run hard_draft05_dense 0.15 $PY scripts/eval_hard.py --method dense --model Qwen/Qwen2.5-0.5B-Instruct \
  --ids results/hard/ids_sub.txt --gpu-util 0.15 --out $R/hard/draft05_dense.jsonl
for v in "${VARIANTS[@]}"; do
  n=${v%%|*}; f=${v#*|}
  run "hard_$n" 0.5 $PY scripts/eval_hard.py --method layersieve --ids results/hard/ids_sub.txt --gpu-util 0.5 \
    $f --out $R/hard/$n.jsonl
done
MARGIN_MIB=1024 run kv50_unit 0.68 $PY scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --no-ttft --mlp-chunk 16384 \
  --gpu-util 0.68 --max-model-len 170000 --unit --out $R/kv50/unit.jsonl
