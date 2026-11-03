#!/bin/bash
# Ablation cac cai tien cua doi top 1 (MOE, docs/MOE_VIETTEL_AI_RACE_2026_Technical_Report.pdf) ap vao LayerSieve:
#   unit  : bo ca don vi (dong/cau/menh de) thay vi tung token
#   accum : xep hang bang diem cong don qua cac layer da cham
#   bm25  : khong bao gio bo cac don vi khop cau hoi nhat theo BM25 (toi da 2% prompt)
#   bm25only : BM25 thay han diem attention (ablation: phan nao den tu khop tu khoa)
# Bo sung cho scripts/run_moe.sh: prompt vt da doi (build_hard.py truoc day khong tai lap duoc) nen tren ids_sub.txt target
# dense chi dung 17/40 vt. Lam lai nhu quy trinh goc: target dense tren ca 120 mau vt, lay cac mau dung chua co trong
# ids_sub.txt, chay draft 0.5B dense va cac bien the tren do. Ket qua: results/moe/hard_vt (summarize_moe.py gop vao).
# Moi job doi den khi co GPU bat ky du bo nho trong (+ du phong) moi chay; job loi thi doi va chay lai (cac script
# eval tu chay tiep tu mau da xong). Chay nen: nohup bash scripts/run_moe_vt.sh > results/moe/run_vt.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/moe; LOGDIR=$R/logs; mkdir -p $R/hard_vt $LOGDIR
source scripts/gpu_queue.sh

VARIANTS=(
  "base|"
  "unit|--unit"
  "accum|--accum"
  "bm25|--bm25 0.02"
  "all|--unit --accum --bm25 0.02"
  "bm25only|--unit --score bm25"
)

V=$R/hard_vt
run vt_target_dense 0.5 $PY scripts/eval_hard.py --method dense --tasks vt --gpu-util 0.5 --out $V/target_dense_all.jsonl
$PY - <<'PY'
import json
sub = set(open("results/hard/ids_sub.txt").read().split())
ids = [r["id"] for r in map(json.loads, open("results/moe/hard_vt/target_dense_all.jsonl")) if r["ok"] and r["id"] not in sub]
open("results/moe/hard_vt/ids.txt", "w").write("\n".join(ids) + "\n")
print(len(ids), "mau vt bo sung")
PY
run vt_draft05_dense 0.15 $PY scripts/eval_hard.py --method dense --model Qwen/Qwen2.5-0.5B-Instruct --ids $V/ids.txt \
  --gpu-util 0.15 --out $V/draft05_dense.jsonl
for v in "${VARIANTS[@]}"; do
  n=${v%%|*}; f=${v#*|}
  run "vt_$n" 0.5 $PY scripts/eval_hard.py --method layersieve --ids $V/ids.txt --gpu-util 0.5 $f --out $V/$n.jsonl
done
