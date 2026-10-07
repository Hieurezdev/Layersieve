#!/bin/bash
# Sang KV cho decode (decode_ref = r): sau prefill, moi layer < r chi giu KV cua cac token da toi layer r, nen decode doc
# it KV hon (layer nong khong con giu ca prompt).
#   Tap case kho -> results/decode/hard/<ten>.jsonl
#     token l0 = 17, rho = 0.12 voi r = 17 / 22 / 27; unit voi r = 27; unit rho = 0.25 voi r = 27 / 22
#   kv50, unit + r = 27 -> results/decode/kv50_unit_r27.jsonl
#   Toc do: scripts/run_speed_all.sh
# Tong hop: python scripts/summarize_decode.py. Chay nen:
#   setsid nohup bash scripts/run_decode.sh > results/decode/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/decode; LOGDIR=$R/logs; mkdir -p $R/hard $LOGDIR
source scripts/gpu_queue.sh
IDS=results/moe/sp/ids.txt
for v in "r17|--decode-ref 17" "r22|--decode-ref 22" "r27|--decode-ref 27" "unit_r27|--unit --decode-ref 27" \
         "unit_d0.25_r27|--unit --drop-ratio 0.25 --decode-ref 27" "unit_d0.25_r22|--unit --drop-ratio 0.25 --decode-ref 22"; do
  n=${v%%|*}; f=${v#*|}
  run "hard_$n" 0.5 $PY scripts/eval_hard.py --method layersieve --ids $IDS --gpu-util 0.5 $f --out $R/hard/$n.jsonl
done
MARGIN_MIB=1024 run kv50_unit_r27 0.68 $PY scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --no-ttft --mlp-chunk 16384 \
  --gpu-util 0.68 --max-model-len 170000 --unit --decode-ref 27 --out $R/kv50_unit_r27.jsonl
