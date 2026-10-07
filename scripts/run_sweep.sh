#!/bin/bash
# Quet lich sang cua LayerSieve: giu nguyen layer 0..l0-1 (l0 = 17 / 14 / 7), roi bo rho moi layer.
#   Do chinh xac: tap case kho (results/moe/sp/ids.txt = ids_sub + mau vt bo sung) -> results/sweep/hard/s<l0>_d<rho>.jsonl
#     (l0=17, rho=0.12 = base, da co trong results/moe/hard + hard_vt)
#   Toc do: scripts/run_speed_all.sh
# Tong hop: python scripts/summarize_sweep.py. Chay nen:
#   setsid nohup bash scripts/run_sweep.sh > results/sweep/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/sweep; LOGDIR=$R/logs; mkdir -p $R/hard $LOGDIR
source scripts/gpu_queue.sh
IDS=results/moe/sp/ids.txt
[ -f $IDS ] || cat results/hard/ids_sub.txt results/moe/hard_vt/ids.txt > $IDS
for c in 17:0.25 14:0.06 14:0.10 14:0.15 14:0.25 7:0.03 7:0.05 7:0.08; do
  s=${c%:*}; d=${c#*:}
  run "hard_s${s}_d${d}" 0.5 $PY scripts/eval_hard.py --method layersieve --ids $IDS --gpu-util 0.5 \
    --start-layer $s --drop-ratio $d --out $R/hard/s${s}_d${d}.jsonl
done
