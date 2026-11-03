#!/bin/bash
# Quet lich sang voi unit (bo nguyen don vi, inmodel_prune/units.py) - phuong phap chinh sau ablation MOE. Cung tap
# case kho voi scripts/run_sweep.sh -> results/sweep/hard_unit/s<l0>_d<rho>.jsonl (l0=17, rho=0.12 = results/moe unit).
#   setsid nohup bash scripts/run_sweep_unit.sh > results/sweep/run_unit.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=results/sweep; LOGDIR=$R/logs; mkdir -p $R/hard_unit $LOGDIR
source scripts/gpu_queue.sh
IDS=results/moe/sp/ids.txt
for c in 17:0.25 17:0.35 14:0.10 14:0.15 14:0.25 7:0.05 7:0.08 7:0.12 7:0.20; do
  s=${c%:*}; d=${c#*:}
  run "unit_s${s}_d${d}" 0.5 $PY scripts/eval_hard.py --method layersieve --unit --ids $IDS --gpu-util 0.5 \
    --start-layer $s --drop-ratio $d --out $R/hard_unit/s${s}_d${d}.jsonl
done
