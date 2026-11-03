#!/usr/bin/env bash
# Evidence-loss pilot (Qwen2.5-7B-Instruct-1M, 1 GPU, ~0.32 of an H200). Re-running skips prompts already written.
#   PY=/path/to/vllm-0.6.3/python bash scripts/run_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=results/pilot
mkdir -p $OUT/logs
[ -f data/pilot.jsonl ] || $PY data/build_pilot.py --lengths 32000,64000 --n 12 --n-vt 36 --out data/pilot.jsonl
$PY scripts/pilot_evidence.py --mode dense --out $OUT/dense.jsonl > $OUT/logs/dense.log 2>&1
# l0 = 17 (paper default, 0.61L), 14 (0.5L, "fast descent" of Table VI), 11 (0.39L, stress)
for L0 in 17 14 11; do
  $PY scripts/pilot_evidence.py --mode ls --start-layer $L0 --drop-ratio 0.4 --min-keep 0.25 --trace-from 6 \
      --dense $OUT/dense.jsonl --out $OUT/ls_l$L0.jsonl --npz-dir $OUT/npz/ls_l$L0 > $OUT/logs/ls_l$L0.log 2>&1
done
# counterfactual dense scores, only for the prompts that lost evidence in some LayerSieve run
$PY -c "import json,glob; print(chr(10).join(sorted({r['id'] for f in glob.glob('$OUT/ls_l*.jsonl') for r in map(json.loads, open(f)) if not r['ev_all_kept']})))" > $OUT/lost_ids.txt
$PY scripts/pilot_evidence.py --mode dense-trace --trace-from 6 --ids $OUT/lost_ids.txt --out $OUT/dense_trace.jsonl \
    --npz-dir $OUT/npz/dense_trace > $OUT/logs/dense_trace.log 2>&1
