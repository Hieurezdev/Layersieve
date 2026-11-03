#!/bin/bash
# Toc do LayerSieve vs lich cua FastKV / ASL, cung khung vLLM (decode CUDA graph), 7B-1M, GPU trong.
# Toc do chi phu thuoc lich sang (layer nao xu ly bao nhieu token), nen FastKV / ASL duoc dung lai bang che do "cat mot
# lan" cua vllm_patch: cat o layer s, giu 37.5% (muc cung chi phi), cac layer truoc s giu toan bo KV nhu ban goc.
#   FastKV: s = 15. ASL: s = trung vi (va tu phan vi 1, 3) cua layer ASL tu chon tren InfiniteBench (select_layer).
# Chi phi cham diem tung layer tu L_min toi s cua ASL khong tinh (co loi cho ASL).
# -> results/bench/speed_vs.jsonl ; tong hop: python scripts/summarize_scaling.py style (speed_table)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
read Q1 MED Q3 < <($PY - <<'EOF'
import json, statistics
ls = [r["select_layer"] for f in ["results/bench/infbench/m_asl_t0.3_f0.375.jsonl"]
      for r in map(json.loads, open(f)) if r.get("select_layer") is not None]
q = statistics.quantiles(ls, n=4)
print(round(q[0]), round(statistics.median(ls)), round(q[2]))
EOF
)
echo "$(date '+%F %T') [speed_vs] ASL select_layer q1/median/q3 = $Q1/$MED/$Q3"
CFG="dense,17:0.25:22,17:0.12:27,o15:0.375,o$MED:0.375"
[ "$Q1" != "$MED" ] && CFG="$CFG,o$Q1:0.375"
[ "$Q3" != "$MED" ] && CFG="$CFG,o$Q3:0.375"
$PY scripts/bench_speed.py --graph --model Qwen/Qwen2.5-7B-Instruct-1M --prompts data/speed.jsonl \
  --lengths 131072,262144 --n 2 --decode 256 --gpu-util 0.93 --configs "$CFG" \
  --out results/bench/speed_vs.jsonl
