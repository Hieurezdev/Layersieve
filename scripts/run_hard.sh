#!/bin/bash
# Hard cases cho SpecPrefill (results/hard): dense draft/target, SpecPrefill (draft 1.5B/0.5B, giu 10%/30%), LayerSieve.
set -e
cd "$(dirname "$0")/.."
P=data/hard.jsonl; R=results/hard; C=baselines/SpecPrefill/configs; mkdir -p $R
[ -f $P ] || .venv/bin/python data/build_hard.py --out $P
run() { .venv/bin/python scripts/eval_hard.py --prompts $P "$@"; }
run --method dense --out $R/target_dense.jsonl
# chi chay tiep tren tap con: moi task toi da 30 mau target dense dung (vt lay het), seed 0
[ -f $R/ids_sub.txt ] || .venv/bin/python - <<'PY'
import json, random
rs = [json.loads(l) for l in open("results/hard/target_dense.jsonl")]
rng, ids = random.Random(0), []
for t, n in [("kv", 30), ("vt", 40), ("hotpotqa", 30), ("2wikimqa", 30)]:
    ids += rng.sample(ok := [r["id"] for r in rs if r["task"] == t and r["ok"]], min(n, len(ok)))
open("results/hard/ids_sub.txt", "w").write("\n".join(ids) + "\n")
PY
run() { .venv/bin/python scripts/eval_hard.py --prompts $P --ids $R/ids_sub.txt "$@"; }
run --method layersieve --out $R/layersieve.jsonl
run --method dense --model Qwen/Qwen2.5-1.5B-Instruct --out $R/draft15_dense.jsonl
run --method dense --model Qwen/Qwen2.5-0.5B-Instruct --out $R/draft05_dense.jsonl
run --method specprefill --spec-model Qwen/Qwen2.5-1.5B-Instruct --spec-config $C/config_p1_full_lah8.yaml --out $R/sp15_p1.jsonl
run --method specprefill --spec-model Qwen/Qwen2.5-1.5B-Instruct --spec-config $C/config_p3_full_lah8.yaml --out $R/sp15_p3.jsonl
run --method specprefill --spec-model Qwen/Qwen2.5-0.5B-Instruct --spec-config $C/config_p1_full_lah8.yaml --out $R/sp05_p1.jsonl
