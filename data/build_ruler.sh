#!/bin/bash
# RULER (NVIDIA, github.com/NVIDIA/RULER, commit c3f5e3b) cho Qwen2.5 (template chat 'qwen2.5' them vao
# scripts/data/template.py), 13 task x do dai x 500 mau, seed 42 (mac dinh cua RULER).
#   RULER=/root/PromptPruningPaper/ext/RULER bash data/build_ruler.sh [do_dai ...]   (mac dinh 32768 65536 131072)
# -> data/ruler/<do_dai>/<task>/validation.jsonl ({"index", "input", "outputs", "length", ...});
# can tai truoc essay va QA: scripts/data/synthetic/json/{download_paulgraham_essay.py,download_qa_dataset.sh}.
# Mau 100 cau / task (seed 42) duoc lay luc chay danh gia (scripts/eval_bench.py), khong o day.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
RULER=${RULER:-/root/PromptPruningPaper/ext/RULER}
LENS=${*:-32768 65536 131072}
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue
       niah_multiquery vt cwe fwe qa_1 qa_2"
export PATH=$ROOT/.venv/bin:$PATH                     # prepare.py goi "python" qua subprocess
mkdir -p data/ruler/logs
for L in $LENS; do for t in $TASKS; do echo "$L $t"; done; done |
  xargs -P ${JOBS:-12} -n 2 sh -c 'cd '"$RULER"'/scripts/data && python prepare.py --save_dir '"$ROOT"'/data/ruler/$0 \
    --benchmark synthetic --task $1 --tokenizer_path Qwen/Qwen2.5-7B-Instruct-1M --tokenizer_type hf \
    --max_seq_length $0 --model_template_type qwen2.5 --num_samples 500 > '"$ROOT"'/data/ruler/logs/$0_$1.log 2>&1 \
    && echo "xong $0 $1" || echo "loi $0 $1"'
