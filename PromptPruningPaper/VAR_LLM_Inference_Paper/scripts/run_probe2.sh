#!/bin/bash
# Query-only remap / logit-scale screen on KV retrieval (scripts/remap_probe.py), then the query-only generation lane
# (run_remap9.sh, resumes where it stopped), then the same screen on LongDialogue.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/remap_probe.py --task kv --n 20 --out results/selfext/remap_probe_kv.json > results/selfext/remap_probe_kv.log 2>&1
nohup bash scripts/run_remap9.sh > /dev/null 2>&1 &
