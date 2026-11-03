#!/bin/bash
# After the remap probe (PID $1): the fast-dim damping screen on KV, then the query-only generation lane
# (run_remap9.sh, resumes where it stopped), then the damping screen on LongDialogue.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 $1 2>/dev/null; do sleep 20; done
V="--variants fastoff4k,fasthalf4k,fastoff32k,fastoff1k,scale1.3"
.venv/bin/python scripts/remap_probe.py --task kv --n 20 $V --out results/selfext/fast_probe_kv.json > results/selfext/fast_probe_kv.log 2>&1
nohup bash scripts/run_remap9.sh > /dev/null 2>&1 &
