#!/bin/bash
# Stop run_insight.sh (PID $1) right after its last HF job, so its trailing vLLM jobs at gpu-util 0.25 (which cannot
# start: not enough memory for 128k) do not block the lanes waiting on it.
cd "$(dirname "$0")/.."
R=results/selfext
until grep -qE "\[cwe_128k_copy_b32768\] (xong|loi)" $R/run.log; do sleep 30; done
C=$(pgrep -P $1); kill $1; sleep 1; [ -n "$C" ] && kill $C
echo "$(date '+%F %T') [insight] dung sau cwe b32768 (phan vLLM chuyen sang run_insight_vllm2)" >> $R/run.log
