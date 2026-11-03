#!/bin/bash
# Watchdog for run_repos*.sh on a shared GPU: exit 1 if a repos queue is alive but none of its eval_bench jobs has run
# for > 10 minutes (stalled queue); exit 0 when no repos queue is left.
cd "$(dirname "$0")/.."
idle=0
while true; do
  q=$(pgrep -fc "scripts/run_repos")
  [ "$q" -eq 0 ] && { echo "$(date '+%F %T') het hang doi repos"; exit 0; }
  if pgrep -f "eval_bench.py.*--out results/repos/|retrieval_heads.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  [ $idle -ge 3 ] && { echo "$(date '+%F %T') CANH BAO: hang doi repos song nhung khong co job >10 phut"; tail -3 results/repos/run.log; exit 1; }
  sleep 300
done
