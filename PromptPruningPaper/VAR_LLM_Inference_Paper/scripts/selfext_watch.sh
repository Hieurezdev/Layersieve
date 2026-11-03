#!/bin/bash
# Watchdog for run_selfext*.sh on a shared GPU: exit 1 if a selfext queue is alive but none of its jobs has run
# for > 10 minutes (stalled queue); exit 0 when no selfext queue is left.
cd "$(dirname "$0")/.."
idle=0
while true; do
  q=$(pgrep -fc "scripts/run_(selfext|remap)")
  [ "$q" -eq 0 ] && { echo "$(date '+%F %T') het hang doi selfext"; exit 0; }
  if pgrep -f "selfextend_eval.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  [ $idle -ge 3 ] && { echo "$(date '+%F %T') CANH BAO: hang doi selfext song nhung khong co job >10 phut"; tail -3 results/selfext/run.log; exit 1; }
  sleep 300
done
