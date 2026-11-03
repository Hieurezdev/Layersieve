#!/bin/bash
# One hourly check: sleep an hour, then log GPU use, this repo's running jobs and the last queue events, and exit
# (the exit wakes the session, which relaunches it).
cd "$(dirname "$0")/.."
sleep 3600
LOG=results/selfext/hourly.log
{
  echo "=== $(date -u '+%F %T') UTC"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  echo "jobs: $(pgrep -fc '(recall|selfextend)_eval.py|qlen_probe.py') eval, $(pgrep -fc 'scripts/run_') queues"
  pgrep -af "(recall|selfextend)_eval.py" | grep -v pgrep | sed 's/.*--out //' | cut -c1-80
  tail -n 6 results/selfext/run.log
} | tee -a $LOG
