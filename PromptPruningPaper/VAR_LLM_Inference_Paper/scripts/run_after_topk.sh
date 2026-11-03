#!/bin/bash
# Relaunch the queues paused to give the top-k test priority (each job skips what is already "xong").
cd "$(dirname "$0")/.."
until grep -q "\[topk_A\] het" results/selfext/run.log && grep -q "\[topk_B\] het" results/selfext/run.log; do sleep 60; done
for q in "run_gem2.sh A" "run_gem2.sh B" "run_mvfix.sh" "run_q3.sh"; do setsid nohup bash scripts/$q >/dev/null 2>&1 & sleep 5; done
