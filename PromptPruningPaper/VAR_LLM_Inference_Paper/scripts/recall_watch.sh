#!/bin/bash
# Watchdog for the remap/recall queues: exit 1 if a queue is alive but none of its jobs has run for > 10 minutes;
# exit 0 when no queue is left.
cd "$(dirname "$0")/.."
idle=0
while true; do
  q=$(pgrep -fc "scripts/run_(selfext|remap|recall|bias|sel|fuse|focus|night9|night10|night11|night12|night13|night14|calib|reorder|gem|gem3|gemls|gemls_ld|ld_move|insight|insight_vllm|insight_vllm2|cause|cause2|cause3|depth_a|depth_b|rerot0|ld50b|move_sel|move_one|move_vt|move_cwe|dual1|dual2|selprobe|rel1|rel2)")
  [ "$q" -eq 0 ] && { echo "$(date '+%F %T') het hang doi"; exit 0; }
  if pgrep -f "(selfextend|recall)_eval.py|asl_driver.py|gem_ls.py|ld_move_probe.py|eval_bench.py|selector_probe.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  [ $idle -ge 3 ] && { echo "$(date '+%F %T') CANH BAO: hang doi song nhung khong co job >10 phut"; tail -3 results/selfext/run.log; exit 1; }
  sleep 300
done
