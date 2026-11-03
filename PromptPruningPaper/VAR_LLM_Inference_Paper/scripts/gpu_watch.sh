#!/bin/bash
# Theo doi GPU: moi 5 phut lay mau, moi tieng ghi 1 dong tom tat vao results/floor/gpu_watch.log.
# Thoat (exit 1) neu GPU ranh lien tuc > 10 phut trong khi hang doi (run_night*/run_bench*) van con song,
# de nguoi chay nhan duoc thong bao va xu ly ngay. Thoat 0 khi khong con hang doi nao.
#   bash scripts/gpu_watch.sh
cd "$(dirname "$0")/.."
LOG=results/floor/gpu_watch.log; idle=0; n=0
while true; do
  read u m <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')"
  q=$(pgrep -fc "scripts/run_(night|bench)")
  if [ "$m" -lt 1000 ] && [ "$u" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
  if [ $((n % 12)) -eq 0 ]; then
    echo "$(date '+%F %T') util=${u}% mem=${m}MiB queues=$q last: $(tail -1 results/floor/run.log)" >> $LOG
  fi
  if [ "$q" -eq 0 ]; then echo "$(date '+%F %T') het hang doi" | tee -a $LOG; exit 0; fi
  if [ $idle -ge 3 ]; then
    echo "$(date '+%F %T') CANH BAO: GPU ranh >10 phut, queues=$q" | tee -a $LOG; tail -3 results/floor/run.log; exit 1
  fi
  n=$((n + 1)); sleep 300
done
