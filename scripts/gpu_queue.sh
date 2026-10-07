#!/bin/bash
# Ham dung chung cho cac script chay hang doi GPU (source file nay). Bien: LOGDIR (bat buoc), MARGIN_MIB, POLL, MAX_UTIL,
# GPUS (danh sach index cach nhau dau phay, mac dinh moi GPU).
#   pick_gpu <util>  : in index GPU dau tien co memory.free >= util * memory.total + MARGIN_MIB va utilization.gpu
#                      <= MAX_UTIL (%), thoa 2 lan do lien tiep cach POLL giay
#   run <ten> <util> <lenh...> : doi GPU, chay lenh (log o $LOGDIR/<ten>.log), loi thi doi 5 phut va chay lai (toi da 5 lan);
#                      xong thi tao $LOGDIR/<ten>.done, lan chay lai script se bo qua job do
# Nhieu runner chay song song dung chung mot khoa: job chon GPU khong giu khoa, roi lay khoa, kiem tra lai GPU do va
# khoi dong, giu khoa STARTUP giay de no kip chiem bo nho. Hai job khong the cung thay mot GPU con trong roi cung OOM,
# va job doi GPU lon khong chan job nho.
MARGIN_MIB=${MARGIN_MIB:-4096}
POLL=${POLL:-60}
MAX_UTIL=${MAX_UTIL:-100}
GPUS=${GPUS:-}
STARTUP=${STARTUP:-240}
LOCK=${LOCK:-/tmp/gpu_queue_$USER.lock}

_fits() {  # _fits <util> [index]: in cac GPU (hoac chi GPU index) thoa dieu kien
  nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' -v u="$1" -v m="$MARGIN_MIB" -v x="$MAX_UTIL" -v g="${2:-}" -v a=",$GPUS," \
      '(g == "" || $1 == g) && (a == ",," || index(a, "," $1 ",")) && $2 >= u * $3 + m && $4 <= x {print $1; exit}'
}

pick_gpu() {
  local prev="" cur
  while true; do
    cur=$(_fits "$1")
    if [ -n "$cur" ] && [ "$cur" = "$prev" ]; then echo "$cur"; return; fi
    prev=$cur; sleep "$POLL"
  done
}

run() {
  local name=$1 util=$2 g pid lk; shift 2
  [ -f "$LOGDIR/$name.done" ] && return 0
  for try in 1 2 3 4 5; do
    while true; do
      g=$(pick_gpu "$util")
      exec {lk}>"$LOCK"; flock "$lk"
      [ -n "$(_fits "$util" "$g")" ] && break
      flock -u "$lk"; exec {lk}>&-
    done
    echo "$(date '+%F %T') [$name] GPU $g, lan $try"
    CUDA_VISIBLE_DEVICES=$g "$@" >> "$LOGDIR/$name.log" 2>&1 {lk}>&- &
    pid=$!
    sleep "$STARTUP"
    flock -u "$lk"; exec {lk}>&-
    if wait "$pid"; then
      echo "$(date '+%F %T') [$name] xong"; touch "$LOGDIR/$name.done"; return 0
    fi
    echo "$(date '+%F %T') [$name] loi (xem $LOGDIR/$name.log), doi roi chay lai"; sleep 300
  done
  echo "$(date '+%F %T') [$name] bo qua sau 5 lan loi"
}
