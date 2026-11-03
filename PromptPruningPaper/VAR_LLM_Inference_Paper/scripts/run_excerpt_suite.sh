#!/bin/bash
# Excerpt-recall suite: every method on the task set of results/selfext/EXCERPT_RECALL.md (+ mv, math_find of LD_CAUSE.md).
# Every run but fuse uses --select, so each row also holds the plain HF dense answer (pred_dense) next to the method's
# answer (pred_recall): the dense column comes for free and the gate analyses (analysis/gates.py) can reuse the runs.
#
#   bash scripts/run_excerpt_suite.sh                        # pilot sizes, Qwen2.5-7B-1M, every method x every task
#   SIZE=full bash scripts/run_excerpt_suite.sh              # KV86, LD200, RULER/InfBench 100 each
#   METHODS="gem_win gem_tail" TASKS="code_debug math_find" bash scripts/run_excerpt_suite.sh
#   MODEL=q3 bash scripts/run_excerpt_suite.sh               # Qwen3-4B-Instruct-2507 (transformers 4.51, its own heads)
#   DRY=1 bash scripts/run_excerpt_suite.sh                  # print the commands only
#   .venv/bin/python scripts/analysis/suite_table.py results/selfext/suite/q25_pilot   # the table
#
# Methods (selector: retrieval heads + whole lines unless the name says otherwise):
#   copy      EXCERPT_RECALL.md copy: mean over heads at L17, 64-token blocks +-96, budget 2048
#   random    copy with random blocks (control, EXCERPT_RECALL.md sec. 5)
#   copy_retr copy with the retrieval-head selector
#   move      one 2k window moved next to the question by RoPE only (LD_CAUSE.md sec. 9)
#   gem_win   one 2k window re-encoded alone, rest of the context dropped (LD_CAUSE.md sec. 15)
#   gem_topk  top 2k tokens of retrieval-head blocks re-encoded alone, SEP between pieces
#   gem_tail  gem_win keeping the last 512 prompt tokens verbatim instead of 128 (options / instructions of the task)
#   draft     copy_retr whose selection also uses the tokens of the plain answer (ATTRIEVAL-like)
#   focus     +log(ctx/picked) on the picked window in the retrieval heads, true positions (DySCO-like)
#   reorder   the picked window moved (not copied) to the end of the context and re-encoded (Attention-Sorting-like)
#   fuse      plain and copy_retr decoded in lockstep, sum of log-probs (no --select)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SIZE=${SIZE:-pilot}; MODEL=${MODEL:-q25}; CAP=${CAP:-70000}
METHODS=${METHODS:-"copy random copy_retr move gem_win gem_topk gem_tail draft focus reorder fuse"}
TASKS=${TASKS:-"kv ld mk3 mv fwe qa_1 code_debug qa_2 vt cwe math_find"}
# one start-up lock for the whole team: the QuangNM queues use this file too
LOCK=${LOCK:-/root/team_1/QuangNM/PromptPruningPaper/VAR_LLM_Inference_Paper/results/selfext/.team1_gpu.lock}
case $MODEL in
  q25) HEADS=results/repos/heads_7b1m.json; NEED=26000 ;;
  q3)  HEADS=results/repos/heads_q3_4b.json; NEED=32000
       export PYTHONPATH=/root/team_1/hieurezdev/pylib_tf451 SE_MODEL=Qwen/Qwen3-4B-Instruct-2507 ;;
  *) echo "MODEL must be q25 or q3"; exit 1 ;;
esac
R=results/selfext/suite/${MODEL}_${SIZE}; [ -z "$DRY" ] && mkdir -p $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- task -> prompt arguments -------------------------------------------------------------------------------------
RU="--prompts data/ruler_eval.jsonl --lengths 131072"
if [ $SIZE = full ]; then NR=100; NI=100; KV="--prompts data/infbench_kv86.jsonl"; LD="--prompts data/infbench_ld200.jsonl"
else NR=20; NI=20; KV="--prompts data/infbench_kv86.jsonl --limit 20"; LD="--prompts data/infbench_ld50.jsonl"; fi
targs() {
  case $1 in
    kv)         echo "$KV" ;;
    ld)         echo "$LD" ;;
    mk3)        echo "$RU --tasks niah_multikey_3 --limit $([ $SIZE = full ] && echo $NR || echo 40)" ;;   # doc: 40
    mv)         echo "$RU --tasks niah_multivalue --limit $NR" ;;
    fwe|cwe|vt|qa_1|qa_2) echo "$RU --tasks $1 --limit $NR" ;;
    code_debug|math_find) echo "--prompts data/infbench.jsonl --tasks $1 --limit $NI" ;;
    *) echo "BAD_TASK" ;;
  esac; }

# --- method -> recall_eval.py arguments ---------------------------------------------------------------------------
SEL="--agg retr --heads $HEADS --snap"
WIN="$SEL --pad 992"                                   # one window of about 2k tokens around the best block
margs() {
  case $1 in
    copy)      echo "--select" ;;
    random)    echo "--select --random" ;;
    copy_retr) echo "--select $SEL" ;;
    move)      echo "--select --move $WIN" ;;
    gem_win)   echo "--select --gem --gem-sep $WIN" ;;
    gem_topk)  echo "--select --gem --gem-sep --agg retr --heads $HEADS --budget 2048 --pad 0" ;;
    gem_tail)  echo "--select --gem --gem-sep $WIN --ins 512" ;;
    draft)     echo "--select --draft $SEL" ;;
    focus)     echo "--select --focus $HEADS --head-thr 0.1 $WIN" ;;
    reorder)   echo "--select --reorder $WIN" ;;
    fuse)      echo "--fuse $SEL" ;;
    *) echo "BAD_METHOD" ;;
  esac; }

# --- queue (same gate as scripts/run_day_c.sh) --------------------------------------------------------------------
log() { echo "$(date '+%F %T') [$1] $2" >> $R/run.log; }
retry() { grep -qiE "out of memory|must be on CUDA|CUDA error" $R/$1.log; }
team_mem() {                                   # MiB used by GPU processes started from /root/team_1
  local s=0
  while IFS=, read -r p m; do
    p=${p// /}; m=${m// MiB/}; m=${m// /}
    { readlink /proc/$p/cwd; tr '\0' ' ' < /proc/$p/cmdline; } 2>/dev/null | grep -q /root/team_1 && s=$((s + m))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
  echo $s; }
gate() {                                       # wait for room, start "$@" in the background, hold the lock 3 min
  exec 9>> $LOCK; flock 9
  local w=0
  until [ $(team_mem) -le $((CAP - NEED)) ]; do
    [ $w -eq 0 ] && log $N "cho VRAM team_1 ($(team_mem) MiB)"; w=1; sleep 60; done
  "$@" 9>&- & local pid=$!; sleep 180; flock -u 9; exec 9>&-
  wait $pid; }
job() { N=$1; shift
  if [ -n "$DRY" ]; then echo "$PY scripts/recall_eval.py $* --out $R/$N.jsonl"; return; fi
  [ -s $R/$N.jsonl ] && grep -qE "\[$N\] xong" $R/run.log && return; log $N "bat dau"
  until gate $PY scripts/recall_eval.py "$@" --out $R/$N.jsonl >> $R/$N.log 2>&1; do
    retry $N || { log $N "loi khong phai OOM"; return; }
    log $N "OOM, thu lai sau 2 phut"; : > $R/$N.log; sleep 120; done
  log $N "xong"; }

for t in $TASKS; do ta=$(targs $t); [ "$ta" = BAD_TASK ] && { echo "unknown task $t"; exit 1; }; done
for m in $METHODS; do ma=$(margs $m); [ "$ma" = BAD_METHOD ] && { echo "unknown method $m"; exit 1; }; done
[ -z "$DRY" ] && log suite "bat dau MODEL=$MODEL SIZE=$SIZE METHODS=[$METHODS] TASKS=[$TASKS]"
for m in $METHODS; do                          # method-major: one finished column at a time
  for t in $TASKS; do job ${t}_$m $(margs $m) $(targs $t); done
done
[ -z "$DRY" ] && log suite het
