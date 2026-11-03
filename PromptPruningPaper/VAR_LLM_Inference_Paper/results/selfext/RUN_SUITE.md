# Bộ thực nghiệm excerpt recall: cách chạy (24/10/2026)

Chạy mọi phương pháp trên đúng bộ task của `EXCERPT_RECALL.md` (thêm mv, math_find theo `LD_CAUSE.md`).

- Script chạy: `scripts/run_excerpt_suite.sh`
- Bảng điểm: `scripts/analysis/suite_table.py`
- Kết quả: `results/selfext/suite/<model>_<size>/<task>_<method>.jsonl` (+ `.log` từng job, `run.log` của hàng đợi)

## 1. Môi trường

Thư mục gốc: `/root/team_1/hieurezdev/PromptPruningPaper/VAR_LLM_Inference_Paper` (bản sao của QuangNM, 24/10).

| Thành phần | Đường dẫn | Ghi chú |
|---|---|---|
| venv chính | `.venv/bin/python` | Python 3.10, torch 2.4.0 (CUDA), transformers 4.50.2, vllm 0.6.3.post1 |
| transformers cho Qwen3 | `/root/team_1/hieurezdev/pylib_tf451` | 4.51.3, script tự đặt `PYTHONPATH` khi `MODEL=q3` |
| ASL, FastKV, RULER | `../ext/` | ASL có `.venv` riêng (transformers 4.45.0) |
| Model | `~/.cache/huggingface` | dùng chung, không chép |
| Retrieval heads | `results/repos/heads_7b1m.json`, `results/repos/heads_q3_4b.json` | dò trên KV tổng hợp, không trùng tập đánh giá |

Mọi lệnh dưới đây chạy từ thư mục gốc:

```bash
cd /root/team_1/hieurezdev/PromptPruningPaper/VAR_LLM_Inference_Paper
```

## 2. Task

| Task | Dữ liệu | pilot (mặc định) | `SIZE=full` |
|---|---|---|---|
| `kv` | `data/infbench_kv86.jsonl` | 20 câu đầu | 86 |
| `ld` | `data/infbench_ld50.jsonl` / `data/infbench_ld200.jsonl` | LD50 | LD200 |
| `mk3` | `data/ruler_eval.jsonl`, `niah_multikey_3@131072` | 40 (như tài liệu) | 100 |
| `mv` | RULER `niah_multivalue@131072` | 20 | 100 |
| `fwe`, `cwe`, `vt`, `qa_1`, `qa_2` | RULER `@131072` | 20 | 100 |
| `code_debug`, `math_find` | `data/infbench.jsonl` | 20 | 100 |

## 3. Phương pháp

Bộ chọn mặc định của các bản mới: retrieval heads + cắt theo dòng (`--agg retr --heads <file> --snap`, gọi là SEL).
Một cửa sổ khoảng 2k token quanh khối mạnh nhất: SEL + `--pad 992` (gọi là WIN).

| Tên | Cờ `recall_eval.py` | Nguồn |
|---|---|---|
| `copy` | `--select` (mean qua head, L17, khối 64 ±96, budget 2048) | EXCERPT_RECALL.md, bản Copy |
| `random` | `--select --random` | EXCERPT_RECALL.md mục 5, đối chứng |
| `copy_retr` | `--select` + SEL | copy với bộ chọn retrieval heads |
| `move` | `--select --move` + WIN | LD_CAUSE.md mục 9 (dời một đoạn) |
| `gem_win` | `--select --gem --gem-sep` + WIN | LD_CAUSE.md mục 15 (bản mạnh nhất) |
| `gem_topk` | `--select --gem --gem-sep --agg retr --budget 2048 --pad 0` | LD_CAUSE.md mục 14–15 |
| `gem_tail` | `gem_win` + `--ins 512` | **mới**: giữ nguyên 512 token cuối (lựa chọn, hướng dẫn); nhắm code_debug, math_find |
| `draft` | `--select --draft` + SEL | giống ATTRIEVAL: token của câu trả lời nháp cũng chấm điểm |
| `focus` | `--select --focus <heads> --head-thr 0.1` + WIN | giống DySCO: cộng logit cho cửa sổ ở retrieval heads, vị trí thật |
| `reorder` | `--select --reorder` + WIN | giống Attention Sorting: dời cửa sổ về cuối context, mã hóa lại |
| `fuse` | `--fuse` + SEL (không `--select`) | decode song song bản thường và bản copy, cộng log-prob |

`--select` khiến mỗi dòng kết quả có cả `pred_dense` (dense HF, không can thiệp) lẫn `pred_recall` (phương pháp). Vì vậy
cột dense có sẵn, không cần chạy riêng, và `scripts/analysis/gates.py` dùng lại được các run này cho phần gate.

## 4. Lệnh chạy

```bash
# xem trước các lệnh, không chạy gì
DRY=1 bash scripts/run_excerpt_suite.sh

# toàn bộ pilot (11 phương pháp x 11 task = 121 job), chạy nền
nohup bash scripts/run_excerpt_suite.sh > /dev/null 2>&1 &

# chỉ một nhóm
METHODS="gem_tail gem_win" TASKS="code_debug math_find kv ld" nohup bash scripts/run_excerpt_suite.sh > /dev/null 2>&1 &

# bản đầy đủ (KV86, LD200, 100 câu mỗi task khác)
SIZE=full nohup bash scripts/run_excerpt_suite.sh > /dev/null 2>&1 &

# Qwen3-4B-Instruct-2507
MODEL=q3 nohup bash scripts/run_excerpt_suite.sh > /dev/null 2>&1 &
```

Biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SIZE` | `pilot` | `pilot` hoặc `full` |
| `MODEL` | `q25` | `q25` (Qwen2.5-7B-Instruct-1M) hoặc `q3` (Qwen3-4B-Instruct-2507) |
| `METHODS` | cả 11 | danh sách tên ở mục 3, cách nhau bằng dấu cách |
| `TASKS` | cả 11 | danh sách tên ở mục 2 |
| `CAP` | `70000` | trần VRAM (MiB) cho mọi tiến trình của `/root/team_1` |
| `LOCK` | lock của hàng đợi QuangNM | file khóa lúc khởi động job; đổi nếu muốn tách hàng đợi |
| `DRY` | rỗng | đặt `1` để chỉ in lệnh |

Chạy một job lẻ bằng tay (không qua hàng đợi):

```bash
.venv/bin/python scripts/recall_eval.py --select --gem --gem-sep --agg retr --heads results/repos/heads_7b1m.json \
  --snap --pad 992 --ins 512 --prompts data/infbench.jsonl --tasks code_debug --limit 20 --out /tmp/x.jsonl
```

## 5. Theo dõi và tổng hợp

```bash
tail -f results/selfext/suite/q25_pilot/run.log          # bat dau / xong / cho VRAM / OOM
tail -f results/selfext/suite/q25_pilot/kv_gem_tail.log  # từng câu: id, câu trả lời, đáp án có trong đoạn chọn không
.venv/bin/python scripts/analysis/suite_table.py results/selfext/suite/q25_pilot
```

Bảng in ra markdown, mỗi ô: `điểm (chênh lệch so với dense [CI 95% bootstrap ghép cặp], n)`. Dense được gộp từ mọi run
của cùng task. LD chấm theo tên đầu tiên (`selscore.py`). Điểm chính thức của InfiniteBench LD: `scripts/analysis/ld_first.py`.

## 6. Hàng đợi hoạt động thế nào

- Trước mỗi job: chờ tới khi tổng VRAM của các tiến trình `/root/team_1` ≤ `CAP − NEED` (NEED 26 GB với q25, 32 GB với
  q3), giữ lock 3 phút để hai job không khởi động cùng lúc.
- Job đã xong (có `.jsonl` và dòng `[tên] xong` trong `run.log`) được bỏ qua, nên chạy lại script là tiếp tục từ chỗ dừng.
- OOM hoặc lỗi CUDA: tự chạy lại sau 2 phút. Lỗi khác: ghi `loi khong phai OOM` vào `run.log` rồi sang job sau.
- Thứ tự: hết một phương pháp trên mọi task rồi mới sang phương pháp sau, để luôn có trọn cột.

## 7. Thời gian và thứ tự đề xuất

- Theo log cũ, một job 20 câu ở 128k mất khoảng 20 phút; LD50 lâu hơn. Toàn bộ pilot khoảng 40 giờ GPU.
- Thứ tự đề xuất:
  1. `METHODS="gem_tail" TASKS="code_debug math_find"`: kiểm tra nhanh xem giữ 512 token cuối có gỡ được phần hại.
  2. `METHODS="gem_win gem_tail move copy_retr"`, mọi task: bảng chính so sánh các cách dùng bằng chứng.
  3. `METHODS="copy random"`: tái lập EXCERPT_RECALL.md trên cùng tập câu.
  4. `METHODS="draft focus reorder fuse"`: baseline kiểu các bài báo liên quan.
  5. `SIZE=full` cho các ô sẽ đưa vào paper; `MODEL=q3` để kiểm tra trên mô hình thứ hai.

## 8. Lưu ý

- Một số ô trùng với run cũ của QuangNM (ví dụ `kv20_v` tương ứng `kv_copy`). Bộ này chạy lại để mọi ô cùng cấu hình,
  cùng tập câu, cùng thư mục.
- `vllm_patch.py` của bản này là bản của QuangNM. Bản riêng ở `/root/team_1/hieurezdev/VAR_LLM_Inference_Paper` (có
  `trace`, `protect_pos`, `set_intervention`) chưa được gộp; bộ này không dùng tới file đó.
- Các đường dẫn tuyệt đối trong `scripts/` và shebang của `.venv/bin` đã được đổi sang `/root/team_1/hieurezdev/...`.
