# LayerSieve nhiều lượt: E1 (thời gian) và E2 (độ chính xác), 30/10/2026

Tài liệu đứng trước câu hỏi và attention nhân quả, nên 17 layer dense đầu của tài liệu không phụ thuộc câu hỏi.
Sau lượt đầu giữ KV dense của tài liệu + hidden state vào layer 17. Câu hỏi sau:

- **A (resieve)**: câu hỏi qua layer 0–16 trên KV đã giữ, rồi chạy lại phần sàng (layer 17–27) từ hidden state đã lưu.
  Bằng LayerSieve chạy lại từ đầu.
- **B (dense)**: bù layer 17–27 dense cho cả tài liệu khi rảnh → prefix cache dense; câu hỏi sau chỉ tốn ε.
- **naive**: dùng lại nguyên cache đã cắt (và đã sàng KV) của câu hỏi đầu.

Code: `inmodel_prune/multiturn.py` (HF), `scripts/multiturn_hf.py`, `scripts/bench_multiturn.py` (vLLM),
`scripts/analysis/multiturn_table.py`. Dữ liệu: `data/build_multiq.py` → `data/multiq.jsonl`.

## E2: độ chính xác (HF, Qwen2.5-7B-Instruct-1M, 128k, 50 tài liệu mỗi task)

Mỗi tài liệu 3 câu: `orig` (câu gốc), `same` (cùng key, diễn đạt khác), `switch` (key khác, cách ≥ 1/4 tài liệu).

| chế độ | kv orig | kv same | kv switch | mk3 orig | mk3 same | mk3 switch |
|---|---|---|---|---|---|---|
| dense (= B) | 58 | 64 | 68 | 84 | 88 | 84 |
| A (resieve) | 62 | 66 | 70 | 84 | 90 | 86 |
| naive | – | 50 | 18 | – | 86 | 26 |

- naive sụp khi đổi evidence: câu dense đúng mà naive còn giữ được: kv switch 7/34, mk3 switch 12/42 (A: 33/34, 41/42).
  McNemar A vs naive: kv switch 28:2 (p = 9e-7), mk3 switch 30:0 (p = 2e-9); kv same 9:1 (p = 0.02); mk3 same 2:0 (p = 0.5).
- A ngang dense ở mọi ô (McNemar p = 1.0).
- A = LayerSieve từ đầu (30 câu, 5 tài liệu đầu mỗi task): cùng câu trả lời 30/30, cùng token đầu 30/30, Jaccard tập giữ
  trung vị 0.976; logit lệch trung vị 0.56 = mức nhiễu nền (dense có/không prefix cache: 0.57). Lệch do bf16.

## C: chỉ phục hồi phần câu hỏi mới cần (`--modes resieve,partial`, `results/multiturn/hf_partial.jsonl`)

Câu hỏi mới qua layer 0–16 như A; chọn unit một lần ở layer 17 (giữ 25%); layer 17–27 chỉ tính cho unit mới được chọn
(lượt đầu đã bỏ) + token câu hỏi, đọc KV sâu **cũ** của các unit lượt đầu giữ. dense, A, naive lấy từ E2 (A chạy lại
trùng E2 300/300).

| | kv same | kv switch | mk3 same | mk3 switch |
|---|---|---|---|---|
| độ chính xác dense / A / **C** / naive | 64 / 66 / **62** / 50 | 68 / 70 / **70** / 18 | 88 / 90 / **88** / 86 | 84 / 86 / **82** / 26 |
| câu dense đúng còn giữ: A / **C** / naive | 30 / **29** / 21 (32) | 33 / **32** / 7 (34) | 44 / **44** / 42 (44) | 41 / **41** / 12 (42) |
| McNemar A vs C | 2:0, p = 0.5 | 1:1, p = 1 | 1:0, p = 1 | 2:0, p = 0.5 |
| unit mới (% tài liệu) | 9.0 | 13.7 | 8.8 | 12.3 |
| token×layer sâu của C / A | 0.31 | 0.47 | 0.30 | 0.42 |

- C ngang A về độ chính xác; trạng thái sâu cũ không làm hỏng câu trả lời trên hai task truy xuất này.
- C chỉ tính 30–47% token×layer sâu của A. Thời gian HF của C lại chậm hơn A (3.1–5.0 s so với 2.6–3.4 s) vì attention
  có mask theo vị trí (KV cũ và mới xen kẽ) không dùng được kernel flash; cần kernel block-sparse mới lấy được phần lợi.
- Ngay cả `same` vẫn có ~9% unit mới: chọn một lần ở layer 17 khác với sàng dần của lượt đầu.

Sửa lỗi 31/10: unit lượt đầu bỏ giữa chừng (có KV cũ ở vài layer sâu) mà được chọn lại thì KV cũ phải bỏ, không thì
cùng vị trí có hai key. Chạy lại (`hf_partial_fix.jsonl`): 2/200 câu trả lời của C đổi chữ, số câu đúng không đổi ở ô nào.

## Mở rộng 1: thêm task RULER 128k (`data/multiq2.jsonl`, `results/multiturn/hf2.jsonl`)

mk1 (essay + 4 needle), mk2 (~6000 needle số), mq (4 key; `subset` = hỏi 1 trong 4 key), qa1 (SQuAD; `same` = câu
hỏi SQuAD khác trên cùng đoạn, `switch` = câu hỏi trên đoạn khác của ngữ cảnh). Điểm RULER (`all` / `part`).

| | dense | A | C | naive |
|---|---|---|---|---|
| mk1 same / switch | 100 / 100 | 100 / 98 | 100 / 98 | 98 / 100 |
| mk2 same / switch | 98 / 96 | 94 / 94 | 98 / 94 | 98 / **22** |
| mq same / subset | 80 / 100 | 82 / 100 | 80 / 100 | 91 / 100 |
| qa1 same / switch | 66 / 74 | 64 / 72 | 66 / 76 | 70 / **24** |

- naive sụp khi đổi evidence ở mk2 (dense đúng 48 câu: A giữ 47, naive 11; 36:0, p = 3e-11) và qa1 (37: A 35, naive 11;
  25:1, p = 8e-7). Không sụp ở mk1 (4 needle nổi bật, câu đầu đã giữ cả 4) và khi evidence không đổi (same, subset).
- A = LayerSieve từ đầu: cùng câu trả lời 21/24, 3 câu còn lại chỉ khác chữ, cùng điểm.

## Bù dần (`fill<p>`, `hf_fill_check.jsonl`, `hf_fill_multiq*.jsonl`)

Phần bù nền đã tới vị trí p của tài liệu: token [0, pT) giữ KV dense ở mọi layer sâu, câu hỏi mới chỉ sàng [pT, T).
Kiểm tra: fill0 = A và fill1 = dense, cùng câu trả lời 8/8 mỗi đầu.

| câu dense đúng còn giữ | A (p = 0) | p = 1/4 | p = 1/2 | p = 3/4 |
|---|---|---|---|---|
| kv same / switch (32 / 34) | 30 / 33 | 31 / 33 | 31 / 32 | 32 / 33 |
| mk3 same / switch (44 / 42) | 44 / 41 | 44 / 41 | 44 / 41 | 44 / 41 |
| mk2 same / switch (49 / 48) | 47 / 47 | 49 / 47 | 48 / 47 | 49 / 47 |
| qa1 same / switch (33 / 37) | 32 / 35 | 31 / 35 | 32 / 35 | 33 / 37 |
| token×layer sâu lúc hỏi / A | 1 | 0.75 | 0.50 | 0.25 |

Độ chính xác ngang A và dense ở mọi p; phần phải tính khi câu hỏi đến giảm tuyến tính (1 − p).

## SCBench 128k có lịch sử hội thoại, Qwen (`data/scbench128k.jsonl`, `results/multiturn/scbench_qwen.jsonl`)

Prompt, cắt giữa và bộ chấm điểm lấy từ code SCBench (`../ext/MInference/scbench`, commit 29ef197); lịch sử = đáp án đúng
lượt trước (mặc định của SCBench); 6 task × 20 phiên × ≤ 5 lượt (RepoQA bỏ: cần sinh 1024 token và tree-sitter).
Token sau tài liệu (lịch sử, câu hỏi) không bao giờ bị cắt; naive hội thoại tiếp trên cache đã cắt; C cộng dồn tập đã tính.

| task (lượt ≥ 2, n) | dense (= B) | A | C | naive |
|---|---|---|---|---|
| kv (80) | 88.8 | 88.8 | 87.5 | 66.2 |
| prefix_suffix (80) | 81.2 | 77.5 | 80.0 | 17.5 |
| vt (80) | 45.8 | 42.3 | 41.3 | 34.3 |
| qa_eng (80) | 30.0 | 30.0 | 28.8 | 18.8 |
| choice_eng (60) | 60.0 | 60.0 | 60.0 | 48.3 |
| mf (80) | 13.8 | 15.0 | 11.2 | 12.5 |

Chênh lệch so với dense, trung bình đều theo task, lượt ≥ 2, bootstrap theo phiên (2000 lần, seed 0, CI 95%):
A −1.0 [−2.4, +0.4]; C −1.8 [−3.2, −0.5]; naive −20.3 [−24.0, −16.7].
A = LayerSieve từ đầu trên cả hội thoại: cùng câu trả lời 49/58, cùng điểm 58/58.

## E1: thời gian (vLLM 0.6.3, H200 **dùng chung** với job khác, 2 prompt mỗi độ dài, 256 token ra)

| | 128k | 256k |
|---|---|---|
| TTFT dense / LayerSieve | 17.1 / 12.3 s | 73.2 / 52.3 s |
| layer 17–27: dense / LayerSieve | 5.8 / 1.1 s | 28.7 / 3.9 s |
| câu hỏi sau, dense + prefix cache (ε) | 0.13 s | 0.17 s |
| câu hỏi sau, A | 1.21 s (7% TTFT dense) | 4.04 s (5.5%) |
| bù cache B (chạy nền) | 5.8 s (34%) | 28.7 s (39%) |

Tổng thời gian phiên (s, TTFT + 256 × TPOT mỗi lượt; lượt sau của dense/B dùng TPOT dense):

| số lượt | dense + prefix cache | LayerSieve vứt cache | A | B |
|---|---|---|---|---|
| 128k, 1 | 19.5 | 14.5 | 14.5 | 14.5 |
| 128k, 3 | 24.6 | 43.6 | 21.4 | 19.7 |
| 128k, 10 | 42.6 | 145.4 | 45.4 | 37.7 |
| 256k, 1 | 77.8 | 55.6 | 55.6 | 55.6 |
| 256k, 3 | 87.4 | 166.7 | 70.1 | 65.2 |
| 256k, 10 | 121.1 | 555.7 | 121.1 | 98.8 |

GPU bị job khác dùng 80–100% nên số tuyệt đối chậm hơn Bảng IV của paper khoảng 1.3–1.5×; cần đo lại khi GPU trống
trước khi đưa vào paper. Tỷ lệ (A ≈ 5–7%, B ≈ 34–39% TTFT dense) khớp ước lượng từ Bảng IV.

```bash
.venv/bin/python scripts/multiturn_hf.py --verify 5 --out results/multiturn/hf.jsonl
.venv/bin/python scripts/bench_multiturn.py --mode layers --lengths 131072 --gpu-util 0.4 --out results/multiturn/layers.jsonl
.venv/bin/python scripts/bench_multiturn.py --mode prefix --lengths 131072 --gpu-util 0.4 --out results/multiturn/prefix.jsonl
.venv/bin/python scripts/analysis/multiturn_table.py --acc results/multiturn/hf.jsonl \
    --layers results/multiturn/layers.jsonl --prefix results/multiturn/prefix.jsonl
```
