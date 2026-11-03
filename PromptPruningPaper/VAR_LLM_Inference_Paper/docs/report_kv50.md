# Report: kv50, LayerSieve vs SlimInfer (2026-10-03)

Ký hiệu: [ĐO] = số đo; [SUY] = suy luận.

## Thiết lập
- Model Qwen2.5-7B-Instruct-1M (28 layer). vLLM 0.6.3.post1, eager, 1 request, 1 GPU.
- Dữ liệu: InfiniteBench kv_retrieval, id 0, 10, …, 490 (50 mẫu, ~169k token), prompt chính thức + chat template (`data/build_kv.py`).
- TTFT = thời gian `generate` 1 token. Trả lời greedy 100 token, đúng nếu chứa đáp án.
- So cặp bằng McNemar trên cùng id (`scripts/summarize_kv.py`).

## Kết quả [ĐO]
| Phương pháp | Đúng /50 | TTFT | vs dense (thắng:thua, p) | vs LayerSieve (thắng:thua, p) |
|---|---|---|---|---|
| dense | 35 | 20.8 s | – | 7:1, 0.07 |
| **LayerSieve** (ℓ₀ = 17, ρ = 0.12) | **29** | 15.5 s | 1:7, 0.07 | – |
| SlimInfer giữ 0.9/0.75/0.5 | 26 | 16.2 s | 1:10, 0.01 | 5:8, 0.58 |
| SlimInfer giữ 0.8/0.6/0.4 | 24 | 13.9 s | 3:14, 0.01 | 6:11, 0.33 |
| SlimInfer giữ 0.5/0.25/0.15 | 10 | 9.4 s | 1:26 | 1:20 |
| SlimInfer gốc 12k/6k/4k | 3 | 6.9 s | 1:33 | 1:27 |
| SlimInfer tỉ lệ (ref 32k → ~64k/32k/21k) | 9 | 8.4 s | 1:27 | 2:22 |

- SlimInfer "giữ a/b/c" = giữ tỉ lệ a, b, c của prompt sau L8, L17, L25.
- Tổng token×layer so với dense: LayerSieve ≈ 0.80; SlimInfer 0.8/0.6/0.4 ≈ 0.78; 0.9/0.75/0.5 ≈ 0.86.
- Bản HF tham chiếu của LayerSieve (`inmodel_prune/prune.py`): 28/50 so với dense 34/50.

## Kiểm chứng bản SlimInfer cài lại
- Chạy code HF gốc của SlimInfer (`baselines/SlimInfer/run_kv.py`, cấu hình Qwen 12k/6k/4k, flash-attn) trên 10 mẫu đầu: đúng 1/10.
- Đúng/sai khớp 10/10 mẫu với bản vLLM cài lại.
- Hai bản chỉ khác cách sai: bản gốc thường trả lời "không có key", bản vLLM trả về một uuid sai.

## Khác biệt cách chọn token
| | LayerSieve | SlimInfer |
|---|---|---|
| Query quan sát | 128 token cuối | 4 token cuối |
| Key | từng token | trung bình key mỗi cụm 8 token |
| Điểm | softmax attention, max theo head, cộng theo hàng | logit q·k/√d, trung bình theo head và hàng |
| Đơn vị bỏ | từng token (điểm avg-pool 13) | block 64 token (điểm block = max của cụm) |
| Lịch | từ L17 mỗi layer bỏ 12% | 3 lần cắt mạnh sau L8, L17, L25 |

[SUY] Probe trước đó cho thấy ở L8 evidence chỉ ở percentile ~0.55 (gần ngẫu nhiên), còn từ L17 trở đi là ~0.86–0.99. Vì vậy cắt sớm từ L8 và lấy trung bình theo head dễ làm mất evidence.

## Nhận xét
- [ĐO] Ở compute ngang hoặc cao hơn, SlimInfer vẫn thấp hơn LayerSieve (24 và 26 so với 29), nhưng chưa có ý nghĩa thống kê (6:11, 5:8).
- [SUY] Cần thêm mẫu hoặc task khó hơn mới khẳng định được.

## Việc tiếp
- Ablation: lịch cắt của SlimInfer + điểm của LayerSieve, và ngược lại.
- Thêm mẫu và task khó (agent trace, câu hỏi không ở cuối).
- SpecPrefill chạy trên cùng vLLM.
- LongBench.
