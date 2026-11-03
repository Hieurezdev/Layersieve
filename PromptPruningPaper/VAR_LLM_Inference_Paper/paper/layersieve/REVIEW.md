# Tự review LayerSieve (vai reviewer, mục tiêu: main track top tier)

Cập nhật: 11/10, 17h–21h giờ VN.

## Reviewer sẽ hỏi gì và đã xử lý tới đâu

| # | Điểm yếu reviewer sẽ nêu | Mức | Tình trạng |
|---|---|---|---|
| W1 | Chỉ một model (Qwen2.5-7B-1M) cho bảng chính | **Nặng**, dễ bị reject | Đang chạy Llama-3.1-8B-UltraLong-1M, cũng là model chính trong paper ASL, với dense / LayerSieve / FastKV / ASL trên InfiniteBench + RULER 128k và đo tốc độ. Patch vLLM đã chuyển sang Llama. |
| W2 | Độ chính xác chỉ *ngang* ASL, không hơn | Nặng | Đã thêm 3 lập luận: (a) bootstrap cặp cho thấy LS = ASL = dense, LS > FastKV/SlimInfer (p<0.02); (b) hình Pareto (Fig. pareto); (c) **giới hạn cấu trúc của ASL**: kể cả giữ 0 token sau khi cắt, C của ASL vẫn là 0.75 (IB) / 0.79 (RULER), cao hơn 0.72 của LS. Tức là ASL không thể nhanh bằng LS bằng cách giảm budget. |
| W3 | Tốc độ ASL đo bằng replay lịch, không phải code gốc | Vừa | Đã ghi rõ trong paper. Replay *có lợi* cho ASL vì không tính chi phí monitor. |
| W4 | Siêu tham số (ℓ0, ρ, φ, r) chọn cho model mới thế nào | Nặng | **Đã giải quyết** (mục "How Many Tokens to Keep at the Last Layer", Bảng `tab:floor`): quy tắc theo độ sâu tương đối ℓ0 = 0.61L, ρ_L = 1 − 0.6^{28/L}, φ = 0.25. 32B trở lại mức dense trên RULER 32k (cwe/fwe/mk3 94/99/100 so với dense 96/98/100), hard set giữ 95–99% với mọi model từ 7B trở lên, C = 0.72–0.73 với mọi kích cỡ. Còn yếu: r ≈ 0.8L mới kiểm trên 7B; 14B cwe 76 (so với 82). |
| W5 | Novelty so với LazyLLM / PyramidInfer (đã có progressive pruning) | Vừa | Ablation mới tách được phần đóng góp: cùng scorer, cùng unit, cùng chi phí, chỉ đổi progressive thành một lần cắt ở L15 thì mất 1.5 IB / 1.1 RULER, cwe 30.9 → 20.0. Cần viết positioning sắc hơn ở Related Work. |
| W6 | Benchmark nặng synthetic (RULER) | Vừa | IB có code/dialogue/math. Nếu còn thời gian GPU, LongBench v2 là ứng viên tiếp theo. |
| W7 | Chỉ batch size 1 | Nhẹ | Ghi ở phần Limitations. |
| W8 | KV sieving làm giảm fwe (78 so với 84) | Nhẹ | Đã thừa nhận, có biến thể LS_prefill. |
| W9 | Template IEEE, 9 trang | Format | Nếu nộp ACL/EMNLP phải chuyển sang template ACL, 8 trang + appendix. |
| W10 | SpecPrefill cấu hình p50 có công bằng không | Vừa | Các run d_* (mặc định của từng paper) vẫn trong hàng đợi p2. |

## Đã làm tối nay

- Ablation: điền xong các dòng "một lần cắt" (unit 32.0/82.8, token 25.5/79.6) và "progressive token" (IB 28.2).
- Kiểm định bootstrap: `scripts/significance.py`.
- Hình Pareto: `scripts/plot_paper.py` → `figures/pareto.pdf`.
- Đoạn "The speed gap to ASL is structural" trong mục tốc độ.
- Patch vLLM hỗ trợ Llama; dữ liệu `data/{infbench,ruler_eval,speed}_llama.jsonl`.
- Sweep φ (`scripts/run_floor.sh`, `scripts/summarize_floor.py`) và quy tắc tương đối, viết thành mục mới trong paper (paper hiện dài 10 trang).

## Câu hỏi dự kiến: "sao cắt từ 0.5L là ổn mà lại chọn 0.61L?"

0.5L là điểm bắt đầu sớm nhất khi cắt chậm đến layer cuối (C ≈ 0.76). Lịch mặc định cắt nhanh, chạm sàn sau 0.11L (C = 0.72), nên phải bắt đầu muộn hơn. Thứ cố định là điểm kết thúc cắt (≈ 0.72L). Bằng chứng: 32B cắt nhanh từ 0.5L chỉ giữ 85% hard set và mk3 = 55, so với 95% và 100 khi bắt đầu từ 0.61L. Trên 7B, ℓ0 = 14 với ρ .25 và φ .3 thua cấu hình mặc định 2.1 điểm ở RULER 128k (p = 0.015). Đã viết vào paper ở đoạn "Choosing ℓ0" và đoạn "Why not start the fast descent at 0.5L?". Sweep ℓ0 ∈ {.55, .68}L cùng lịch cắt chậm (`scripts/run_night5.sh`) chạy sau Llama để vẽ đường accuracy theo C.

## Review bên ngoài 11/10 (4/10, weak reject): kế hoạch cứu

| Điểm | Việc làm | Tình trạng |
|---|---|---|
| W1 novelty, units là thành phần trực giao | Biên Pareto "một lần cắt + units" cùng C ≈ 0.72: (L13, 50%), (L15, 37.5%), (L17, 25%), (L19, 10%), chạy IB + RULER 128k. Nếu LayerSieve hơn mọi điểm thì đóng góp của progressive được tách riêng. Đã trích dẫn PyramidKV, ChunkKV, SepLLM; thừa nhận units không mới. | Queue P1 (~10 h) |
| W2 ASL không cùng C | Đổi tên mục thành "Accuracy", ghi rõ ASL tốn hơn. Lập luận cấu trúc giới hạn ở τ mặc định. Biên một lần cắt ở trên thay cho Pareto của ASL theo τ (ASL ở mọi τ là hỗn hợp các điểm một lần cắt). | Text xong; số liệu chờ P1 |
| W3 FastKV không có nén KV, thiếu tốc độ SpecPrefill/SlimInfer | Đo tốc độ FastKV có nén KV trước điểm cắt (o15:0.375:15), SlimInfer, SpecPrefill (target 50% + draft 1.5B), mỗi độ dài 5 prompt. Đã sửa câu sai ở Discussion. | Queue P0c |
| W4 SpecPrefill bị nhiễu do YaRN | IB cắt 32k (draft trong cửa sổ gốc, không YaRN): dense / LS / SpecPrefill 50% và 30%. Hard set làm lại trên H200 với bộ 120 câu, có SpecPrefill 90/50/30/10 và LS cùng C. | Queue P0a, P0b |
| W5 agentic / prefix cache / chunked prefill | Định vị lại: nhắm prefill chưa có trong cache (tài liệu mới, repo mới, lượt đầu). Thêm đoạn "Deployment scope" nói thẳng giới hạn. | Text xong |
| W6 trần 1.6×, so với sparse attention | Ghi trần 1/0.61 vào Limitations, nói tính trực giao với MInference/FlexPrefill. Đo tốc độ 5 prompt. | Text xong; chưa có SnapKV/PyramidKV cho decode |
| W7 1M không có accuracy | Abstract ghi "accuracy evaluated up to 128k". Đo decode 1M của cấu hình mặc định. | Text xong; queue night5 |
| W8 một họ model, IB 4 task, synthetic | Llama đang chạy. **Cảnh báo:** trên Llama IB, LS thua dense 3 điểm (23.3 vs 26.3, p = 0.001), thua cả ASL và FastKV. Đang chẩn đoán: bỏ KV sieving, ℓ0 = 22, một lần cắt + units. LongBench v2 chưa xếp hàng (~10 h). | Queue P0- |
| Q1 scorer token yếu hơn FastKV trên IB | Không đồng đều: IB 25.5 vs 29.5 nhưng RULER 79.6 vs 77.5. | Trả lời bằng số |
| Q3 cửa sổ quan sát / sink | Hard set với n_obs 32/512, n_sink 16/512. | Queue P0a |
| Q6, minor RTX 5880 | Làm lại Table II trên H200, bộ 120 câu. | Queue P0a |
| Minor | Algorithm ℓ > ℓ0; appendix có chữ cái (\appendices); ẩn danh tác giả (\anontrue); tỷ lệ no-cut ASL thống nhất "một phần tư"; Q5 ghi rõ khoảng cách ở 32k/64k. | Xong |

## Review lần 2 (5/10): việc chính là đối chứng cắt một lần ở L18/L19

- Đối chứng cắt một lần + units, cùng C = 0.72: s17 giữ 29.3% (reviewer gọi là "L18") và s18 giữ 22% ("L19"), chạy IB + RULER 128k. Xếp đầu queue sau chẩn đoán Llama.
- Kiểm định hiện có giữa progressive và một lần cắt ở L15: RULER 128k +1.13 (p = 0.001) nhưng không tính cwe chỉ còn +0.31 (p = 0.36); IB +1.5 (p = 0.32). Lợi thế hiện tại gần như toàn bộ đến từ cwe, nên rủi ro hòa với L18 là có thật.
- Phương án dự phòng nếu hòa: đóng góp chuyển thành "điểm kết thúc cắt (~0.72L) là thứ quyết định, progressive và một lần cắt ở đúng độ sâu là tương đương". Phần còn lại vẫn đứng: quy tắc theo kích cỡ, mức sàn theo tỷ lệ, units, KV sieving cho decode.
- ASL theo τ (0.45, 0.6): IB + RULER 128k. FastKV và ASL có nén KV trước điểm cắt (cache-k 0.375): cùng một cấu hình cho cả accuracy và tốc độ. SnapKV 25% sau prefill dense.
- Claim 1M: bỏ 2.2× khỏi abstract, sẽ dùng số đo của cấu hình mặc định. r cho 14B-1M theo quy tắc 0.8L (r = 38), đã queue.

## Chẩn đoán Llama IB (11/10, 21h UTC)

| Cấu hình | C | IB |
|---|---|---|
| dense | 1.00 | 26.2 |
| một lần cắt + units L15 giữ 37.5% | 0.67 | 26.8 (vs dense +0.5, p = 0.70) |
| ASL | > 0.72 | 25.0 |
| LS ℓ0 = 22, noref | ~0.75 | 24.5 (vs dense −1.75, p = 0.017) |
| LS ℓ0 = 19, noref | 0.71 | 23.8 (một lần cắt L15 hơn 3.0, p < 0.001) |
| LS ℓ0 = 19, r = 25 | 0.71 | 23.2 (vs dense −3.0, p = 0.001; vs ASL −1.75, p = 0.08; vs FastKV −0.25, p = 0.92) |
| FastKV | | 23.5 |

KV sieving không phải nguyên nhân. Lịch progressive mới là nguyên nhân: một lần cắt ở L15 vừa rẻ hơn vừa tốt hơn, chủ yếu ở kv_retrieval (37 vs 29). Chờ frontier Qwen để chốt câu chuyện.

## Cập nhật paper 14/10 (số mới đã điền, không còn \pend)

- Table II (hard set) làm lại trên H200, 30 câu/task: draft 1.5B vt 9/30, 0.5B 1/30; SpecPrefill 90/50/30/10%; LS unit 119/120, token 101/120.
- Bảng chính: thêm ASL τ = .45 (C .80/.81) và .6 (C .77/.77), FastKV + nén KV, SnapKV 25% (RULER chỉ 128k). ASL τ .6 thua LS 2.0 RULER 128k (p < .001, mk3 53 vs 85). SnapKV thua LS 5.0 IB / 5.3 RULER.
- W4 (YaRN): IB cắt 32k, SpecPrefill 50% = 12.0 so với dense 21.0, LS 20.8 (p < .001). Lỗi không do YaRN.
- Tốc độ: thêm FastKV + nén KV, SlimInfer, SpecPrefill có tính draft (1.45× ở 128k). Speed chỉ có 2 prompt/độ dài (data/speed.jsonl chỉ có 2), không phải 5.
- Decode mặc định: TPOT 1.20 / 1.27 / 1.64 / 1.86× (128k → 1M), hard 119/120. 14B-1M r = 38: 1.66× ở 256k.
- tab:rule TTFT 128k: 1.47–1.49× ở mọi kích cỡ (7B không đo).
- Ablation: token RULER 83.3; không sàn 32.8 / 81.0 (cwe 7.7); thêm dòng cắt một lần L17 / L18; thêm n_obs / n_sink.
- **Đổi claim:** bỏ "progressive hơn mọi cách cắt một lần". Thay bằng: ở cùng C, độ sâu kết thúc cắt mới quyết định; cắt một lần + units ở L17–18 ngang LS. Đã sửa abstract, contributions, ablation, discussion.
- Còn chạy: cắt một lần L13 50%, sweep ℓ0, cấu hình mặc định từng paper (d_*), ASL τ .3 + nén KV RULER.
