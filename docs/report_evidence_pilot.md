# Pilot: LayerSieve có mất độ chính xác vì cắt quá sớm bằng chứng không?

Ngày 2026-10-16. Model Qwen2.5-7B-Instruct-1M (L = 28), vLLM 0.6.3, 1 H200 dùng chung (GPU đang chạy job khác ở 100%, nên số đo thời gian chỉ để tham khảo).

## Kết luận ngắn

1. **Lịch mặc định của paper (ℓ0 = 17, ρ = 0.4, φ = 0.25) không cắt mất bằng chứng trên pilot này.**
   - Cả 144/144 prompt giữ đủ mọi đơn vị bằng chứng đến layer cuối.
   - LayerSieve làm đúng 87/87 câu mà dense làm đúng. Cận trên 95% của tỷ lệ lỗi là khoảng 3.4%.
   - Ở mọi layer có cắt, điểm của bằng chứng gấp ≥ 2.9 lần điểm cao nhất của đơn vị bị loại (trung vị 15–500 lần), tức không có ca nào sát ngưỡng.
2. **Cắt sớm hơn ở ℓ0 = 14 (0.5L) vẫn an toàn trên pilot này**: không mất bằng chứng, 87/87 câu đúng, với C = 0.644 thay vì 0.723.
3. **Ở ℓ0 = 11 (0.39L), lỗi do mất bằng chứng xuất hiện và được xác nhận bằng can thiệp nhân quả.**
   - 25/144 prompt mất bằng chứng, tất cả là kv và mk.
   - Có 24 ca dense đúng, LayerSieve sai và mất bằng chứng. Buộc giữ bằng chứng ở cùng ngân sách sửa đúng cả 24/24. Placebo (giữ cùng số token ngẫu nhiên) sửa 0/24.
   - Ở lịch này không có ca nào mất bằng chứng mà vẫn trả lời đúng.
   - 3 ca sai còn lại (vt 2, qa2 1) vẫn giữ đủ bằng chứng, nên không phải do cắt mất bằng chứng.
4. **Giả thuyết "bằng chứng bị cắt có điểm chưa ổn định" chỉ đúng một phần.**
   - Đúng: thứ hạng của bằng chứng bị mất dao động mạnh trước lúc cắt (AUC 0.92).
   - Sai: mức bất đồng giữa các head không phân biệt được (AUC 0.57). Lúc bị cắt, không head nào xếp bằng chứng vào top 25%.
   - Tín hiệu mạnh nhất là **"bằng chứng từng ở rất cao rồi rơi"**. Cả 25/25 đơn vị bị mất đều đạt đỉnh ở **L8**, với percentile ≥ 0.991 (trung vị 0.998), rồi rơi xuống percentile ~0.13 đúng ở L12–L13. Đặc trưng "đỉnh percentile trước đó" tách bằng chứng bị mất khỏi các đơn vị bị loại khác với AUC 1.00.
   - Lượt dense đối chứng cho thấy chỗ rơi này là **tạm thời**: ngay tại layer cắt, model không cắt cũng xếp bằng chứng thấp (trung vị 0.32), nhưng từ layer sau đó trở đi thì xếp lại ở top (trung vị 0.998).
5. **Go/no-go.**
   - **No-go** cho việc dùng "hoãn cắt khi điểm chưa ổn định" để sửa lịch mặc định, vì ở đó không có lỗi để sửa.
   - **No-go** cho phiên bản dựa trên phương sai thứ hạng hoặc bất đồng giữa head (kém chọn lọc: hoãn 10% token chỉ bắt được 44% bằng chứng, head-spread gần ngẫu nhiên).
   - **Go có điều kiện** cho phiên bản hẹp hơn, *"đừng loại đơn vị từng ở top cao vừa mới rơi"*, như một cách để **bắt đầu cắt sớm hơn ℓ0 = 14**. Điều kiện là phải kiểm chứng bằng chạy thật trước khi đề xuất thuật toán, vì mọi con số ở mục 4 mới là ước lượng offline.

## 1. Đối chiếu tài liệu: điều đã công bố và giả thuyết mới

| Paper (nguồn) | Điều đã công bố liên quan | Không có trong paper |
|---|---|---|
| LazyLLM ([arXiv 2407.14057](https://arxiv.org/abs/2407.14057)) | Cắt dần theo layer bằng attention của layer trước (token cuối, trung bình qua head), chọn theo top-k percentile, giữ nhiều token ở layer nông. Token đã cắt có thể được "hồi sinh" ở các bước decode sau qua Aux Cache. Trong prefill, token bị cắt bị loại khỏi mọi layer sau. Có nói "later layers are less sensitive to token pruning". | Không phân tích bằng chứng bị cắt sớm hay độ ổn định điểm. |
| ASL ([arXiv 2601.07667](https://arxiv.org/abs/2601.07667), Findings ACL 2026) | Chọn **một** layer cắt cho mỗi input từ phương sai thứ hạng của tập top-k qua cửa sổ 8 layer, bắt đầu từ ⌊L/3⌋, ngưỡng tương đối τ = 0.3. Cắt một lần, không sửa lại. | Phương sai chỉ đo trên tập top-k đang theo dõi, **không đo trên token bị cắt**. Không có quy tắc hoãn cắt theo từng token. |
| FastKV ([arXiv 2502.01068](https://arxiv.org/abs/2502.01068)) | Một layer TSP (L15 với Llama-3.1-8B), cửa sổ 8 token, trung bình qua head, pool 7. Lấy độ trùng top-512 giữa các layer làm bằng chứng "importance ổn định ở layer sau", nhưng chỉ trên 1 model và 1 input 128k. | Không phân tích từng token. |
| SlimInfer ([arXiv 2508.06447](https://arxiv.org/abs/2508.06447), AAAI 2026) | Cắt theo block ở vài layer cố định. Giả thuyết "information diffusion": thông tin của token quan trọng lan ra cả chuỗi, nên cắt *muộn* token then chốt vẫn đúng, còn cắt sớm (L5) thì sai. Minh họa bằng **một** ví dụ probe. KV của block bị cắt được offload sang CPU để có thể khôi phục khi decode. | Không có kiểm tra nhân quả trên tập lớn. |

**Giả thuyết mới, chưa thấy trong bốn paper trên:** quyết định hoãn cắt **theo từng đơn vị** dựa trên lịch sử thứ hạng của chính đơn vị đó (dao động, hoặc từng ở top rồi rơi). Gần nhất là ASL, nhưng ASL dùng tín hiệu ổn định ở *mức layer* để chọn *một* điểm cắt.

Pilot này kiểm tra trực tiếp điểm khác biệt đó. Ở mức layer, thứ hạng xáo trộn mạnh ở mọi layer (Spearman giữa hai layer liền nhau 0.2–0.7), và prompt bị mất bằng chứng không xáo trộn khác biệt rõ so với prompt giữ được. Tín hiệu phân biệt chỉ thấy ở mức đơn vị.

Chi tiết LazyLLM, FastKV và SlimInfer lấy từ bản HTML trên arXiv. Tôi chưa chạy mã gốc của bốn paper trong pilot này.

## 2. Thiết lập

- **Dữ liệu** `data/build_pilot.py` → `data/pilot.jsonl`: 144 prompt, 32k và 64k token (27k–62k thực tế), bằng chứng gán nhãn ở mức token.
  - **kv**: truy xuất UUID kiểu InfiniteBench, 24 prompt. Bằng chứng là cặp được hỏi.
  - **mk**: multi-key kiểu RULER niah_multikey_2, 24 prompt. Haystack toàn needle, khóa là cặp từ. Bằng chứng là needle được hỏi.
  - **vt**: variable tracking, 4 chuỗi × 5 câu, 72 prompt. Bằng chứng là 5 câu gán của chuỗi được hỏi.
  - **qa2**: QA hai bước tổng hợp trong văn bản gov_report, 24 prompt. Bằng chứng là "X sinh ở thị trấn T" và "T thuộc tỉnh P". Bước 2 không chung từ nào với câu hỏi.
  - Dense chỉ đúng 19/72 câu vt vì hay chọn nhầm chuỗi gây nhiễu, nên tôi tăng số prompt vt. Mọi tỷ lệ lỗi chỉ tính trên **87 câu dense làm đúng**: kv 23, mk 24, qa2 21, vt 19.
- **Lịch cắt**: LayerSieve_prefill (theo đơn vị, có floor, không sàng KV), ρ = 0.4, φ = 0.25. ℓ0 = 17 (mặc định, 0.61L), 14 (0.5L), 11 (0.39L, để ép lỗi xuất hiện).
- **Ghi vết theo layer**:
  - Từ L6 trở đi: các token còn sống, điểm Eq. (2), attention của từng head trong cửa sổ quan sát, và đơn vị bị loại ở mỗi layer.
  - Layer < ℓ0 chạy dense nên chấm điểm ở đó là hợp lệ.
  - Sau khi đơn vị bị loại, **không suy diễn** điểm của nó trong lượt có cắt. Điểm ở các layer sau chỉ lấy từ một lượt dense riêng (mục D).
- **Kiểm tra nhân quả**, áp dụng cho mọi ca dense đúng, LayerSieve sai và mất bằng chứng:
  - **force**: chạy lại, không cho loại các đơn vị bằng chứng.
  - **placebo**: giữ các đơn vị không phải bằng chứng (đã bị loại trong lượt gốc), chọn ngẫu nhiên, có cùng số token.
  - Cả hai lượt dùng **đúng số token mỗi layer** như lượt gốc (`set_intervention(budget=...)`), nên các đơn vị điểm thấp khác bị loại thay vào. Số token còn ở layer cuối của cả hai lượt luôn ≤ lượt gốc.

## 3. Kết quả

### A. Giữ bằng chứng, lỗi và kiểm tra nhân quả

| Lịch | C | Giữ đủ bằng chứng (144 prompt) | Đúng / dense-đúng | Sai | Sai và mất bằng chứng | force sửa đúng | placebo sửa đúng | Sai dù giữ đủ bằng chứng | Mất bằng chứng nhưng vẫn đúng |
|---|---|---|---|---|---|---|---|---|---|
| ℓ0 = 17 (mặc định) | 0.723 | 144 (100%) | 87/87 | 0 | 0 | – | – | 0 | 0 |
| ℓ0 = 14 | 0.644 | 144 (100%) | 87/87 | 0 | 0 | – | – | 0 | 0 |
| ℓ0 = 11 | 0.564 | 119 (83%) | 60/87 | 27 | 24 | **24/24** | **0/24** | 3 | 0 |

Theo task ở ℓ0 = 11:

| Task | Sai | Mất bằng chứng | force sửa đúng |
|---|---|---|---|
| kv | 12/23 | 12 | 12 |
| mk | 12/24 | 12 | 12 |
| vt | 2/19 | 0 | – |
| qa2 | 1/21 | 0 | – |

vt và qa2 chưa bao giờ mất bằng chứng. Bằng chứng của chúng là các câu ngắn khác biệt hẳn với phần nền nhiễu.

### B. Nơi bằng chứng bị mất (ℓ0 = 11)

- **Layer mất đầu tiên**: L13 với 24 prompt, L12 với 1 prompt. Đây là lần cắt thứ 2–3, ngay trước mốc L14 mà probe trong README ghi là "ổn định từ".
- **Ngay trước khi bị cắt**:
  - Percentile trung vị là 0.131 (dao động 0.002–0.294).
  - Thứ hạng trung vị là 292.
  - Tỷ số điểm so với đơn vị điểm cao nhất bị loại có trung vị 0.67, tức bằng chứng bị loại không phải vì nằm sát ngưỡng.
- **Ở ℓ0 = 17**, khoảng cách an toàn nhỏ nhất là 2.86 lần (mk_32000_5 tại L18, thứ hạng 120/759).

### C. Kiểm tra giả thuyết "điểm chưa ổn định"

Mọi đặc trưng chỉ tính từ các layer mà đơn vị còn sống trong lượt có cắt. So sánh 25 đơn vị bằng chứng bị mất với 614,663 đơn vị khác bị loại ở cùng layer và cùng prompt:

| Đặc trưng | Bằng chứng bị mất (trung vị) | Bằng chứng được giữ, cùng layer | Đơn vị khác bị loại | AUC | AUC so với đơn vị sát ngưỡng |
|---|---|---|---|---|---|
| Độ lệch chuẩn percentile, 4 layer gần nhất | 0.321 | 0.004 | 0.188 | 0.92 | 0.94 |
| Đỉnh percentile trước − percentile lúc cắt | 0.860 | 0.007 | 0.517 | 0.96 | 0.99 |
| **Đỉnh percentile trước lúc cắt** | **0.998** | 0.999 | 0.722 | **1.00** | **1.00** |
| Head tốt nhất − head trung vị (bất đồng giữa head) | 0.499 | 0.019 | 0.433 | 0.57 | 0.55 |
| Tỷ lệ head xếp vào top 25% | 0.000 | 0.964 | 0.000 | 0.39 | 0.31 |
| Khoảng cách đến ngưỡng cắt (điểm / điểm cao nhất bị loại) | 0.667 | – | 0.622 | 0.55 | 0.37 |

**Mô phỏng hoãn cắt offline, chỉ ở lần cắt đầu tiên.** Câu hỏi: cần hoãn bao nhiêu phần trăm số token bị loại ở layer đó (xếp theo đặc trưng) để hoãn được cả đơn vị bằng chứng? Đây chỉ là đo độ chọn lọc, không phải độ chính xác.

| Xếp theo | Phần token cần hoãn (trung vị) | Bắt được khi hoãn 5% | Khi hoãn 10% | Khi hoãn 20% |
|---|---|---|---|---|
| Đỉnh percentile trước | 0.011 | 1.00 | 1.00 | 1.00 |
| Mức rơi | 0.116 | 0.28 | 0.48 | 0.64 |
| Độ lệch chuẩn 4 layer | 0.191 | 0.28 | 0.44 | 0.56 |
| Bất đồng giữa head | 0.391 | 0.04 | 0.20 | 0.32 |
| Khoảng cách đến ngưỡng | 0.573 | 0.04 | 0.04 | 0.08 |

### D. Đối chứng dense

Lượt dense riêng, chạy trên 25 prompt bị mất bằng chứng.

- **Tại layer cắt**: dense cũng xếp bằng chứng thấp (percentile trung vị 0.321). Hố thứ hạng ở L12–L13 là của chính model, không phải do các lượt cắt trước gây ra.
- **Các layer sau đó**: percentile trung bình có trung vị 0.998. Tỷ lệ layer sau có percentile ≥ 0.9 có trung vị 1.00.

Nếu bằng chứng sống qua được L12–L13, các layer sâu hơn của model dense sẽ giữ nó. Đây là điều kiện cần cho ý tưởng hoãn cắt, chưa phải điều kiện đủ: trong lượt có cắt, phân bố attention đã khác vì nhiều token khác đã bị loại.

### Ví dụ điển hình (percentile của đơn vị bằng chứng, lượt ℓ0 = 11)

| Prompt | L6 | L7 | L8 | L9 | L10 | L11 (cắt) | L12 (cắt) | L13 (cắt) | Trả lời |
|---|---|---|---|---|---|---|---|---|---|
| kv_32000_1 | 0.44 | 0.87 | **0.995** (hạng 3/429) | 0.93 | 0.41 | 0.76 | 0.56 | **0.17 → bị loại** | sai → force: đúng, placebo: sai |
| kv_64000_3 | 0.44 | 0.95 | **0.998** (hạng 3/861) | 0.90 | 0.43 | 0.82 | 0.76 | **0.15 → bị loại** | sai → force: đúng, placebo: sai |
| mk_64000_0 | 0.05 | 0.11 | **1.000** (hạng 2/2546) | 0.45 | 0.26 | 0.88 | **0.18 → bị loại** | – | sai → force: đúng, placebo: sai |

Cả 25/25 đơn vị bị mất đều đạt đỉnh ở L8 (percentile ≥ 0.991), dao động ở L9–L10, rồi rơi ở L12–L13. Vì đỉnh luôn nằm ở cùng một layer, tín hiệu "đỉnh" ở đây thực chất là "L8 xếp đơn vị này ở top". Điều này dễ là đặc thù của model hoặc task (xem mục 4 và 6).

Ba ca sai mà vẫn giữ đủ bằng chứng (qa2_32000_4, vt_64000_1, vt_64000_26) đều chọn nhầm đáp án của chuỗi gây nhiễu. Nguyên nhân chưa được kiểm tra; có thể là mất ngữ cảnh của các token khác.

### E. Chi phí đo đạc

Toàn bộ pipeline chạy khoảng 1 giờ 25 phút trên GPU dùng chung.

| Lượt chạy | Số prompt | Thời gian generate / prompt | Phân tích CPU / prompt | Can thiệp (tổng) |
|---|---|---|---|---|
| dense | 144 | 5.5 s | – | – |
| ℓ0 = 17, có ghi vết | 144 | 4.3 s | 4.5 s | – |
| ℓ0 = 14, có ghi vết | 144 | 3.9 s | 4.1 s | – |
| ℓ0 = 11, có ghi vết | 144 | 3.5 s | 3.6 s | 174 s (48 lượt) |
| dense-trace | 25 | 5.7 s | 4.0 s | – |

- Dữ liệu vết theo đơn vị (`results/pilot/npz/`) chiếm 263 MB.
- **Chưa đo tách riêng** chi phí của bản thân việc ghi vết (attention theo head, chấm điểm ở các layer dense L6..ℓ0−1, chép về CPU), vì GPU dùng chung làm thời gian nhiễu. Thời gian generate ở trên đã gồm phần này.

## 4. Giới hạn và phần chưa chạy

- **Phạm vi**: một model (7B-1M), độ dài ≤ 64k, dữ liệu tổng hợp. Chưa chạy 128k, RULER mk3 thật hay InfiniteBench. Paper báo lịch fast descent từ 0.5L mất 2.1 điểm ở các task RULER 128k khó nhất, nhưng pilot ≤ 64k này không tái hiện được điều đó. Kết luận "ℓ0 = 14 an toàn" chỉ đúng trong phạm vi pilot.
- **Cỡ mẫu**: 25 đơn vị bằng chứng bị mất, chỉ thuộc hai loại task (kv, mk) và gần như cùng một layer (L13). AUC 1.00 của đặc trưng "đỉnh" có thể quá lạc quan trên dữ liệu tự nhiên, nơi có nhiều đơn vị không phải bằng chứng nhưng cũng từng được chú ý mạnh.
- **Tính offline**: mô phỏng hoãn cắt chỉ xét lần cắt đầu tiên. Chưa có lượt chạy thật nào kiểm tra bằng chứng được hoãn có sống đến cuối và trả lời đúng hay không.
- **Đặc trưng "đỉnh"** dựa vào điểm ở các layer dense L6–L10. Bản LayerSieve gốc không chấm điểm ở các layer này, nên dùng nó sẽ tốn thêm chi phí chấm điểm (chưa đo).
- **Môi trường**: dùng interpreter vLLM 0.6.3 có sẵn ở `/root/PromptPruningPaper/VAR_LLM_Inference_Paper/.venv`, không sửa gì ở thư mục đó. Mọi mã, dữ liệu và kết quả nằm trong repo này. Chưa tạo git commit.

## 5. Mã, lệnh và file

Thay đổi trong repo (chưa commit):

- `inmodel_prune/vllm_patch.py`: thêm
  - `min_keep` (floor φ; bản trong repo này chưa có),
  - `trace_from` và `last_trace()`: vết theo layer gồm điểm, attention theo head và đơn vị bị loại,
  - `set_intervention(protect_pos, budget)`: buộc giữ đơn vị với ngân sách khớp.
- `data/build_pilot.py`: dựng dữ liệu pilot có nhãn bằng chứng.
- `scripts/pilot_evidence.py`: driver chạy dense, dense-trace, LayerSieve và can thiệp force/placebo.
- `scripts/run_pilot.sh`: toàn bộ pipeline.
- `scripts/pilot_report.py`: tổng hợp bảng.

Kết quả:

- `results/pilot/{dense,ls_l17,ls_l14,ls_l11,dense_trace}.jsonl`: từng prompt, gồm vết bằng chứng theo layer và kết quả can thiệp.
- `results/pilot/npz/`: mảng theo đơn vị.
- `results/pilot/report_tables.md`: bảng đầy đủ.
- `results/pilot/summary.json`.
- `results/pilot/logs/`.

Tái lập:

```bash
cd /root/hieurezdev/VAR_LLM_Inference_Paper
PY=/root/PromptPruningPaper/VAR_LLM_Inference_Paper/.venv/bin/python bash scripts/run_pilot.sh
$PY scripts/pilot_report.py > results/pilot/report_tables.md
```

`run_pilot.sh` tự dựng `data/pilot.jsonl` (seed 0, `--lengths 32000,64000 --n 12 --n-vt 36`), dùng `--gpu-util 0.32`, và bỏ qua các prompt đã có kết quả.

## 6. Thí nghiệm tiếp theo đề xuất

1. **Kiểm chứng thật quy tắc "giữ đơn vị từng ở top vừa mới rơi"**, chưa cần coi là thuật toán mới.
   - Ở mỗi layer có cắt, không loại đơn vị có đỉnh percentile ≥ p* (thử p* = 0.99 và 0.995) trên các layer đã chấm, nhưng chỉ giữ nó tối đa k = 2 layer.
   - Giữ C bằng nhau bằng cách loại thêm đơn vị khác.
   - Chạy ở ℓ0 = 11 và 12 trên pilot này. Thành công nghĩa là: đạt ≥ 86/87 câu đúng, C ≤ 0.60, và chi phí chấm điểm thêm được đo riêng.
2. **So với các phương án rẻ hơn** cùng chi phí C:
   - (a) không cắt ở L12–L13, tức một lịch bỏ qua layer có hố thứ hạng;
   - (b) tích lũy điểm qua các layer (`accum=True`, đã có trong repo);
   - (b') dùng phiếu của riêng L8: giữ top ~1% đơn vị theo điểm L8. Đây là quy tắc cố định theo layer, rẻ hơn quy tắc theo lịch sử từng đơn vị, và cũng là đối chứng xem tín hiệu "đỉnh" có khái quát được ra ngoài L8 hay không;
   - (c) đơn giản là ℓ0 = 14 với ρ hoặc φ chỉnh lại để có cùng C.

   Nếu (a) hoặc (c) đạt cùng kết quả thì quy tắc theo từng đơn vị không đáng thêm độ phức tạp.
3. **Mở rộng phạm vi trước khi kết luận cho paper**:
   - 128k cùng RULER mk2/mk3/vt và InfiniteBench kv thật (dữ liệu đã dựng sẵn ở thư mục kia, cần chép sang repo này);
   - n ≈ 50 mỗi ô;
   - thêm 14B-1M hoặc 32B, nơi paper báo ℓ0 = 0.5L mất độ chính xác;
   - kiểm tra xem đặc trưng "đỉnh" có còn tách tốt trên văn bản tự nhiên không.
4. **Kiểm tra ba ca sai mà vẫn giữ đủ bằng chứng** ở ℓ0 = 11: buộc giữ thêm các câu của chuỗi gây nhiễu hoặc ngữ cảnh xung quanh, để biết nguyên nhân là mất ngữ cảnh hay thay đổi hidden state.
