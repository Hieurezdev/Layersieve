# Excerpt recall: tóm tắt phương pháp và kết quả (19/10/2026)

Mô hình: Qwen2.5-7B-Instruct-1M. Tất cả thí nghiệm dưới đây là **dense + excerpt recall so với dense**, chưa dùng LayerSieve.
Code: `scripts/recall_eval.py`, `scripts/qlen_probe.py`. Kết quả: `results/selfext/`.

## 1. Vấn đề

Prompt dài 128k token, bằng chứng nằm cách câu hỏi khoảng 100k token. Đo trực tiếp (`scripts/kv_attn_probe.py`,
`scripts/remap_probe.py`, `scripts/rope_logit_profile.py`) thấy:

- Mô hình **có** chú ý vào bằng chứng nhiều hơn vào token nhiễu. Trên KV, tại token trả lời, key đúng nhận 0,0048 attention,
  key sai nhận 0,0036. Nhưng mỗi head chỉ dành khoảng 0,5% attention cho bằng chứng, vì hơn 100k token khác cùng chia phần.
- Token ở xa còn bị RoPE trừ khoảng 3 nat logit so với token ở gần (khoảng cách 512–4k so với > 65k, ở các layer < 17).

Sửa RoPE cho mọi token thì thất bại:

| Cách sửa RoPE | KV20 (dense 16) | mk3 @128k (dense 33/40) | LD200 chấm theo tên (dense 14,0) |
|---|---|---|---|
| SelfExtend g8 (gộp vị trí xa) | – | ở 32k: 76 so với dense 95 | 14,0 |
| Nén khoảng cách ở các chiều chậm, freq 4k g8 | 12 | 30 | 13,0 |
| freq 32k g4 (nén nhẹ hơn) | 14 | 34 | 12,0 |
| Kẹp khoảng cách (giống ReRoPE) | 3/13 (dense 11/13) | – | – |
| Tắt thành phần chiều nhanh với key xa | không đổi (probe) | – | – |
| Làm sắc logit ×1.3 cho câu hỏi | 12 | – | – |

Lý do:

- Cộng điểm cho mọi token ở xa thì nhiễu tăng cùng mức với bằng chứng, tỉ lệ bằng chứng/nhiễu không đổi.
- Mô hình dùng khoảng cách RoPE thật như một "con trỏ" để chép đáp án. Mọi phép sửa khoảng cách phía query đều làm
  attention vào key đúng giảm 4–16 lần.

## 2. Ý tưởng

**Chỉ đưa phần bằng chứng lại gần câu hỏi.** RoPE không đổi, mô hình không đổi, không cần huấn luyện.

## 3. Các bước

1. **Prefill context bình thường.**
2. **Cho câu hỏi đọc context một lần.**
   - Lấy attention của các token câu hỏi ở layer 17 (cùng layer chấm điểm với LayerSieve).
   - Cộng điểm theo khối 64 token, bỏ qua 64 token đầu (sink).
   - Chọn các khối cao nhất, mỗi khối mở rộng ±96 token, gộp các khối chồng nhau, tổng tối đa 2048 token.
3. **Đưa các đoạn được chọn lại gần câu hỏi.** Có hai cách:
   - **Copy** (đã chạy): chép token của các đoạn vào ngay trước câu hỏi, kèm khung `[Excerpts…]`. Chạy lại phần này qua
     28 layer rồi sinh câu trả lời. KV của context được giữ nguyên, chỉ tốn thêm khoảng 2k token.
   - **Move** (đã viết và test đơn vị, chưa chạy thí nghiệm): không chép. Giữ nguyên KV của các đoạn đó, chỉ đổi vị trí
     RoPE của chúng thành vị trí ngay trước câu hỏi; vị trí cũ ở xa bị bỏ trống. Không token nào bị nhân đôi, không thêm
     token nào, vị trí tương đối bên trong mỗi đoạn được giữ nguyên.

Vì sao cách này giúp dù context vẫn dài 128k:

- **Tăng điểm có chọn lọc.** Chỉ khoảng 2k token đã chọn (1,5% context) được hưởng logit cao của khoảng cách gần. Tử số
  softmax của bằng chứng tăng, mẫu số gần như không đổi.
- **Các head cục bộ đọc được bằng chứng.** Khoảng 41% attention rơi vào 4k token cuối. Nhiều head gần như không bao giờ
  nhìn tới vị trí cách 100k token, nhưng đọc được đoạn đã đưa lại gần.
- **Tìm chỗ khác với dùng được bằng chứng.** Cộng attention qua các token câu hỏi và các head thì nhiễu triệt tiêu, nên bước
  xếp hạng khối tìm đúng chỗ (đáp án nằm trong đoạn chép ở KV 86/86, LD 199/200), dù từng head tại từng token rất loãng.

Ghi chú: các cơ chế trên mới là giải thích dựa trên số liệu trung bình, chưa đo trực tiếp attention lên đoạn chép so với
bản gốc.

## 4. Kết quả bản Copy

| Task | Dense | Copy | Nhận xét |
|---|---|---|---|
| KV86 (InfiniteBench) | 66 | **85** | 0 câu tệ đi |
| LD200, chấm theo tên | 14,0 | **31,0** | +17, CI [+10,5; +24], p<0,001; bản dùng marker, xem mục 6 |
| LD200, scorer chính thức | 9,5 | 22,0 | |
| multikey-3 @128k | 33/40 | **37/40** | 6 câu tốt lên, 2 câu tệ đi |
| fwe @128k | 80 | **88** | |
| qa_1 @128k | 70 | 70 | trung tính |
| code_debug | 50 | 50 | trung tính |
| qa_2 @128k (multi-hop) | 55 | 45 | hơi hại |
| vt @128k (chuỗi gán biến) | 97 | **85** | hại |
| cwe @128k (đếm từ) | 24,5 | **11** | hại, vì nhân đôi từ làm sai phép đếm |

- Dense là bản HF. Riêng KV86 so với dense chạy bằng vLLM.
- Các task RULER và code_debug: 20 câu mỗi task.

Ablation trên LD100 (chấm theo tên, dense 20,0):

| Cấu hình | Điểm |
|---|---|
| Layer 14/17/20, ngân sách 2048 | 34,0 |
| Chỉ layer 17, ngân sách 2048 | 35,0 |
| Ngân sách 512 | 14,0 |
| Ngân sách 4096 | 30,0 |

Trên KV20, ngân sách 512 cho 18/20. Vậy ngân sách tốt nhất phụ thuộc task, và phương pháp chưa tự chọn được.

## 5. Các phép kiểm tra bias

- **Chép đoạn ngẫu nhiên** (cùng khung, cùng ngân sách):

  | | Dense | Chép ngẫu nhiên | Chép theo attention |
  |---|---|---|---|
  | KV20 | 16 | 17 | 20 |
  | LD50, chấm theo tên | 18 | 16 | 30 |

  Lợi ích đến từ việc **chọn đúng đoạn**, không phải từ format.
- **Phần chạy sau khi đã cố định phương pháp** (coi như held-out):
  - LD (100 câu sau): 28,0 so với 8,0 (chấm theo tên).
  - KV (66 câu sau): 65 so với 49.
- **Scorer chính thức của InfiniteBench LD** không bỏ dấu `**`, nên câu trả lời đúng nhưng in đậm bị chấm 0. Khung
  `[Excerpts…]` làm mô hình đổi format (chép ngẫu nhiên vẫn được +8 theo scorer này). Vì vậy luôn báo kèm cách chấm theo tên.
- **Đã bỏ hoàn toàn** BM25 và mọi so chuỗi câu hỏi với context, kể cả marker câu hỏi riêng của từng dataset. Đáp án chỉ được
  dùng sau khi chạy xong, để chấm điểm và ghi log.

## 6. Vấn đề đang xử lý

Khi bỏ marker và lấy **128 token cuối** làm câu hỏi (không biết task):

- KV86 vẫn tốt: 85/86.
- **LD200 tụt về đúng mức dense:** 14,0 (chấm theo tên). Câu hỏi LD chỉ dài khoảng 20 token; 100 token còn lại là đoạn
  cuối kịch bản, và attention của chúng làm loãng tín hiệu. Đoạn chép chỉ còn chứa `$$MASK$$` ở 30/200 câu, so với 199/200
  khi dùng marker.

Vậy **kết quả LD +17 phụ thuộc vào việc biết chính xác câu hỏi nằm ở đâu.**

Đang đo (`scripts/qlen_probe.py`) xem lấy bao nhiêu token cuối (1, 4, 8, 16, 32, 64, 128) thì chọn đoạn tốt cho mọi task:
KV, LD, qa_1, qa_2, multikey-3. Mục tiêu là một con số chung, không chỉnh theo từng task.

## 7. Bước tiếp theo

1. Chọn độ dài câu hỏi chung từ phép đo ở mục 6, rồi chạy lại LD.
2. Chạy bản **Move** trên cả các task được lợi (KV, LD) lẫn các task bị hại (cwe, vt, qa_2). Kỳ vọng là Move giữ được phần
   lợi mà bớt hại, vì không nhân đôi token.
3. Nếu vẫn còn task bị hại: thêm cơ chế tự quyết định có chép hay không, dựa trên độ tập trung của attention (đã bắt đầu ghi
   lại số liệu `share`/`frac`). Ngưỡng phải được chọn chung, không theo task.
4. Ghép vào LayerSieve (dùng lại điểm ở ℓ₀=17), đo cả độ chính xác lẫn tốc độ.

**Còn chạy:** Summarize (InfiniteBench En.Sum, 20 cuốn), kiểm tra chỉ giữ khung không chép đoạn, multivalue/multiquery,
các task ở 32k.

## 8. Đánh giá hiện tại

- Hiệu quả rõ với các bài có **một vùng bằng chứng cục bộ**: KV, multikey, và LD (khi xác định đúng câu hỏi).
- **Chưa đủ tổng quát:** có hại với các task cần tổng hợp (vt, cwe, qa_2), và trên LD còn phụ thuộc vào việc xác định đúng
  câu hỏi.
- Bản **Move** và cách chọn độ dài câu hỏi chung là hai thứ sẽ quyết định phương pháp có dùng được hay không.
- Mới thử trên một mô hình. Các công trình liên quan cần kiểm tra trước khi trích dẫn: Attention Sorting
  (Peysakhovich & Lerer, 2023), STRING, Ms-PoE, Found-in-the-Middle.
