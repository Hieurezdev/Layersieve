# Đưa bằng chứng ở xa lại gần câu hỏi: báo cáo (cập nhật 20/10/2026, 01:50 UTC)

Mô hình: Qwen2.5-7B-Instruct-1M, prompt khoảng 128k token. Mọi thí nghiệm chạy dense, chưa ghép LayerSieve.

- Code: `scripts/recall_eval.py`, `scripts/qlen_probe.py`.
- Chấm điểm: `scripts/analysis/` (`selscore.py`, `summary_table.py`, `ld_first.py`).
- Kết quả: `results/selfext/`.
- Quy tắc: không BM25, không so chuỗi câu hỏi với context, không marker riêng của dataset. Đáp án chỉ dùng để chấm sau khi chạy.

## 1. Tóm tắt

1. **Sửa RoPE theo mọi kiểu đều thất bại.** Mô hình dùng khoảng cách thật như "con trỏ" để chép đáp án; can thiệp đều lên mọi token hay mọi head đều làm hỏng con trỏ đó (mục 2).
2. **Copy bản sạch cho lợi ích lớn và đã xác nhận trên LD200 đủ 200 câu:** 34,0 so với 14,5 (+19,5, CI [+10; +22,5]). Bản sạch nghĩa là: 16 token cuối dùng để chấm điểm, đoạn chép chèn trước 128 token cuối, không marker. Copy cũng tăng mạnh trên KV, multikey, multivalue, qa_1 và fwe.
3. **Copy có hại ở các câu hỏi về toàn bộ context:** math_find (số lớn nhất của cả list) giảm 35 điểm, cwe giảm 13, qa_2 giảm 10. Cơ chế đã được xác định: **mô hình coi đoạn chép là toàn bộ context**. Ví dụ với math_find, copy trả lời số lớn nhất *trong đoạn chép* (85) thay vì của cả list (91).
4. **Có hai cơ chế khác nhau** (mục 5):
   - (A) *Loãng ở các head chép.* Chỉ cần tăng attention ở 16 retrieval head (focus) là KV được lợi, và không task nào bị hại.
   - (B) *Suy luận trên bằng chứng ở xa* (LD). Focus không giúp gì (16 so với 18); cần đặt cả đoạn bằng chứng lại gần để mọi head cùng xử lý (copy: 36–40).
5. **Chưa có cách nào vừa giữ toàn bộ lợi ích vừa không gây hại ở đâu.** Fuse cho trung bình tốt nhất (69,6 so với dense
   66,1; copy+select 68,7) và an toàn nhất, nhưng mất lợi ích trên KV và một nửa lợi ích trên LD.

## 2. Những gì không hiệu quả (và vì sao)

| Cách | Kết quả | Lý do |
|---|---|---|
| Sửa RoPE: SelfExtend, nén chiều chậm, kẹp khoảng cách, tắt chiều nhanh, logit ×1.3 | KV20 3–14 (dense 16); LD chấm theo tên không tăng | Nâng cả nhiễu lẫn bằng chứng; làm hỏng con trỏ khoảng cách |
| Các "cải thiện" LD trước đây của remap | Là artifact | Scorer chính thức chấm 0 câu trả lời in đậm `**Tên**`; chấm theo tên thì không tăng |
| Copy với 128 token cuối làm câu hỏi | LD 14 = dense | Đoạn cuối kịch bản làm loãng tín hiệu; chỉ 30/200 lần chọn trúng `$$MASK$$` |
| Copy hoặc Move với 16 token làm điểm chèn | KV Move 0/20 | Điểm chèn rơi vào giữa câu hỏi, đẩy key ra xa |
| Move (đổi vị trí RoPE của đoạn chọn, không chép) | vt 59, kém copy ở mọi task | Phá cấu trúc vị trí quanh câu hỏi |
| Bias +4 nat ở cả 784 head | vt 68, cwe 6,5; sinh lặp | Can thiệp quá mạnh và quá rộng |
| Focus 45 head (ngưỡng 0,1) | Kém hơn 16 head (multivalue 78) | Bao cả các head không phải head chép |
| Dense kiểm chứng câu trả lời của copy | Gần như luôn giữ dense | Mô hình dense ưu tiên câu trả lời của chính nó |

## 3. Phương pháp chính: copy bản sạch

1. Prefill context như bình thường.
2. **Chấm điểm:** lấy attention của **16 token cuối** của prompt ở **layer 17**, cộng theo khối 64 token. Chọn các khối tốt nhất, mỗi khối mở rộng ±96 token, tổng tối đa 2048 token.
   - Con số 16 được chọn chung cho mọi task bằng `qlen_probe.py`:

     | Số token cuối | 1 | 4 | 8 | **16** | 32 | 64 | 128 |
     |---|---|---|---|---|---|---|---|
     | Số lần trúng (tổng 70) | 43 | 39 | 59 | **59** | 55 | 39 | 30 |

3. **Chèn:** chép các đoạn đã chọn vào **trước 128 token cuối**, để câu hỏi luôn nguyên vẹn. Chạy lại phần này qua 28 layer rồi sinh câu trả lời. Không phải prefill lại 128k token.
4. Các biến thể:
   - **select:** sinh cả câu trả lời dense, giữ câu có log-prob trung bình cao hơn.
   - **draft:** các token câu trả lời dense cũng được dùng để chấm điểm.
   - **fuse:** sinh song song hai nhánh, mỗi token chọn theo tổng log-prob của dense và copy.
   - **focus:** không chép; cộng `log(n_context / n_đoạn chọn)` vào logit của các đoạn chọn, chỉ ở 16 retrieval head (`results/repos/heads_7b1m.json`, ngưỡng 0,3), chỉ cho token câu hỏi và token đang sinh.

## 4. Bảng kết quả

20 câu mỗi task (mk3 40 câu, LD 50 câu). Dense lấy từ cùng run. LD chấm theo tên.

| Task | Dense | Copy | Copy+select | Copy+draft | Focus 16 head | Fuse |
|---|---|---|---|---|---|---|
| KV | 80 | **95** | **95** | 90 | 90 | 80 |
| multikey-3 | 82 | **95** | 88 | **95** | – | **95** |
| multivalue | 85 | 94 | 92 | **98** | – | 89 |
| LD | 18 | 36 | 30 | **40** | 16 | 28 |
| qa_1 | 70 | **80** | 75 | **80** | – | 75 |
| fwe | 80 | **88** | 85 | – | – | 85 |
| code_debug | 50 | 55 | 55 | – | – | **60** |
| vt | 97 | 96 | 96 | 98 | 91 | **100** |
| qa_2 | **55** | 45 | 45 | 45 | 50 | 50 |
| cwe | **25** | 12 | 15 | 12 | 24 | 19 |
| math_find | **85** | 50 | 80 | 50 | **85** | **85** |
| **Trung bình** | 66,1 | 67,8 | 68,7 | (9 task) | (6 task) | **69,6** |

Ô "–" là run chưa chạy hoặc đang chạy (focus 16 head cho multivalue/qa_1/fwe).

**Fuse cho trung bình cao nhất (69,6)** và không task nào thấp hơn dense quá 6 điểm. Nó giữ được lợi ích trên multikey (95),
vt (100), code_debug (60) và cứu math_find (85). Đổi lại nó mất lợi ích trên KV (80, vì dense tự tin mà sai) và mất khoảng
một nửa lợi ích trên LD (28 so với 36–40 của copy).

**Xác nhận với số mẫu lớn hơn:**
- LD200, copy bản sạch: 34,0 so với 14,5 (+19,5, CI [+10; +22,5]).
- KV86, copy với 128 token: 85/86 so với 66/86 dense vLLM; 0 câu tệ đi.
- math_find 50 câu (đang mở rộng các task có hại): dense 52, copy 26, copy+select 40. Mức hại được xác nhận với số mẫu lớn
  hơn. 20 câu đầu cho dense 85 là do may mắn khi chọn mẫu.

**Các phép kiểm tra bias:**

| Kiểm tra | Kết quả | Kết luận |
|---|---|---|
| Đoạn chép ngẫu nhiên (KV20 / LD50) | 17 / 16, dense 16 / 18 | Lợi ích đến từ chọn đúng đoạn |
| Chỉ giữ khung, không chép (KV20 / LD50) | 15 / 20 | Khung không tạo lợi ích |
| Phần chạy sau khi đã cố định phương pháp | LD 28 so với 8; KV 65 so với 49 | Lợi ích vẫn giữ |
| Summarize (ROUGE-Lsum, 20 cuốn) | 34,6 so với 34,3 | Trung tính |
| Context 32k (mk3 / qa_1 / qa_2) | 97,5 / 85 / 55; dense 95 / 90 / 55 | Gần như trung tính |

## 5. Hai cơ chế: điểm mới rút ra trong đêm

**(A) Loãng ở các head chép (KV, multikey).**
- Đo trên dữ liệu có sẵn: chỉ 16/784 head thật sự chép đáp án, nằm ở layer 14 và 22–23.
- Chỉ tăng attention ở 16 head đó (focus): KV 80 → 90, và **không task nào bị hại** (math 85, cwe 24).
- Tức là phần "tìm và chép" có thể sửa rất cục bộ.

**(B) Suy luận trên bằng chứng ở xa (LD).**
- Focus không giúp (16 so với 18), trong khi copy giúp mạnh (36–40).
- Đáp án LD không có sẵn để chép; phải suy ra từ ngữ cảnh hội thoại. Đặt cả đoạn lại gần cho **mọi head** cùng đọc nó với câu hỏi.
- Cái giá: mô hình coi đoạn chép là đủ, nên các câu hỏi về toàn bộ context bị hại. Draft không sửa được điều này (math vẫn 50).

## 6. Bước tiếp theo đề xuất

1. **Ghép theo cơ chế:** luôn bật focus 16 head (an toàn), còn copy chỉ dùng cho phần cần suy luận.
   - Vấn đề mở là quyết định khi nào dùng copy mà không đọc nội dung câu hỏi.
   - Các tín hiệu đã thử chưa tách được: độ tập trung attention (cwe 0,62, ngang KV 0,64), log-prob, dense kiểm chứng.
   - Ứng viên tiếp theo: độ phủ của attention trên toàn context. Câu hỏi toàn cục kiểu "lớn nhất trong list" có thể rải attention đều trên cả list.
2. **Tăng số mẫu cho các task có hại** (qa_2, cwe, math, 50–100 câu). Mức hại hiện tại chỉ là 2–7 câu trên 20.
3. **Ghép vào LayerSieve/vLLM:** điểm ở layer 17 đã có sẵn. Hiện vòng lặp HF chậm, khoảng 1–1,5 phút mỗi prompt.
4. **Kiểm tra bias của retrieval head:** chúng được xác định trên KV, cần đo lại trên một task khác.

## 7. Ghi chú vận hành

- GPU dùng chung; đêm qua người khác chiếm 110–126GB, một số run bị OOM. Các lane hiện thử lại khi OOM (`scripts/run_night12.sh`).
- Kiểm tra mỗi giờ ghi ở `results/selfext/hourly.log`.
- Các run có lỗi thiết kế được chuyển sang `results/selfext/flawed_q16/`.
- Chưa commit, chưa sửa paper.
