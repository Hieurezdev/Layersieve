# LayerSieve nhiều lượt: hỏi tiếp trên cùng tài liệu

Báo cáo ngày 01/11/2026. Model chính Qwen2.5-7B-Instruct-1M (L = 28), ngữ cảnh ~128k token, một H200. Số liệu gốc và
lệnh chạy lại nằm trong `results/multiturn/` (xem mục 8). Phần Llama và E1 đo sạch vẫn đang chạy (mục 7).

## 1. Vấn đề

LayerSieve chọn token theo câu hỏi ở cuối prompt: 17 layer đầu chạy dense, từ layer 17 (đánh số từ 0) mỗi layer bỏ
các unit có attention thấp, xuống sàn 25% ở layer 19, rồi KV sieving thu KV của các layer < 22 về tập còn lại. Cache sau
prefill vì vậy **gắn với câu hỏi đó**. Paper (mục "Deployment scope") chỉ đề xuất vứt cache sau mỗi lượt và để
multi-turn là future work. Câu hỏi tiếp theo cần evidence đã bị cắt thì phải prefill lại cả tài liệu.

## 2. Ý tưởng chi tiết

### 2.1 Nhận xét nền tảng

Gọi prompt là tài liệu D (T token, vị trí 0..T−1) nối với câu hỏi Q. Với attention nhân quả, hidden state của token i ở
layer ℓ trong lượt chạy **dense** chỉ phụ thuộc các token 0..i. Mọi token tài liệu đứng trước câu hỏi, nên:

> Ở mọi layer, trạng thái dense của tài liệu là như nhau với mọi câu hỏi.

LayerSieve chạy dense ℓ0 = 17 layer đầu (61% độ sâu) trên toàn prompt, nên **toàn bộ phần nông của tài liệu (KV của
layer 0–16 và hidden state vào layer 17) là phần dùng chung chính xác cho mọi câu hỏi**. Từ layer 17, tập token S_ℓ
được chọn bằng attention của 128 token cuối (tức câu hỏi), nên phần sâu phụ thuộc câu hỏi theo hai cách:

1. token nào được giữ;
2. trạng thái của token được giữ, vì nó chỉ attend tới các token còn lại.

Ba hệ quả:

- **K, V của layer 17 cho mọi token tài liệu cũng không phụ thuộc câu hỏi**, vì đầu vào là hidden state vào layer 17.
  Chấm điểm câu hỏi mới ở layer 17 không cần tính lại gì cho tài liệu.
- **Đường cắt độ sâu ℓ0 của LayerSieve chính là ranh giới** giữa phần dùng chung và phần riêng của từng câu hỏi. Đây là
  tính chất mà phương pháp cắt ở đầu vào (SpecPrefill, Raincheck) không có: ở đó mọi layer đều phụ thuộc lựa chọn.
- **Áp dụng được cho hội thoại.** Lịch sử (câu hỏi, câu trả lời các lượt trước) cũng đứng trước câu hỏi hiện tại, nên
  phần dense của lịch sử cũng tính một lần rồi dùng lại.

### 2.2 Trạng thái giữ lại sau lượt đầu

Kích thước tính cho Qwen2.5-7B (L = 28, d = 3584, 4 KV head × 128, bf16), T = 128k:

| thành phần | công thức | 128k | dùng cho |
|---|---|---|---|
| KV layer 0–16 của mọi token tài liệu | ℓ0 · T · 2 · 512 · 2 B | 4.6 GB | A, B, bù dần, C |
| checkpoint: hidden state vào layer 17 | T · d · 2 B (vLLM: `hidden` + `residual`, ×2) | 0.94 GB (1.9 GB) | A, B, bù dần, C |
| KV dense layer 17–27 (sau khi bù) | (L − ℓ0) · T · 2 · 512 · 2 B | 2.9 GB | B, bù dần |
| KV sâu đã cắt của lượt trước | ≈ Σ_ℓ≥17 κ_ℓ · T · 2 KB | 0.84 GB | C, `naive` |

Hai lưu ý thiết kế:

- **Mâu thuẫn với KV sieving.** KV sieving xóa KV nông của các token bị cắt để decode nhanh, nhưng đó lại là phần cần
  giữ cho câu hỏi sau. Phải giữ một bản đầy đủ riêng: trên GPU, hoặc chép ra CPU (4.6 GB qua PCIe 5, khoảng 0.1–0.2 s)
  rồi chép ngược khi có câu hỏi sau. Decode vẫn đọc bản đã sàng.
- **Chỉ lưu một checkpoint, không lưu hidden state từng layer.** Với GQA, hidden state của một token (3584 số) lớn gấp
  3.5 lần KV của một layer (2 × 512 số). Lưu hidden state cho từng layer như HCache tốn hơn lưu KV. Chỉ ranh giới ℓ0
  cần hidden state, vì phần sâu luôn phải tính lại từ đó.

### 2.3 Chế độ A: sàng lại khi câu hỏi đến

```
lượt k ≥ 2, câu hỏi Q_k (s token):
  1. Q_k qua layer 0–16, attend vào KV nông đầy đủ của tài liệu      (rẻ: s token)
  2. h = [checkpoint của tài liệu ; hidden state vào layer 17 của Q_k]
  3. chạy lại LayerSieve ở layer 17–27 trên h với cửa sổ quan sát = Q_k:
       chấm điểm → bỏ 40% unit → 60%, 36%, rồi sàn 25% tài liệu
  4. KV sieving trên một bản sao, decode như LayerSieve thường
```

Ở lượt 2, phần nào được tính lại (layer đánh số từ 0 như trong code; "layer 17" ở đây là layer 18 của paper):

| phần | tài liệu | token câu hỏi mới |
|---|---|---|
| layer 0–16 (nông) | **không tính lại**: dùng nguyên KV giữ từ lượt 1 | **có tính**: chưa có KV ở đâu; vài chục–vài trăm token, đọc KV tài liệu đã giữ |
| layer 17–27 (sàng) | **tính lại** từ checkpoint; KV sâu của lượt 1 bị bỏ | tính cùng tài liệu |

"Tính lại phần sâu" là chạy lại cả quá trình sàng, không chỉ tính lại KV: chấm điểm lại unit bằng attention của câu hỏi
mới, bỏ dần (60% → 36% → 25%), rồi chỉ unit còn lại mới được tính KV. Tập giữ ở lượt 2 vì vậy khác lượt 1; đó là lý do A
đúng khi câu hỏi đổi evidence. KV sieving sau đó làm trên một bản sao; bản KV nông đầy đủ vẫn được giữ cho các lượt sau.

- **Chính xác.** Mọi đầu vào của bước 3 giống hệt khi chạy LayerSieve từ đầu trên (D, Q_k), nên kết quả bằng LayerSieve
  từ đầu, chỉ khác sai số dấu phẩy động do kernel chạy trên độ dài khác nhau.
- **Chi phí lượt sau, phần tuyến tính:** (ℓ0/L)·(s/T) + (1/L)·Σ_ℓ≥17 κ_ℓ = (0.6 + 0.36 + 9 × 0.25)/28 ≈ 11.5% FLOPs
  tuyến tính của prefill dense. Phần attention co bậc hai theo κ nên thời gian thực thấp hơn: đo được 5.5–7% TTFT dense.
- **Ưu điểm:** không cần GPU rảnh, dùng kernel causal chuẩn, bộ nhớ thấp hơn B.
- **Nhược điểm:** trả chi phí này ở **mỗi** lượt, trên đường găng.

### 2.4 Chế độ B: bù cache khi GPU rảnh

```
sau khi trả lời lượt 1 (người dùng đang đọc / gõ):
  chạy layer 17–27 DENSE cho toàn bộ tài liệu từ checkpoint      (≈ (L − ℓ0)/L = 39% prefill dense)
  → KV dense đầy đủ cả 28 layer = một prefix cache dense bình thường
lượt k ≥ 2:
  chỉ prefill Q_k trên prefix cache (ε ≈ 0.13 s ở 128k)
```

- **Chính xác:** sau khi bù xong, kết quả bằng dense có prefix cache.
- **Chi phí bù** trả một lần, ngoài đường găng, nếu khoảng nghỉ giữa hai lượt đủ dài (~5.8 s ở 128k, ~29 s ở 256k).
- **Nhược điểm:**
  - Phần bù vẫn tốn công suất GPU. Trên server bận, nó giành GPU với request khác nên phải chạy ưu tiên thấp và nhường
    khi có request đến.
  - Decode trên KV đầy đủ chậm hơn decode trên KV đã sàng. Có thể gỡ bằng decode thưa kiểu Quest trên cache đầy đủ.
  - Tốn bộ nhớ như dense.
- **Hai cách tìm thời gian bù:**
  - khoảng nghỉ giữa các lượt;
  - xen vào các bước decode của lượt đầu, như Raincheck. Decode bị giới hạn bởi băng thông bộ nhớ nên còn dư công suất
    tính; cách này chưa đo.

### 2.5 Bù dần: nối liền A và B

Phần bù chạy theo thứ tự trong tài liệu, từng khúc: token 0..c, rồi c..2c, … Tại bất kỳ thời điểm nào, nếu đã bù tới vị
trí p·T:

```
câu hỏi Q_k đến khi đã bù tới p:
  - tài liệu [0, pT): KV dense ở mọi layer sâu → giữ nguyên, không chấm, không tính
  - tài liệu [pT, T) + Q_k: sàng LayerSieve ở layer 17–27; khi chấm điểm, câu hỏi nhìn cả phần đã bù
    (softmax trên toàn bộ), nhưng chỉ phần [pT, T) bị cắt; sàn 25% tính trên phần này
```

- **Chính xác.** Token trong [0, pT) chỉ attend tới token trước nó, đều đã được giữ đủ, nên KV dense của chúng đúng.
  Kết quả bằng "LayerSieve với prefix [0, pT) được bảo vệ khỏi bị cắt", đúng cách mà đoạn "Deployment scope" của paper
  gợi ý ("apply LayerSieve to the uncached suffix").
- **Chi phí khi câu hỏi đến** ≈ (1 − p) × phần sâu của A + phần câu hỏi qua layer nông. **Đã bỏ ra trước** khi chạy
  nền: p × chi phí bù của B. p = 0 là A, p = 1 là B.
- **Ý nghĩa:** một cơ chế duy nhất, không phải chọn giữa A và B. Người dùng chờ càng lâu thì lượt sau càng rẻ, và lúc
  nào cũng chính xác. Khác Raincheck ở chỗ câu hỏi đến giữa chừng vẫn nhận kết quả chính xác (của LayerSieve với prefix
  được bảo vệ), không phải cache xấp xỉ.

**Diễn biến theo thời gian.**

```
Lượt 1: LayerSieve như thường → trả lời
        ↓ GPU rảnh: bắt đầu bù nền
Bù nền: layer 17–27 DENSE từ checkpoint, từng khúc theo thứ tự tài liệu: 0–16k → 16k–32k → …
        ↓ câu hỏi 2 đến khi đã bù tới p
Lượt 2: dừng bù, trả lời (bảng dưới)
        ↓ trả lời xong: bù tiếp từ p cho lượt 3
```

| phần | tài liệu đã bù [0, p) | tài liệu chưa bù [p, T) | token câu hỏi mới |
|---|---|---|---|
| layer 0–16 | không tính, dùng KV đã giữ | không tính, dùng KV đã giữ | có tính (rẻ) |
| layer 17–27 | **không tính, không chấm, không cắt**; dùng luôn KV dense đã bù | **sàng lại như A** từ checkpoint, sàn 25% tính trên phần này | tính cùng phần chưa bù |

**Tính phần bù nền.** Đây là chunked prefill bình thường, chỉ bắt đầu từ layer 17. Mỗi khúc chỉ cần KV của các khúc
trước, nên dừng ở đâu cũng được và tiếp tục sau.

```
for khúc [a, b) theo thứ tự tài liệu:
    h = checkpoint[a:b]
    for layer ℓ = 17 … 27:
        q, k, v = chiếu(h)                                    # chỉ token của khúc
        K, V = KV dense layer ℓ của [0, a)  nối  k, v
        h = layer ℓ(h), attention nhân quả trên K, V
        lưu k, v vào KV dense của layer ℓ
    p = b
```

**Tính khi câu hỏi đến (đã bù tới p).**

```
h_q = câu hỏi qua layer 0–16, nhìn KV nông đầy đủ của tài liệu
h   = [checkpoint[p:T] ; h_q]                                 # phần [0, p) không có trong h
for layer ℓ = 17 … 27:
    K_bù, V_bù = KV dense layer ℓ của [0, p)                  # có sẵn
    điểm = softmax(q của 128 token cuối · [K_bù ; k(h)])     # softmax trên cả phần đã bù
    bỏ unit điểm thấp, chỉ trong h (phần [0, p) không bao giờ bị bỏ)
    h = layer ℓ(h): query = token còn lại trong h,
                    key/value = [K_bù, V_bù] + KV của chính chúng (nhân quả)
KV sieving trên bản sao, decode
```

So với A chỉ khác hai điểm: ít token phải tính hơn, và phần [0, p) góp vào attention và softmax qua KV có sẵn.

**Ước lượng thời gian.** Prefill khi câu hỏi đến ≈ phần nông của câu hỏi + (1 − p) × phần sâu của A. Số lấy từ E1, vLLM,
128k, GPU dùng chung:

- phần sâu của A = tổng layer 17–27 của LayerSieve = 1.13 s;
- phần nông của câu hỏi ≈ ε × 17/28 = 0.13 × 17/28 ≈ 0.08 s.

| p | phép tính | prefill lượt sau | đã bù nền trước (p × 5.8 s) |
|---|---|---|---|
| 0 (A) | 0.08 + 1.00 × 1.13 | 1.21 s | 0 |
| 1/4 | 0.08 + 0.75 × 1.13 | 0.93 s | 1.5 s |
| 1/2 | 0.08 + 0.50 × 1.13 | 0.65 s | 2.9 s |
| 3/4 | 0.08 + 0.25 × 1.13 | 0.36 s | 4.4 s |
| 1 (B) | ≈ ε | 0.13 s | 5.8 s |

Hệ số (1 − p) đã đo trên bản HF: token×layer sâu tính lúc câu hỏi đến bằng 0.75 / 0.50 / 0.25 lần của A. Ước lượng giả định
thời gian tỷ lệ với token×layer. Giả định này hơi lạc quan: attention còn phải đọc KV đã bù, nên ở p cao thời gian thật có
thể lớn hơn một chút. Cần cài bù dần vào vLLM để đo trực tiếp.

**Bản thử chưa có tiến trình bù nền thật.** KV dense của [0, p) được cắt ra từ lần prefill dense tài liệu; vì attention
nhân quả, kết quả giống hệt bù nền. Độ chính xác là số đo thật; thời gian là ước lượng.

### 2.6 Chế độ C: phục hồi từng phần (ý tưởng ban đầu)

Giữ một tập U các token đã có KV sâu (ban đầu là tập giữ của lượt 1), cùng KV sâu của chúng ở từng layer.

```
lượt k ≥ 2:
  1. Q_k qua layer 0–16 (như A)
  2. chọn MỘT LẦN ở layer 17: chấm điểm mọi unit bằng attention của Q_k lên K của layer 17
     (đã có sẵn, không phụ thuộc câu hỏi), giữ 25% → S_k
  3. N_k = S_k \ U   (unit mới cần, khoảng 9–14% tài liệu trong thí nghiệm)
  4. layer 17–27: chỉ tính N_k ∪ Q_k; key/value = KV CŨ của U ∪ KV mới của N_k ∪ Q_k,
     mặt nạ nhân quả theo vị trí gốc (KV cũ và mới xen kẽ)
  5. U ← U ∪ N_k (hội thoại: cộng dồn qua các lượt); KV sieving rồi decode
```

- **Chi phí:** |N_k|/T × (L − ℓ0)/L, đo được 30–47% phần sâu của A. Có cận trên: cộng dồn cả phiên không vượt chi phí bù
  toàn bộ của B.
- **Vì sao chỉ xấp xỉ:**
  1. KV sâu của unit trong U được tính khi chỉ một tập token khác còn lại. Nếu tập token đứng trước nó đổi, KV đó "cũ".
  2. Chọn một lần ở layer 17 khác với sàng dần ở lượt đầu. Ngay cả khi hỏi lại cùng evidence, vẫn có ~9% unit mới.
- **Một unit bị bỏ giữa chừng ở lượt trước** (có KV cũ ở vài layer đầu của phần sâu) mà được chọn lại thì phải bỏ KV cũ
  của nó trước khi tính lại. Nếu không, cùng một vị trí có hai key. Lỗi này đã gặp và sửa ngày 31/10.
- **Biến thể làm mới** (C+R trong SessionSieve): tính lại một phần U có tập token đứng trước thay đổi nhiều nhất. Chưa
  làm ở đây.
- **Yêu cầu kernel:** attention trên KV cũ và mới xen kẽ theo vị trí cần kernel block-sparse hoặc varlen có mask, khác
  A và B chỉ dùng kernel causal chuẩn. Nếu không có kernel phù hợp, C tính ít hơn mà không nhanh hơn. Bản HF hiện tại
  chậm hơn A vì lý do này.

### 2.7 Vì sao bỏ "chấm điểm block theo chi phí phục hồi"

Ý tưởng ban đầu muốn ưu tiên cắt các block rẻ phục hồi (có checkpoint gần, ít token phụ thuộc). Với LayerSieve, tiền đề
này không đứng.

**Phân bố layer bị loại do lịch cắt quyết định.** Trước layer 17 bỏ 40% tài liệu, trước layer 18 bỏ thêm 24%, trước
layer 19 bỏ thêm 11%, rồi chạm sàn 25% và không bỏ thêm. Mọi token bị loại đều bị loại ở layer 17–19.

Hệ quả:

- **Phục hồi từ checkpoint tốn như nhau cho mọi block:** 11 layer sâu.
- **Checkpoint ngay tại layer bị loại tiết kiệm tối đa 2/11 layer**, mà lại "cũ": hidden state sau layer 17 đã gắn với
  tập token của câu hỏi trước.
- **Không có khác biệt chi phí giữa các block để tối ưu.**

Ý tưởng vẫn được giữ, nhưng **ở mức lịch cắt** thay vì từng block: chi phí hoàn tác = phần layer phải tính lại.

| phương pháp | cắt bắt đầu từ | phần phải tính lại để về dense | chi phí lượt đầu C |
|---|---|---|---|
| SpecPrefill, Raincheck (cắt ở đầu vào) | layer 0 | 100% | 0.70 |
| SlimInfer | sau layer 8 | 71% | 0.74 |
| FastKV | layer 15 | 50% | 0.71 |
| **LayerSieve** | layer 17 (0.61L) | **39%** | **0.72** |
| ASL | trung vị layer 20 | ~29% | 0.84–0.87 |

> Độ sâu bắt đầu cắt quyết định cả độ chính xác lẫn chi phí hoàn tác.

LayerSieve cắt đủ muộn để vừa chính xác vừa hoàn tác rẻ, nhưng vẫn đủ sớm để lượt đầu nhanh hơn ASL. Câu này nối thẳng
với luận điểm chính của paper về độ sâu bắt đầu cắt.

### 2.8 Hội thoại nhiều lượt

Lịch sử (câu hỏi và câu trả lời các lượt trước) được xử lý như phần nối dài của tài liệu:

- **Lớp nông:** sau mỗi lượt, phần mới (đáp án lượt trước + câu hỏi mới) đi qua mọi layer dense một lần rồi gắn vào
  prefix (`extend`). Rẻ, vì chỉ vài trăm token mỗi lượt.
- **Lịch sử không bao giờ bị cắt.** Chỉ token tài liệu bị cắt; câu hỏi hiện tại luôn nằm trong cửa sổ quan sát.
- **A:** mỗi lượt sàng lại tài liệu với câu hỏi mới; bằng LayerSieve từ đầu trên cả hội thoại.
- **C:** U cộng dồn qua các lượt, gồm cả token lịch sử đã tính.
- **`naive`:** hội thoại tiếp trên cache đã cắt của lượt 1, phần mới được tính chồng lên cache đó.

### 2.9 Chọn chế độ nào khi phục vụ

| tình huống | nên dùng | vì sao |
|---|---|---|
| câu hỏi đến ngay sau câu trả lời | A, hoặc bù dần với p hiện có | chính xác, không cần chờ |
| có khoảng nghỉ ≥ thời gian bù | B (bù xong trước khi câu hỏi đến) | lượt sau gần như miễn phí |
| server bận, không có GPU rảnh | A | không tốn tính toán nền |
| câu hỏi mới cần cùng evidence | dùng lại cache (`naive`) là đủ | thí nghiệm: không mất điểm |
| cần tiết kiệm tính toán, có kernel phù hợp | C | 30–47% phần sâu của A, mất ~2 điểm trên SCBench |

**Hướng mở: tự quyết khi nào phải sàng lại.** Chấm điểm câu hỏi mới ở layer 17 (rẻ, vì K layer 17 đã có), rồi so tập
được chọn với tập đang giữ. Trùng nhiều thì dùng lại cache; lệch nhiều thì sàng lại (A) hoặc phục hồi phần thiếu (C).
Chưa thử.

### 2.10 So với công trình liên quan

| | cắt ở đâu | phục hồi từ | kết quả sau phục hồi | câu hỏi đến sớm | dữ liệu |
|---|---|---|---|---|---|
| Raincheck | đầu vào | layer 0 | xấp xỉ (KV token đã giữ không tính lại) | không có chế độ riêng | ảnh, video |
| HCache, CacheFlow | không cắt | hidden state đã lưu | chính xác (cache dense) | – | – |
| CacheBlend, ProphetKV, Budgeted Cache Repair | không cắt (ghép đoạn) | tính lại một phần token | xấp xỉ | – | RAG |
| KVzip | nén sau prefill, không theo câu hỏi | – | xấp xỉ | – | nhiều câu hỏi |
| LazyLLM | trong mô hình, từ layer đầu | Aux Cache trong một lần sinh | – | – | một lượt |
| SessionSieve (cùng team) | trong mô hình, cắt một lần ở ℓ0 | checkpoint ℓ0 | "B" chính xác, "C" xấp xỉ | có | SCBench 32k |
| **LayerSieve nhiều lượt (đây)** | trong mô hình, sàng dần từ 0.61L | checkpoint ℓ0 | A, B, bù dần chính xác; C xấp xỉ | A, bù dần | kv, RULER, SCBench 128k |

### 2.11 Câu đóng góp đề xuất cho paper

> *Because LayerSieve prunes only after a dense, question-independent prefix, its pruning can be undone exactly for a
> new question by re-running only the pruned layers from a single mid-depth checkpoint: on demand (about 6% of a dense
> prefill), in the background (about 40%, once), or progressively between the two, whereas input-level or early-layer
> pruning must recompute far more and only approximately.*

**Câu hỏi mở:**

- Tích hợp vào scheduler của vLLM V1 (chunked prefill, prefix caching, request ưu tiên thấp cho phần bù).
- Kernel cho C.
- Sparse attention ở phần dense: pattern phải không phụ thuộc câu hỏi thì mới dùng lại được cache.
- Quy tắc tự quyết khi nào sàng lại.
- Lịch cắt cho họ model khác (Llama).

## 3. Cài đặt

- **Bản HF tham chiếu:** `inmodel_prune/multiturn.py`, lớp `MultiTurn`. LayerSieve giống bản của paper: unit, sàn
  `min_keep` 0.25, `drop_ratio` 0.4, chấm điểm bằng attention của 128 token cuối (max theo head), 128 sink, KV sieving tại
  layer 22.
- **Phần dùng chung.** `load_document` prefill dense tài liệu một lần (KV mọi layer + hidden state vào layer 17). Các
  chế độ sau đó dùng lại phần này. B dùng nguyên cache dense; trong bản thử, KV dense các layer sâu đã có sẵn từ lần
  prefill này. Bù dần cắt lấy KV của [0, p): vì attention nhân quả, KV đó không phụ thuộc phần sau.
- **C và bù dần** cần attention có mask theo vị trí (KV cũ và mới xen kẽ), tự viết bằng SDPA có mask (`_layer_over`),
  nên không dùng được kernel flash. Thời gian HF của hai chế độ này chậm hơn mức thực tế có thể đạt.
- **Hội thoại** (`load_document(..., history=True)` rồi `extend()` sau mỗi lượt): prefix tăng thêm phần mới của mỗi
  lượt (đáp án lượt trước + câu hỏi mới) qua mọi layer dense; token sau tài liệu không bao giờ bị cắt; `naive` hội thoại
  tiếp trên cache đã cắt; C cộng dồn tập token đã tính qua các lượt.
- **Đo thời gian vLLM:** `inmodel_prune/vllm_patch.py` thêm tùy chọn `layer_times=True` (CUDA event quanh mỗi layer,
  gồm cả chấm điểm). `scripts/bench_multiturn.py` đo:
  - TTFT, TPOT và thời gian từng layer của dense và LayerSieve;
  - ε: TTFT của câu hỏi sau trên vLLM gốc có prefix caching.

  Chi phí lượt sau của A = layer 17–27 của LayerSieve + câu hỏi qua 17 layer (ước bằng ε·17/28). Chi phí bù của B =
  layer 17–27 của dense.

## 4. Cách đo

**Kiểm tra tính đúng** (vì A, B, bù dần phải bằng chính xác một phương pháp đã biết):

- A so với LayerSieve chạy lại từ đầu (`fresh`): cùng câu trả lời, Jaccard tập token giữ ở layer cuối, sai lệch logit
  token đầu.
- Mức nhiễu nền: dense có và không có prefix cache (`dense_fresh`), cùng các chỉ số.
- Bù dần: `fill0` phải trùng A, `fill1` phải trùng dense.

**Độ chính xác:**

- Điểm theo benchmark; "số câu dense đúng còn giữ được" theo từng chế độ, giống cột "Hard" của paper.
- So cặp từng câu bằng McNemar chính xác. Với SCBench: chênh lệch so với dense trên lượt ≥ 2, trung bình đều theo task,
  bootstrap theo phiên (2000 lần, seed 0, CI 95%).

**Chi phí:**

- Token×layer sâu phải tính khi câu hỏi đến (C và bù dần so với A).
- Thời gian vLLM từng phần (E1), ghép thành tổng thời gian cả phiên theo số lượt: TTFT + 256 token × TPOT mỗi lượt.
- Bộ nhớ tính tay ở 128k.

**Ba kiểu chuyển câu hỏi** trên cùng tài liệu: `orig` (câu gốc), `same` (cùng evidence), `switch` (evidence khác, cách
câu gốc ≥ 1/4 tài liệu), `subset` (evidence là tập con, chỉ với multiquery).

## 5. Benchmark

| bộ | nguồn | tài liệu / phiên | câu hỏi | chấm |
|---|---|---|---|---|
| `data/multiq.jsonl` | InfiniteBench kv (cắt giữa 128k), RULER niah_multikey_3 @128k | 50 + 50 | orig / same (diễn đạt lại) / switch (key khác) | đáp án trong câu trả lời |
| `data/multiq2.jsonl` | RULER @128k: niah_multikey_1 (essay + 4 needle), niah_multikey_2 (~6000 needle số), niah_multiquery, qa_1 (SQuAD) | 4 × 50 | qa1: same = câu SQuAD khác trên cùng đoạn, switch = câu trên đoạn khác; mq: subset = 1 trong 4 key | RULER `all` / `part` |
| `data/scbench128k.jsonl` | SCBench (code chính thức, commit 29ef197), cắt giữa xuống 128k − max_new·lượt | 6 task × 20 phiên × ≤ 5 lượt | lịch sử = đáp án đúng lượt trước (mặc định SCBench) | `get_score_one` của SCBench |
| `data/multiq_llama.jsonl`, `data/scbench128k_llama.jsonl` | như trên, tokenizer và template của Llama-3.1-8B-UltraLong-1M | | | |

**Cách dựng.**

- `data/build_multiq.py` tách prompt tại chỗ câu hỏi bắt đầu. Phần tài liệu được tokenize một lần, dùng chung cho mọi
  câu hỏi. Câu hỏi mới được suy từ chính câu hỏi gốc (giữ template), đáp án lấy từ chính tài liệu.
- `data/build_scbench.py` dùng `create_multiturn_prompt` của SCBench. Token ghép lại đúng bằng token SCBench gửi cho
  model.
- SCBench RepoQA bị bỏ: cần sinh 1024 token mỗi lượt và tree-sitter để chấm.

## 6. Kết quả

### 6.1 Tính đúng

| kiểm tra | kết quả |
|---|---|
| A so với LayerSieve từ đầu (kv, mk3; 30 câu) | cùng câu trả lời 30/30, Jaccard trung vị 0.976, logit lệch trung vị 0.56 |
| mức nhiễu nền: dense có / không prefix cache (10 câu) | cùng câu trả lời 10/10, logit lệch trung vị 0.57 |
| A so với LayerSieve từ đầu (mk1, mk2, mq, qa1; 24 câu) | 21/24 trùng nguyên văn, 24/24 cùng điểm |
| A so với LayerSieve từ đầu trên cả hội thoại (SCBench; 58 lượt) | 49/58 trùng nguyên văn, 58/58 cùng điểm |
| bù dần: fill0 so với A, fill1 so với dense (8 + 8 câu) | trùng 8/8 mỗi đầu |

Độ lệch của A nằm đúng ở mức nhiễu bf16 của chính dense.

### 6.2 Dùng lại nguyên cache sẽ hỏng khi đổi evidence (Qwen, 128k)

Độ chính xác (%), n = 50 mỗi ô:

| | kv same | kv switch | mk3 same | mk3 switch | mk2 switch | qa1 switch |
|---|---|---|---|---|---|---|
| dense (= B) | 64 | 68 | 88 | 84 | 96 | 74 |
| A | 66 | 70 | 90 | 86 | 94 | 72 |
| C | 62 | 70 | 88 | 82 | 94 | 76 |
| naive | 50 | **18** | 86 | **26** | **22** | **24** |

Câu dense đúng còn giữ ở `switch` (A / C / naive):

- kv: 33 / 32 / 7 trên 34
- mk3: 41 / 41 / 12 trên 42
- mk2: 47 / 47 / 11 trên 48
- qa1: 35 / 35 / 11 trên 37

McNemar A so với naive ở `switch`: p từ 9e-7 (kv) tới 3e-11 (mk2). A và C ngang dense ở mọi ô.

**Khi naive không hỏng:**

- evidence không đổi: mk3 same 86 (A 90); mq same 91 (A 82: câu diễn đạt lại làm việc chọn kém hơn câu gốc); mq subset
  100;
- evidence nổi bật hẳn so với nền: mk1 (4 needle trong essay), naive switch 100.

Vậy chỉ cần phục hồi khi câu hỏi mới cần evidence khác và tài liệu có nhiều đoạn giống nhau cạnh tranh.

### 6.3 C: phục hồi từng phần

| | kv same | kv switch | mk3 same | mk3 switch |
|---|---|---|---|---|
| câu dense đúng còn giữ: A / C | 30 / 29 (32) | 33 / 32 (34) | 44 / 44 (44) | 41 / 41 (42) |
| unit mới (% tài liệu) | 9.0 | 13.7 | 8.8 | 12.3 |
| token×layer sâu của C / A | 0.31 | 0.47 | 0.30 | 0.42 |

- C ngang A trên RULER, kv và qa1 (McNemar p ≥ 0.5) mà chỉ tính 30–47% phần sâu của A.
- Thời gian HF của C lại chậm hơn A (3.1–5.0 s so với 2.6–3.4 s) vì attention có mask không dùng được kernel flash.
  Cần kernel block-sparse mới lấy được phần lợi.
- Lỗi KV trùng (unit bị bỏ giữa chừng rồi được chọn lại) đã sửa ngày 31/10. Chạy lại: số câu đúng không đổi.

### 6.4 Bù dần

Câu dense đúng còn giữ:

| | A (p = 0) | p = 1/4 | p = 1/2 | p = 3/4 |
|---|---|---|---|---|
| kv same / switch (32 / 34) | 30 / 33 | 31 / 33 | 31 / 32 | 32 / 33 |
| mk3 same / switch (44 / 42) | 44 / 41 | 44 / 41 | 44 / 41 | 44 / 41 |
| mk2 same / switch (49 / 48) | 47 / 47 | 49 / 47 | 48 / 47 | 49 / 47 |
| qa1 same / switch (33 / 37) | 32 / 35 | 31 / 35 | 32 / 35 | 33 / 37 |
| phần sâu tính khi hỏi / A | 1 | 0.75 | 0.50 | 0.25 |

Độ chính xác ngang A và dense ở mọi p; phần phải tính khi câu hỏi đến giảm đúng theo (1 − p). Ghép với E1 ở 128k, prefill
lượt sau ước tính:

| p | 0 (A) | 1/4 | 1/2 | 3/4 | 1 (B) |
|---|---|---|---|---|---|
| prefill lượt sau | ~1.2 s | ~0.93 s | ~0.65 s | ~0.36 s | ~0.13 s |

### 6.5 SCBench 128k có lịch sử hội thoại (Qwen, 120 phiên)

Điểm lượt ≥ 2 (%):

| task (n lượt) | dense (= B) | A | C | naive |
|---|---|---|---|---|
| kv (80) | 88.8 | 88.8 | 87.5 | 66.2 |
| prefix_suffix (80) | 81.2 | 77.5 | 80.0 | 17.5 |
| vt (80) | 45.8 | 42.3 | 41.3 | 34.3 |
| qa_eng (80) | 30.0 | 30.0 | 28.8 | 18.8 |
| choice_eng (60) | 60.0 | 60.0 | 60.0 | 48.3 |
| mf (80) | 13.8 | 15.0 | 11.2 | 12.5 |

Chênh lệch so với dense (trung bình đều theo task, CI 95%):

| | chênh lệch | CI 95% |
|---|---|---|
| A | −1.0 | [−2.4, +0.4] |
| C | **−1.8** | [−3.2, −0.5] |
| naive | **−20.3** | [−24.0, −16.7] |

- Trên benchmark chuẩn có lịch sử thật, naive mất 20 điểm; A ngang dense.
- C lần đầu thua dense có ý nghĩa, dù nhỏ: trạng thái cũ hoặc việc chọn một lần ở layer 17, cộng dồn qua 5 lượt, bắt
  đầu gây hại (rõ ở mf, vt).
- qa_eng và mf có dense thấp ở 128k nên ít thông tin.

### 6.6 Thời gian (E1, vLLM, **H200 dùng chung**, 2 prompt mỗi độ dài)

| | 128k | 256k |
|---|---|---|
| TTFT dense / LayerSieve | 17.1 / 12.3 s | 73.2 / 52.3 s |
| layer 17–27: dense / LayerSieve | 5.8 / 1.1 s | 28.7 / 3.9 s |
| câu hỏi sau, dense + prefix cache (ε) | 0.13 s | 0.17 s |
| câu hỏi sau, A | 1.21 s (7% TTFT dense) | 4.04 s (5.5%) |
| bù cache B, một lần, chạy nền | 5.8 s (34%) | 28.7 s (39%) |

Tổng thời gian cả phiên (s; lượt sau của dense và B dùng TPOT dense):

| | dense + prefix cache | LayerSieve vứt cache | A | B |
|---|---|---|---|---|
| 128k, 1 lượt | 19.5 | 14.5 | 14.5 | 14.5 |
| 128k, 3 lượt | 24.6 | 43.6 | 21.4 | **19.7** |
| 128k, 10 lượt | 42.6 | 145.4 | 45.4 | **37.7** |
| 256k, 3 lượt | 87.4 | 166.7 | 70.1 | **65.2** |
| 256k, 10 lượt | 121.1 | 555.7 | 121.1 | **98.8** |

- GPU bị job khác dùng 80–100% nên số tuyệt đối chậm hơn Bảng IV của paper khoảng 1.3–1.5×.
- Tỷ lệ (A ≈ 5–7%, B ≈ 34–39% TTFT dense) khớp ước lượng từ Bảng IV.
- B thắng ở mọi số lượt nếu giữa hai lượt có đủ thời gian rảnh để bù.
- A thắng dense tới khoảng 5–10 lượt.

**Bộ nhớ ở 128k (tính tay):**

| | bộ nhớ |
|---|---|
| KV dense | ~7.5 GB |
| LayerSieve có KV sieving | ~1.9 GB |
| A: KV nông đầy đủ + checkpoint + KV sâu đã sàng | ~6.2 GB |
| B | ~7.5 GB |

### 6.7 Model thứ hai: Llama-3.1-8B-UltraLong-1M (RULER / kv: 300 tài liệu, xong; SCBench đang chạy)

Lịch cắt theo quy tắc độ sâu tương đối của paper (L = 32): ℓ0 = 20, ρ = 0.361, KV sieving tại layer 25. Paper chưa kiểm
chứng lịch này trên Llama. Bản HF cần ~62 GB lúc cao nhất (Qwen 39 GB).

Câu dense đúng còn giữ (A / C / naive):

| | same | switch | McNemar A vs naive (switch) |
|---|---|---|---|
| kv | 18 / 21 / 15 (21) | 12 / 12 / **3** (13) | 10:2, p = 0.04 |
| mk3 | 23 / 26 / 22 (28) | 24 / 25 / **5** (25) | 23:1, p = 3e-6 |
| mk2 | 46 / 50 / 41 (50) | 35 / 37 / **8** (40) | 29:5, p = 4e-5 |
| mk1 | 47 / 47 / 45 (47) | 46 / 46 / 45 (46) | 2:1 |
| mq (cột switch là subset) | 46 / 46 / 46 (46) | 44 / 44 / 44 (44) | 0:1 |
| qa1 | 31 / 31 / 29 (31) | 32 / 32 / **16** (32) | 16:0, p = 3e-5 |

- Hiện tượng lặp lại trên model thứ hai: naive sụp khi đổi evidence (kv, mk3, mk2, qa1), không sụp với mk1 và mq; A và C
  giữ được.
- A = LayerSieve từ đầu: cùng câu trả lời 36/36.
- Dense của Llama yếu hơn Qwen ở 128k trên kv và mk3 (42%, 56%).
- Bản thân LayerSieve với lịch cắt này mất vài câu trên Llama, ví dụ mk2 orig: dense 98, A 86; mk3 same: A giữ 23/28. A
  vẫn trùng LayerSieve từ đầu, nên phần mất đến từ lịch cắt chứ không từ việc phục hồi. C giữ nhiều hơn A ở `same`.

### 6.8 Tốc độ từng phương pháp

**vLLM (E1; H200 dùng chung; 128k / 256k).**

| phương pháp | TTFT lượt đầu | prefill mỗi lượt sau | tính nền (một lần) | TPOT |
|---|---|---|---|---|
| dense + prefix cache | 17.1 / 73.2 s | **0.13 / 0.17 s** | – | 9.5 / 18.1 ms |
| LayerSieve vứt cache | 12.3 / 52.3 s | 12.3 / 52.3 s | – | 8.7 / 12.7 ms |
| A | 12.3 / 52.3 s | **1.21 / 4.04 s** | – | 8.7 / 12.7 ms |
| B (đã bù xong) | 12.3 / 52.3 s | **0.13 / 0.17 s** | **5.8 / 28.7 s** | 9.5 / 18.1 ms (KV đầy đủ) |
| bù dần p = 1/4, 1/2, 3/4 (ước lượng, chỉ có 128k) | 12.3 s | ~0.93 / 0.65 / 0.36 s | 1/4, 1/2, 3/4 của 5.8 s | 8.7–9.5 ms |
| C, naive | chưa đo trên vLLM | | | |

Tổng thời gian cả phiên theo số lượt: xem 6.6.

**HF bản thử** (trung vị prefill câu hỏi sau, kv + mk3, 128k, GPU dùng chung):

| chế độ | naive | dense (= B) | A | C | bù dần 1/4 / 1/2 / 3/4 | LayerSieve từ đầu | dense từ đầu |
|---|---|---|---|---|---|---|---|
| prefill | 0.22 s | 0.81 s | 3.32 s | 4.65 s | 9.3 / 8.3 / 5.5 s | 26.0 s | ~38 s |

C và bù dần dùng attention có mask tự viết (`_layer_over`) và phần chấm điểm trên toàn bộ prefix, không có kernel flash,
nên chậm hơn A trên HF dù tính ít hơn (C: 30–47% token×layer sâu của A; bù dần: 1 − p). Thời gian HF của hai chế độ này
không đại diện.

**Tóm tắt.**

- Lượt sau nhanh nhất là B đã bù xong (~0.13 s, ngang dense có prefix cache), rồi bù dần, rồi A (~1.2 s ở 128k).
- Lượt đầu mọi chế độ nhanh như LayerSieve (~1.4× dense).
- Cả phiên: B tốt nhất; A thắng dense tới khoảng 5–10 lượt; vứt cache tệ nhất.
- Số tuyệt đối sẽ đổi khi đo lại trên GPU trống.

## 7. Giới hạn và việc còn lại

- **E1 chưa có bản đo sạch.** Hàng đợi `scripts/run_e1_clean.sh` chờ GPU không còn tiến trình nào trong 3 phút (tối
  đa 72 h).
- **Thời gian của C và bù dần trên vLLM chưa đo.** Bản HF dùng attention có mask nên không đại diện; bù dần ước được từ
  thời gian từng layer, C cần kernel block-sparse.
- **B chưa có bản chạy nền thật trong engine**, cũng chưa xen vào decode như Raincheck. Chi phí bù lấy từ thời gian các
  layer sâu của dense.
- **Một request mỗi lúc, batch 1.** Trên server bận, phần bù nền giành GPU với request khác.
- **Llama (RULER, kv, SCBench) đang chạy.** Lịch cắt của Llama suy theo quy tắc của paper, chưa kiểm chứng; khác
  SessionSieve (ℓ0 = 19, cắt một lần).
- **Còn thiếu:** SCBench RepoQA, sparse attention cho phần dense (pattern phải không phụ thuộc câu hỏi để dùng lại được
  cache), ngữ cảnh 256k và 1M cho phần độ chính xác.

## 8. File và lệnh

| việc | code | kết quả |
|---|---|---|
| dữ liệu nhiều câu hỏi | `data/build_multiq.py` | `data/multiq.jsonl`, `data/multiq2.jsonl`, `data/multiq_llama.jsonl` |
| dữ liệu SCBench | `data/build_scbench.py` | `data/scbench128k.jsonl`, `data/scbench128k_llama.jsonl` |
| E2, C, bù dần, mở rộng | `scripts/multiturn_hf.py` | `hf.jsonl`, `hf_partial_fix.jsonl`, `hf2.jsonl`, `hf_fill_*.jsonl`, `hf_llama.jsonl` |
| SCBench | `scripts/session_hf.py` | `scbench_qwen.jsonl`, `scbench_llama.jsonl` |
| E1 | `scripts/bench_multiturn.py`, `scripts/run_e1_clean.sh` | `layers.jsonl`, `prefix.jsonl`, `clean/` |
| hàng đợi | `scripts/run_multiturn_ext.sh`, `scripts/run_multiturn_resume.sh` | `ext.log` |
| bảng | `scripts/analysis/multiturn_table.py` | `MULTITURN.md` (ghi chép theo từng lần chạy) |

```bash
cd /root/team_1/hieurezdev/PromptPruningPaper/VAR_LLM_Inference_Paper
.venv/bin/python scripts/multiturn_hf.py --data data/multiq.jsonl --verify 5 --out results/multiturn/hf.jsonl
.venv/bin/python scripts/multiturn_hf.py --data data/multiq.jsonl --modes dense,resieve,fill0.25,fill0.5,fill0.75 \
    --verify 0 --out results/multiturn/hf_fill_multiq.jsonl
.venv/bin/python scripts/session_hf.py --data data/scbench128k.jsonl --verify 2 --out results/multiturn/scbench_qwen.jsonl
.venv/bin/python scripts/bench_multiturn.py --mode layers --lengths 131072 --gpu-util 0.4 --out results/multiturn/layers.jsonl
.venv/bin/python scripts/analysis/multiturn_table.py --acc results/multiturn/hf.jsonl \
    --layers results/multiturn/layers.jsonl --prefix results/multiturn/prefix.jsonl
```
