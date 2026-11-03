# Nguyên nhân tăng điểm LongDialogue: báo cáo đêm 21–22/10/2026

Mô hình: Qwen2.5-7B-Instruct-1M. Số liệu là LD50 (50 câu LongDialogue đầu), chấm theo tên (first-name), trừ khi ghi khác.

- Code: `scripts/ld_move_probe.py` (các phép thử oracle), `scripts/recall_eval.py` (`--move`, `--hide-from`),
  `scripts/gem_ls.py`, `inmodel_prune/vllm_patch.py` (`n_score`).
- Chấm điểm: `scripts/analysis/ld_cause.py`, `ld50b_cause.py`, `ld_first.py`.
- "Oracle" nghĩa là dùng vị trí biết trước của `$$MASK$$` để chẩn đoán. Đây không phải phương pháp.

## 1. Kết luận

**Khoảng cách RoPE giữa câu hỏi và bằng chứng là nguyên nhân khiến LD thấp. Giả thuyết ban đầu là đúng.**

Chỉ quay lại RoPE của key thuộc đoạn bằng chứng về vị trí ngay trước câu hỏi, **ở mọi layer**, cho token câu hỏi và token câu
trả lời (không chép, không tính lại context, không đổi gì khác), là lấy lại toàn bộ lợi ích: **18 → 34**.

Các cách sửa RoPE trước đây thất bại vì chúng gỡ khoảng cách cho **mọi** key (cả hơn 100k token nhiễu), hoặc làm hỏng cấu trúc vị
trí cục bộ. Ở đây chỉ đoạn bằng chứng được kéo lại gần, khoảng cách nội bộ trong đoạn giữ nguyên.

## 2. Các phép thử nhân quả

Cùng một đoạn bằng chứng: ±1024 token quanh chỗ `$$MASK$$`, cắt ở ranh giới dòng.

| Phép thử | Gì thay đổi | LD50 | Bằng chứng xa (> 48k) |
|---|---|---|---|
| Dense | — | 18 | 12,5 |
| Dời đoạn tới ngay trước câu hỏi | thứ tự (mã hóa lại ở gần) | 34 | 31 |
| Dời một đoạn ngẫu nhiên (đối chứng) | thứ tự | 14 | 16 |
| copy_near | thêm bản chép ở gần | **40** | 47 |
| copy_prefix | bản chép ở gần, nhưng khi mã hóa chỉ thấy phần đứng trước nó trong bản gốc | 34 | 38 |
| copy_far17 | bản chép mã hóa ở gần, nhưng ở layer ≥ 17 câu hỏi đọc nó từ vị trí xa gốc | 24 | 22 |
| **rerot0** | **chỉ quay RoPE của key đoạn gốc về gần, mọi layer** | **34** | 34 |
| rerot17 | như trên, chỉ ở layer ≥ 17 | 16 | 16 |
| rerot17 với đoạn ngẫu nhiên (đối chứng) | | 18 | 12,5 |
| Copy chỉ hiện ở layer < 17 | | 20 | |
| Copy chỉ hiện ở layer ≥ 17 | | 34 | |

**Đọc bảng:**
- Lợi ích đến từ **vị trí**, không đến từ việc nhân đôi nội dung (dời đoạn ≈ copy) hay từ việc có thêm ngữ cảnh khi mã hóa
  (copy_prefix ≈ copy_near).
- Chỉ cần đổi RoPE là đủ (rerot0 = 34). Nhưng việc này phải có ở cả các layer đầu (rerot17 = 16): câu hỏi phải đọc được bằng
  chứng ở gần ngay từ các layer đầu để tích hợp nó vào biểu diễn của mình.
- Nếu bằng chứng đã được mã hóa lại ở gần (copy), thì chỉ cần câu hỏi thấy nó ở layer ≥ 17 (34). Thấy ở layer < 17 thôi thì
  không đủ (20).

## 3. Bằng chứng phải gần tới mức nào

Dời đoạn MASK tới các độ sâu khác nhau trước câu hỏi:

| Độ sâu | 0 | 1k | 2k | 4k | 8k | 16k | 48k | (dense) |
|---|---|---|---|---|---|---|---|---|
| LD50 | **34** | 26 | 30 | 24 | 28 | 18 | 18 | 18 |

- Trong khoảng 8k token: tăng +6 đến +12; sát câu hỏi: +16.
- Từ 16k trở đi: không tăng.

Khớp với số đo độ cục bộ của attention trên LD (profile 2 prompt):
- 341/784 head dồn ≥ 75% attention (không tính sink) vào 4k token gần nhất.
- Chỉ 92 head dồn ≥ 50% ra xa hơn 32k; nhóm này gồm phần lớn các retrieval head.

## 4. Hệ quả cho phương pháp

**(a) Quay RoPE có chọn lọc (`recall_eval.py --move`).**
- Chọn đoạn bằng attention của 16 token cuối ở layer 17, rồi chỉ chạy lại khoảng 128 token câu hỏi với key của đoạn được chọn
  quay về sát câu hỏi.
- Không chép, không tính lại context, nên chi phí gần như bằng 0 so với prefill.

| | LD50 |
|---|---|
| Dense | 18 |
| `--move` với bộ chọn thật, khoảng 8 mảnh, tổng 2048 token | 26 (+8) |
| Oracle một khối liền (rerot0) | 34 |
| Một khối liền 2048 token quanh khối điểm cao nhất (`--pad 992`) | 24 (+6) |

Bộ chọn tìm đúng vùng MASK ở 49/50 câu. Một khối liền không tốt hơn nhiều mảnh (24 so với 26), nên khoảng cách tới oracle
(34) không đến từ tính liền mạch. Có thể đến từ việc đoạn chọn không căn đúng vào MASK (oracle là ±1024 token quanh chính
MASK, cắt theo dòng), hoặc từ việc vùng 128 token cuối được giữ ở vị trí thật xen giữa đoạn và câu hỏi. Cần kiểm tra.

**(b) LayerSieve: chấm điểm bằng 16 token cuối thay vì 128** (`--n-score 16`, vLLM, LD200 đủ 200 câu):

| | Tất cả | Bằng chứng xa |
|---|---|---|
| Dense (vLLM) | 14,5 | 12,1 |
| LayerSieve mặc định | 10,5 | 5,6 |
| LayerSieve, chấm bằng 16 token | **15,0** | **14,5** |

So với bản mặc định: +4,5 điểm, CI [0; +9], p một phía 0,034. LayerSieve trên LD từ thua dense thành ngang dense. Có thể đưa vào
paper.

**(c) GemFilter** (chạy lại mô hình chỉ trên các token được chọn):
- Bản gốc (1 query cuối, 2048 token): KV 0, multikey 0, LD 12, math 20.
- Với bộ chọn LayerSieve (unit, giữ 25%): KV 95, math 90 (cao hơn dense 85), nhưng LD 8. Mở rộng ±16 unit lân cận thì LD lên 18.
- Unit của LayerSieve trên LD chỉ dài khoảng 7 token (một câu thoại), quá vụn để mã hóa lại riêng.

## 5. Những gì đã bác bỏ trong đêm

- **Copy với budget lớn hơn không hết hại trên math/cwe.** math 50 → 45 khi budget tăng từ 2048 lên 32768; cwe 12 → 7. Hiện
  tượng neo không phải do chép thiếu: mô hình trả lời theo phần nằm gần câu hỏi.
- **Lợi ích không đến từ việc mã hóa bằng chứng sau khi đã đọc toàn bộ câu chuyện.** Dời đoạn tới độ sâu 16k: không tăng.
  copy_prefix vẫn đạt 34.

## 6. Kết quả hoàn tất lúc gần sáng

**`--move` trên các task khác (20 câu, bộ chọn thật):**

| Task | Dense | Copy | `--move` |
|---|---|---|---|
| LD50 | 18 | 36 | 26 |
| KV20 | 80 | 95 | 85 |
| **math_find** | 85 | **50** | **85** |
| **vt** | 97 | 96 | **59** |
| cwe | 24,5 | 12 | 9 |

**`--move` không làm hại math_find (85 = dense), trong khi copy làm sập (50).** Copy đặt một bản sao nội dung sát câu hỏi nên
mô hình chỉ đọc phần đó. `--move` không thêm nội dung nào, chỉ kéo đoạn được chọn lại gần về vị trí, nên mô hình vẫn đọc được
cả list.

**Nhưng `--move` làm hại vt nặng (59 so với 97) và cwe (9 so với 24,5).** Bài vt yêu cầu lần theo một chuỗi phép gán biến rải
khắp context. Dời vị trí của một số đoạn trong chuỗi làm sai khoảng cách tương đối giữa đoạn được dời và đoạn không được dời,
trong khi mô hình dùng chính khoảng cách đó để lần theo chuỗi. Vậy quay RoPE có chọn lọc giữ được nội dung, nhưng làm vỡ quan
hệ vị trí giữa các phần bằng chứng nằm ở nhiều chỗ.

**Lặp lại trên LD 51–100** (dense HF 22):
- dời đoạn MASK 30 (+8, CI [−4; +20]);
- rerot17 24 (+2).

Cùng chiều với LD 1–50 nhưng mức tăng nhỏ hơn.

## 7. Bước tiếp theo đề xuất

1. Đóng khoảng cách giữa `--move` với bộ chọn thật (26) và oracle (34). Một khối liền không giúp; cần thử căn đoạn theo
   token có điểm cao nhất, và đặt đoạn sát token câu hỏi thay vì trước 128 token cuối.
2. Sửa tác hại trên vt/cwe. Ý tưởng: chỉ quay RoPE của đoạn được chọn khi đoạn đó là một vùng bằng chứng tập trung; hoặc
   giữ nguyên thứ tự vị trí giữa các đoạn được dời và rút ngắn khoảng cách một cách đồng đều (co toàn bộ phần context giữa
   đoạn và câu hỏi), thay vì xếp chồng các đoạn ngay trước câu hỏi.
3. Ghép vào LayerSieve/vLLM: điểm ở layer 17 có sẵn, và việc quay key chỉ ảnh hưởng tới lượt decode câu hỏi.
4. Đưa kết quả `n_score = 16` của LayerSieve vào paper, sau khi kiểm tra lại trên RULER.

## 8. 22/10 (chiều): bước chọn và mép cửa sổ

**Đo độ chính xác của bước chọn** (`scripts/selector_probe.py`, `results/selfext/selprobe/`, đánh giá offline bằng
`scripts/analysis/selector_eval.py`). Tỉ lệ prompt có khối bằng chứng đứng hạng 1:

| Cách chấm | LD | KV | mk3 | mv |
|---|---|---|---|---|
| mean qua head, L17, 16 token (bộ chọn cũ của `--move`) | 0,00 | 0,00 | 0,45 | 0,85 |
| max qua head, L17, 16 token | 0,93 | 0,00 | 0,60 | 1,00 |
| retrieval head (`heads_7b1m.json`, ngưỡng 0,3) | 1,00 | 0,80 | 1,00 | 1,00 |

Retrieval head được hiệu chỉnh trên prompt KV tổng hợp 8k (`kv_8000_*`), không trùng tập đánh giá. Bộ chọn mới:
`recall_eval.py --agg retr`. Bản dời một khối liền cũ (`ld50_move_sel_one`, 24) chưa lần nào chứa MASK (0/50).

**Dời một đoạn bằng bộ chọn retrieval head (`--pad 992 --agg retr`).** Đoạn trùng 73–99% với cửa sổ oracle, chứa MASK
50/50. Nhưng LD50 chỉ 26, so với oracle `rerot0` 34; oracle đúng thêm 4 câu, chiều ngược lại 0 câu.

**Nguyên nhân là mép cửa sổ.** Oracle căn cửa sổ theo dòng; `--move` cắt theo khối 64 token, nên đoạn bắt đầu giữa một
câu thoại. Kiểm tra cả hai chiều trên LD50 (chấm theo tên đầu tiên; dense 18):

| Chạy | Điểm | So với bản đối chiếu |
|---|---|---|
| `--move` (retr, cắt theo khối) | 26 | |
| `--move` + `--snap` (căn theo dòng) | **32** (p = 0,013 so với dense) | đúng thêm 3, sai thêm 0 |
| oracle `rerot0` (căn theo dòng) | 34 | đúng thêm 1 so với `--snap` |
| oracle `rerot0` bỏ căn dòng (`--nosnap`) | 28 | mất 3, thêm 0 |

Vậy với LD, khoảng cách giữa bộ chọn thật và oracle đã gần như đóng (32 so với 34) khi đoạn được dời gồm trọn các dòng.
`--select` (giữ câu trả lời có log-prob cao hơn) làm mất phần lợi: 26 thay vì 32.

**Trên các task khác** (dời một đoạn, chưa căn dòng; câu trả lời sau khi dời, không `--select`):
mk3 128k 100 so với dense 80 (tốt hơn 4, kém hơn 0); KV20 75–90; math_find 85 = dense; vt 55 (dense 97);
cwe 12,5 (24,5); mv 75 (85). Bài có bằng chứng rải khắp context vẫn bị hại, kể cả khi chỉ dời một đoạn: tác hại không
đến từ đoạn nhiễu bị dời theo, mà từ việc chỉ một phần bằng chứng được kéo lại gần. Đang chạy `--snap` trên các task
này (`scripts/run_snap2.sh`).

## 9. Kết quả 22/10 chiều: LD200, phân tích theo ô, bộ chọn `rel`

**Sửa lại mục 8:** trên vt, khối attention cao nhất là *ví dụ mẫu* ở đầu prompt (19/20 câu), nên lần dời một đoạn
chuyển 0% dòng của chuỗi thật. Kết luận "chỉ một phần bằng chứng được kéo lại gần" ở mục 8 sai với vt.

**LD200, `--move` một đoạn (retr) + `--snap`** (câu trả lời sau khi dời, metric tên đầu tiên, so với HF dense):
21,5 so với 14,0 (+7,5, p=.007). Xa >48k: 22,6 so với 11,3. Gần <32k: 25,5 so với 27,5. Scorer chính thức 20,0 so với 9,5.
Đoạn dời chứa câu MASK ở 199/200 câu. Cứu được 23/172 câu dense sai, làm hỏng 8/28 câu dense đúng (5 câu ở xa). Với
`--select` còn 18,5. Copy cũng trên LD200 đạt 34,0.
Ở 157 câu vẫn sai sau khi dời, tên đúng xuất hiện trong đoạn đã dời ở cả 157 câu. 73 câu chọn một tên khác cũng có
trong đoạn, 84 câu chọn một tên không có trong đoạn. Như vậy ở các câu này, bằng chứng vừa nằm gần vừa được attention
nhìn tới, nhưng mô hình vẫn không suy ra được tên.

**Phân tích theo ô** (20 câu mỗi task, LD 30; "nhìn đúng" = khối vàng là top-1 và mọi đơn vị vàng nằm trong top-8 của
retr_w16):

| task | bộ chọn | đơn vị vàng được dời | nhìn đúng, dense sai → cứu | dense đúng → làm hỏng |
|---|---|---|---|---|
| LD30 | một đoạn | 100% | 5/25 | 0/5 |
| mk3 | một đoạn / rel | 100% / 100% | 4/4 / 4/4 | 0/16 / 0/16 |
| KV20 | một đoạn / rel | 75% / 90% | 2/3 / 1/3 | 3/16 / 3/16 |
| mv | một đoạn / rel | 25% / 81% | 2/11 / 3/11 | 4/9 / 6/9 |
| vt | một đoạn / rel | 0% / 79% | — / — | 13/17 / 17/17 |

Bộ chọn `rel` (L14, max/16 + contrast/128, khối ≥0,3 so với khối cao nhất, pad 32, `--snap`) dời được phần lớn bằng
chứng của mv và vt (5–6 đoạn mỗi câu), nhưng tác hại không giảm mà còn tăng: vt hỏng 17/17. Vậy trên vt, việc dời
chính các dòng bằng chứng (nhiều đoạn, không phải ví dụ mẫu) cũng làm mô hình hỏng. Các task chỉ có một chỗ bằng
chứng (LD, mk3) được lợi. Các task có bằng chứng ở nhiều chỗ thì bị hại khi dời nhiều đoạn về sát nhau.

## 10. Vì sao dời nhiều đoạn lại làm hỏng: các đoạn bị dồn sát nhau (22/10 tối)

Dùng đoạn oracle, tức đúng các câu bằng chứng (`results/selfext/oracle_spans.json`, `--spans-file`), 20 câu mỗi task,
`pred_recall`, so với dense (vt 97, mv 85). "Cứu" = câu dense sai được sửa đúng, "hỏng" = câu dense đúng bị làm sai:

| cách đặt vị trí | vt | cứu / hỏng | mv | cứu / hỏng |
|---|---|---|---|---|
| pack (sát nhau, mặc định) | 82 | 0/3, 13/17 | 67,5 | 0/11, 11/9 |
| perm (hoán vị, không trùng vị trí) | 84 | 0/3, 12/17 | 68,8 | 2/11, 13/9 |
| `--vgap 4096` (4k vị trí trống giữa hai đoạn) | 91 | 0/3, 6/17 | 86,2 | 5/11, 4/9 |
| `--vgap 16384` | **98** | 2/3, 1/17 | **95,0** | **9/11**, 2/9 |
| copy oracle (mã hóa lại thành văn bản) | 100 | 3/3, 0/17 | 83,8 | 7/11, 5/9 |

Nguyên nhân không phải bằng chứng còn sót ở xa (oracle dời đủ 100%), cũng không phải trùng vị trí (perm không đổi
gì). Nguyên nhân là các đoạn được mã hóa độc lập bị đặt liền nhau: mô hình đọc chúng như một chuỗi liền, gộp hoặc bỏ
mục (mv liệt kê 2–3 giá trị rồi dừng; vt bỏ một biến hoặc bịa thêm tên biến). Khi để trống 16k vị trí giữa các đoạn,
mv được cứu 9/11 câu thuộc nhóm "nhìn đúng nhưng trả lời sai", tốt hơn cả copy. Hàng đợi tiếp theo
(`scripts/run_gap2.sh`) chạy bộ chọn thật `--sel rel --snap --vgap 16384` trên mọi task, kèm cwe, math_find và một
sweep oracle gap 8k/32k trên mv.

## 11. Bộ chọn thật + khoảng trống, thứ tự theo điểm (22/10 đêm)

`--sel rel --snap --move`, 20 câu mỗi task, `pred_recall`; dense vt 97, mv 85, KV 80, mk3 80, LD50 18 (tên đầu tiên):

| | vt | mv | KV20 | mk3 | LD50 | cwe | math_find |
|---|---|---|---|---|---|---|---|
| pack (dồn sát) | 58 | 80 | 75 | 100 | (một đoạn 32) | (một đoạn 12,5) | (một đoạn 85) |
| `--vgap 16384`, thứ tự văn bản | 88 | 86 | **30** | 80 | 22 | 19,5 | 85 |
| `--vgap 16384 --vorder score` | **34** | 77,5 | **15** | 100 | – | – | – |

- Mv oracle theo khoảng trống: 4k 86, 8k 89, 16k 95, 32k 95.
- Khoảng trống rộng sửa được vt và mv, nhưng làm KV sụp: mô hình trả về key hoặc value của cặp khác trong các đoạn đã dời.
- Xếp đoạn theo điểm, đoạn mạnh nhất đặt gần câu hỏi nhất, làm vt sụp. Thứ tự văn bản của các mảnh bằng chứng mang
  thông tin (chuỗi gán của vt, danh sách giá trị của mv).
- KV lại cần các cặp nằm sát nhau. Vậy không có một kiểu đặt vị trí nào hợp cho mọi task: thứ tự và khoảng cách tương
  đối giữa các mảnh bằng chứng là một phần của thông tin.
- Đang chạy `--vpos scale` (`scripts/run_scale.sh`): giữ thứ tự và tỉ lệ khoảng cách, chỉ nhân mọi khoảng cách tới câu
  hỏi với 0,25 hoặc 0,1.

## 12. Thu nhỏ khoảng cách giữ nguyên hình học (`--vpos scale`) và phép thử trùng vị trí (23/10 sáng)

Bộ chọn thật `--sel rel --snap --move`, 20 câu mỗi task (LD50 dùng metric tên đầu tiên):

| cách đặt | vt (dense 97) | mv (85) | KV20 (80) | mk3 (80) | LD50 (18) |
|---|---|---|---|---|---|
| pack (dồn sát) | 58 | 80 | 75 | 100 | một đoạn: 32 |
| perm (dồn sát, không trùng vị trí) | – | – | 75 | – | – |
| `--vgap 16384` | 88 | 86 | 30 | 80 | 22 |
| `--vgap 16384 --vorder score` | 34 | 77,5 | 15 | 100 | – |
| `--vpos scale --vscale 0.25` | 80 | 86 | 5 | 70 | 14 |
| `--vpos scale --vscale 0.1` | 67 | 84 | 50 | 85 | 12 |
| `--vpos scaleperm --vscale 0.25` | 83 | 86 | 5 | 70 | – |

- Trùng vị trí không phải nguyên nhân: KV perm bằng pack (75), scaleperm bằng scale (5).
- Không có một cách đặt vị trí nào hợp cho mọi task:
  - Bằng chứng ở một chỗ: dời một đoạn dồn sát là tốt nhất (LD200 +7,5, p=.007; mk3 100).
  - Bằng chứng ở nhiều chỗ, nếu biết đúng các mảnh (oracle): để khoảng trống giữa chúng thì cứu được (mv 95, vt 98).
  - Bộ chọn thật còn kéo theo các đoạn không phải bằng chứng (ví dụ mẫu của vt; header và đuôi JSON của KV). Mọi cách
    trải các đoạn ra đều phá KV: mô hình trả về key hoặc value của cặp khác.

## 13. Mô hình mất bằng chứng ở đâu: đầu dò tại token quyết định và activation patching (23/10)

`scripts/use_probe.py` ép câu trả lời dense đến đúng chỗ mô hình bắt đầu viết tên (token quyết định), rồi đo ở token
đó: logit lens, mức đóng góp trực tiếp vào logit (DLA) của từng head và MLP, và lượng attention vào các chỗ nhắc tên
đúng (gần hoặc xa MASK), vào tên dense đã trả lời, và vào MASK. Đánh giá bằng `scripts/analysis/use_probe_eval.py`.
Dữ liệu LD200: 28 câu dense đúng, 172 câu sai.

- Mức attention vào MASK như nhau ở câu đúng và câu sai (retrieval heads khoảng 0,02): mô hình đọc đúng cảnh.
- Ở 75% số câu sai, retrieval heads nhìn vào tên sai nhiều hơn tên đúng (0,031 so với 0,005–0,010). Ở câu đúng,
  attention vào tên đúng cao gấp khoảng 3 lần so với câu sai.
- Tên sai là một nhân vật khác trong cùng cảnh (cách MASK dưới khoảng 2k token) ở 62% số câu.
- Logit lens trước layer 24 không cho thông tin: tên đúng vượt tên sai ngay từ layer 0, như nhau ở câu đúng và câu
  sai, tức chỉ là thiên lệch sẵn có của token. Biên độ chỉ sụp ở layer 24–27.

**Trên lần chạy `--move`** (cùng 200 câu; tên sai = tên dense đã trả lời):

| ô | n | gần-tên-đúng dense → move | tên sai dense → move | tên đúng > tên sai |
|---|---|---|---|---|
| được cứu | 23 | 0,009 → 0,050 | 0,037 → 0,022 | 30% → 100% |
| vẫn sai | 148 | 0,0045 → 0,013 | 0,030 → 0,025 | 26% → 45% |

**Activation patching** (`scripts/use_patch.py`, trên 23 câu được cứu). Margin = log p(tên đúng) − log p(tên dense
đã trả lời): dense −2,27, move +1,75. Phần margin hồi phục, trung vị theo câu:

- Cấy trạng thái move vào dense chỉ ở hàng quyết định, sau layer l: 0 tới tận l=17, rồi 0,14 (l=19), 0,37 (l=22),
  0,78 (l=24), 1,0 (l=27). Hàng quyết định gom thông tin "là ai" dần dần trong các layer 18–27.
- Ngược lại, cấy trạng thái dense vào move ở mọi hàng câu hỏi sau layer l, các layer sau đó vẫn dùng vị trí đã dời:
  l≤3 hồi phục 1,0; l=14 còn 0,48; l=17 còn 0,26; l=24 còn 0,06. Việc đọc cảnh ở gần trong các layer 4–17 đóng góp
  khoảng một nửa hiệu quả, các layer 17–27 đóng góp nửa còn lại. Điều này khớp với kết quả cũ: rerot17, chỉ dời từ
  layer 17 trở lên, không cứu được.
- Hình ảnh: khoảng cách RoPE làm câu hỏi đọc cảnh yếu đi ở các layer giữa. Đây cũng là chỗ có hình phạt khoảng cách
  (rope_logit_profile: layers < 17, các chiều quay chậm). Vì vậy truy vấn tại token quyết định, hình thành ở layer
  18–27, trỏ vào nhầm nhân vật.

**Đối chứng trần (prompt ngắn, `ld_move_probe.py --mode short`):** chỉ giữ dòng hướng dẫn, cửa sổ quanh MASK (cắt
theo dòng) và câu hỏi. Kết quả trên LD200, metric tên đầu tiên (chính thức):

| | điểm |
|---|---|
| ngắn ±1k | 32,0 (38,5) |
| ngắn ±4k | 26,5 (19,0) |
| dense | 14,0 (9,5) |
| `--move` | 21,5 (20,0) |

- Nhóm "giải được khi ngắn" (±1k) có 64 câu. Trong đó dense đúng 16, `--move` đúng 20. Nhóm "ngắn cũng sai" có 136
  câu: dense đúng 12, `--move` đúng 23.
- Vậy nhóm mục tiêu thật (giải được khi ngắn, sai khi dài) có 48 câu, và `--move` chỉ cứu được 4 câu trong đó. Phần
  lớn lợi ích của `--move` rơi vào các câu mà bản ngắn cũng sai.
- Mở rộng cửa sổ từ ±1k lên ±4k, mọi khoảng cách vẫn dưới 8k, mà điểm vẫn tụt từ 32 xuống 26,5. Thêm lời thoại xung
  quanh, nghĩa là thêm ứng viên, đã đủ làm hỏng, không cần đến khoảng cách xa. Ở LD, mất mát khi context dài chủ yếu là
  do nhiễu từ các nhân vật cạnh tranh, không phải do khoảng cách RoPE.

**Đầu dò trên nhóm mục tiêu** (`useprobe/target_short`, `target_long`): 64 câu mà prompt ngắn ±1k giải được, ép cùng
câu trả lời dense trong cả hai prompt. Số liệu là tổng mass của retrieval heads tại token quyết định.

| | n | tên đúng | tên dense đã trả lời |
|---|---|---|---|
| dense sai, prompt NGẮN | 48 | 1,60 | 0,22 |
| dense sai, prompt DÀI | 48 | 0,07 trong cảnh + 0,17 cảnh khác | 0,09 trong cảnh + **0,37 cảnh khác** |
| dense đúng, prompt dài | 16 | 0,35 + 0,47 | 0,04 + 0,06 |

- Ở 83% câu sai, mass vào tên sai ở các cảnh khác lớn hơn mass vào tên sai trong cảnh.
- Câu hỏi định vị đúng cảnh (khối MASK đứng top-1 ở 100% số câu), nhưng tại token quyết định attention trôi sang chỗ
  nhắc nhân vật khác ở các cảnh khác.
- Cách sửa đang thử (`scripts/run_focus2.sh`): `--focus` trên cảnh đã định vị bằng bộ chọn retr một đoạn + `--snap`.
  Retrieval heads (thr 0,1) được +log(ctx/picked) cho câu hỏi và câu trả lời, không dời vị trí nào. Lần focus cũ trên
  LD (16) dùng bộ chọn mean-over-heads, bộ chọn này chưa bao giờ chọn đúng cảnh.

**Lỗi công cụ (sửa 23/10 07:10 UTC):** ánh xạ ký tự → token bằng cách giải mã từng token một bị trôi khi gặp ký tự
nhiều byte. LD200: 134/200 prompt không trôi, p90 11 ký tự, tối đa 291. Mv: một số needle bị cắt trước khi hết giá
trị, nên `oracle_spans.json` cũ (đã lưu thành `oracle_spans_drift.json`) cắt cụt 17/40 needle của mv và mk3. Spans của
vt và KV không bị ảnh hưởng. Đã sửa: `scripts/analysis/tokmap.py` (ánh xạ chính xác theo byte); `selector_eval.py`,
`use_probe.py`, `ld_move_probe.py` giờ dùng nó.
- Kiểm tra lại trên 134 prompt LD không trôi: các kết luận ở mục 13 vẫn giữ. Nhóm mục tiêu (n=37): prompt ngắn tên
  đúng 1,67, tên sai 0,20; prompt dài tên đúng trong cảnh 0,08, tên sai ở cảnh khác 0,41. LD200 câu sai: tên đúng >
  tên sai chỉ ở 21%.
- Các run oracle trên mv (pack, perm, gap, copy, gem) dùng spans bị cắt, đang chạy lại (`scripts/run_mvfix.sh`, hậu
  tố `_v2`). Kết quả oracle mv ở mục 10 phải đọc với lưu ý này.

## 14. Chỉ giữ phần retrieve được, mã hóa lại từ đầu (`--gem --gem-sep`), và chọn top-k theo retrieval heads (23/10)

Độ phủ bằng chứng của retrieval heads (`retr_w16`, khối 64 token; scratchpad `coverage.py`):

| task | top 1k phủ | top 4k phủ | budget cần để phủ 100% (trung vị) |
|---|---|---|---|
| vt | 99% | 100% | 768 |
| mv | 96% (phần còn lại nằm trong câu hỏi) | 96% | 1056 |
| KV | 91% | 99% | 832 |
| mk3 | 92% | 96% | 928 |

Bộ chọn `rel` sót mắt chuỗi của vt là do ngưỡng (30% so với khối cao nhất, mà khối cao nhất là ví dụ mẫu), không phải
do retrieval heads không thấy.

Điểm (`pred_recall`, 20 câu mỗi task; dense vt 97, mv 85, KV 80):

| | vt | mv | KV20 |
|---|---|---|---|
| gem oracle (chỉ các câu bằng chứng; spans mv đã sửa) | 94 | **100** (cứu 11/11, hỏng 0) | (chạy) |
| gem, bộ chọn `rel` | 85 | (chạy) | **90** (cứu 3/4, hỏng 1) |
| **gem, top-1k khối retrieval heads (`--agg retr --budget 1024 --pad 0`)** | **100** (cứu 3/3, hỏng 0) | (chạy) | (chạy) |
| `--move --vgap 16384`, top-1k | 99 (cứu 3/3, hỏng 1) | (chạy) | (chạy) |

Nghĩa là chỉ cần lấy top khoảng 1k token mà retrieval heads nhìn vào, bỏ phần còn lại của context, rồi mã hóa lại
với dấu ngăn cách giữa các mảnh: vt đạt 100, không hỏng câu nào. KV với bộ chọn `rel` (có cả đoạn rác) vẫn được 90.
Thứ gây hại chính là phần context dài còn lại, không phải đoạn rác nhỏ đi kèm.

## 15. Kết quả đêm 23→24/10: "gem" (chỉ giữ phần retrieval heads chọn, tính lại từ đầu)

LD chấm trên `pred_recall` (câu trả lời gem), metric tên đầu tiên bỏ qua đoạn in đậm chứa MASK (Qwen3 hay in đậm
`**$$MASK$$**`); điểm chính thức trong ngoặc. Scorer: scratchpad `ld_rec2.py` (BASE=baseline dense).

| LD200 | Qwen2.5-7B-1M | Qwen3-4B-2507 |
|---|---|---|
| dense | 14.0 (9.5) | 12.5 (0.5) |
| move 1 đoạn | 21.5 (20.0) | 25.0 (2.0) |
| gem top-1k khối | 30.5 (15.0) | 18.0 (1.0) |
| **gem 1 cửa sổ 2k quanh khối retr mạnh nhất** (`--pad 992 --snap`) | **42.0 (30.5)** | **40.5 (4.5)** |
| trần prompt ngắn ±1k quanh MASK | 32.0 (38.5) | 41.5 (3.0) |

Điểm chính thức của Qwen3 gần 0 ở mọi cách (kể cả trần), nên không dùng được để so cho Qwen3.
LD50 kích thước cửa sổ: 1k (budget 2k) 40, 2k 50. Lần chạy w4k sai cấu hình (budget 2048 cắt mất cửa sổ 4k) -> chạy lại
`run_day_c.sh` với budget khớp.

RULER/KV gem top-2k (Qwen2.5): qa_1 70→80, qa_2 55→60, KV n=50 76→90 (cứu 10/12, hỏng 3/38), mv n=50 83.5→97.
Hại trên task cần cả văn bản: cwe 24.5→6.5 (top-1k), fwe 80→71.7, math_find 85→80 (top-1k). Qwen3 KV 95→85.
Chọn giữa dense và gem theo logprob (`--select`) không cứu được phần bị hại (cwe 7.0, fwe 71.7).

## 16. Đối chứng độ dài và nhiễu (24/10, `scripts/build_ld_filler.py`)

Cửa sổ 2k do retrieval heads chọn được giữ đúng vị trí cũ, cùng độ dài prompt, phần còn lại thay bằng essay Paul Graham
(Qwen2.5, n=181, tên đầu tiên): 13.8, ngang dense 14.4; gem 42.5. 87/158 câu sai trả lời "Robert Morris" (tên hay xuất
hiện trong essay), tức là essay KHÔNG trung tính về tên: mô hình lấy tên xuất hiện nhiều ở chỗ khác, khác cả thể loại.
Đang chạy (`run_filler2.sh`): RULER noise không có tên (vị trí cũ / sát câu hỏi), noise dài 8k/32k/full sát câu hỏi,
essay sát câu hỏi; `run_filler.sh` còn bản kịch bản khác và Qwen3.
Kích thước cửa sổ LD50 (budget khớp): 1k 38, 2k 50, 4k 28 (8k đang chạy) -> thêm nội dung cùng truyện lại làm hỏng.
Các task khác: qa_1/qa_2 một cửa sổ 70/50 (top-2k 80/60); code_debug 50 -> 10 (cửa sổ) / 30 (top-2k);
fwe 2k 71.7, 4k 76.7 (dense 80); cwe 1k 6.5, 4k 18.0 (dense 24.5); Qwen3 qa_1 top-2k 55 -> 80, qa_2 50 -> 50,
math_find cửa sổ 75 -> 55.
