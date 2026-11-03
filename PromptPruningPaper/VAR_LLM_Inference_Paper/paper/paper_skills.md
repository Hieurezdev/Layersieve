# Playbook viết paper — khung dùng chung cho mọi loại bài

Khung này không gắn với số trang của venue nào. Nó hoạt động bằng cách: **xác định loại đóng góp → chọn module thân bài → phân bổ chỗ theo tỉ lệ → viết theo quy ước chung**.

---

## 0. Bước đầu tiên: xác định loại đóng góp

Trả lời một câu duy nhất: *"Nếu người đọc chỉ nhớ được một thứ từ bài này, đó là gì?"*

| Loại | Claim của bài | Phần chiếm nhiều chỗ nhất |
|---|---|---|
| **Dataset** | "Đây là tập dữ liệu mới, và nó lấp khoảng trống X" | Xây dựng + kiểm định chất lượng dữ liệu |
| **Benchmark / đánh giá** | "Đây là cách đo đúng cho bài toán X, và đây là điều ta học được khi đo lại" | Giao thức đánh giá + phát hiện từ kết quả |
| **Phương pháp** | "Đây là cách làm mới, tốt hơn vì lý do Y" | Mô tả phương pháp |
| **Phân tích thực nghiệm** | "Niềm tin phổ biến X thật ra đúng/sai trong điều kiện Z" | Thiết kế thí nghiệm + phân tích |
| **Hệ thống / công cụ** | "Đây là thứ dùng được thật, giải quyết khó khăn kỹ thuật Z" | Kiến trúc + đánh giá vận hành |

**Bài hỗn hợp**: chọn **đúng một** loại làm claim chính (thường là thứ mới nhất, khó nhất, hoặc thứ mà nếu bỏ đi thì bài không còn lý do tồn tại). Các phần còn lại tự động xuống hàng phụ trợ — chúng vẫn được viết đầy đủ nhưng ngắn hơn và không được tranh chỗ với phần chính. Ví dụ: bài vừa curate dataset vừa propose framework thì nếu framework là điểm mới → dataset viết gọn lại thành phần chuẩn bị; nếu dataset mới là đóng góp thật → framework chỉ đóng vai baseline chứng minh dataset dùng được.

---

## 1. Lắp bài từ module

Khung xương tiêu chuẩn (dùng khi không có yêu cầu khác):

> Abstract → Introduction → Related Work → **[các module thân bài]** → Conclusion

**Yêu cầu của người dùng / giảng viên / venue luôn đứng trên khung này.** Nếu đã quy định số chương, tên chương hay thứ tự thì giữ đúng như vậy và đặt nội dung của khung vào các chương đó (vd Introduction + Related Work thành mục con của chương đầu). Tỉ lệ ở §2 khi đó áp theo khối nội dung, không theo chương.

Thân bài lấy từ kho module dưới đây, chỉ chọn module bài thực sự cần, đánh số theo thứ tự đọc tự nhiên và **để module mang claim chính lên trước**.

| Module | Nội dung cốt lõi | Bắt buộc với loại |
|---|---|---|
| **D. Dữ liệu** | Nguồn, thống kê, quy trình xử lý, chia tập | Dataset, Benchmark |
| **P. Giao thức đánh giá** | Metric, cách chia, đối tượng được so, điều kiện công bằng | Benchmark, Phân tích |
| **M. Phương pháp** | Ý tưởng, ký hiệu, kiến trúc, hàm mục tiêu | Phương pháp, Hệ thống |
| **E. Thực nghiệm & kết quả** | Setup, bảng chính, ablation | Gần như mọi loại |
| **A. Phân tích & thảo luận** | Vì sao kết quả như vậy, case study, hạn chế | Phân tích, Benchmark |
| **S. Hệ thống** | Kiến trúc triển khai, chi phí, độ trễ, khả năng mở rộng | Hệ thống |

Vai trò của module **E** đổi theo loại bài: với paper phương pháp, E là bằng chứng bài mình tốt hơn; với paper dataset, E là bằng chứng dữ liệu dùng được và có độ khó hợp lý; với paper benchmark, E chính là nội dung phát hiện chứ không phải phần phụ.

---

## 2. Phân bổ chỗ theo tỉ lệ

Đặt toàn bài = 100%, rồi nhân với số trang venue cho phép. Tỉ lệ giữ nguyên dù bài 4 trang hay 12 trang.

| Khối | Tỉ lệ |
|---|---|
| Abstract + Introduction (kèm hình teaser) | 12–15% |
| Related Work | 10–15% |
| **Module mang claim chính** | **35–45%** |
| Các module phụ trợ (chia đều) | 20–30% |
| Conclusion | 3–5% |

Ba ví dụ phân bổ khác nhau cho cùng khung:

- **Paper phương pháp**: M 40% · E 20% · D 8%
- **Paper dataset**: D 40% (phân tích + xây dựng + kiểm định chất lượng) · E 18% (baseline) · M 5% hoặc bỏ
- **Paper benchmark**: P 25% + A 20% · D 12% · M 0%

**Thứ tự cắt khi tràn** (từ trên xuống): hình phân tích phụ → bảng nhỏ cùng cấu trúc (gộp thành một bảng nhiều nhóm cột) → ví dụ minh họa dư → câu chuyển đoạn → mô tả baseline đã quen thuộc.
**Không cắt**: định nghĩa ký hiệu, quy trình xây dữ liệu, giao thức đánh giá, hạn chế. Đây là những thứ quyết định bài có tái lập được hay không.

---

## 3. Quy ước trình bày (áp cho mọi bài)

- **Thì động từ**: hiện tại đơn cho toàn bài; riêng Related Work dùng quá khứ đơn.
- **Viết hoa**: tên section viết hoa toàn bộ, subsection chỉ viết hoa chữ đầu.
- **Metadata**: kiểm tra tên tác giả, affiliation, email đúng định dạng và còn hoạt động.
- **Công thức theo luồng**: mỗi công thức được dẫn bằng một câu nối với công thức trước ("từ Eq. (n) ta có…"), không đổ một loạt rồi giải thích sau; mỗi section nên kết bằng một công thức "chốt" để phần sau bám vào. Công thức không bao giờ được `\ref`/`\eqref` → **inline** vào câu văn hoặc thêm câu tham chiếu, đừng để trôi lơ lửng.
- **Citation**: mọi câu khẳng định hiện tượng, con số hay xu hướng đều phải có reference. Không có nguồn thì đổi thành quan sát của chính mình hoặc bỏ.
- **Footnote**: dùng cho link dataset, repo, công cụ, ví dụ thực tế từ báo chí — không nhét chú thích lằng nhằng vào ngoặc đơn giữa dòng.
- **Số lượng citation**: theo kỳ vọng của venue (paper ngắn thường ≥30), phân bố đều giữa Introduction và Related Work, ưu tiên bài 2–3 năm gần nhất.
- **Danh sách tác giả dài**: rút thành "et al." theo chuẩn venue (IEEEtran: thêm control entry `@IEEEtranBSTCTL` + `\bstctlcite`).
- **Tên section mô tả** > chung chung ("Scaling with Context Length on RULER" > "Results on RULER").
- **Thao tác sửa an toàn**: grep trước khi xóa label/công thức/hình (xem có `\ref` không); phần của cộng sự = **review/flag**, chỉ sửa khi được cho phép rõ ràng; đang sửa file có sẵn thì **append, đừng `Write` đè cả file**; placeholder hình phải **biên dịch được ngay**.

---

## 4. Abstract — ba khối

Ba khối luôn giống nhau, chỉ đổi nội dung theo loại bài. 150–200 từ, viết cuối cùng.

| Khối | Paper phương pháp | Paper dataset | Paper benchmark |
|---|---|---|---|
| **1. Vấn đề** | Cách làm hiện tại thiếu gì | Dữ liệu hiện có thiếu gì (quy mô, nhiễu, chồng lấn, thiên lệch) | Cách đo hiện tại sai/thiếu ở đâu |
| **2. Đã làm gì** | Propose framework, gồm những thành phần nào, mỗi cái giải quyết phần nào | Nguồn, quy trình xây, quy mô, cách đảm bảo chất lượng | Thiết kế giao thức, phạm vi đối tượng đánh giá |
| **3. Bằng chứng** | Cải thiện bao nhiêu so với baseline mạnh nhất | Baseline chạy trên tập này cho thấy điều gì | Phát hiện chính, nhất là phát hiện đi ngược trực giác |

Kiểm tra: abstract phải làm nổi **2–3 contribution**, không chỉ nêu tên framework/dataset. Nếu người đọc chỉ nhớ được cái tên thì chưa đạt.

---

## 5. Introduction — bản phóng to của Abstract

Giữ nguyên thứ tự ba khối, mỗi khối giãn thành 1–2 đoạn.

**Nguyên tắc bao trùm:** Abstract + Introduction phải đọc như **một câu chuyện thuyết phục** — dựng bài toán sao cho reviewer thấy hứng thú và tin nó có tiềm năng/tác động thật; neo vào một động lực cụ thể (ứng dụng, ngành, người dùng, bối cảnh), tránh phát biểu chung chung.

**Đoạn 1 — Bối cảnh.** Câu đầu nêu tầm quan trọng của bài toán, **bắt buộc có citation**. Nếu có nguồn thật, kèm một **footnote dẫn ví dụ tác động thực tế trích từ báo chí** (sự cố, thiệt hại, quy mô) — thứ này khiến reviewer tin bài toán có giá trị thật. **Không có nguồn thật thì không viết thay** — để note cho tác giả tìm nguồn.

**Đoạn 2 — Khoảng trống.** Hiện trạng làm gì, hạn chế ở đâu, mỗi hạn chế có citation. Nếu có giải pháp/công cụ hiển nhiên ("sao không dùng X?"), **chặn trước** bằng lý do cụ thể + một ví dụ thất bại thật. Kết đoạn bằng một câu chốt gap.

**Đoạn 3 — Cách tiếp cận.** Ý tưởng cốt lõi ở mức trực giác, chưa ký hiệu. Trỏ tới hình teaser. **Nếu** bài có "sản phẩm/tầm nhìn" tách khỏi "kỹ thuật/đóng góp", đặt tên riêng cho từng cái — sản phẩm dẫn Abstract/Intro, kỹ thuật giới thiệu ở phần kỹ thuật.

**Đoạn 4 — Contribution list.** 3–4 gạch đầu dòng. Quy tắc cứng: **mỗi contribution phải có một phần tương ứng được trình bày đầy đủ trong bài**. Chưa trình bày được thì bỏ khỏi danh sách, đừng để lửng.

**Hình teaser.** Đặt đầu Introduction, thu nhỏ, gộp các hình liên quan thành panel a) b) trong cùng một figure. Nội dung hình đổi theo loại bài: paper phương pháp vẽ pipeline rút gọn; paper dataset vẽ thống kê/so sánh với dữ liệu cũ; paper benchmark vẽ phát hiện chính. Caption phải tự đọc hiểu được.

---

## 6. Related Work — quá khứ đơn

- Nhóm theo **hướng tiếp cận**, không theo thời gian, không liệt kê mỗi bài một dòng. Mỗi nhóm một đoạn: nhóm này làm gì, đại diện là ai, chung nhau điểm yếu gì.
- Mỗi đoạn kết bằng câu nối về gap của bài mình.
- Với paper dataset/benchmark, thêm một đoạn riêng so sánh với các dataset/benchmark đã có — tốt nhất là một bảng đối chiếu (quy mô, nhãn, nguồn, giấy phép).
- Nên **khép Related Work bằng một mục định vị** (mọi loại bài): bảng đối chiếu — cột = các chiều mà đóng góp của mình phủ, mỗi hướng prior work đánh dấu phủ chiều nào, dòng cuối là bài mình — kèm một đoạn nói mình nằm ở **giao điểm** không hướng nào phủ một mình.
- Không so sánh số liệu kết quả ở đây, để dành cho module E.

---

## 7. Các module thân bài

### D. Dữ liệu

**Phân tích.** Nguồn lấy từ đâu, phiên bản nào — **mỗi nguồn một link footnote**. Thống kê cơ bản: quy mô, phân bố nhãn, độ dài mẫu. Kết bằng quan sát dẫn dắt sang bước xử lý (mất cân bằng, trùng lặp, chồng lấn giữa các nguồn, nhiễu nhãn). Nếu hết chỗ phải chọn giữa hình phân tích và chữ phân tích, **giữ chữ**.

**Xây dựng / xử lý.** Mô tả theo đúng thứ tự thao tác: lọc → khử trùng lặp → chuẩn hóa → gán/hợp nhất nhãn → chia tập. Mỗi bước nêu tiêu chí cụ thể và số mẫu còn lại. Kết bằng bảng thống kê tập cuối.

Với paper dataset, thêm phần **kiểm định chất lượng**: độ đồng thuận giữa người gán nhãn hoặc quy trình xác minh tự động, tỉ lệ lỗi ước tính, giấy phép và vấn đề đạo đức/riêng tư. Đây là phần reviewer soi kỹ nhất.

### P. Giao thức đánh giá

- Metric dùng gì và **vì sao metric đó phù hợp** với bài toán — nếu chỉ dùng metric mặc định thì nói rõ lý do.
- Cách chia dữ liệu, và cách tránh rò rỉ giữa các tập.
- Đối tượng được đánh giá: chọn theo tiêu chí nào, cấu hình lấy ở đâu, đã tinh chỉnh công bằng chưa.
- Điều kiện chạy giống nhau: phần cứng, ngân sách tính toán, số lần lặp và cách báo cáo độ lệch.

### M. Phương pháp

Mỗi thành phần đi theo thứ tự **cần gì → vì sao → dùng gì (+citation, lý do chọn)**, nhưng thứ tự này thể hiện **ngầm trong văn xuôi liền mạch**:

- Không viết nhãn kiểu "Muốn gì:", "Ví dụ:", "Dùng gì:" / "What is needed:", "Example:", "What we use:" — kiểu gạch đầu mục đó chỉ hợp với Abstract/Introduction.
- Chỉ đưa ví dụ minh họa khi nó mang thông tin mà lập luận không tự nói được; ví dụ không thêm gì thì bỏ.
- Số liệu chứng minh (ablation, đo đạc) đặt ở Experiments; phần phương pháp chỉ nêu lý do và trỏ tới đó.

Điều quan trọng nhất vẫn giữ: người đọc phải hiểu *tại sao cần* trước khi nghe *dùng cái gì*.

**Hình pipeline**: kích thước theo hình mẫu hoặc yêu cầu của tác giả (mặc định khoảng 1/3 trang); ưu tiên rõ ràng, không chồng chữ; có legend, **nhúng ký hiệu công thức ngay vào các khối trong hình** để hình khớp với phần ký hiệu bên dưới. Phân biệt luồng dữ liệu với luồng gradient bằng nét/màu.

**Ký hiệu**: khai báo theo luồng từ đầu vào tới hàm mục tiêu, mỗi ký hiệu định nghĩa ngay lần xuất hiện đầu, không dùng lại một chữ cái cho hai nghĩa. Khung mẫu cho bài toán phân loại:

- Tập mẫu đầu vào: `S = {s₁, s₂, …, sₙ}`
- Tập đơn vị trích xuất từ mẫu: `F = {f₁, f₂, …, f_m}`
- Encoder → biểu diễn: `V = Enc(F)`, `V ∈ ℝ^{m×d}`
- Cấu trúc quan hệ: đồ thị `G = (F, E)` xây từ quan hệ giữa các đơn vị
- Tổng hợp theo cấu trúc: `V' = GNN(G, V)`
- Kết hợp mức batch (nếu có): `V'' = Agg(V')`
- Dự đoán: `ŷ = σ(W·V'' + b)`
- Hàm mục tiêu: `L = −(1/N) Σ [y log ŷ + (1−y) log(1−ŷ)]`, kèm giải thích từng số hạng regularization nếu có

Bài không phải học máy vẫn giữ nguyên tinh thần: định nghĩa đầu vào, phép biến đổi, đầu ra, tiêu chí tối ưu.

**Preliminaries — gọn & không đạo văn.** Đừng chép gần nguyên văn preliminaries/definition của baseline mình xây trên: nói **ngắn + cite nguồn từng definition**, chỉ giữ cái **được dùng ở phần sau**, mỗi paradigm nền (kể cả baseline được cite) có **một Definition hình thức**. Overview + hình kiến trúc thuộc **Methodology**, đừng nhét vào Preliminaries hay để cạnh Motivation (kẻo tưởng là "phát hiện").

**Ranh giới Motivation–Method.** Motivation/Observations = **chỉ thí nghiệm + phát hiện**, cơ chế để hết ở Method (nếu hai chỗ tả cùng cơ chế → cắt ở Motivation, chỉ giữ "vì sao cần" + trỏ `§\ref`). Đừng gọi một mục là "empirical observation" nếu nó không có số đo — đó là design rationale.

**Ghi công ý tưởng có trước.** Thành phần kế thừa phát hiện/kỹ thuật của bài khác thì nói ý đó trong câu và đặt citation — không viết nhãn kiểu "(from X)", "inspired by X:" ở tiêu đề hay đầu đoạn.

**Độ phức tạp công thức.** Không bắt buộc dạng đơn giản nhất, nhưng tránh công thức quá rối: đại lượng phụ có thể định nghĩa bằng lời hoặc tách thành công thức riêng; tránh chỉ số chồng chất, tổng lồng nhau khi có cách viết tương đương dễ đọc hơn. Vẫn phải đúng với cái đã cài đặt.

**Sửa ký hiệu & công thức.** Chủ động quét đụng độ ký hiệu (vd `θ` threshold vs params, `k`/`K`/text, `v` fingerprint vs embedding); **đụng độ chéo giữa hai đóng góp là tệ nhất — rà cả hai cùng lúc**. Công thức lõi phát biểu **một lần**, chỗ khác refer (đừng viết 3–4 kiểu). Không overclaim so với toán: dùng "controls/approximates" thay "bounds/guarantees" nếu chưa chứng minh.

### E. Thực nghiệm & kết quả

- **Setup**: dữ liệu, cách chia, baseline (kèm citation), metric, siêu tham số, phần cứng — đủ để tái lập.
- **Bảng chính**: gộp các bảng cùng cấu trúc thành một bảng nhiều nhóm cột thay vì nhiều bảng nhỏ. In đậm kết quả tốt nhất.
- **Ablation** (paper phương pháp): bỏ lần lượt từng thành phần đã propose. Thành phần nào không cải thiện thì cân nhắc bỏ khỏi contribution list.
- **Baseline khảo sát** (paper dataset/benchmark): chạy nhiều họ phương pháp khác nhau để cho thấy tập dữ liệu có độ khó hợp lý và chưa bão hòa.
- Mỗi bảng/hình đều phải được nhắc và diễn giải trong thân bài — không để bảng mồ côi.
- **Bảng đừng teo**: KHÔNG `\resizebox{\columnwidth}{!}{...}` (nhòe, vô hiệu `\small`) → font thật + `\setlength{\tabcolsep}`; bề rộng do **số cột + độ dài nhãn** → rút nhãn, viết tắt header theo quy ước chuẩn (+cite nguồn quy ước); bỏ cột không phân biệt (bão hòa) + ghi chú trong text; bảng lớn → `table*` hoặc lưới đầy đủ xuống Appendix; `\phantom{0}` canh số.
- **Baseline & số liệu trung thực**: cite từng method **ngay trên hàng bảng** (không nhãn "Prior/This work"); so ở **điều kiện công bằng do mình đo lại** (cùng setup, cùng ngân sách tính toán/tốc độ), không báo cấu hình làm lệch so sánh; chỉ lấy số từ paper khác khi không chạy lại được và ghi rõ; **bold cột-best trung thực** kể cả khi baseline thắng; sau khi sửa số/thêm baseline **rà lại mọi câu bị lệch**; claim phải khớp với thứ đã chạy thật.
- **Không nhồi số liệu thừa**: thân bài, caption, nhãn trục chỉ giữ con số người đọc cần để theo lập luận. Chi tiết chạy (số mẫu, id, seed, cấu hình phụ) để ở note/appendix. Không chú thích lặp trên từng hàng bảng (vd "(8.7%)") — thông số chung nêu một lần ở Setup.
- **Nhiều đóng góp → cấu trúc song song**: mỗi đóng góp một `\subsection`, Setup+Results cùng độ sâu, RQ đặt đầu mỗi nhánh; đừng dán experiments đóng góp sau lủng lẳng.

### A. Phân tích & thảo luận

Một hoặc hai phát hiện, mỗi phát hiện gồm: quan sát → giả thuyết giải thích → bằng chứng phụ (case study, hình trực quan, thí nghiệm nhỏ kiểm chứng). Đây là phần khiến bài có chiều sâu chứ không chỉ là bảng số. Khi có nhiều nguồn lợi ích, **gán mỗi hiệu ứng cho đúng nguyên nhân** (vd amortization→latency, cửa sổ rộng→accuracy) để chặn trước phản biện fairness của reviewer. Kèm phần hạn chế viết thật, nêu điều kiện mà kết luận không còn đúng.

### S. Hệ thống

Kiến trúc triển khai, các ràng buộc thực tế phải xử lý, và số liệu vận hành: độ trễ, thông lượng, chi phí, khả năng mở rộng. So sánh với cách làm thủ công hoặc hệ thống sẵn có.

---

## 8. Conclusion

3–5 câu: nhắc lại vấn đề, tóm tắt cái đã làm và kết quả chính, nêu hạn chế thật, một hướng phát triển cụ thể. Không số liệu mới, không citation mới.