# VAR_LLM_Inference_Paper

Paper cho Viettel AI Race, task Long Context Inference (prompt pruning).

Đội top 1 dùng SpecPrefill: một draft model nhỏ chấm điểm token rồi model lớn chỉ prefill phần được giữ.
Luận điểm của paper: draft model nhỏ bỏ sót token mà model lớn cần ở task khó, nên cắt token **bên trong chính model lớn** sẽ chính xác hơn. Phương pháp của repo này tên là **LayerSieve**.

## Phương pháp: LayerSieve

Ý tưởng: không dùng draft model. Chính model lớn chấm điểm token ở từng layer sâu, rồi "sàng" bớt dần token qua các layer.

**Ký hiệu**
- Prompt dài $T$ token, model có $L$ layer (Qwen2.5-7B: $L=28$), $H$ query head, head dim $d$.
- $S_\ell$: tập token đi vào layer $\ell$. Mỗi token $j$ giữ vị trí gốc $p_j$ (dùng cho RoPE).
- $Z$: $n_{\text{sink}}=128$ token đầu (attention sink). $O$: $n_{\text{obs}}=128$ token cuối prompt (cửa sổ quan sát). Hai tập này luôn được giữ.
- $\ell_0=17:$  layer bắt đầu sàng. $\rho=0.12:$  tỉ lệ bỏ mỗi layer.

**1. Layer nông chạy đủ**

```math
S_\ell = \{1,\dots,T\}, \qquad \ell < \ell_0.
```

**2. Chấm điểm trước mỗi layer $\ell \ge \ell_0$** (trên các token còn lại $S_{\ell-1}$, dùng $q, k$ của chính layer $\ell$)

```math
a^{h}_{ij}
=
\frac{\exp\!\left(\dfrac{q^{(h,\ell)}_i \cdot k^{(h,\ell)}_j}{\sqrt{d}}\right)}
{\displaystyle
\sum_{\substack{j'\in S_{\ell-1}\\ p_{j'}\le p_i}}
\exp\!\left(\dfrac{q^{(h,\ell)}_i \cdot k^{(h,\ell)}_{j'}}{\sqrt{d}}\right)},
\qquad
\substack{i\in O,\ j\in S_{\ell-1}\\ p_j\le p_i}.
```
```math
s_j
=
\sum_{i\in O}\max_{h\in\{1,\dots,H\}} a^{h}_{ij},
\qquad
\hat{s}_j
=
\frac{1}{|\mathcal{N}(j)|}
\sum_{j'\in\mathcal{N}(j)} s_{j'}.
```
- $a^h_{ij}$: attention thật (đã softmax, causal) từ token quan sát $i$ tới token $j$ ở head $h$.
- $\max_h$: chỉ cần một head nhìn mạnh là token được giữ, nên các head tra cứu không bị các head khác pha loãng.
- $\sum_{i\in O}:$ gom phiếu của cả 128 token cuối thay vì chỉ token cuối cùng.
- $\hat{s_j}$ : làm mượt trên $\mathcal{N}(j)$ = 13 token liền kề $j$ trong $S_{\ell-1}$, để giữ trọn cụm token liền nhau (một key/giá trị thường dài nhiều token).

**3. Bỏ token điểm thấp**

```math
P_\ell = S_{\ell-1} \setminus (Z \cup O),
\qquad
S_\ell = S_{\ell-1} \setminus D_\ell.
```
```math
D_\ell \subseteq P_\ell,
\qquad
|D_\ell| = \left\lfloor \rho\,|P_\ell| \right\rfloor,
```
```math
\hat{s}_j \le \hat{s}_{j'}
\qquad
\forall\, j\in D_\ell,
\quad
j'\in P_\ell\setminus D_\ell.
```
- $P_\ell$: token được phép bỏ; $D_\ell$: $\lfloor \rho |P_\ell| \rfloor$ token có $\hat{s}$ nhỏ nhất.
- Token bị bỏ ra khỏi cả attention lẫn MLP từ layer $\ell$ trở đi. Hidden state của token giữ lại đi tiếp bình thường, vị trí RoPE không đổi.
- Số token còn lại: $|S_\ell| \approx |Z \cup O| + (1-\rho)^{\ell-\ell_0+1}\,(T-256)$. Với $\rho=0.12$, đến L27 còn khoảng 24%.

**4. Decode**
- Mỗi layer chỉ giữ KV của $S_\ell$. Token sinh ra có vị trí tiếp nối $T, T+1, \dots$
- Tùy chọn sàng KV (`decode_ref` $= r$): ngay sau prefill, mỗi layer $\ell < r$ chỉ giữ KV của $S_r$. Mỗi bước decode khi đó đọc ít KV hơn (xem mục Case khó).

**Chi phí**
- Tổng token×layer so với dense:

```math
\frac{1}{L}
\left(
\ell_0
+
\sum_{k=1}^{L-\ell_0}(1-\rho)^k
\right)
\approx 0.80.
```
- Chấm điểm thêm chỉ tốn $O(|O| \cdot |S_{\ell-1}|)$ mỗi layer, rất nhỏ so với attention đầy đủ $O(|S_\ell|^2)$.

**Vì sao bắt đầu ở L17**
- Probe trên kv cho thấy ở layer nông (vd. L8) evidence chỉ đứng quanh percentile 0.55 theo điểm attention, tức gần ngẫu nhiên.
- Từ L17 trở đi evidence đứng ở percentile 0.86–0.99.
- Vì vậy chấm điểm từ L17 mới đáng tin, và bỏ ít mỗi layer thì lỗi chấm sai ở một layer không làm mất hẳn token.

## Cấu trúc
```
inmodel_prune/        code LayerSieve
  prune.py              bản HF tham chiếu
  vllm_patch.py         bản vLLM 0.6.3 (1 request), có kèm SlimInfer cài lại (method="slim"); unit/accum/bm25, sàng KV decode
  units.py              chia đơn vị (dòng/câu), BM25, bỏ nguyên đơn vị
baselines/            mỗi thư mục là một paper baseline
  SpecPrefill/          code gốc SpecPrefill (MIT), không sửa
  specprefill_qwen.py   patch để SpecPrefill chạy với Qwen2 (gốc chỉ hỗ trợ Llama), ghi lại token được giữ
  SlimInfer/            official/ = submodule code gốc (commit 39640a5), run_kv.py = driver chạy kv
data/build_kv.py      tạo prompt InfiniteBench kv_retrieval đã token hoá (data/*.jsonl không commit)
data/build_hard.py    tạo prompt ≤30k cho thí nghiệm case khó: kv, variable tracking, HotpotQA/2WikiMQA
scripts/              chạy thí nghiệm và tổng hợp
results/              kết quả benchmark (jsonl từng mẫu)
docs/                 paper liên quan và report
```

## Cài đặt (uv, Python 3.10)
```bash
git submodule update --init                          # code gốc SlimInfer
uv venv -p 3.10 .venv && uv pip install -p .venv -r requirements.txt
```
Baseline SlimInfer gốc cần env riêng (`baselines/SlimInfer/README.md`).

### Container: lưu và khôi phục môi trường
Tắt container sẽ mất mọi thứ ngoài `/data`. Code, `.venv` và HF cache (`HF_HOME=/data/env/hf`) nằm trong `/data` nên không mất. Phiên Claude Code (hội thoại, memory) thì phải lưu tay.

Trước khi tắt container:
```bash
bash /data/LongContextInferPaper/scripts/save_claude.sh    # lưu ~/.claude vào /data/env/claude-state (không lưu token đăng nhập)
```

Sau khi bật lại:
```bash
bash /data/LongContextInferPaper/scripts/restore_env.sh    # khôi phục ~/.claude, binary claude, CLAUDE_CODE_TMPDIR
cd /data/speculative_prefill                               # phải đúng thư mục này mới thấy hội thoại cũ
claude                                                     # /login để đăng nhập lại
claude --resume                                            # chọn hội thoại cần mở lại
```
Git báo "dubious ownership" vì chủ sở hữu của `/data` bị đổi thành uid 65534. Chạy một lần sau mỗi lần khôi phục:
```bash
git config --global --add safe.directory /data/speculative_prefill
```

## Chạy
```bash
python data/build_kv.py --out data/kv50.jsonl
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --out results/kv50/kv50_catdan.jsonl        # LayerSieve
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --dense --out results/kv50/kv50_dense.jsonl  # dense
python scripts/summarize_kv.py
```
Tái lập toàn bộ bảng kv50: `scripts/run_kv50.sh`.

Thí nghiệm case khó (SpecPrefill, cải tiến từ MOE, quét lịch sàng, sàng KV cho decode). Các runner tự chờ GPU đủ chỗ trống qua `scripts/gpu_queue.sh`; biến `GPUS=1` giới hạn job vào GPU 1.
- Chạy nền (`setsid nohup bash scripts/<tên>.sh >> <log> 2>&1 < /dev/null &`):
  - Target/draft chạy đầy đủ và biến thể MOE: `run_moe.sh`, `run_moe_vt.sh`.
  - SpecPrefill: `run_sp_check.sh`.
  - Quét lịch: `run_sweep.sh` (chọn theo token), `run_sweep_unit.sh` (chọn theo unit).
  - Sàng KV: `run_decode.sh`.
  - Tốc độ: `run_speed_all.sh`.
- Tổng hợp: `python scripts/summarize_moe.py`, `summarize_sweep.py`, `summarize_decode.py`.

## Kết quả hiện có (kv50, Qwen2.5-7B-Instruct-1M, ~170k token, 1×GPU)
| Phương pháp | Đúng /50 | TTFT |
|---|---|---|
| dense | 35 | 20.8 s |
| **LayerSieve** ($\ell_0=17,\ \rho=0.12$) | **29** | 15.5 s |
| SlimInfer giữ 0.9/0.75/0.5 | 26 | 16.2 s |
| SlimInfer giữ 0.8/0.6/0.4 (compute ≈ LayerSieve) | 24 | 13.9 s |
| SlimInfer giữ 0.5/0.25/0.15 | 10 | 9.4 s |
| SlimInfer cấu hình gốc 12k/6k/4k | 3 | 6.9 s |

Chi tiết: `docs/report_kv50.md`.

## Case khó: SpecPrefill, cải tiến từ MOE, quét lịch sàng, sàng KV cho decode

Câu hỏi:
- Khi model nháp nhỏ không trả lời được câu mà model lớn trả lời được, SpecPrefill có sập không?
- Cải tiến nào của đội MOE dùng được cho LayerSieve?
- Sàng mạnh hơn hoặc sớm hơn thì nhanh thêm bao nhiêu và mất bao nhiêu?
- Có thể tận dụng việc sàng để decode nhanh hơn không?

**Thiết lập**
- Target Qwen2.5-7B-Instruct-1M. Model nháp Qwen2.5-1.5B-Instruct hoặc 0.5B-Instruct.
- Prompt ≤ 27k token, vì model nháp chỉ có context 32k. Nếu cho prompt dài hơn, model nháp sẽ sập vì tràn context chứ không phải vì kém.
- SpecPrefill dùng config gốc: look-ahead 8, chunk 32, pool 13.
- Ba loại task (`data/build_hard.py`):
  - kv: kv_retrieval tự sinh.
  - vt: variable tracking 4 bước, câu hỏi theo chiều xuôi ("giá trị của VAR Q?"), có chuỗi gây nhiễu.
  - QA: HotpotQA và 2WikiMQA lấy từ LongBench.
- Mẫu:
  - kv, Hotpot, 2Wiki: mỗi task 30 mẫu chọn ngẫu nhiên (seed 0) trong các mẫu target trả lời đúng (`results/hard/ids_sub.txt`).
  - vt: 17 mẫu trong `ids_sub.txt` mà target vẫn đúng, cộng 21 mẫu bổ sung (`results/moe/hard_vt/ids.txt`).
  - Lý do vt phải bổ sung: `build_hard.py` từng duyệt `set` nên dữ liệu vt phụ thuộc PYTHONHASHSEED, không tái lập được. Đã sửa bằng `sorted()` và tạo lại dữ liệu.
  - Chỉ đếm câu target làm đúng trên dữ liệu hiện tại: kv /30, vt /38, hotpot /30, 2wiki /29.
- "Giữ evidence": mọi span evidence còn ≥ 1/2 số token sau khi cắt (chỉ đo cho kv và vt).
- Ký hiệu:
  - ℓ₀: layer bắt đầu sàng; ρ: tỉ lệ token bỏ ở mỗi layer.
  - r: mốc sàng KV. Sau prefill, layer < r chỉ giữ KV của các token còn sống tới layer r, nên decode đọc ít KV hơn.
  - "Giữ cuối": tỉ lệ token còn lại ở layer cuối. "token×layer": khối lượng tính khi prefill so với chạy đầy đủ. "KV đọc": lượng KV mỗi bước decode phải đọc so với chạy đầy đủ.
- Các biến thể lấy từ MOE (`inmodel_prune/units.py`):
  - **unit**: bỏ nguyên đơn vị. Đơn vị kết thúc ở xuống dòng hoặc `. , ; ! ?`, dài tối đa 128 token; điểm đơn vị là max điểm các token trong nó.
  - **accum**: xếp hạng theo điểm cộng dồn qua các layer.
  - **bm25**: bảo vệ các đơn vị khớp BM25 với câu hỏi, tối đa 2% prompt; term là bigram token.
  - **chỉ BM25**: dùng BM25 thay cho điểm attention.
- So cặp từng câu bằng McNemar (thắng:thua).
- GPU RTX 5880 48GB, dùng chung với người khác.

**Bảng 1. Tập case khó**

| # | Nhóm | Setting | kv /30 | vt /38 | hotpot /30 | 2wiki /29 | Giữ cuối | token×layer | KV đọc/bước | vs dòng 4 (thắng:thua) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Chạy đầy đủ | Target 7B | 30 | 38 | 30 | 29 | 100% | 1.00 | 100% | – |
| 2 | Chạy đầy đủ | Model nhỏ 1.5B | 17 | 7 | 24 | 17 | 100% | 1.00 | 100% | – |
| 3 | Chạy đầy đủ | Model nhỏ 0.5B | 15 | 1 | 11 | 12 | 100% | 1.00 | 100% | – |
| 4 | MOE | LayerSieve gốc (token) | 23 | 38 | 30 | 29 | 27% | 0.80 | 80% | – |
| 5 | MOE | unit | 30 | 38 | 30 | 29 | 26% | 0.80 | 80% | 7:0 |
| 6 | MOE | accum | 25 | 38 | 30 | 29 | 27% | 0.80 | 80% | 2:0 |
| 7 | MOE | bm25 (bảo vệ 2%) | 30 | 38 | 30 | 29 | 28% | 0.80 | 80% | 7:0 |
| 8 | MOE | all = unit+accum+bm25 | 30 | 38 | 30 | 29 | 27% | 0.80 | 80% | 7:0 |
| 9 | MOE | chỉ BM25 (thay attention) | 30 | 2 | 28 | 22 | 26% | 0.80 | 80% | 7:45 |
| 10 | SpecPrefill | Nháp 1.5B giữ 90% | 30 | 29 | 30 | 29 | 90% | ~90% | ~90% | 7:9 |
| 11 | SpecPrefill | Nháp 1.5B giữ 30% | 13 | 25 | 30 | 27 | 30% | ~30% | ~30% | 2:27 |
| 12 | SpecPrefill | Nháp 1.5B giữ 10% | 4 | 18 | 27 | 21 | 10% | ~10% | ~10% | 0:50 |
| 13 | SpecPrefill | Nháp 0.5B giữ 10% | 2 | 11 | 25 | 18 | 10% | ~10% | ~10% | 0:64 |
| 14 | Sweep token | token ℓ₀=17 ρ=0.25 | 13 | 37 | 30 | 28 | 7% | 0.71 | 71% | 0:12 |
| 15 | Sweep token | token ℓ₀=14 ρ=0.06 | 26 | 38 | 30 | 29 | 44% | 0.82 | 82% | 3:0 |
| 16 | Sweep token | token ℓ₀=14 ρ=0.10 | 19 | 37 | 30 | 29 | 25% | 0.75 | 75% | 0:5 |
| 17 | Sweep token | token ℓ₀=14 ρ=0.15 | 11 | 38 | 29 | 27 | 13% | 0.68 | 68% | 0:15 |
| 18 | Sweep token | token ℓ₀=14 ρ=0.25 | 2 | 32 | 29 | 28 | 5% | 0.61 | 61% | 0:29 |
| 19 | Sweep token | token ℓ₀=7 ρ=0.03 | 16 | 38 | 29 | 27 | 54% | 0.80 | 80% | 0:10 |
| 20 | Sweep token | token ℓ₀=7 ρ=0.05 | 10 | 33 | 30 | 27 | 36% | 0.70 | 70% | 0:20 |
| 21 | Sweep token | token ℓ₀=7 ρ=0.08 | 7 | 34 | 30 | 28 | 20% | 0.59 | 59% | 0:21 |
| 22 | Sweep unit | unit ℓ₀=17 ρ=0.25 | 30 | 37 | 30 | 29 | 6% | 0.71 | 71% | 7:1 |
| 23 | Sweep unit | unit ℓ₀=17 ρ=0.35 | 30 | 36 | 30 | 26 | 4% | 0.67 | 67% | 7:5 |
| 24 | Sweep unit | unit ℓ₀=14 ρ=0.10 | 30 | 36 | 29 | 29 | 24% | 0.75 | 75% | 7:3 |
| 25 | Sweep unit | unit ℓ₀=14 ρ=0.15 | 30 | 36 | 30 | 27 | 12% | 0.68 | 68% | 7:4 |
| 26 | Sweep unit | unit ℓ₀=14 ρ=0.25 | 30 | 35 | 30 | 27 | 4% | 0.61 | 61% | 7:5 |
| 27 | Sweep unit | unit ℓ₀=7 ρ=0.05 | 28 | 34 | 29 | 28 | 34% | 0.70 | 70% | 6:7 |
| 28 | Sweep unit | unit ℓ₀=7 ρ=0.08 | 25 | 34 | 29 | 26 | 18% | 0.59 | 59% | 4:10 |
| 29 | Sweep unit | unit ℓ₀=7 ρ=0.12 | 23 | 31 | 29 | 24 | 9% | 0.49 | 49% | 4:17 |
| 30 | Sweep unit | unit ℓ₀=7 ρ=0.20 | 9 | 17 | 24 | 17 | 4% | 0.39 | 39% | 1:54 |
| 31 | Sàng KV decode | token r=17 | 23 | 38 | 30 | 29 | 27% | 0.80 | 73% | 0:0 |
| 32 | Sàng KV decode | token r=22 | 23 | 38 | 30 | 29 | 27% | 0.80 | 44% | 0:0 |
| 33 | Sàng KV decode | token r=27 | 19 | 38 | 30 | 28 | 27% | 0.80 | 25% | 0:5 |
| 34 | Sàng KV decode | unit r=27 | 30 | 38 | 30 | 29 | 26% | 0.80 | 25% | 7:0 |
| 35 | Sàng KV decode | unit ρ=0.25 r=22 | 30 | 37 | 30 | 29 | 6% | 0.71 | 16% | 7:1 |
| 36 | Sàng KV decode | unit ρ=0.25 r=27 | 30 | 32 | 28 | 23 | 6% | 0.71 | 4% | 7:14 |

Với SpecPrefill (dòng 10–13), "token×layer" chưa tính chi phí chạy model nháp.

**Bảng 2. kv50 (50 câu, ~170k token)**

| # | Setting kv50 (50 câu, ~170k token) | Đúng /50 | vs LayerSieve gốc | vs dense | Giữ cuối (token) | Đo ở |
|---|---|---|---|---|---|---|
| 1 | Chạy đầy đủ (dense) | 35 | 7:1 | – | 169,160 | H100, 10-03 |
| 2 | LayerSieve gốc (token, ℓ₀=17 ρ=0.12) | 29 | – | 1:7 | 41,654 | H100, 10-03 |
| 3 | unit, ℓ₀=17 ρ=0.12 | 34 | 6:1 | 0:1 | 41,439 | 5880, xong 08:18 |
| 4 | unit, ℓ₀=17 ρ=0.12, sàng KV r=27 | 34 | 7:2 | 0:1 | 41,439 | 5880, xong 07:09 |

**Bảng 3. Tốc độ trên kv50** (mỗi cấu hình 3 prompt ~170k token; `scripts/bench_speed.py`)

Tốc độ chỉ phụ thuộc lịch sàng, không phụ thuộc chọn theo token hay theo unit.

| # | ℓ₀ | ρ | r (sàng KV) | Số prompt | TTFT (s) | TTFT nhanh | TPOT (ms) | TPOT nhanh | Lần đo |
|---|---|---|---|---|---|---|---|---|---|
| 1 | dense | 0.00 | – | 3 | 117.7 | 1.00x | 32.2 | 1.00x | 1 |
| 2 | 17 | 0.12 | – | 3 | 88.7 | 1.33x | 31.4 | 1.03x | 1 |
| 3 | 17 | 0.12 | 17 | 3 | 87.7 | 1.34x | 33.2 | 0.97x | 1 |
| 4 | 17 | 0.12 | 22 | 3 | 87.1 | 1.35x | 29.6 | 1.09x | 1 |
| 5 | 17 | 0.12 | 27 | 3 | 87.6 | 1.34x | 27.2 | 1.18x | 1 |
| 6 | 17 | 0.25 | – | 3 | 75.8 | 1.55x | 30.1 | 1.07x | 1 |
| 7 | 17 | 0.25 | 22 | 3 | 64.8 | 1.82x | 26.7 | 1.21x | 2* |
| 8 | 17 | 0.25 | 27 | 3 | 78.3 | 1.50x | 25.3 | 1.27x | 1 |
| 9 | 14 | 0.06 | – | 3 | 88.2 | 1.33x | 32.1 | 1.00x | 1 |
| 10 | 14 | 0.10 | – | 3 | 79.3 | 1.48x | 31.1 | 1.04x | 1 |
| 11 | 14 | 0.15 | – | 3 | 70.5 | 1.67x | 30.2 | 1.07x | 1 |
| 12 | 14 | 0.25 | – | 3 | 65.6 | 1.79x | 29.4 | 1.10x | 1 |
| 13 | 7 | 0.03 | – | 3 | 83.5 | 1.41x | 32.9 | 0.98x | 1 |
| 14 | 7 | 0.05 | – | 3 | 69.8 | 1.69x | 31.7 | 1.02x | 1 |
| 15 | 7 | 0.08 | – | 3 | 56.7 | 2.08x | 30.4 | 1.06x | 1 |
| 16 | 7 | 0.08 | 27 | 3 | 56.3 | 2.09x | 26.3 | 1.22x | 1 |
| 17 | 7 | 0.12 | – | 3 | 48.5 | 2.43x | 29.4 | 1.10x | 1 |

\* Đo ở lần 2 (lần 1 lúc 03:50–05:04, lần 2 lúc 08:25), khi GPU rảnh hơn, và mốc dense không đo lại. Dòng 6 và 7 có prefill giống hệt nhau mà TTFT lệch 75.8 s so với 64.8 s, nên số của dòng 7 lạc quan khoảng 15%.

**Nhận xét**
- **unit** sửa hết lỗi kv của bản gốc (7:0). Nguyên nhân lỗi: uuid bị cắt mất vài token ở giữa dù evidence vẫn được giữ quá nửa.
  - Trên kv50, unit được 34/50, gần dense (35/50); bản gốc được 29/50.
  - BM25 không thêm gì so với unit. Chỉ dùng BM25 thì vt sụp còn 2/38, nên cần điểm attention bên trong model.
- **SpecPrefill**:
  - Giữ 90% thì đúng gần hết, nên bản port không lỗi.
  - Giữ 30% hoặc 10% thì thua LayerSieve rõ, và model nháp càng nhỏ càng tệ.
  - Cần nói rõ đây là so ở cùng số token giữ lại. SpecPrefill tốn ít FLOP hơn và ở ≤ 27k token có TTFT nhanh hơn LayerSieve, nên so sánh chưa ngang tốc độ.
- **Lịch sàng**:
  - Bắt đầu sàng sớm (ℓ₀ = 7 hoặc 14) luôn thua ℓ₀ = 17 khi cùng khối lượng tính. Ví dụ ℓ₀=7 ρ=0.03 tốn cùng tính toán với bản gốc, giữ gấp đôi token mà kv chỉ 16/30.
  - Unit chịu được sàng mạnh hơn: ℓ₀=17 ρ=0.25 vẫn đúng 30/30 kv, TTFT nhanh 1.55x.
- **Sàng KV**:
  - Chọn theo token, r=27 mất 5 câu kv (chép sai value). Unit + r=27 không mất câu nào, cả trên kv50.
  - Unit ρ=0.25 + r=22 chỉ đọc 16% KV mà không mất câu nào. Unit ρ=0.25 + r=27 (đọc 4% KV) là quá tay.
  - TPOT chỉ nhanh khoảng 1.2x vì ở 170k token, trọng số (~15 GB) vẫn chiếm phần lớn lượng đọc mỗi bước, và bản vá chạy eager (không có CUDA graph).
- Chưa chạy: kv50 cho accum, bm25, all và chỉ BM25. Bản HF `prune.py` chưa có các tính năng mới.
