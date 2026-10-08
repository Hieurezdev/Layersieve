# VAR_LLM_Inference_Paper

Paper cho Viettel AI Race, task Long Context Inference (prompt pruning). Phương pháp: **LayerSieve**.

- Đội top 1 (MOE) dùng SpecPrefill: một draft model nhỏ chấm điểm token, model lớn chỉ prefill phần được giữ. Ở task khó, draft nhỏ bỏ sót token mà model lớn cần.
- Cắt token ngay bên trong model lớn không còn là ý mới: FastKV (Findings ACL 2026) và ASL (Findings ACL 2026) đều cắt **một lần** ở một layer. LayerSieve khác ở ba điểm:
  1. **Sàng dần** qua nhiều layer, bắt đầu từ khoảng nửa độ sâu model ($\ell_0\approx 0.5L$). Layer sau sửa được lỗi chấm sai của layer trước.
  2. **Bỏ nguyên đơn vị** (dòng, câu, mệnh đề) thay vì từng token.
  3. **Sàng KV khi decode**: dùng lựa chọn của một layer sâu để cắt KV ở các layer nông.

## Phương pháp: LayerSieve

**Ký hiệu**
- Prompt dài $T$ token, model có $L$ layer (Qwen2.5-7B: $L=28$), $H$ query head, head dim $d$.
- $S_\ell$: tập token đi vào layer $\ell$. Mỗi token $j$ giữ vị trí gốc $p_j$ (dùng cho RoPE).
- $Z$: $n_{\text{sink}}=128$ token đầu (attention sink). $O$: $n_{\text{obs}}=128$ token cuối prompt (cửa sổ quan sát). Hai tập này luôn được giữ.
- $\ell_0$: layer bắt đầu sàng (mặc định 17 với $L=28$; quy tắc chung $\ell_0\approx 0.5L$, xem Q1). $\rho$: tỉ lệ bỏ mỗi layer (0.12 hoặc 0.25).

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
\sum_{i\in O}\max_{h\in\{1,\dots,H\}} a^{h}_{ij}.
```
- $a^h_{ij}$: attention thật (đã softmax, causal) từ token quan sát $i$ tới token $j$ ở head $h$.
- $\max_h$: chỉ cần một head nhìn mạnh là token được giữ, nên các head tra cứu không bị các head khác pha loãng.
- $\sum_{i\in O}$: gom phiếu của cả 128 token cuối thay vì chỉ token cuối cùng.

**3. Bỏ nguyên đơn vị có điểm thấp**
- Prompt được chia thành đơn vị $u$. Mỗi đơn vị kết thúc ở xuống dòng hoặc `. , ; ! ?`, dài tối đa 128 token (`inmodel_prune/units.py`). Điểm của đơn vị là $\hat{s}_u = \max_{j\in u} s_j$.

```math
P_\ell = S_{\ell-1} \setminus (Z \cup O),
\qquad
S_\ell = S_{\ell-1} \setminus D_\ell,
\qquad
|D_\ell| \gtrsim \left\lfloor \rho\,|P_\ell| \right\rfloor.
```
- $D_\ell$: hợp các đơn vị (nằm trọn trong $P_\ell$) có $\hat{s}_u$ nhỏ nhất, lấy dần cho tới khi đủ $\lfloor \rho |P_\ell| \rfloor$ token.
- Token bị bỏ ra khỏi cả attention lẫn MLP từ layer $\ell$ trở đi. Token giữ lại đi tiếp bình thường, vị trí RoPE không đổi.
- Số token còn lại ở layer cuối: $(1-\rho)^{L-\ell_0}$ của phần được phép bỏ. Với $\ell_0=17$, $\rho=0.12$ thì còn khoảng 24.5%; với $\rho=0.25$ thì còn khoảng 4%.
- Bản chọn theo token (dùng ở các thí nghiệm đầu) làm mượt $s_j$ bằng avg-pool 13 rồi bỏ từng token. Bản theo đơn vị sửa được hết lỗi kv của bản này: uuid bị cắt mất vài token ở giữa.

**4. Decode và sàng KV**
- Mỗi layer chỉ giữ KV của $S_\ell$. Token sinh ra có vị trí tiếp nối $T, T+1, \dots$
- Sàng KV (`decode_ref` $= r$): ngay sau prefill, mỗi layer $\ell < r$ chỉ giữ KV của $S_r$, nên mỗi bước decode đọc ít KV hơn.
- Decode chạy được trong CUDA graph: độ dài cache và slot của từng layer được tính trên GPU từ một buffer cố định, kết quả giống hệt bản eager.

**Chi phí prefill** (tổng token×layer so với dense)

```math
\frac{1}{L}\left(\ell_0 + \sum_{k=1}^{L-\ell_0}(1-\rho)^k\right)
```
- Với $L=28$, $\ell_0=17$: bằng 0.80 khi $\rho=0.12$ và 0.71 khi $\rho=0.25$.
- Chấm điểm chỉ tốn $O(|O|\cdot|S_{\ell-1}|)$ mỗi layer, rất nhỏ so với attention đầy đủ $O(|S_\ell|^2)$.

## Kết quả

Mọi số đo đều trên **1×H200**, vLLM 0.6.3. Model mặc định: Qwen2.5-7B-Instruct-1M. Độ chính xác là greedy, chỉ tính trên các câu mà model dense làm đúng.

### Q1. Layer bắt đầu sàng tỉ lệ theo độ sâu ($\ell_0\approx 0.5L$), không cố định

- **Bộ case khó** (`data/build_hard.py`, ≤27k token): kv, variable tracking (vt), HotpotQA, 2WikiMQA. Mỗi task lấy 30 câu mà dense làm đúng, tổng 120 câu (3B: 117 câu).
- **Lịch sàng**: chọn theo đơn vị; $\rho$ được chỉnh để layer cuối luôn giữ 24.5% token, nên các cột chỉ khác nhau ở $\ell_0$.
- **Probe**: nếu chọn token một lần ở layer $\ell$ (giữ top 25% theo đơn vị), evidence của kv/vt có còn không. Cột "Probe ổn định từ" là layer đầu tiên mà từ đó trở đi recall ≥ 0.85, không tính 2 layer cuối.

Ô ghi phần trăm câu đúng so với dense:

| Model ($L$) | 0.25L | 0.33L | 0.4L | **0.5L** | 0.6L | 0.7L | 0.8L | $\ell_0=17$ | Probe ổn định từ |
|---|---|---|---|---|---|---|---|---|---|
| 7B-1M (28) | 92 | 95 | 97 | **100** | 100 | 100 | 100 | 100 | L14 (0.50L) |
| 7B (28) | 90 | 89 | 92 | **96** | 96 | 97 | 98 | 96 | L14 (0.50L) |
| 3B (36) | 85 | 85 | 89 | **95** | 97 | 99 | 100 | 97 | L19 (0.53L) |
| 14B (48) | 89 | 94 | 96 | **99** | 100 | 100 | 100 | 96 | L22 (0.46L) |
| 14B-1M (48) | 91 | 96 | 96 | **100** | 100 | 100 | 100 | 96 | L18–22 |
| 32B (64) | 88 | 91 | 94 | **98** | 99 | 100 | 100 | 88 | L38 (0.59L) |

- Model cùng cỡ có cùng mốc: 7B và 7B-1M đều ở L14; 14B và 14B-1M đều ở L24.
- Model khác cỡ thì mốc dịch theo số layer.
- Nếu giữ nguyên layer 0–16 cho mọi model ($\ell_0=17$), 32B chỉ còn 88%. Nếu giữ luôn $\rho=0.12$ thì 32B chỉ còn **37%**.
- 3B và 7B (bản thường) yếu hơn ở context dài, nên vẫn lệch vài câu tới các $\ell_0$ cao.

### Q2. Tốc độ prefill theo cỡ model (TTFT so với dense, ở 128k)

| Model | $\ell_0\approx 0.5L$ (giữ độ chính xác) | $\ell_0=17$, $\rho=0.12$ cố định | Độ chính xác của "$\ell_0=17$, $\rho=0.12$" |
|---|---|---|---|
| 3B | 1.47x | 1.70x | 90% |
| 7B-1M | 1.45x | 1.33x | 100% |
| 14B | 1.45x | 2.22x | 88% (14B-1M: 82%) |
| 32B | 1.43x | 2.90x | 37% |

TTFT bám sát tỉ lệ token×layer ở mọi cỡ model. Khi $\ell_0$ tỉ lệ theo độ sâu, mức tăng tốc gần như không đổi theo cỡ model, khoảng 1.45x. Với 3B, 14B và 32B, các số này đo bằng YaRN ×4 (chỉ để đo tốc độ).

### Q3. Sàng KV khi decode, context tới 1M (7B-1M)

Tốc độ đo với decode chạy trong CUDA graph (decode 512 token, 2 prompt mỗi ô). Độ chính xác đo trên 120 câu case khó và kv50 (InfiniteBench kv_retrieval, 50 câu, ~170k token):

| Setting (unit, $\ell_0=17$) | Case khó /120 | kv50 /50 | Prefill nhanh (1M) | TPOT nhanh 128k | 256k | 512k | **1M** |
|---|---|---|---|---|---|---|---|
| dense | 120 | 35 | 1.00x | 1.00x (9.1 ms) | 1.00x (11.4 ms) | 1.00x (14.7 ms) | 1.00x (23.9 ms) |
| $\rho=0.12$ | 120 | 35 | 1.37x | 1.01x | 1.11x | 1.06x | 1.19x |
| **$\rho=0.12$, $r=27$** | 119 | 34 | 1.38x | 1.29x | 1.48x | 1.55x | **1.83x** |
| $\rho=0.25$ | 118 | 33 | 1.52x | 1.16x | 1.15x | 1.14x | 1.26x |
| **$\rho=0.25$, $r=22$** | 117 | 33 | 1.53x | 1.27x | 1.54x | 1.70x | **2.17x** |
| $\rho=0.25$, $r=27$ | 103 | 34 | 1.53x | 1.28x | 1.57x | 1.97x | 2.49x |

- Hai setting được chọn:
  - $\rho=0.12$, $r=27$: **an toàn**, chỉ mất 1 câu ở mỗi bộ.
  - $\rho=0.25$, $r=22$: **cân bằng**, nhanh hơn rõ.
- $\rho=0.25$, $r=27$ mất 15 câu ở vt và QA, nên bị loại.
- Chỉ sàng prefill (không sàng KV) thì decode gần như không nhanh hơn. Sàng KV mới là nguồn chính của mức tăng tốc decode, và mức tăng lớn dần theo độ dài context.
- 14B-1M (48 layer, $\ell_0=29\approx0.6L$, $r=47$): TPOT nhanh 1.38x ở 128k và 1.69x ở 256k.
- Decode ở chế độ eager trên H200 bị nghẽn ở CPU (TPOT 13–25 ms, dao động mạnh), nên mọi số TPOT ở đây đều đo với CUDA graph.

### 4. Draft model nhỏ thất bại ở task khó (SpecPrefill)

Đo trên RTX 5880, case khó ≤27k token. Mẫu là các câu mà target 7B-1M làm đúng (kv /30, vt /38, hotpot /30, 2wiki /29).

| | kv | vt | hotpot | 2wiki |
|---|---|---|---|---|
| Target 7B-1M (dense) | 30 | 38 | 30 | 29 |
| Draft 1.5B (dense) | 17 | 7 | 24 | 17 |
| Draft 0.5B (dense) | 15 | 1 | 11 | 12 |
| LayerSieve, chọn theo token ($\rho=0.12$, giữ 27%) | 23 | 38 | 30 | 29 |
| **LayerSieve, chọn theo unit** ($\rho=0.12$, giữ 26%) | **30** | **38** | **30** | **29** |
| SpecPrefill, draft 1.5B, giữ 90% | 30 | 29 | 30 | 29 |
| SpecPrefill, draft 1.5B, giữ 30% | 13 | 25 | 30 | 27 |
| SpecPrefill, draft 1.5B, giữ 10% | 4 | 18 | 27 | 21 |

Draft càng nhỏ và giữ càng ít thì SpecPrefill càng sụp, nhất là ở kv và vt. Hai số liệu khác từ các thí nghiệm này:
- Thêm BM25 không cải thiện so với chọn theo unit.
- Chỉ dùng BM25 (không dùng attention) thì vt sụp còn 2/38.

### 5. So sánh với baseline (đang chạy)

**Bộ đánh giá:**
- **InfiniteBench**: Math.Find, Code.Debug, LongDialogue QA, KV Retrieval. Mỗi task 100 câu (seed 42), cắt về 128k theo giao thức của repo ASL. Code.Debug sinh 64 token thay vì 5, vì với 5 token mọi phương pháp đều được khoảng 0%.
- **RULER**: đủ 13 task ở 32k, 64k, 128k. Mỗi ô sinh 500 mẫu, chạy 100 mẫu (seed 42).

**Baseline:**
- SpecPrefill (ICML 2025): draft Qwen2.5-1.5B, bật YaRN ×4 cho draft ở 64k/128k.
- SlimInfer (AAAI 2026).
- FastKV (Findings ACL 2026).
- ASL (Findings ACL 2026).
- SnapKV (NeurIPS 2024).

FastKV, ASL và SnapKV dùng code Qwen2 của repo ASL. PyramidInfer bị bỏ vì tràn bộ nhớ từ 32k trở lên. GemFilter và EHPC bị bỏ: GemFilter không rõ nơi đăng, EHPC không có code.

**Hai mức ngân sách:**
- **Cùng chi phí prefill** với LayerSieve $\rho=0.25$ (token×layer ≈ 0.71): FastKV cắt ở L15 giữ 37.5%; ASL τ=0.3 giữ 37.5%; SlimInfer giữ 0.75/0.55/0.35; SpecPrefill giữ 50%.
- **Mặc định của từng paper**: giữ 2.048 token, SpecPrefill 10%, SlimInfer 12k/6k/4k.

Dense hiện có (đối chiếu với full KV trong paper ASL):

| Task | Mình | Paper ASL |
|---|---|---|
| Math.Find | 37.0 | 38.57 |
| En.Dia | 11.0 | 9.0 |
| Retr.KV | 66.0 | 66.4 |
| RULER 32k | 86.7 | – |
| RULER 64k | 82.9 | – |

**Hàng đợi** (`scripts/run_bench_priority.sh`):
- **P1**: LayerSieve (2 setting), FastKV, ASL, SpecPrefill trên InfiniteBench + RULER 128k.
- **P2**: ablation {cắt một lần, sàng dần} × {token, unit}, cùng chi phí.
- **P3**: RULER 32k/64k, SlimInfer, mức mặc định của paper, SnapKV.

**Còn cần làm:**
- Đường Pareto: 3–4 mức ngân sách cho LayerSieve, FastKV, ASL, SpecPrefill.
- Họ model thứ hai: nvidia/Llama-3.1-8B-UltraLong-1M-Instruct (cùng model với paper ASL).
- Kiểm định McNemar hoặc bootstrap cho mọi bảng.

## Cấu trúc
```
inmodel_prune/          code LayerSieve
  vllm_patch.py           bản vLLM 0.6.3 (1 request): sàng dần, unit, sàng KV, decode CUDA graph, probe,
                          cắt một lần (ablation), SlimInfer cài lại (method="slim")
  units.py                chia đơn vị, BM25, bỏ nguyên đơn vị
  prune.py                bản HF tham chiếu (chỉ có bản chọn theo token)
baselines/
  SpecPrefill/            code gốc SpecPrefill (MIT), không sửa
  specprefill_qwen.py     cho SpecPrefill chạy với Qwen2; YaRN riêng cho draft
  asl_driver.py           FastKV / ASL / SnapKV bằng code của repo ASL (env riêng ext/ASL/.venv)
  SlimInfer/              submodule code gốc + driver kv
data/                   dựng prompt (data/*.jsonl, data/ruler/ không commit)
  build_hard.py           case khó ≤30k
  build_kv.py             kv50
  build_speed.py          prompt đo tốc độ 32k–1M
  build_infbench.py       InfiniteBench 4 task × 100
  build_ruler.sh          sinh RULER (cần repo NVIDIA/RULER)
  build_ruler_eval.py     chọn và tokenize mẫu RULER
scripts/
  run_scaling.sh          Q1–Q3 (speed: Q2, Q3; acc: Q1) -> results/scaling, tổng hợp bằng summarize_scaling.py
  run_decode_acc.sh       độ chính xác của sàng KV -> results/scaling/decode_acc
  run_bench_priority.sh   so sánh baseline -> results/bench, tổng hợp bằng score_bench.py
  bench_speed.py, eval_hard.py, eval_bench.py, probe_layers.py   các driver
results/                kết quả từng mẫu (jsonl)
docs/                   paper liên quan, report kv50 cũ, technical report của đội MOE
```

## Cài đặt (uv, Python 3.10)
```bash
git submodule update --init                          # code gốc SlimInfer
uv venv -p 3.10 .venv && uv pip install -p .venv -r requirements.txt
```
Baseline từ repo ASL cần env riêng (torch 2.1, transformers 4.45, flash-attn 2.6.3) trong `ext/ASL/.venv`.

## Chạy
```bash
# dữ liệu
python data/build_hard.py --out data/hard.jsonl
python data/build_kv.py --out data/kv50.jsonl
python data/build_speed.py --out data/speed.jsonl
python data/build_infbench.py --out data/infbench.jsonl
bash data/build_ruler.sh && python data/build_ruler_eval.py --out data/ruler_eval.jsonl
# thí nghiệm (chạy nền, mỗi job có file .done nên chạy lại sẽ bỏ qua phần đã xong)
bash scripts/run_scaling.sh speed     # Q2, Q3 (cần GPU trống)
bash scripts/run_scaling.sh acc       # Q1
bash scripts/run_decode_acc.sh        # độ chính xác cho Q3
bash scripts/run_bench_priority.sh    # mục 5
# tổng hợp
python scripts/summarize_scaling.py
python scripts/score_bench.py
```
