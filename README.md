# VAR_LLM_Inference_Paper

Paper cho Viettel AI Race, task Long Context Inference (prompt pruning).

Đội top 1 dùng SpecPrefill: một draft model nhỏ chấm điểm token rồi model lớn chỉ prefill phần được giữ.
Luận điểm của paper: draft model nhỏ bỏ sót token mà model lớn cần ở task khó, nên cắt token **bên trong chính model lớn** sẽ chính xác hơn. Phương pháp của repo này tên là **LayerSieve**.

## Phương pháp: LayerSieve

Ý tưởng: không dùng draft model. Chính model lớn chấm điểm token ở từng layer sâu, rồi "sàng" bớt dần token qua các layer.

**Ký hiệu**
- Prompt dài T token, model có L layer (Qwen2.5-7B: L = 28), H query head, head dim d.
- `S_ℓ`: tập token đi vào layer ℓ. Mỗi token giữ vị trí gốc `p_j` (dùng cho RoPE).
- `Z`: n_sink = 128 token đầu (attention sink). `O`: n_obs = 128 token cuối prompt (cửa sổ quan sát). Hai tập này luôn được giữ.
- `ℓ₀ = 17`: layer bắt đầu sàng. `ρ = 0.12`: tỉ lệ bỏ mỗi layer.

**1. Layer nông chạy đủ**

```
S_ℓ = {1, …, T}        với ℓ < ℓ₀
```

**2. Chấm điểm trước mỗi layer ℓ ≥ ℓ₀** (trên các token còn lại `S_ℓ₋₁`, dùng q/k của chính layer ℓ)

```
a_ij^h = softmax_{j ∈ S_ℓ₋₁, p_j ≤ p_i} ( q_i^h · k_j^h / √d )      i ∈ O, j ∈ S_ℓ₋₁
s_j    = Σ_{i ∈ O} max_h a_ij^h
ŝ_j    = trung bình của s trên 13 token liền kề j trong S_ℓ₋₁
```
- `a_ij^h`: attention thật (đã softmax, causal) từ token quan sát i tới token j ở head h.
- `max_h`: chỉ cần một head nhìn mạnh là token được giữ, nên các head tra cứu không bị các head khác pha loãng.
- `Σ_{i ∈ O}`: gom phiếu của cả 128 token cuối thay vì chỉ token cuối cùng.
- `ŝ_j`: làm mượt để giữ trọn cụm token liền nhau (một key/giá trị thường dài nhiều token).

**3. Bỏ token điểm thấp**

```
P_ℓ   = S_ℓ₋₁ \ (Z ∪ O)                     token được phép bỏ
D_ℓ   = ⌊ρ · |P_ℓ|⌋ token có ŝ nhỏ nhất trong P_ℓ
S_ℓ   = S_ℓ₋₁ \ D_ℓ
```
- Token bị bỏ ra khỏi cả attention lẫn MLP từ layer ℓ trở đi. Hidden state của token giữ lại đi tiếp bình thường, vị trí RoPE không đổi.
- Số token còn lại ở layer ℓ là `|Z ∪ O| + (1 − ρ)^(ℓ − ℓ₀ + 1) · (T − 256)`. Với ρ = 0.12, đến L27 còn khoảng 24%.

**4. Decode**
- Mỗi layer chỉ giữ KV của `S_ℓ`. Token sinh ra có vị trí tiếp nối `T, T+1, …`.

**Chi phí**
- Tổng token×layer khoảng `(ℓ₀ + Σ_{k=1}^{L−ℓ₀} (1−ρ)^k) / L ≈ 0.80` so với dense.
- Chấm điểm thêm chỉ tốn `|O| × |S_ℓ|` mỗi layer, rất nhỏ so với attention đầy đủ.

**Vì sao bắt đầu ở L17**
- Probe trên kv cho thấy ở layer nông (vd. L8) evidence chỉ đứng quanh percentile 0.55 theo điểm attention, tức gần ngẫu nhiên.
- Từ L17 trở đi evidence đứng ở percentile 0.86–0.99.
- Vì vậy chấm điểm từ L17 mới đáng tin, và bỏ ít mỗi layer thì lỗi chấm sai ở một layer không làm mất hẳn token.

## Cấu trúc
```
inmodel_prune/        code LayerSieve
  prune.py              bản HF tham chiếu
  vllm_patch.py         bản vLLM 0.6.3 (1 request), có kèm SlimInfer cài lại (method="slim")
baselines/            mỗi thư mục là một paper baseline
  SpecPrefill/          code gốc SpecPrefill (MIT), không sửa
  SlimInfer/            official/ = submodule code gốc (commit 39640a5), run_kv.py = driver chạy kv
data/build_kv.py      tạo prompt InfiniteBench kv_retrieval đã token hoá (data/*.jsonl không commit)
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

## Chạy
```bash
python data/build_kv.py --out data/kv50.jsonl
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --out results/kv50/kv50_catdan.jsonl        # LayerSieve
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --dense --out results/kv50/kv50_dense.jsonl  # dense
python scripts/summarize_kv.py
```
Tái lập toàn bộ bảng kv50: `scripts/run_kv50.sh`.

## Kết quả hiện có (kv50, Qwen2.5-7B-Instruct-1M, ~170k token, 1×GPU)
| Phương pháp | Đúng /50 | TTFT |
|---|---|---|
| dense | 35 | 20.8 s |
| **LayerSieve** (ℓ₀ = 17, ρ = 0.12) | **29** | 15.5 s |
| SlimInfer giữ 0.9/0.75/0.5 | 26 | 16.2 s |
| SlimInfer giữ 0.8/0.6/0.4 (compute ≈ LayerSieve) | 24 | 13.9 s |
| SlimInfer giữ 0.5/0.25/0.15 | 10 | 9.4 s |
| SlimInfer cấu hình gốc 12k/6k/4k | 3 | 6.9 s |

Chi tiết: `docs/report_kv50.md`.
