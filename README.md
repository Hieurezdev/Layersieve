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
  vllm_patch.py         bản vLLM 0.6.3 (1 request), có kèm SlimInfer cài lại (method="slim")
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

## Chạy
```bash
python data/build_kv.py --out data/kv50.jsonl
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --out results/kv50/kv50_catdan.jsonl        # LayerSieve
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --dense --out results/kv50/kv50_dense.jsonl  # dense
python scripts/summarize_kv.py
```
Tái lập toàn bộ bảng kv50: `scripts/run_kv50.sh`.

Thí nghiệm case khó cho SpecPrefill: `scripts/run_hard.sh`, tổng hợp bằng `python scripts/summarize_hard.py`.

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

## Case khó cho SpecPrefill (draft nhỏ sai, target đúng)

Câu hỏi: khi draft nhỏ không trả lời được câu mà model lớn trả lời được, SpecPrefill có sập không?

**Thiết lập**
- Target Qwen2.5-7B-Instruct-1M. Draft Qwen2.5-1.5B-Instruct hoặc 0.5B-Instruct.
- Prompt ≤ 27k token, vì draft chỉ có context 32k. Nếu cho prompt dài hơn, draft sẽ sập vì tràn context chứ không phải vì kém, và kết quả không còn đo đúng điều cần đo.
- SpecPrefill dùng config gốc: look-ahead 8, chunk 32, pool 13, giữ 10% hoặc 30%. LayerSieve dùng ℓ₀ = 17, ρ = 0.12.
- Ba loại task (`data/build_hard.py`):
  - kv: kv_retrieval tự sinh.
  - vt: variable tracking 4 bước, câu hỏi theo chiều xuôi ("giá trị của VAR Q?"), có chuỗi gây nhiễu.
  - QA: HotpotQA và 2WikiMQA lấy từ LongBench.
- Chỉ dùng mẫu target dense trả lời đúng: kv 30, vt 40, Hotpot 30, 2Wiki 30, chọn ngẫu nhiên với seed 0 (`results/hard/ids_sub.txt`).
  - **Tập khó**: draft dense trả lời sai. **Tập dễ**: draft dense trả lời đúng.
- "Giữ evidence" nghĩa là mọi span evidence còn ≥ 1/2 số token sau khi cắt (chỉ đo cho kv và vt).

**Draft 1.5B** (đúng / n)

| Task | Tập | n | SP 10% | SP 30% | LayerSieve |
|---|---|---|---|---|---|
| kv | khó | 13 | 0 | 4 | 10 |
| kv | dễ | 17 | 4 | 9 | 14 |
| vt | khó | 25 | 12 | 17 | **25** |
| vt | dễ | 15 | 11 | 10 | 14 |
| 2WikiMQA | khó | 13 | 7 | 11 | 12 |
| 2WikiMQA | dễ | 17 | 14 | 16 | 17 |
| HotpotQA | khó | 6 | 5 | 6 | 6 |
| HotpotQA | dễ | 24 | 22 | 24 | 24 |

**Draft 0.5B, SpecPrefill giữ 10%, tập khó**

| Task | n | SP 10% | LayerSieve | thắng:thua, p (McNemar) |
|---|---|---|---|---|
| kv | 15 | 0 | 10 | 0:10, 0.002 |
| vt | 39 | 11 | 38 | 1:28, <0.001 |
| 2WikiMQA | 19 | 7 | 18 | 0:11, 0.001 |
| HotpotQA | 18 | 14 | 18 | 0:4, 0.13 |

TTFT trung bình: dense 0.40 s, LayerSieve 0.36 s, SP 1.5B giữ 30% 0.32 s, SP 1.5B giữ 10% 0.25 s, SP 0.5B giữ 10% 0.16 s.

**Nhận xét**
- Draft càng yếu thì SpecPrefill sập càng nặng. LayerSieve không dùng draft nên vẫn đạt 97–100% trên tập khó.
- Ở vt, SpecPrefill thường giữ hop đầu và hop cuối nhưng cắt các hop ở giữa chuỗi, vì draft không lần theo được chuỗi.
- Target chép lại đáp án sai của draft. Với SP 1.5B giữ 10%, 10/17 câu vt sai trùng đúng con số sai mà draft dense đưa ra; khi giữ 30% còn 4/13. Có thể do các token look-ahead (câu trả lời sai của draft) kéo ngân sách giữ token về phía chuỗi gây nhiễu.
- Khi giữ 30%, tập khó và tập dễ của vt gần bằng nhau (17/25 so với 10/15). Như vậy chênh lệch do draft kém chỉ rõ ở ngân sách thấp, nhưng SpecPrefill vẫn thua LayerSieve trên vt khó.
- kv sập ở cả tập dễ, tức là do ngân sách quá thấp chứ không phải do draft. Không nên dùng kv làm bằng chứng cho luận điểm về draft.

**Hạn chế**
- Mẫu nhỏ: Hotpot khó chỉ có 6 mẫu với draft 1.5B.
- Ở ≤ 27k token, SpecPrefill có TTFT nhanh hơn LayerSieve, nên so sánh chưa ngang tốc độ.
- Ngưỡng giữ evidence 1/2 có thể bỏ sót trường hợp mất đúng tên biến.
