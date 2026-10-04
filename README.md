# VAR_LLM_Inference_Paper

Paper cho Viettel AI Race, task Long Context Inference (prompt pruning).

Đội top 1 dùng SpecPrefill: một draft model nhỏ chấm điểm token rồi model lớn chỉ prefill phần được giữ.
Luận điểm của paper: draft model nhỏ bỏ sót token mà model lớn cần ở task khó, nên cắt token **bên trong chính model lớn** sẽ chính xác hơn.

## Phương pháp: cắt dần theo layer (in-model progressive pruning)
- L0–16 chạy đủ prompt.
- Từ L17, trước mỗi layer bỏ 12% token còn lại có điểm thấp nhất.
- Điểm = attention mà token nhận từ 128 token cuối prompt ở chính layer đó (softmax, max theo head, cộng theo hàng, avg-pool 13).
- Luôn giữ 128 token đầu (sink) và 128 token cuối.
- Token bị bỏ ra khỏi cả attention lẫn MLP; token giữ lại giữ nguyên vị trí RoPE gốc.
- KV cache riêng cho từng layer khi decode.

## Cấu trúc
```
inmodel_prune/        code phương pháp
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

## Cài đặt
```bash
python -m venv .venv && .venv/bin/pip install vllm==0.6.3.post1 transformers==4.50.2 datasets==2.21.0 setuptools
git submodule update --init          # code gốc SlimInfer
```
Baseline SlimInfer gốc cần env riêng (`baselines/SlimInfer/README.md`).

## Chạy
```bash
python data/build_kv.py --out data/kv50.jsonl
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --out results/kv50/kv50_catdan.jsonl        # cắt dần
python scripts/eval_kv_vllm.py --prompts data/kv50.jsonl --dense --out results/kv50/kv50_dense.jsonl  # dense
python scripts/summarize_kv.py
```
Tái lập toàn bộ bảng kv50: `scripts/run_kv50.sh`.

## Kết quả hiện có (kv50, Qwen2.5-7B-Instruct-1M, ~170k token, 1×GPU)
| Phương pháp | Đúng /50 | TTFT |
|---|---|---|
| dense | 35 | 20.8 s |
| **cắt dần** (L17, 12%/layer) | **29** | 15.5 s |
| SlimInfer giữ 0.9/0.75/0.5 | 26 | 16.2 s |
| SlimInfer giữ 0.8/0.6/0.4 (compute ≈ cắt dần) | 24 | 13.9 s |
| SlimInfer giữ 0.5/0.25/0.15 | 10 | 9.4 s |
| SlimInfer cấu hình gốc 12k/6k/4k | 3 | 6.9 s |

Chi tiết: `docs/report_kv50.md`.
