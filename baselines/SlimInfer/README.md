# SlimInfer (Long et al., AAAI 2026, arXiv 2508.06447)

- `official/`: submodule trỏ tới code gốc https://github.com/Longxmas/SlimInfer (commit 39640a5). Repo gốc không có LICENSE nên chỉ tham chiếu, không chép code vào đây.
- `run_kv.py`: chạy bản HF gốc trên prompt kv đã token hoá (`data/build_kv.py`), cùng token với bản vLLM.
- Bản cài lại trên vLLM nằm trong `inmodel_prune/vllm_patch.py` (`method="slim"`). Chỉ cài lại phần chọn block lúc prefill; phần chọn lại block lúc decode và offload CPU không cài.

Env riêng:
```bash
python -m venv .venv && .venv/bin/pip install torch==2.5.1 transformers==4.53.0 accelerate pyyaml datasets matplotlib setuptools
.venv/bin/pip install flash-attn==2.7.4.post1 --no-build-isolation
cd baselines/SlimInfer && .venv/bin/python run_kv.py --prompts ../../data/kv50.jsonl --out out.jsonl
```
Phải dùng flash_attention_2: nhánh sdpa của code gốc gọi prefill với `is_causal=False` và mask `None`.
