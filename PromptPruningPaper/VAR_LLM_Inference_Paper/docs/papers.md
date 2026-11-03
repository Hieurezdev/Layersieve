# Paper liên quan

| Paper | Link | Code | Ghi chú |
|---|---|---|---|
| SpecPrefill (Liu et al., 2025) | https://arxiv.org/abs/2502.02789 | https://github.com/Jingyu6/speculative_prefill | Phương pháp đội top 1 dùng. Draft model chấm token, target prefill phần giữ lại. `baselines/SpecPrefill` |
| SlimInfer (Long et al., AAAI 2026) | https://arxiv.org/abs/2508.06447 | https://github.com/Longxmas/SlimInfer | Bỏ hidden state theo block tại vài layer cố định. Gần nhất với LayerSieve. `baselines/SlimInfer` |
| GemFilter (Shi et al., 2024) | https://arxiv.org/abs/2409.17422 | https://github.com/SalesforceAIResearch/GemFilter | Chọn token ở một layer giữa rồi chạy lại cả model trên token đã chọn, đánh lại vị trí. Không LayerSieve. |
| LazyLLM (Fu et al., 2024) | https://arxiv.org/abs/2407.14057 | – | Cắt token động theo layer, có hồi sinh token. SlimInfer xây trên ý này. |
| InfiniteBench (Zhang et al., 2024) | https://arxiv.org/abs/2402.13718 | https://github.com/OpenBMB/InfiniteBench | Nguồn task kv_retrieval. |
