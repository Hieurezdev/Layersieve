"""SpecPrefill (baselines/SpecPrefill, unmodified) on Qwen2 drafts/targets, vLLM 0.6.3.

The original worker only accepts Llama: it asserts Llama classes, reads config.head_dim (absent in Qwen2Config) and
hard-codes Llama EOS ids for the look-ahead. This module patches those three spots and records the kept indices of
the most recent prefill. Call enable(...) before importing vllm.LLM.
"""
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpecPrefill"))

_STATE = {"kept": None}
QWEN_EOS = [151643, 151645]


def enable(spec_model: str, spec_config_path: str):
    from speculative_prefill import enable_prefill_spec
    enable_prefill_spec(spec_model=spec_model, spec_config_path=spec_config_path)

    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
    from speculative_prefill.vllm_patch.worker import look_ahead_spec_worker as w

    w.LlamaForCausalLM = (LlamaForCausalLM, Qwen2ForCausalLM)      # only used in isinstance asserts
    w.LlamaDecoderLayer = (LlamaDecoderLayer, Qwen2DecoderLayer)
    W = w.LookAheadSpecWorker

    orig_load = W.load_model

    def load_model(self):
        orig_load(self)
        cfg = self.model_runner.model.config
        if getattr(cfg, "head_dim", None) is None:                 # read lazily by _reshape_query
            cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    W.load_model = load_model

    orig_cnt = W._get_actual_look_ahead_cnts
    W._get_actual_look_ahead_cnts = lambda self, outs, stop_ids: orig_cnt(self, outs, QWEN_EOS)

    orig_kept = W._get_kept_indices_from_token_importance

    def kept(self, token_importance):
        out = orig_kept(self, token_importance)
        _STATE["kept"] = out[0].cpu()
        return out
    W._get_kept_indices_from_token_importance = kept


def last_kept() -> Optional[List[int]]:
    """Original prompt positions kept by the most recent SpecPrefill prefill."""
    return _STATE["kept"].tolist() if _STATE["kept"] is not None else None
