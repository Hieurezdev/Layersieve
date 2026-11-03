# SlimInfer goc (HF, flash-attn, code chinh thuc trong official/) tren prompt kv da token hoa (data/build_kv.py),
# cung token voi ban vLLM. --scale-ref 0: cau hinh goc Qwen (L8/17/25 giu 12k/6k/4k); >0: nhan so token giu voi T/scale_ref.
import argparse, json, os, sys, time, yaml, copy, torch
OFFICIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "official")
sys.path.insert(0, OFFICIAL)
from transformers import AutoTokenizer
from models.modeling_qwen2 import Qwen2ForCausalLM
from utils.cache_utils import SlimInferCache, HsCache
ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--scale-ref", type=int, default=0); ap.add_argument("--n", type=int, default=0)
ap.add_argument("--cfg", default=os.path.join(OFFICIAL, "prune_configs/b64_t09_w4_prune_fx_8_12_17_6_25_4.yaml")); a = ap.parse_args()
name = "Qwen/Qwen2.5-7B-Instruct-1M"; tok = AutoTokenizer.from_pretrained(name)
model = Qwen2ForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
base = yaml.safe_load(open(a.cfg))
for k, line in enumerate(open(a.prompts)):
    if a.n and k >= a.n: break
    r = json.loads(line); ids = torch.tensor([r["prompt_ids"]], device="cuda"); T = ids.shape[1]
    cfg = copy.deepcopy(base)
    if a.scale_ref:
        for L in cfg["pruning_config"]: cfg["pruning_config"][L]["fixed_keep_tokens"] *= T / a.scale_ref
    kw = {"past_key_values": SlimInferCache(model.device), "hs_cache": HsCache(), "sliminfer_config": cfg,
          "all_original_prompt_indices_abs": list(range(T))}
    torch.cuda.synchronize(); t0 = time.time()
    out = model.generate(ids, max_new_tokens=100, num_beams=1, do_sample=False, use_cache=True, temperature=1.0,
                         eos_token_id=tok.eos_token_id, **kw)[0]
    torch.cuda.synchronize(); e2e = time.time() - t0
    pred = tok.decode(out[T:], skip_special_tokens=True)
    rec = {"id": r["id"], "pred": pred, "ok": any(x in pred for x in r["answers"]), "e2e": e2e}
    print(rec, flush=True); open(a.out, "a").write(json.dumps(rec) + "\n"); torch.cuda.empty_cache()
