"""Baselines from the ASL repo (github.com/TANIGUCHIREI/ASL, ACL 2026 Findings), which ships Qwen2 versions of
FastKV, ASL, GemFilter, PyramidInfer and SnapKV, run on our tokenized eval sets (data/build_infbench.py,
data/build_ruler_eval.py). Model loading and generation follow eval/infinite_bench/run_infinitebench.py of that repo
(HF transformers 4.45, flash-attn 2, greedy); only the data loading is ours. Run with the ASL env (ext/ASL/.venv).
    ext/ASL/.venv/bin/python baselines/asl_driver.py --method fastkv --select-layer 15 --select-k 2048 \
        --prompts data/infbench.jsonl --out results/bench/infbench/fastkv_k2048.jsonl
method: vanilla (dense; with --cache-k = SnapKV), fastkv (TSP at --select-layer), asl (adaptive layer, L_min =
--select-layer, threshold --tau), gemfilter (filter at --select-layer, second pass on the selected tokens), pyramidinfer.
--select-k: tokens kept (int) or, if < 1, a fraction of each prompt.
Output lines: {"id", "task", "pred", "len", "select_k"}; scored by scripts/score_bench.py.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument("--asl", default="/root/PromptPruningPaper/ext/ASL")
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--method", required=True, choices=["vanilla", "fastkv", "asl", "gemfilter", "pyramidinfer"])
ap.add_argument("--select-layer", type=int, default=None, help="FastKV TSP layer / GemFilter filter layer / ASL L_min")
ap.add_argument("--select-k", type=float, default=None, help="tokens kept after selection (< 1: fraction of prompt)")
ap.add_argument("--tau", type=float, default=None, help="ASL rank-variance threshold (paper default 0.3)")
ap.add_argument("--cache-k", type=float, default=None, help="KV budget per layer (SnapKV-style); None = full KV")
ap.add_argument("--cache-p", type=float, default=None)
ap.add_argument("--prompts", required=True)
ap.add_argument("--tasks", default="", help="comma list of tasks (default all)")
ap.add_argument("--lengths", default="", help="RULER only: comma list of lengths to run (task suffix @<len>)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.asl)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.generation.utils  # noqa: E402
from baseline.asl.customized_cache import DynamicCache  # noqa: E402

# every method of the repo expects its DynamicCache (set_cache_kwargs, mlp_chunk_size, ...); generate() builds the cache
# from the name bound in generation.utils, so replace it there too (their harness gets this from its import order)
transformers.cache_utils.DynamicCache = DynamicCache
transformers.generation.utils.DynamicCache = DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402

torch.manual_seed(42)
dtype = torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
if a.method in ("asl", "fastkv"):
    from baseline.asl.monkeypatch import replace_qwen2
    replace_qwen2()
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", attn_implementation="flash_attention_2",
                                                 torch_dtype=dtype)
elif a.method == "gemfilter":
    from baseline.gemfilter.monkeypatch import replace_qwen2
    replace_qwen2()
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", attn_implementation="flash_attention_2",
                                                 torch_dtype=dtype)
elif a.method == "pyramidinfer":
    from baseline.pyramidinfer.utils import get_model, load_pyramid_config
    cfg = "qwen2.5_14b.json" if "14B" in a.model else "qwen2.5_7b.json"
    model = get_model(a.model, torch_dtype=dtype, device_map="auto", attn_implementation="eager", cache_dir=None,
                      load_in_8bit=False)
    model.config.pad_token_id = tok.pad_token_id
    model = load_pyramid_config(model, json.load(open(os.path.join(a.asl, "baseline/pyramidinfer/configs", cfg))))
else:
    from baseline.vanilla.monkeypatch import get_model
    model = get_model(a.model, torch_dtype=dtype, device_map="auto", attn_implementation="flash_attention_2")
model.eval()

# the per-example arguments the ASL hijacks read (names as in run_infinitebench_per_task.sh: L_obs=8, kernel 7,
# window 32, avgpool)
base = dict(default_select_layer=a.select_layer, layer_th=a.tau, window_size=32, kernel_size=7, use_layer_num=8,
            pooling="avgpool", cache_k=a.cache_k, cache_p=a.cache_p, method=a.method)
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
tasks = set(a.tasks.split(",")) if a.tasks else None
lens = set(a.lengths.split(",")) if a.lengths else None
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
for line in open(a.prompts):
    r = json.loads(line)
    if r["id"] in done or (tasks and r["task"].split("@")[0] not in tasks) or \
            (lens and r["task"].split("@")[-1] not in lens):
        continue
    T = len(r["prompt_ids"])
    k = None if a.select_k is None else (int(a.select_k * T) if a.select_k < 1 else int(a.select_k))
    model.cache_kwargs = SimpleNamespace(select_k=k, **base)
    ids = torch.tensor([r["prompt_ids"]], device=model.device)
    with torch.no_grad():
        if a.method == "gemfilter":
            from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection, set_topk
            set_topk(model, k, mode="gemfilter")
            pred = gemfilter_generate_selection(ids, None, model, tok, max_gen_len=r["max_new_tokens"],
                                                select_layer_idx=a.select_layer)[0]
        else:
            gc = GenerationConfig(max_new_tokens=r["max_new_tokens"], num_return_sequences=1, do_sample=False,
                                  temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
            out = model.generate(input_ids=ids, generation_config=gc)
            pred = tok.decode(out[0, T:], skip_special_tokens=True).strip()
    rec = {"id": r["id"], "task": r["task"], "pred": pred, "len": T, "select_k": k}
    print(rec["id"], repr(pred[:80]), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    torch.cuda.empty_cache()
