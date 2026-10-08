"""vLLM methods on the baseline-comparison eval sets (data/build_infbench.py, data/build_ruler_eval.py): dense,
LayerSieve (inmodel_prune/vllm_patch.py), SlimInfer (re-implemented there, method="slim") and SpecPrefill (original
code, Qwen port baselines/specprefill_qwen.py). Greedy, max_new_tokens per example; decode runs in a CUDA graph
(LayerSieve's patch supports it; output identical to eager). Output lines as baselines/asl_driver.py:
{"id", "task", "pred", "len", "kept"}; scored by scripts/score_bench.py.
    python scripts/eval_bench.py --method layersieve --unit --drop-ratio 0.25 --decode-ref 22 \
        --prompts data/infbench.jsonl --out results/bench/infbench/ls_unit_d0.25_r22.jsonl
    python scripts/eval_bench.py --method specprefill --spec-model Qwen/Qwen2.5-1.5B-Instruct --spec-yarn 4 \
        --spec-config baselines/SpecPrefill/configs/config_p3_full_lah8.yaml --prompts ... --out ...
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--method", required=True, choices=["dense", "layersieve", "slim", "specprefill"])
ap.add_argument("--start-layer", type=int, default=17)
ap.add_argument("--drop-ratio", type=float, default=0.12)
ap.add_argument("--unit", action="store_true")
ap.add_argument("--decode-ref", type=int, default=None)
ap.add_argument("--oneshot-keep", type=float, default=None, help="ablation: one cut at --start-layer to this fraction")
ap.add_argument("--slim-frac", default="", help="SlimInfer: kept fractions after L8/L17/L25, e.g. 0.8,0.6,0.4")
ap.add_argument("--spec-model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--spec-config", default=os.path.join(ROOT, "baselines/SpecPrefill/configs/config_p3_full_lah8.yaml"))
ap.add_argument("--spec-yarn", type=float, default=0.0, help="YaRN factor for the draft only (over its 32k)")
ap.add_argument("--prompts", required=True)
ap.add_argument("--tasks", default="")
ap.add_argument("--lengths", default="", help="RULER only: comma list of lengths to run (task suffix @<len>)")
ap.add_argument("--max-model-len", type=int, default=131072, help="= YaRN x4 draft limit; RULER 128k counts output tokens in it")
ap.add_argument("--gpu-util", type=float, default=0.85)
ap.add_argument("--eager", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

kept_fn = lambda: None  # noqa: E731
if a.method == "specprefill":
    import specprefill_qwen
    specprefill_qwen.enable(a.spec_model, a.spec_config, draft_yarn=a.spec_yarn)
    kept_fn = lambda: (lambda k: len(k) if k else None)(specprefill_qwen.last_kept())  # noqa: E731
elif a.method in ("layersieve", "slim"):
    from inmodel_prune.vllm_patch import enable_inmodel_prune, last_kept
    enable_inmodel_prune(start_layer=a.start_layer, drop_ratio=a.drop_ratio, unit=a.unit, decode_ref=a.decode_ref,
                         oneshot_keep=a.oneshot_keep,
                         mlp_chunk=16384, method="ours" if a.method == "layersieve" else "slim",
                         slim_layers=dict(zip((8, 17, 25), map(float, a.slim_frac.split(",")))) if a.slim_frac else None)
    kept_fn = lambda: (lambda k: k[-1] if k else None)(last_kept())  # noqa: E731

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

graph = not a.eager and a.method != "specprefill"          # SpecPrefill's worker is eager-only
llm = LLM(a.model, enforce_eager=not graph, max_seq_len_to_capture=a.max_model_len, enable_chunked_prefill=False,
          enable_prefix_caching=False, max_num_seqs=1, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util)
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
tasks = set(a.tasks.split(",")) if a.tasks else None
lens = set(a.lengths.split(",")) if a.lengths else None
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
for line in open(a.prompts):
    r = json.loads(line)
    if r["id"] in done or (tasks and r["task"].split("@")[0] not in tasks) or \
            (lens and r["task"].split("@")[-1] not in lens):
        continue
    sp = SamplingParams(temperature=0.0, max_tokens=r["max_new_tokens"])
    pred = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"])], sp, use_tqdm=False)[0].outputs[0].text
    rec = {"id": r["id"], "task": r["task"], "pred": pred.strip(), "len": len(r["prompt_ids"]), "kept": kept_fn()}
    print(rec["id"], repr(rec["pred"][:80]), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
