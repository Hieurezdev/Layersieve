"""LongBench predictions with in-model progressive pruning (HF), same prompts / output format as SpecPrefill's
baselines/SpecPrefill/eval/long_bench/pred_vllm.py, so its eval.py can score them. --dense runs the same code path without pruning (baseline).

Run from the repo root:
    python scripts/longbench_hf.py --model Qwen/Qwen2.5-7B-Instruct-1M --start-layer 17 --drop-ratio 0.12 --exp inmodel_s17_d12
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inmodel_prune import InModelPruneConfig, load_model, prefill_and_generate  # noqa: E402

NO_TEMPLATE = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
ALL = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
       "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
       "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
ALL_E = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
         "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--start-layer", type=int, default=17)
    p.add_argument("--drop-ratio", type=float, default=0.12)
    p.add_argument("--n-obs", type=int, default=128)
    p.add_argument("--n-sink", type=int, default=128)
    p.add_argument("--pool-kernel", type=int, default=13)
    p.add_argument("--dense", action="store_true", help="no pruning (same code path)")
    p.add_argument("--datasets", default="", help="comma list; default = all LongBench (or LongBench-E with --e)")
    p.add_argument("--e", action="store_true", help="LongBench-E")
    p.add_argument("--max-len", type=int, default=0, help=">0: middle-truncate prompts to this many tokens (LongBench style)")
    p.add_argument("--exp", default=None)
    return p.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)


def main():
    seed_everything(227)
    a = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.join(here, "../baselines/SpecPrefill/eval/long_bench/configs")
    dataset2prompt = json.load(open(os.path.join(cfg_dir, "dataset2prompt.json")))
    dataset2maxlen = json.load(open(os.path.join(cfg_dir, "dataset2maxlen.json")))
    cfg = InModelPruneConfig(start_layer=a.start_layer, drop_ratio=a.drop_ratio, n_obs=a.n_obs,
                             n_sink=a.n_sink, pool_kernel=a.pool_kernel)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model, cfg)
    eos = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>")
                                if tok.convert_tokens_to_ids(t) is not None}

    names = a.datasets.split(",") if a.datasets else (ALL_E if a.e else ALL)
    exp = a.exp or (a.model.replace("/", "_") + ("_dense" if a.dense else f"_inmodel_s{a.start_layer}_d{a.drop_ratio}"))
    out_dir = os.path.join(here, "../results/long_bench", "pred_e" if a.e else "pred", exp)
    os.makedirs(out_dir, exist_ok=True)

    for name in names:
        data = load_dataset("THUDM/LongBench", f"{name}_e" if a.e else name, split="test")
        out_path = os.path.join(out_dir, f"{name}.jsonl")
        done = sum(1 for _ in open(out_path)) if os.path.exists(out_path) else 0          # resume
        print(f"Evaluating {name}: {len(data)} samples ({done} done)")
        for i, obj in enumerate(tqdm(data)):
            if i < done:
                continue
            prompt = dataset2prompt[name].format(**obj)
            if name not in NO_TEMPLATE:
                prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, add_special_tokens=name in NO_TEMPLATE).input_ids
            if a.max_len and len(ids) > a.max_len:
                half = a.max_len // 2
                ids = ids[:half] + ids[-half:]
            ids = torch.tensor([ids], device="cuda")
            out, st = prefill_and_generate(model, ids, cfg, max_new_tokens=dataset2maxlen[name], eos_token_ids=eos, prune=not a.dense)
            pred = tok.decode(out, skip_special_tokens=True)
            if name == "samsum":
                pred = pred.split("\n")[0]
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": obj["answers"], "all_classes": obj["all_classes"], "length": obj["length"],
                           "ttft": st.ttft, "prompt_len": st.prompt_len, "kept_last_layer": st.kept_per_layer[-1]}, f, ensure_ascii=False)
                f.write("\n")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
