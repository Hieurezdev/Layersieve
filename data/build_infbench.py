"""InfiniteBench prompts for the baseline comparison: math_find, code_debug, longdialogue_qa_eng, kv_retrieval, 100
random examples per task (seed 42). Same protocol as the ASL repo (github.com/TANIGUCHIREI/ASL, which reports Qwen2.5-
7B-Instruct-1M), whose eval code is imported rather than copied: default (yarn-mistral) task template, the whole prompt
as one chat message, Qwen chat template, then middle truncation to max_len - max_new_tokens tokens (128k by default).
Output lines: {"id", "task", "idx", "len", "prompt_ids", "answers", "max_new_tokens"}; scored with
compute_scores.get_score_one of the same repo (scripts/eval_bench.py).
    python data/build_infbench.py --out data/infbench.jsonl
"""
import argparse
import json
import random
import sys

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--asl", default="/root/PromptPruningPaper/ext/ASL/eval/infinite_bench")
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--tasks", default="math_find,code_debug,longdialogue_qa_eng,kv_retrieval")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-len", type=int, default=131072)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.asl)
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS, create_prompt, get_answer, truncate_input  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
with open(a.out, "w") as f:
    for task in a.tasks.split(","):
        rows = [json.loads(l, strict=False) for l in open(hf_hub_download(
            "xinrongzhang2022/InfiniteBench", f"{task}.jsonl", repo_type="dataset"))]
        idx = sorted(random.Random(a.seed).sample(range(len(rows)), min(a.n, len(rows))))
        # Code.Debug: 64 instead of 5 tokens. With 5 the model only gets out "The correct option is:" (the template's
        # last line) and every method scores ~0 (dense 0/100 here, 0.76% in the ASL paper); the official scorer reads
        # the letter after "is:", so a longer answer is scored as intended.
        gen = 64 if task == "code_debug" else DATA_NAME_TO_MAX_NEW_TOKENS[task]
        for i in idx:
            eg = rows[i]
            text = create_prompt(eg, task, "qwen", a.asl)                           # unknown name -> default templates
            text = tok.apply_chat_template([{"role": "system", "content": text}], add_generation_prompt=True,
                                           tokenize=False)                         # role as in the ASL repo
            ids = truncate_input(tok.encode(text), a.max_len - gen, manner="middle")
            ans = get_answer(eg, task)
            f.write(json.dumps({"id": f"{task}_{i}", "task": task, "idx": i, "len": len(ids), "prompt_ids": ids,
                                "answers": ans if isinstance(ans, list) else [ans], "max_new_tokens": gen}) + "\n")
        print(task, len(idx), "examples", flush=True)
