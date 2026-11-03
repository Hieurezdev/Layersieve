"""Build the evidence-loss pilot (scripts/pilot_evidence.py): long prompts whose evidence spans are known exactly.
  kv   InfiniteBench-style kv_retrieval over uuid pairs (as data/build_hard.py), evidence = the queried pair
  vt   RULER-style variable tracking, 4 chains x 5 statements in noise (as data/build_hard.py), evidence = the 5
       statements of the queried chain (hop k+1 shares no word with the question)
  mk   RULER niah_multikey_2-style: the haystack is made only of needles "One of the special magic numbers for
       <key> is: <7 digits>." with word-pair keys, evidence = the queried needle
  qa2  synthetic two-hop QA in natural filler (LongBench gov_report): "<person> was born in the town of <town>." and
       "The town of <town> lies in the province of <province>.", 4 chains, question names only the person, evidence =
       both facts of the queried chain (the second hop shares no word with the question)
Output lines: {"id", "task", "len", "prompt_ids", "answers", "evidence": [[tok_start, tok_end), ...], "max_new_tokens"}
    python data/build_pilot.py --lengths 32000,64000 --n 12 --n-vt 36 --out data/pilot.jsonl
"""
import argparse
import json
import random
import re
import string
import uuid
import zipfile

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
ap.add_argument("--lengths", default="32000,64000")
ap.add_argument("--n", type=int, default=12, help="prompts per task and length")
ap.add_argument("--n-vt", type=int, default=36, help="vt prompts per length (dense solves only ~1/4 of them)")
ap.add_argument("--tasks", default="kv,vt,mk,qa2")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = random.Random(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)

z = zipfile.ZipFile(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
GOV = [json.loads(l)["context"] for l in z.open("data/gov_report.jsonl")]
FILLER = [s.strip() for doc in GOV for s in re.split(r"(?<=[.!?])\s+", doc) if 40 <= len(s.strip()) <= 300]
WORDS = sorted({w for doc in GOV[:50] for w in re.findall(r"\b[a-z]{5,10}\b", doc)})


def chat(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def ntok(s):
    return len(tok(s, add_special_tokens=False).input_ids)


def record(rid, task, prompt, answers, ev_strings, max_new):
    """Tokenize; map each evidence substring (unique in the prompt) to its token span."""
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    ev = []
    for s in ev_strings:
        assert prompt.count(s) == 1, (rid, s)
        c0 = prompt.index(s)
        c1 = c0 + len(s)
        t = [i for i, (x, y) in enumerate(enc.offset_mapping) if y > c0 and x < c1]
        ev.append([t[0], t[-1] + 1])
    return {"id": rid, "task": task, "len": len(enc.input_ids), "prompt_ids": enc.input_ids, "answers": answers,
            "evidence": ev, "max_new_tokens": max_new}


def pseudo(k=None):
    """Capitalized pronounceable pseudo-name (not a real word, so the answer cannot come from prior knowledge)."""
    c, v = "bcdfghklmnprstvz", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) for _ in range(k or rng.choice([2, 3]))).capitalize()


def place(statements, filler, rng):
    """Insert statements (order kept) at random slots between filler sentences."""
    slots = sorted(rng.sample(range(len(filler) + 1), len(statements)))
    out, si = [], 0
    for i in range(len(filler) + 1):
        while si < len(slots) and slots[si] == i:
            out.append(statements[si])
            si += 1
        if i < len(filler):
            out.append(filler[i])
    return " ".join(out)


def build_kv(target, k):
    kv_tok = ntok(json.dumps({str(uuid.UUID(int=1)): str(uuid.UUID(int=2))})) - 2
    pairs = [(str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128))))
             for _ in range(max(2, (target - 100) // kv_tok))]
    key, val = pairs[rng.randrange(len(pairs))]
    ctx = "JSON data:\n" + json.dumps(dict(pairs), indent=0)
    q = f'Key: "{key}"\nThe value associated with the specified Key is: '
    user = f"Extract the value corresponding to the specified key in the JSON object below.\n\n{ctx}\n\n{q}"
    return record(f"kv_{target}_{k}", "kv", chat(user), [val], [f'"{key}": "{val}"'], 128)


NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


def build_vt(target, k, hops=4, n_chains=4):
    names = set()
    while len(names) < n_chains * (hops + 1):
        names.add("".join(rng.choices(string.ascii_uppercase, k=5)))
    names = sorted(names)
    vals = rng.sample(range(10000, 99999), n_chains)
    chains = []
    for c in range(n_chains):
        v = names[c * (hops + 1):(c + 1) * (hops + 1)]
        chains.append([f"VAR {v[0]} = {vals[c]}."] + [f"VAR {v[i]} = VAR {v[i - 1]}." for i in range(1, len(v))])
    head = "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    tail = (f"\n\nQuestion: Following the chain(s) of variable assignment in the text above, what is the numerical "
            f"value of VAR {names[hops]}? Answer with only the number.")
    n_noise = max(1, (target - ntok(head + tail) - 15 * n_chains * (hops + 1)) // ntok(NOISE + " "))
    seq = [c for c in range(n_chains) for _ in range(hops + 1)]
    rng.shuffle(seq)
    nxt = [0] * n_chains
    stm = []
    for c in seq:
        stm.append(chains[c][nxt[c]])
        nxt[c] += 1
    user = head + place(stm, [NOISE] * n_noise, rng) + tail
    return record(f"vt_{target}_{k}", "vt", chat(user), [str(vals[0])], chains[0], 32)


def build_mk(target, k):
    def key():
        return f"{rng.choice(WORDS)}-{rng.choice(WORDS)}"
    one = "One of the special magic numbers for abcdefg-hijklmn is: 1234567. "
    n = (target - 150) // ntok(one)
    keys = set()
    while len(keys) < n:
        keys.add(key())
    keys = sorted(keys)
    rng.shuffle(keys)
    vals = [str(rng.randrange(1000000, 9999999)) for _ in keys]
    needles = [f"One of the special magic numbers for {kk} is: {v}." for kk, v in zip(keys, vals)]
    qi = rng.randrange(n)
    user = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz "
            "you about the numbers afterwards.\n" + " ".join(needles) +
            f"\nWhat is the special magic number for {keys[qi]} mentioned in the provided text? "
            f"The special magic number for {keys[qi]} mentioned in the provided text is")
    return record(f"mk_{target}_{k}", "mk", chat(user), [vals[qi]], [needles[qi]], 32)


def build_qa2(target, k, n_chains=4):
    people, towns, provs = set(), set(), set()
    while len(people) < n_chains:
        people.add(f"{pseudo()} {pseudo()}")
    while len(towns) < n_chains:
        towns.add(pseudo(3))
    while len(provs) < n_chains:
        provs.add(pseudo(3))
    people, towns, provs = sorted(people), sorted(towns), sorted(provs)
    rng.shuffle(towns)
    rng.shuffle(provs)
    hop1 = [f"{p} was born in the town of {t}." for p, t in zip(people, towns)]
    hop2 = [f"The town of {t} lies in the province of {v}." for t, v in zip(towns, provs)]
    stm = hop1 + hop2
    rng.shuffle(stm)
    head = "Read the following documents and answer the question at the end.\n\n"
    tail = (f"\n\nQuestion: In which province is the town where {people[0]} was born? "
            f"Answer with only the name of the province.")
    budget = target - ntok(head + tail) - 20 * len(stm)
    start = rng.randrange(len(FILLER))
    fill, used = [], 0
    while used < budget:
        s = FILLER[(start + len(fill)) % len(FILLER)]
        fill.append(s)
        used += ntok(s) + 1
    user = head + place(stm, fill, rng) + tail
    return record(f"qa2_{target}_{k}", "qa2", chat(user), [provs[0]], [hop1[0], hop2[0]], 32)


BUILD = {"kv": build_kv, "vt": build_vt, "mk": build_mk, "qa2": build_qa2}
recs = []
for L in map(int, a.lengths.split(",")):
    for t in a.tasks.split(","):
        recs += [BUILD[t](L, k) for k in range(a.n_vt if t == "vt" else a.n)]
with open(a.out, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print({t: sum(r["task"] == t for r in recs) for t in a.tasks.split(",")},
      "len", min(r["len"] for r in recs), "-", max(r["len"] for r in recs))
