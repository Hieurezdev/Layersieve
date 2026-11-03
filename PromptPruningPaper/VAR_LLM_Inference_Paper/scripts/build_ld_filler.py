"""Length vs. interference control for LongDialogue. Keep the window the model's own retrieval heads picked (the
--gem --pad 992 --snap span of recall_eval) at its ORIGINAL position, keep the instruction and the question, keep the
total length, and replace every other context token with filler:
  essay   Paul Graham essays (no screenplay, no character names)
  other   the context of a different LD200 screenplay (its $$MASK$$ replaced by that script's own answer)
  noise   the RULER noise haystack ("The grass is green. The sky is blue. ..."): no names at all
--place near puts the window right before the question instead; --total sets the prompt length (default: original).
If "essay" scores like gem (~42) the long context itself is harmless; if "other" drops, look-alike content (other
names, other scenes) is what pulls the answer away.

    python scripts/build_ld_filler.py --filler essay --out data/ld200_filler_essay.jsonl
"""
import argparse
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="data/infbench_ld200.jsonl")
ap.add_argument("--spans", default="results/selfext/ld50_gem_retr.jsonl,results/selfext/ld200b_gem_retr.jsonl,"
                "results/selfext/ld200c_gem_retr.jsonl")
ap.add_argument("--filler", required=True, choices=["essay", "other", "noise"])
ap.add_argument("--place", default="orig", choices=["orig", "near"])
ap.add_argument("--total", type=int, default=0)
ap.add_argument("--essays", default="/root/team_1/hieurezdev/PromptPruningPaper/ext/RULER/scripts/data/synthetic/json/"
                "PaulGrahamEssays.json")
ap.add_argument("--qlen", type=int, default=128)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
rows = [json.loads(l) for l in open(a.prompts)]
span = {}
for f in a.spans.split(","):
    for l in open(f):
        x = json.loads(l); span[x["id"]] = x["spans"]


def head_end(ids):                                     # end of the instruction line, as in ld_move_probe --mode short
    txt = tok.decode(ids[:400]); c = txt.find("guess who that character is.") + len("guess who that character is.")
    return next(k for k in range(1, 400) if len(tok.decode(ids[:k])) >= c)


if a.filler == "essay":
    E = json.load(open(a.essays))
    pool = tok.encode(E["text"])
if a.filler == "noise":
    pool = tok.encode("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 50)
print("filler pool", a.filler)
with open(a.out, "w") as fo:
    for k, r in enumerate(rows):
        ids = r["prompt_ids"]; h = head_end(ids); q0 = len(ids) - a.qlen
        (lo, hi), = span[r["id"]]
        need = (a.total - h - (hi - lo) - a.qlen) if a.total else (lo - h) + (q0 - hi)
        cut = need if a.place == "near" else min(lo - h, need)       # filler tokens before the window
        if a.filler == "other":
            o = rows[(k + 100) % len(rows)]; oi = o["prompt_ids"]
            txt = tok.decode(oi[head_end(oi):len(oi) - a.qlen]).replace("$$MASK$$", o["answers"][0])
            pool = tok.encode(txt)
        fill = (pool * (need // len(pool) + 1))[:need]
        new = ids[:h] + fill[:cut] + ids[lo:hi] + fill[cut:] + ids[q0:]
        assert len(new) == (a.total or len(ids))
        w0 = h + cut
        fo.write(json.dumps({**r, "prompt_ids": new, "len": len(new), "filler": a.filler, "place": a.place,
                             "window": [w0, w0 + hi - lo], "orig_window": [lo, hi]}) + "\n")
print("wrote", a.out)
