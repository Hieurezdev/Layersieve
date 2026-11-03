"""GemFilter runs vs the plain and copy answers of the *_v runs, per task, on shared prompts.
usage: gem_table.py gem_l14_k2048 [gem_l17_k2048 ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); names, sys.argv = sys.argv[1:], sys.argv[:1]
import selscore as S
R = "results/selfext/"
TASKS = [("KV", "kv20"), ("multikey-3", "mk3_128k"), ("multivalue", "niah_multivalue_128k"), ("LD (first name)", "ld50"),
         ("qa_1", "qa_1_128k"), ("fwe", "fwe_128k"), ("code_debug", "code_debug20"), ("vt", "vt_128k"),
         ("qa_2", "qa_2_128k"), ("cwe", "cwe_128k"), ("math_find", "math_find20")]
G = {n: {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(R + n + ".jsonl")} for n in names}
print("| Task | n | Dense | Copy | " + " | ".join(names) + " |"); print("|---|---|---|---|" + "---|" * len(names))
for t, base in TASKS:
    v = {json.loads(l)["id"]: json.loads(l) for l in open(R + base + "_v.jsonl")}
    ids = [i for i in v if all(i in g for g in G.values())]
    if not ids: print(f"| {t} | 0 |"); continue
    m = lambda xs: f"{100 * sum(xs) / len(xs):.0f}"
    row = [m([S.score(i, v[i]["pred_dense"]) for i in ids]), m([S.score(i, v[i]["pred_recall"]) for i in ids])]
    row += [m([S.score(i, g[i]) for i in ids]) for g in G.values()]
    print(f"| {t} | {len(ids)} | " + " | ".join(row) + " |")
