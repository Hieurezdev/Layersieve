"""One table over the 11 tasks: plain, copy, copy+select, copy+draft, focus (16 heads), fuse, on the prompts each run
shares with the plain answers of the copy run (*_v). Markdown to stdout."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.argv = sys.argv[:1]
import selscore as S
R = "results/selfext/"
TASKS = [("KV", "kv20"), ("multikey-3", "mk3_128k"), ("multivalue", "niah_multivalue_128k"), ("LD (first name)", "ld50"),
         ("qa_1", "qa_1_128k"), ("fwe", "fwe_128k"), ("code_debug", "code_debug20"), ("vt", "vt_128k"),
         ("qa_2", "qa_2_128k"), ("cwe", "cwe_128k"), ("math_find", "math_find20")]
COLS = [("copy", "_v", "pred_recall"), ("copy+select", "_v", "pred"), ("copy+draft", "_draft", "pred_recall"),
        ("focus 16 heads", "_focus3", "pred_recall"), ("fuse", "_fuse", "pred")]
def load(n):
    p = R + n + ".jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
print("| Task | n | Dense | " + " | ".join(c[0] for c in COLS) + " |")
print("|---|---|---|" + "---|" * len(COLS))
means = {c[0]: [] for c in COLS}; means["dense"] = []
for name, base in TASKS:
    v = load(base + "_v")
    ids = list(v)
    d = [S.score(i, v[i]["pred_dense"]) for i in ids]
    row = [f"{100*sum(d)/len(d):.0f}"]; means["dense"].append(sum(d) / len(d))
    for c, suf, key in COLS:
        r = load(base + suf); com = [i for i in ids if i in r and key in r[i]]
        if len(com) < len(ids):
            row.append("–" if not com else f"({len(com)})"); continue
        x = sum(S.score(i, r[i][key]) for i in com) / len(com); means[c].append(x); row.append(f"{100*x:.0f}")
    print(f"| {name} | {len(ids)} | " + " | ".join(row) + " |")
print("| **Mean** | | " + " | ".join(f"**{100*sum(v)/len(v):.1f}**" if len(v) == len(TASKS) else f"({len(v)} tasks)" for v in [means['dense']] + [means[c[0]] for c in COLS]) + " |")
