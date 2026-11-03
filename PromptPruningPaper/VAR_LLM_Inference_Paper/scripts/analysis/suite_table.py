"""Table of a scripts/run_excerpt_suite.sh directory: per task, the plain answer (pred_dense, pooled over the runs of
that task) and every method (pred_recall; fuse: pred), on the prompts each run shares with the dense answers. Paired
bootstrap 95% CI of method - dense. LD: first-name metric (selscore). Markdown to stdout.
usage: suite_table.py results/selfext/suite/q25_pilot"""
import contextlib, io, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D, sys.argv = sys.argv[1], sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):                   # selscore prints its header on import
    import selscore as S  # noqa: E402

TASKS = ["kv", "ld", "mk3", "mv", "fwe", "qa_1", "code_debug", "qa_2", "vt", "cwe", "math_find"]
METHODS = ["copy", "random", "copy_retr", "move", "gem_win", "gem_topk", "gem_tail", "draft", "focus", "reorder", "fuse"]
runs = {}
for f in os.listdir(D):
    if not f.endswith(".jsonl"): continue
    n = f[:-6]; t = next((t for t in sorted(TASKS, key=len, reverse=True) if n.startswith(t + "_")), None)
    if t is None: continue
    runs[(t, n[len(t) + 1:])] = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, f))}
meths = [m for m in METHODS if any(k[1] == m for k in runs)]
random.seed(0)
def ci(diff):
    bs = sorted(sum(random.choice(diff) for _ in diff) / len(diff) for _ in range(2000))
    return f"[{100 * bs[50]:+.0f};{100 * bs[1949]:+.0f}]"
print("| Task | n | Dense | " + " | ".join(meths) + " |")
print("|---|---|---|" + "---|" * len(meths))
for t in TASKS:
    dense = {}                                                    # plain answers are identical across runs: pool them
    for (tt, m), rs in runs.items():
        if tt == t: dense.update({i: r["pred_dense"] for i, r in rs.items() if "pred_dense" in r})
    if not dense: continue
    d = {i: S.score(i, p) for i, p in dense.items()}
    cells = []
    for m in meths:
        rs = runs.get((t, m))
        if not rs: cells.append("–"); continue
        key = "pred" if m == "fuse" else "pred_recall"
        ids = [i for i in rs if i in d and key in rs[i]]
        if not ids: cells.append("–"); continue
        c = [S.score(i, rs[i][key]) for i in ids]; diff = [x - d[i] for x, i in zip(c, ids)]
        cells.append(f"{100 * sum(c) / len(c):.0f} ({100 * sum(diff) / len(diff):+.0f} {ci(diff)}, n={len(ids)})")
    print(f"| {t} | {len(d)} | {100 * sum(d.values()) / len(d):.0f} | " + " | ".join(cells) + " |")
