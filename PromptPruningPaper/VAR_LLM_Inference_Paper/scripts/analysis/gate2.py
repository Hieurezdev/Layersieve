import json, sys, os, math, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.argv = sys.argv[:1]
import selscore as S
names = "kv20_v niah_multivalue_128k_v cwe_128k_v vt_128k_v qa_2_128k_v mk3_128k_v ld200_v qa_1_128k_v fwe_128k_v math_find20_v code_debug20_v".split()
D = {}
for n in names:
    rs = [json.loads(l) for l in open(f"results/selfext/{n}.jsonl")]
    D[n] = [(r, S.score(r["id"], r["pred_dense"]), S.score(r["id"], r["pred_recall"])) for r in rs if "lp_recall_in_dense" in r]
print("per task: median lp_dense when dense right / wrong; median share")
for n, v in D.items():
    ok = [r["lp_dense"] for r, d, c in v if d >= 0.99]; bad = [r["lp_dense"] for r, d, c in v if d < 0.99]
    f = lambda x: f"{st.median(x):.3f}" if x else "  -  "
    print(f"{n:24s} right {f(ok)} wrong {f(bad)} share {st.median(r['share'] for r,_,_ in v):.2f} agree {sum(r['pred_dense']==r['pred_recall'] for r,_,_ in v)}/{len(v)}")
def run(rule):
    return [sum((c if rule(r) else d) for r, d, c in v) / len(v) for v in D.values()]
rules = {"dense": lambda r: False, "copy": lambda r: True, "own": lambda r: r["lp_recall"] > r["lp_dense"]}
for tau in [-0.01, -0.02, -0.05, -0.1, -0.2]:
    rules[f"dense_conf>{tau}"] = (lambda t: lambda r: r["lp_dense"] < t)(tau)
    rules[f"own|conf{tau}"] = (lambda t: lambda r: r["lp_dense"] < t and r["lp_recall"] > r["lp_dense"])(tau)
print(f"\n{'rule':20s} mean  " + " ".join(n.split('_')[0][:5] for n in names))
for k, f in rules.items():
    v = run(f); print(f"{k:20s} {100*sum(v)/len(v):5.1f} " + " ".join(f"{100*x:5.0f}" for x in v))
