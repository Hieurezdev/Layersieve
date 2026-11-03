"""Ghi tien do hang doi benchmark vao results/bench/PROGRESS.md (gio VN), lap moi --every giay.
  setsid nohup .venv/bin/python scripts/progress.py >> results/bench/progress.err 2>&1 < /dev/null &
Noi dung: runner con song khong, job dang chay (so mau / ETA), job xong / loi gan day, GPU, bang diem (score_bench.py),
phan bo layer ASL tu chon."""
import argparse, collections, datetime as dt, json, os, re, subprocess, time
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results/bench")
VN = ZoneInfo("Asia/Ho_Chi_Minh")
EV = re.compile(r"^(\S+ \S+) \[(\S+)\] (bat dau|xong|loi)")


def vn(t):
    return t.replace(tzinfo=dt.timezone.utc).astimezone(VN).strftime("%m-%d %H:%M")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(loi: {e})"


def target(job):
    """job -> (file ket qua, tien to id can dem, so mau can co)."""
    if job.startswith("ib_"):
        return f"{R}/infbench/{job[3:]}.jsonl", "", 400
    if job.startswith("ru128_"):
        return f"{R}/ruler/{job[6:]}.jsonl", "ruler131072_", 1300
    if job.startswith("ru_"):
        n = job[3:]
        if n.endswith("_32k"):
            return f"{R}/ruler/{n[:-4]}.jsonl", "ruler32768_", 1300
        if n.endswith("_long"):
            return f"{R}/ruler/{n[:-5]}.jsonl", ("ruler65536_", "ruler131072_"), 2600
        return f"{R}/ruler/{n}.jsonl", "", 3900
    return None, "", 0


def count(path, pre):
    if not path or not os.path.exists(path):
        return 0
    n = 0
    for line in open(path):
        try:
            if json.loads(line)["id"].startswith(pre):
                n += 1
        except Exception:  # noqa: BLE001
            pass
    return n


def report():
    now = dt.datetime.utcnow()
    out = [f"# Tien do benchmark - cap nhat {vn(now)} (gio VN)\n"]
    runner = sh("pgrep -fa '^bash scripts/run_bench' ; pgrep -fa 'eval_bench.py|asl_driver.py|bench_speed.py' | grep -v pgrep | cut -c1-160")
    out.append("## Tien trinh\n```\n" + (runner or "KHONG CO TIEN TRINH NAO - hang doi da dung (xong het hoac chet)") + "\n```\n")

    events, state = [], {}
    for line in open(os.path.join(R, "run.log")):
        m = EV.match(line)
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            events.append((t, m.group(2), m.group(3)))
            state[m.group(2)] = (t, m.group(3))
    running = [(j, t) for j, (t, s) in state.items() if s == "bat dau"]
    out.append("## Job dang chay\n| Job | Bat dau | Mau | Toc do | ETA (VN) |\n|---|---|---|---|---|")
    for j, t in running:
        path, pre, need = target(j)
        done = count(path, pre)
        el = (now - t).total_seconds() / 60
        # mau co tu truoc (resume) lam toc do bi cao -> chi uoc luong
        rate = done / el if el > 1 and done else 0
        eta = vn(now + dt.timedelta(minutes=(need - done) / rate)) if rate else "?"
        out.append(f"| {j} | {vn(t)} | {done}/{need} | {rate:.2f}/ph | {eta} |")
    out.append("")
    out.append("## 15 su kien gan nhat\n```")
    out += [f"{vn(t)} [{j}] {s}" for t, j, s in events[-15:]]
    out.append("```\n")
    fails = [j for j, (t, s) in state.items() if s == "loi"]
    if fails:
        out.append("**LOI:** " + ", ".join(fails) + f" (xem {R}/logs/<job>.log)\n")
    out.append("## GPU\n```\n" + sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader") + "\n```\n")

    asl = f"{R}/infbench/m_asl_t0.3_f0.375.jsonl"
    if os.path.exists(asl):
        d = collections.defaultdict(collections.Counter)
        for line in open(asl):
            r = json.loads(line)
            d[r["task"]][r.get("select_layer")] += 1
        out.append("## ASL (37.5%, monitor-k 2048) layer tu chon tren InfiniteBench\n")
        for task, c in d.items():
            out.append(f"- {task}: " + ", ".join(f"L{k}:{v}" for k, v in sorted(c.items(), key=lambda x: (x[0] is None, x[0] or 0))))
        out.append("")
    out.append("## Bang diem (scripts/score_bench.py; '(n)' = moi chay n mau)\n")
    out.append(sh(f"cd {ROOT} && .venv/bin/python scripts/score_bench.py"))
    tmp = os.path.join(R, ".PROGRESS.md.tmp")
    open(tmp, "w").write("\n".join(out) + "\n")
    os.replace(tmp, os.path.join(R, "PROGRESS.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            report()
        except Exception as e:  # noqa: BLE001
            print(dt.datetime.utcnow(), "loi:", e, flush=True)
        if a.once:
            break
        time.sleep(a.every)
