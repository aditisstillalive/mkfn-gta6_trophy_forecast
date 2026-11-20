"""Per-trophy breakdown of a forecast run: mean loss, mean signed logit error of the median (y - median), coverage.
usage: python3 tools/breakdown.py <answers dir> <forecast dir>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import LEVELS, logit, pinball  # noqa: E402

ans_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
loss, bias, cov = defaultdict(list), defaultdict(list), defaultdict(list)
for a in sorted(ans_dir.glob("*.ans")):
    rates = json.loads(a.read_text())["rates"]
    rows = {r["id"]: r["q"] for r in json.loads((out_dir / f"{a.stem}.out").read_text())["trophies"]}
    for tid, y in rates.items():
        q = rows[int(tid)]
        loss[tid].append(sum(pinball(y, qt, t) for t, qt in zip(LEVELS, q)) / 5)
        bias[tid].append(logit(y) - logit(q[2]))
        cov[tid].append(q[0] <= y <= q[4])
for tid in sorted(loss, key=lambda k: -sum(loss[k]) / len(loss[k])):
    n = len(loss[tid])
    print(f"{tid:>3} loss {sum(loss[tid])/n:6.3f}  bias {sum(bias[tid])/n:+6.2f}  cover {sum(cov[tid])/n:4.2f}")
