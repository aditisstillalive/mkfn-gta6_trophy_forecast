"""A re-implementation of the house curve baseline as RULES.txt describes it, for comparison (not the house's code).

For each trophy: least squares of logit(rate) on ln(1 + hours from 2026-11-19T00:00Z) over the case's last 24 hours of
reads, read at the target; spread: the random walk's (0.5 x sqrt(days)) and the fit's residual combined. With fewer
than 3 reads, the house baseline.
"""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

Z = [-1.645, -0.674, 0, 0.674, 1.645]
T0 = datetime(2026, 11, 19, tzinfo=timezone.utc)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def lg(x):
    p = min(99.99, max(0.01, x)) / 100
    return math.log(p / (1 - p))


def main():
    inp = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(inp.glob("*.in")):
        case = json.loads(path.read_text())
        at = ts(case["at"])
        reads = [r for r in case["history"] if (at - ts(r["at"])).total_seconds() <= 86400]
        rw = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
        xt = math.log(1 + (ts(case["target"]) - T0).total_seconds() / 3600)
        rows = []
        for i, t in enumerate(case["trophies"]):
            last = case["history"][-1]["rates"][i] or 0
            if len(reads) < 3:
                c, s = lg(last), rw
            else:
                xs = [math.log(1 + (ts(r["at"]) - T0).total_seconds() / 3600) for r in reads]
                ys = [lg(r["rates"][i] or 0) for r in reads]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                sxx = sum((x - mx) ** 2 for x in xs)
                b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0
                res = math.sqrt(sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys)) / max(1, len(xs) - 2))
                c, s = my + b * (xt - mx), math.sqrt(rw * rw + res * res)
            rows.append({"id": t["id"], "q": [round(100 / (1 + math.exp(-(c + z * s))), 4) for z in Z]})
        (out / f"{path.stem}.out").write_text(json.dumps({"trophies": rows}))


if __name__ == "__main__":
    main()
