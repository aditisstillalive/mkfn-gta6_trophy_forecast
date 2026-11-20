"""Local scorer: the contest's pinball loss on the logit scale, as RULES.txt's Scoring section defines it.

usage: python3 tools/score.py <answers dir> <forecast dir>
  <answers dir>/<id>.ans: {"rates": {"<trophy id>": rate, ...}}  (percent)
  <forecast dir>/<id>.out: the program's forecast file

Prints, per case and overall: the loss (mean pinball over every trophy x level), the typical miss (mean |y - median|
in points) and the 90% coverage. `selftest` reproduces the rules' worked example.
"""

import json
import math
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def logit(x: float) -> float:
    p = min(99.99, max(0.01, x)) / 100
    return math.log(p / (1 - p))


def pinball(y: float, q: float, tau: float) -> float:
    d = logit(y) - logit(q)
    return 2 * max(tau * d, (tau - 1) * d)


def score_case(rates: dict, forecast: dict) -> dict:
    rows = {row["id"]: row["q"] for row in forecast["trophies"]}
    losses, misses, covered = [], [], []
    for tid, y in rates.items():
        q = rows[int(tid)]
        for tau, qt in zip(LEVELS, q):
            losses.append(pinball(y, qt, tau))
        misses.append(abs(y - q[2]))
        covered.append(1.0 if q[0] <= y <= q[4] else 0.0)
    return {"losses": losses, "misses": misses, "covered": covered}


def summarise(parts: list) -> dict:
    losses = [x for p in parts for x in p["losses"]]
    misses = [x for p in parts for x in p["misses"]]
    covered = [x for p in parts for x in p["covered"]]
    return {
        "loss": sum(losses) / len(losses),
        "miss": sum(misses) / len(misses),
        "cover90": sum(covered) / len(covered),
    }


def selftest() -> None:
    fc = {"trophies": [{"id": 1, "q": [10, 15, 20, 25, 35]}, {"id": 2, "q": [10, 15, 20, 25, 35]}]}
    s = summarise([score_case({"1": 22, "2": 8}, fc)])
    assert abs(s["loss"] - 0.403404) < 5e-7, s
    assert abs(s["miss"] - 7) < 1e-9 and s["cover90"] == 0.5, s
    inv = {"trophies": [{"id": 1, "q": [30] * 5}, {"id": 2, "q": [5] * 5}]}
    s = summarise([score_case({"1": 22, "2": 8}, inv)])
    assert abs(s["loss"] - 0.46023) < 5e-6, s
    print("selftest ok: worked example loss 0.403404, invalid-case loss 0.460230")


def main() -> None:
    if sys.argv[1:] == ["selftest"]:
        selftest()
        return
    ans_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    parts = []
    for ans in sorted(ans_dir.glob("*.ans")):
        rates = json.loads(ans.read_text())["rates"]
        part = score_case(rates, json.loads((out_dir / f"{ans.stem}.out").read_text()))
        s = summarise([part])
        print(f"{ans.stem}: loss {s['loss']:.6f}  miss {s['miss']:.3f}  cover90 {s['cover90']:.2f}")
        parts.append(part)
    s = summarise(parts)
    print(f"ALL: loss {s['loss']:.6f}  miss {s['miss']:.3f}  cover90 {s['cover90']:.2f}")


if __name__ == "__main__":
    main()
