"""Approximate answers for the practice cases p00/p01, which the kit does not publish.

What RULES.txt tells us about them: the launch-night page (638 players tracked) was refreshed between the 04:00 and
04:30 reads to 3,868 players tracked, every rate but the 100% one lower. We rebuild that page by keeping each trophy's
earner count (rate x 638) and letting each count grow by a factor `growth` (new players reaching it), then dividing by
3,868. growth=1 is pure dilution. The p01 answer (05:00) is taken to be the same refreshed page.

usage: python3 tools/practice_answers.py <kit input dir> <out dir> [growth]
"""

import json
import sys
from pathlib import Path

N_OLD, N_NEW = 638, 3868


def main() -> None:
    inp, out = Path(sys.argv[1]), Path(sys.argv[2])
    growth = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    out.mkdir(parents=True, exist_ok=True)
    for case_path in sorted(inp.glob("p*.in")):
        case = json.loads(case_path.read_text())
        last = case["history"][-1]["rates"]
        rates = {}
        for trophy, r in zip(case["trophies"], last):
            if r >= 100:
                rates[str(trophy["id"])] = 100.0
                continue
            count = round(r * N_OLD / 100) * growth
            rates[str(trophy["id"])] = min(100.0, round(100 * count / N_NEW, 2))
        (out / f"{case_path.stem}.ans").write_text(json.dumps({"rates": rates}))


if __name__ == "__main__":
    main()
