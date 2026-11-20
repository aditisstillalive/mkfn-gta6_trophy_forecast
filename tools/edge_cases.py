"""Edge-case and format test for main.py: odd histories must still give a valid forecast, inside the time limit,
byte-identical across two runs.

usage: python3 tools/edge_cases.py <a final-shaped case, e.g. a sim .in> [main.py]
"""

import copy
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LIMIT_S = 8.0


def validate(case: dict, out: dict) -> list:
    """The rules' shape for an .out file; returns the problems found."""
    problems = []
    if set(out) != {"trophies"}:
        problems.append(f"keys {sorted(out)}")
    ids = [t["id"] for t in case["trophies"]]
    rows = out.get("trophies", [])
    if sorted(r.get("id") for r in rows) != sorted(ids):
        problems.append("ids differ from the case's trophies")
    for r in rows:
        if set(r) != {"id", "q"}:
            problems.append(f"row keys {sorted(r)}")
        q = r.get("q", [])
        if len(q) != 5 or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 100 for x in q):
            problems.append(f"bad q {q} for {r.get('id')}")
        elif any(b < a for a, b in zip(q, q[1:])):
            problems.append(f"decreasing q {q} for {r.get('id')}")
    return problems


def variants(base: dict) -> dict:
    out = {"as_is": base}
    c = copy.deepcopy(base)
    c["history"] = c["history"][:1]
    out["single_read"] = c
    c = copy.deepcopy(base)
    c["history"] = c["history"][:8]
    out["launch_night_only"] = c
    c = copy.deepcopy(base)
    for i, read in enumerate(c["history"]):
        if i % 7 == 3:
            read["rates"][5] = None
    out["null_rates"] = c
    c = copy.deepcopy(base)
    c["trophies"].append({"id": 56, "name": "New", "description": "Added later.", "tier": "bronze", "points": 15,
                          "secret": False})
    n = len(c["history"])
    for i, read in enumerate(c["history"]):
        if i >= n // 2:
            read["rates"] = read["rates"] + [0.5]
    out["trophy_added_late"] = c
    c = copy.deepcopy(base)
    for read in c["history"]:
        read["rates"][10] = 0.0
    out["zero_trophy"] = c
    c = copy.deepcopy(base)
    for i, read in enumerate(c["history"]):  # every read a refresh: the most work the model can be given
        read["players_tracked"] += i
    out["refresh_every_read"] = c
    return out


def run(bot: Path, case: dict) -> tuple:
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d, "in"), Path(d, "out")
        inp.mkdir()
        outp.mkdir()
        (inp / "x.in").write_text(json.dumps(case))
        t = time.time()
        proc = subprocess.run([sys.executable, "-I", str(bot)], env={"MAKEFUN_INPUT_DIR": str(inp),
                              "MAKEFUN_OUTPUT_DIR": str(outp)}, capture_output=True, text=True)
        dt = time.time() - t
        text = (outp / "x.out").read_text() if (outp / "x.out").exists() else ""
        return proc, dt, text


def main() -> None:
    base = json.loads(Path(sys.argv[1]).read_text())
    bot = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "main.py")
    failed = False
    for name, case in variants(base).items():
        proc, dt, text = run(bot, case)
        _, _, text2 = run(bot, case)
        problems = [] if text else ["no output"]
        if text:
            problems += validate(case, json.loads(text))
            if len(text.encode()) > 65536:
                problems.append("over 65,536 bytes")
        if text != text2:
            problems.append("two runs differ")
        if dt > LIMIT_S / 2:
            problems.append(f"slow: {dt:.2f}s")
        if proc.returncode != 0:
            problems.append(f"exit {proc.returncode}: {proc.stderr[-300:]}")
        note = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        print(f"{name:20s} {dt:5.2f}s  {'OK' if not problems else 'FAIL ' + '; '.join(problems)}  {note}")
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
