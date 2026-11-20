"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01 (Python 3.13, standard library only).

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output):
five quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate at the case's target.

v1: refresh-aware per-trophy trend.
  - The house reads the page far more often than Exophase refreshes it, so the history is collapsed to the distinct
    pages Exophase published (a refresh = a change of players tracked or of any rate), each timed when first seen.
  - Each trophy's logit rate is fitted against log time over the recent refreshes, its slope shrunk toward the
    slopes of trophies at a similar level, damped, and read at the target (less the expected staleness of the page
    at the target: the answer is the last refresh before it).
  - The spread adds a random walk in log time, the slope's uncertainty and a model term proportional to the move.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]

# Time origin for the log-time axis: the launch wave (midnight local time across the world, centred on the Americas
# and Europe, where most Exophase users are).
ORIGIN = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
ORIGIN_OFFSET_H = 6.0  # u = ln(hours since ORIGIN + this)
FIT_WINDOW_H = 30.0
MIN_PLAYERS_FIT = 1500
DAMPING = 0.7
RW_SIGMA = 0.35  # random walk in logit per unit of log time
MODEL_SIGMA = 0.45  # extra spread per unit of forecast move


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_origin(dt: datetime) -> float:
    return (dt - ORIGIN).total_seconds() / 3600


def logu(h: float) -> float:
    return math.log(max(h + ORIGIN_OFFSET_H, 0.5))


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    return 100 / (1 + math.exp(-x))


def snapshots(history: list) -> list:
    """The distinct pages Exophase published, oldest first: [(hours since origin, players, rates)]."""
    out, prev_key, prev_at = [], None, None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        at = hours_since_origin(parse_time(read["at"]))
        if key != prev_key:
            # Exophase refreshed somewhere between the previous read and this one.
            when = at if prev_at is None else (at + prev_at) / 2
            out.append((when, read["players_tracked"], read["rates"]))
        prev_key, prev_at = key, at
    return out


def wls_line(xs: list, ys: list, ws: list) -> tuple:
    """Weighted least squares y = a + b (x - x_last): (a, b, residual sd, slope se)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 1.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / sw * len(xs) / dof
    return my + b * (xs[-1] - mx), b, math.sqrt(s2), math.sqrt(s2 / sxx * sw / len(xs)) if sxx > 0 else 1.0


def quantiles(center: float, sd: float) -> list:
    return [inv_logit(center + z * sd) for z in Z]


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    snaps = snapshots(case["history"])
    target_h = hours_since_origin(parse_time(case["target"]))
    at_h = hours_since_origin(parse_time(case["at"]))
    last_t, last_n, last_rates = snaps[-1]

    # Expected staleness of the page at the target: half the typical gap between refreshes.
    recent = [s[0] for s in snaps if s[0] >= at_h - 24]
    gaps = sorted(b - a for a, b in zip(recent, recent[1:]))
    gap = gaps[len(gaps) // 2] if gaps else 3.0
    target_eff = target_h - min(max(gap, 0.5), 12.0) / 2
    du = logu(target_eff) - logu(last_t)

    fit = [s for s in snaps if s[0] >= last_t - FIT_WINDOW_H and s[1] >= MIN_PLAYERS_FIT]
    rows = []
    if len(fit) < 3:
        # Too young to fit a trend (launch night): the page will refresh with many more players, diluting every
        # trophy that is not at 100%. Expect a dilution, with a wide spread.
        for i, t in enumerate(trophies):
            r = last_rates[i] if last_rates[i] is not None else 0.0
            if r >= 99.995:
                q = [99.9, 100, 100, 100, 100]
            else:
                q = quantiles(logit(r) - 1.3, 0.9)
            rows.append({"id": t["id"], "q": q})
        return {"trophies": rows}

    xs = [logu(s[0]) for s in fit]
    ws = [1.0 + i / len(fit) for i in range(len(fit))]  # later refreshes count more
    fits = []
    for i in range(len(trophies)):
        ys = [logit(s[2][i] if s[2][i] is not None else 0.0) for s in fit]
        fits.append(wls_line(xs, ys, ws))

    # Shrink each slope toward the median slope of the trophies whose last level is within 1.5 logits.
    levels = [logit(last_rates[i] if last_rates[i] is not None else 0.0) for i in range(len(trophies))]
    for i, t in enumerate(trophies):
        r = last_rates[i] if last_rates[i] is not None else 0.0
        a, b, sd, se = fits[i]
        peers = sorted(fits[j][1] for j in range(len(trophies)) if abs(levels[j] - levels[i]) < 1.5)
        pooled = peers[len(peers) // 2]
        w = 0.6
        slope = w * b + (1 - w) * pooled
        if r >= 99.995:
            rows.append({"id": t["id"], "q": [99.9, 100, 100, 100, 100]})
            continue
        center = levels[i] + DAMPING * slope * du
        spread = math.sqrt((RW_SIGMA**2) * du + (se * du) ** 2 + (MODEL_SIGMA * slope * du) ** 2 + sd**2 + 0.02)
        rows.append({"id": t["id"], "q": quantiles(center, spread)})
    return {"trophies": rows}


def finish(q: list) -> list:
    """Round to 4 decimals, clamp to [0, 100], and keep the five non-decreasing."""
    out, prev = [], 0.0
    for x in q:
        v = round(min(100.0, max(0.0, x)), 4)
        v = max(v, prev)
        out.append(v)
        prev = v
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        for row in forecast["trophies"]:
            row["q"] = finish(row["q"])
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
