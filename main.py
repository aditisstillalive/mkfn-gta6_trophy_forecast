"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01 (Python 3.13, standard library only).

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output):
five quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate at the case's target.

The model (v2: pooled progression velocity)
  1. Refreshes. The house reads the page far more often than Exophase refreshes it, so the history is collapsed to
     the distinct pages Exophase published, each timed between the read that first showed it and the one before.
  2. Genuine earners. The launch-night page (638 players, every rate near 1%) is a block of early accounts whose
     counts stay put while real players flood in; their counts are subtracted, leaving each trophy's genuine rate
     among the players tracked since.
  3. Progression. A player earns a trophy once their play passes its requirement, so on a log-time axis every trophy
     follows roughly the same curve shifted by how much play it needs, and in probit space a trophy's velocity
     depends mostly on where it is on that curve. The velocity is fitted as a line in the probit level over the
     well-measured trophies (the cross-section of the last ~30 h), each trophy keeping a shrunk share of its own
     deviation from it, and integrated to the target.
  4. The rate at the target is the early block's counts plus the genuine rate times the new players, over the
     players tracked then (their log trend, damped), as of the last refresh before the target.
  5. Spread: the cross-sectional scatter of velocities, a random walk in log time and the level's own noise, in
     probit space, mapped through the same transform, so quantiles stay ordered and inside 0..100.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ND = NormalDist()
Z = [ND.inv_cdf(t) for t in LEVELS]

ORIGIN = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
ORIGIN_OFFSET_H = 3.0  # u = ln(hours since ORIGIN + this): the launch wave's centre is near the origin
FIT_WINDOW_H = 30.0
EARLY_BLOCK_MAX = 2000  # a first page with fewer players than this is the launch-night block
INFORMATIVE_COUNT = 15  # genuine earners a trophy needs at the last page to inform the pooled velocity
VEL_DAMP = 0.85  # velocity damping (players keep joining with no progress; sessions slow after launch week)
OWN_SHARE = 0.5  # share of a trophy's own deviation from the pooled velocity it keeps
PLAYERS_DAMP = 0.7
RW_SIGMA = 0.25  # probit random walk per unit of log time
SPREAD_SCALE = 1.0


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_origin(dt: datetime) -> float:
    return (dt - ORIGIN).total_seconds() / 3600


def logu(h: float) -> float:
    return math.log(max(h + ORIGIN_OFFSET_H, 0.5))


def probit(p: float) -> float:
    return ND.inv_cdf(min(max(p, 1e-9), 1 - 1e-9))


def logit_pct(pct: float) -> float:
    p = min(99.99, max(0.01, pct)) / 100
    return math.log(p / (1 - p))


def inv_logit_pct(x: float) -> float:
    return 100 / (1 + math.exp(-x))


def snapshots(history: list) -> list:
    """The distinct pages Exophase published, oldest first: [(hours since origin, players, rates)]."""
    out, prev_key, prev_at = [], None, None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        at = hours_since_origin(parse_time(read["at"]))
        if key != prev_key:
            when = at if prev_at is None else (at + prev_at) / 2
            out.append((when, read["players_tracked"], read["rates"]))
        prev_key, prev_at = key, at
    return out


def wls(xs: list, ys: list, ws: list) -> tuple:
    """Weighted least squares y = a + b x: (a, b, residual sd, slope se)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 1.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    n = len(xs)
    rss = sum(w * (y - my - b * (x - mx)) ** 2 for w, x, y in zip(ws, xs, ys))
    s2 = rss / sw * n / max(1, n - 2)
    return my - b * mx, b, math.sqrt(s2), math.sqrt(s2 * sw / n / sxx)


def rate_of(rates: list, k: int) -> float:
    r = rates[k] if k < len(rates) else None
    return 0.0 if r is None else float(r)


def young_forecast(trophies: list, last_rates: list) -> dict:
    """Launch night, too young for a trend: Exophase's next refresh brings many more players, diluting every
    trophy below 100%."""
    rows = []
    for k, t in enumerate(trophies):
        r = rate_of(last_rates, k)
        if r >= 99.995:
            q = [99.95, 100, 100, 100, 100]
        else:
            q = [inv_logit_pct(logit_pct(r) - 1.3 + z * 0.9) for z in Z]
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    K = len(trophies)
    snaps = snapshots(case["history"])
    target_h = hours_since_origin(parse_time(case["target"]))
    at_h = hours_since_origin(parse_time(case["at"]))
    last_t, last_n, last_rates = snaps[-1]

    # The launch-night block: its counts stay put.
    first_t, first_n, first_rates = snaps[0]
    if first_n < EARLY_BLOCK_MAX:
        n0, c0 = first_n, [round(rate_of(first_rates, k) * first_n / 100) for k in range(K)]
    else:
        n0, c0 = 0, [0] * K

    fit = [s for s in snaps if s[0] >= last_t - FIT_WINDOW_H and s[1] - n0 >= max(1000, 0.2 * (last_n - n0))]
    if len(fit) < 3 or fit[-1][0] - fit[0][0] < 4:
        return young_forecast(trophies, last_rates)

    # Expected staleness at the target: half the typical gap between refreshes.
    recent = [s[0] for s in snaps if s[0] >= at_h - 24]
    gaps = sorted(b - a for a, b in zip(recent, recent[1:]))
    gap = gaps[len(gaps) // 2] if gaps else 3.0
    target_eff = target_h - min(max(gap, 0.5), 12.0) / 2
    du = max(0.0, logu(target_eff) - logu(last_t))

    xs = [logu(s[0]) for s in fit]
    ws = [1.0 + i / len(fit) for i in range(len(fit))]

    # Players tracked at the target: the log-log trend of the new players, damped.
    _, bn, _, _ = wls(xs, [math.log(max(1, s[1] - n0)) for s in fit], ws)
    n_new_t = (last_n - n0) * math.exp(PLAYERS_DAMP * max(0.0, bn) * du)
    n_t = n0 + n_new_t

    def genuine(s, k):
        new = max(1, s[1] - n0)
        count = rate_of(s[2], k) * s[1] / 100 - c0[k]
        return max(0.0, count), new

    # Each trophy's probit path over the window.
    paths = []
    for k in range(K):
        zs = []
        for s in fit:
            count, new = genuine(s, k)
            g = (count + 0.5) / (new + 1)
            zs.append(probit(min(g, 1 - 0.5 / new)))
        a, b, sd, se = wls(xs, zs, ws)
        count_last, new_last = genuine(fit[-1], k)
        count_first, _ = genuine(fit[0], k)
        paths.append({"z_last": zs[-1], "z_mid": sum(zs) / len(zs), "b": b, "se": se, "sd": sd,
                      "count": count_last, "count_first": count_first, "g": count_last / new_last})

    informative = [k for k in range(K) if paths[k]["count"] >= INFORMATIVE_COUNT and paths[k]["count_first"] >= 3
                   and paths[k]["g"] < 0.98]

    # Pooled velocity v(z) = alpha + beta z over the informative trophies.
    if len(informative) >= 4:
        zx = [paths[k]["z_mid"] for k in informative]
        vy = [paths[k]["b"] for k in informative]
        vw = [1 / (paths[k]["se"] ** 2 + 0.01) for k in informative]
        alpha, beta, vsd, _ = wls(zx, vy, vw)
        beta = min(0.1, max(-0.6, beta))
        mzx = sum(w * x for w, x in zip(vw, zx)) / sum(vw)
        mvy = sum(w * y for w, y in zip(vw, vy)) / sum(vw)
        alpha = mvy - beta * mzx
        zlo, zhi = min(zx) - 0.5, max(zx) + 0.5
        resid = sorted(abs(vy[i] - alpha - beta * zx[i]) for i in range(len(zx)))
        vsd = max(0.1, resid[len(resid) // 2] / 0.6745)
    elif informative:
        vs = sorted(paths[k]["b"] for k in informative)
        alpha, beta, zlo, zhi, vsd = vs[len(vs) // 2], 0.0, -9.0, 9.0, 0.5
    else:
        alpha, beta, zlo, zhi, vsd = 0.5, 0.0, -9.0, 9.0, 0.8

    def velocity(z: float) -> float:
        return alpha + beta * min(zhi, max(zlo, z))

    rows = []
    for k, t in enumerate(trophies):
        p = paths[k]
        if all(rate_of(s[2], k) >= 99.995 for s in fit):
            rows.append({"id": t["id"], "q": [99.95, 100, 100, 100, 100]})
            continue
        own = (p["b"] - velocity(p["z_mid"])) * (OWN_SHARE if k in informative else 0.0)
        z, steps = p["z_last"], 40
        for _ in range(steps):
            z += VEL_DAMP * (velocity(z) + own) * du / steps
        extra = 0.0 if k in informative else 0.4
        sd = SPREAD_SCALE * math.sqrt((vsd * du) ** 2 + RW_SIGMA**2 * du + p["sd"] ** 2 + (extra * du) ** 2 + 0.01)
        q = []
        for zq in Z:
            g = ND.cdf(z + zq * sd)
            q.append(100 * (c0[k] + g * n_new_t) / n_t)
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def finish(q: list) -> list:
    """Round to 4 decimals, clamp to [0, 100], and keep the five non-decreasing."""
    out, prev = [], 0.0
    for x in q:
        v = max(round(min(100.0, max(0.0, x)), 4), prev)
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
