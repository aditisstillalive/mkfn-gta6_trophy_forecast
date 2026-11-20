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
FIT_WINDOWS_H = [30.0, 42.0, 54.0, 72.0]
EARLY_BLOCK_MAX = 2000  # a first page with fewer players than this is the launch-night block
INFORMATIVE_COUNT = 15  # genuine earners a trophy needs at the last page to inform the pooled velocity
# Velocity decay per unit of log time (players keep joining with no progress; play slows after launch): chosen per
# case by a backtest on its own history, shrunk toward these priors.
PRIOR_DECAY_INF = 1.0
PRIOR_DECAY_UNINF = 0.6
PRIOR_SCALE = 0.9
DECAY_GRID = [0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.2]
SCALE_GRID = [0.6, 0.75, 0.9, 1.1, 1.35, 1.7, 2.2]
PRIOR_PULL = 0.05
BACKTEST_WEIGHT = 0.3
CLASS_SHRINK = 3.0  # pseudo-trophies at zero offset in each class mean
COMPLETE_PACE = 0.5
BACKTEST_CUTS_H = [18.0, 24.0, 30.0]
OWN_SHARE = 0.5  # share of a trophy's own deviation from the pooled velocity it keeps
PLAYERS_DAMP = 0.7
RW_SIGMA = 0.25  # probit random walk per unit of log time


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


def trophy_kind(t: dict) -> str:
    """story: a hidden trophy for completing a mission or chapter; complete: the platinum or a 100% completion;
    other: side content."""
    desc = (t.get("description") or "").lower()
    if t.get("tier") == "platinum" or "100%" in desc:
        return "complete"
    if t.get("secret") and desc.startswith("complete"):
        return "story"
    return "other"


class Fit:
    """The model fitted on the pages up to some moment: each trophy's probit path, the pooled velocity line and
    the trend of the players tracked."""

    def __init__(self, snaps: list, n0: int, c0: list, kinds: list):
        self.ok = False
        K = len(kinds)
        last_t, last_n, _ = snaps[-1]
        fit = []
        for window in FIT_WINDOWS_H:  # widen the window when Exophase refreshed rarely
            fit = [s for s in snaps if s[0] >= last_t - window and s[1] - n0 >= max(1000, 0.2 * (last_n - n0))]
            if len(fit) >= 4 and fit[-1][0] - fit[0][0] >= 6:
                break
        if len(fit) < 3 or fit[-1][0] - fit[0][0] < 4:
            return
        self.ok, self.n0, self.c0, self.K, self.fit, self.kinds = True, n0, c0, K, fit, kinds
        self.last_t, self.last_n = last_t, last_n
        xs = [logu(s[0]) for s in fit]
        ws = [1.0 + i / len(fit) for i in range(len(fit))]
        _, self.bn, _, _ = wls(xs, [math.log(max(1, s[1] - n0)) for s in fit], ws)

        self.paths = []
        for k in range(K):
            zs = []
            for s in fit:
                count, new = self.genuine(s, k)
                zs.append(probit(min((count + 0.5) / (new + 1), 1 - 0.5 / new)))
            _, b, sd, se = wls(xs, zs, ws)
            count_last, new_last = self.genuine(fit[-1], k)
            count_first, _ = self.genuine(fit[0], k)
            self.paths.append({"z_last": zs[-1], "z_mid": sum(zs) / len(zs), "b": b, "se": se, "sd": sd,
                               "count": count_last, "count_first": count_first, "g": count_last / new_last,
                               "full": all(rate_of(s[2], k) >= 99.995 for s in fit)})
        self.informative = {k for k in range(K) if self.paths[k]["count"] >= INFORMATIVE_COUNT
                            and self.paths[k]["count_first"] >= 3 and self.paths[k]["g"] < 0.98}

        inf = sorted(self.informative)
        if len(inf) >= 4:
            zx = [self.paths[k]["z_mid"] for k in inf]
            vy = [self.paths[k]["b"] for k in inf]
            vw = [1 / (self.paths[k]["se"] ** 2 + 0.01) for k in inf]
            _, beta, _, _ = wls(zx, vy, vw)
            beta = min(0.1, max(-0.6, beta))
            alpha = sum(w * (y - beta * x) for w, x, y in zip(vw, zx, vy)) / sum(vw)
            zlo, zhi = min(zx) - 0.5, max(zx) + 0.5
            resid = sorted(abs(vy[i] - alpha - beta * zx[i]) for i in range(len(zx)))
            vsd = max(0.1, resid[len(resid) // 2] / 0.6745)
        elif inf:
            vs = sorted(self.paths[k]["b"] for k in inf)
            alpha, beta, zlo, zhi, vsd = vs[len(vs) // 2], 0.0, -9.0, 9.0, 0.5
        else:
            alpha, beta, zlo, zhi, vsd = 0.5, 0.0, -9.0, 9.0, 0.8
        self.alpha, self.beta, self.zlo, self.zhi, self.vsd = alpha, beta, zlo, zhi, vsd

        # Story trophies (everyone plays the story) and side content may move at different speeds at one level:
        # each class's mean residual from the pooled line, shrunk by how many trophies show it.
        self.offset = {}
        for kind in ("story", "other", "complete"):
            members = [k for k in inf if kinds[k] == kind]
            res = [self.paths[k]["b"] - self.velocity(self.paths[k]["z_mid"]) for k in members]
            self.offset[kind] = sum(res) / (len(res) + CLASS_SHRINK) if res else 0.0

    def genuine(self, s, k):
        new = max(1, s[1] - self.n0)
        return max(0.0, rate_of(s[2], k) * s[1] / 100 - self.c0[k]), new

    def velocity(self, z: float) -> float:
        return self.alpha + self.beta * min(self.zhi, max(self.zlo, z))

    def predict(self, target_h: float, decay_inf: float, decay_uninf: float, scale: float) -> list:
        """Five rate quantiles (percent) per trophy for the page at target_h."""
        du = max(0.0, logu(target_h) - logu(self.last_t))
        n_new_t = (self.last_n - self.n0) * math.exp(PLAYERS_DAMP * max(0.0, self.bn) * du)
        n_t = self.n0 + n_new_t
        out = []
        for k in range(self.K):
            p = self.paths[k]
            if p["full"]:
                out.append([99.95, 100, 100, 100, 100])
                continue
            informative = k in self.informative
            kind = self.kinds[k]
            base = self.offset[kind]
            own = (p["b"] - self.velocity(p["z_mid"]) - base) * OWN_SHARE if informative else 0.0
            # Platinum and 100% completion need nearly everything else: they climb at a fraction of the pace.
            pace = COMPLETE_PACE if kind == "complete" else 1.0
            # The velocity decays exponentially along log time (new players join with no progress, play slows).
            decay = decay_inf if informative else decay_uninf
            z, steps = p["z_last"], 40
            for i in range(steps):
                z += pace * math.exp(-decay * du * (i + 0.5) / steps) * (self.velocity(z) + base + own) * du / steps
            extra = 0.0 if informative else 0.4
            sd = scale * math.sqrt((self.vsd * du) ** 2 + RW_SIGMA**2 * du + p["sd"] ** 2 + (extra * du) ** 2 + 0.01)
            out.append([100 * (self.c0[k] + ND.cdf(z + zq * sd) * n_new_t) / n_t for zq in Z])
        return out


def pinball_logit(y: float, q: float, tau: float) -> float:
    d = logit_pct(y) - logit_pct(q)
    return 2 * max(tau * d, (tau - 1) * d)


def backtest(snaps: list, n0: int, c0: list, kinds: list) -> tuple:
    """Choose the velocity decay (informative and thin trophies) and the spread scale by forecasting the last
    page from earlier cuts of this very history, scored with the contest's own loss; shrunk toward the prior."""
    last_t, _, last_rates = snaps[-1]
    fits = []
    for back in BACKTEST_CUTS_H:
        cut = [s for s in snaps if s[0] <= last_t - back]
        if len(cut) >= 3:
            f = Fit(cut, n0, c0, kinds)
            if f.ok:
                fits.append(f)
    if not fits:
        return PRIOR_DECAY_INF, PRIOR_DECAY_UNINF, PRIOR_SCALE, 0

    def loss(di: float, du_: float, sc: float) -> float:
        total, n = 0.0, 0
        for f in fits:
            for k, q in enumerate(f.predict(last_t, di, du_, sc)):
                y = rate_of(last_rates, k)
                total += sum(pinball_logit(y, qt, tau) for tau, qt in zip(LEVELS, q))
                n += 5
        return total / n

    best = None
    for di in DECAY_GRID:
        for du_ in DECAY_GRID:
            l = loss(di, du_, PRIOR_SCALE) + PRIOR_PULL * ((di - PRIOR_DECAY_INF) ** 2 + (du_ - PRIOR_DECAY_UNINF) ** 2) / 4
            if best is None or l < best[0]:
                best = (l, di, du_)
    _, di, du_ = best
    best_sc = min(SCALE_GRID, key=lambda sc: loss(di, du_, sc) + PRIOR_PULL * math.log(sc / PRIOR_SCALE) ** 2)
    w = BACKTEST_WEIGHT
    return (w * di + (1 - w) * PRIOR_DECAY_INF, w * du_ + (1 - w) * PRIOR_DECAY_UNINF,
            math.exp(w * math.log(best_sc) + (1 - w) * math.log(PRIOR_SCALE)), len(fits))


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    K = len(trophies)
    snaps = snapshots(case["history"])
    target_h = hours_since_origin(parse_time(case["target"]))
    at_h = hours_since_origin(parse_time(case["at"]))
    last_rates = snaps[-1][2]

    # The launch-night block: its counts stay put.
    _, first_n, first_rates = snaps[0]
    if first_n < EARLY_BLOCK_MAX:
        n0, c0 = first_n, [round(rate_of(first_rates, k) * first_n / 100) for k in range(K)]
    else:
        n0, c0 = 0, [0] * K

    kinds = [trophy_kind(t) for t in trophies]
    model = Fit(snaps, n0, c0, kinds)
    if not model.ok:
        return young_forecast(trophies, last_rates)

    # Expected staleness at the target: half the typical gap between refreshes.
    recent = [s[0] for s in snaps if s[0] >= at_h - 24]
    gaps = sorted(b - a for a, b in zip(recent, recent[1:]))
    gap = gaps[len(gaps) // 2] if gaps else 3.0
    target_eff = target_h - min(max(gap, 0.5), 12.0) / 2

    decay_inf, decay_uninf, scale, _ = backtest(snaps, n0, c0, kinds)
    rows = model.predict(target_eff, decay_inf, decay_uninf, scale)
    # The platinum needs every other trophy: no quantile above the rarest other trophy's.
    others = [rows[k] for k in range(K) if trophies[k]["tier"] != "platinum"]
    for k, t in enumerate(trophies):
        if t["tier"] == "platinum" and others:
            rows[k] = [min([v] + [o[i] for o in others]) for i, v in enumerate(rows[k])]
    return {"trophies": [{"id": t["id"], "q": q} for t, q in zip(trophies, rows)]}


def finish(q: list, fallback: list) -> list:
    """Round to 4 decimals, clamp to [0, 100], and keep the five non-decreasing; a non-finite value (which the model
    should never write) takes the whole fallback row."""
    if len(q) != 5 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in q):
        q = fallback
    out, prev = [], 0.0
    for x in q:
        v = max(round(min(100.0, max(0.0, x)), 4), prev)
        out.append(v)
        prev = v
    return out


def safe_rows(case: dict) -> dict:
    """The forecast to fall back on, trophy by trophy: the last rate with a wide logit spread."""
    history = case.get("history") or []
    last = history[-1]["rates"] if history else []
    out = {}
    for k, t in enumerate(case["trophies"]):
        r = rate_of(last, k) if last else 1.0
        out[t["id"]] = [99.95, 100, 100, 100, 100] if r >= 99.995 else [inv_logit_pct(logit_pct(r) + z) for z in Z]
    return out


def run_case(case: dict) -> dict:
    """The model's forecast, made valid whatever happens: one row per trophy, five finite non-decreasing numbers."""
    fallback = safe_rows(case)
    try:
        model_rows = {row["id"]: row["q"] for row in forecast_case(case)["trophies"]}
    except Exception as exc:  # never leave a case invalid
        print(f"{case.get('id')}: model failed ({exc!r}), falling back", file=sys.stderr)
        model_rows = {}
    rows = [{"id": t["id"], "q": finish(model_rows.get(t["id"], fallback[t["id"]]), fallback[t["id"]])}
            for t in case["trophies"]]
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = run_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
