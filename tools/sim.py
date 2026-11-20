"""Synthetic final-shaped cases for developing the forecaster (numpy; a dev tool, not part of the entry).

Each scenario simulates GTA VI's first week on Exophase from the real launch-night page on:
  - 638 launch-night accounts keep the counts the real page showed (rate x 638);
  - new players start playing (a launch wave across time zones, then a long tail), are tracked by Exophase after a
    sync lag, play at their own intensity (hours of play per hour), and earn trophy k once their hours pass the
    trophy's requirement (scaled by the scenario's pace and their own skill) if they engage with that content;
  - Exophase refreshes the counts at random intervals; the house reads every 30 min, every 10 min from 12:00Z on
    11-19, a read occasionally failing; each read shows the last refresh, rates to 2 decimals.
The case is cut at the deadline (2026-11-21T11:30Z) and answered by the page at 2026-11-26T00:00Z.

usage: python3 tools/sim.py <kit input dir> <out dir> <n scenarios> [first seed]
writes <out dir>/s<seed>.in (case) and <out dir>/s<seed>.ans (answer rates)
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

T0 = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)  # simulation clock origin (hours)
DEADLINE = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
FIRST_REFRESH_AFTER = datetime(2026, 11, 19, 4, 0, tzinfo=timezone.utc)

# trophy id -> (hours of play needed by a typical player, share of players who engage with it)
REQ = {
    1: (160, 0.15), 2: (0, 1.0), 3: (3.5, 0.97), 4: (9, 0.95), 5: (16, 0.93), 6: (26, 0.9), 7: (42, 0.88),
    8: (5, 0.97), 9: (20, 0.92), 10: (12, 0.94), 11: (30, 0.9), 12: (36, 0.89), 13: (39, 0.88), 14: (150, 0.2),
    15: (70, 0.3), 16: (18, 0.6), 17: (45, 0.3), 18: (50, 0.25), 19: (40, 0.35), 20: (4, 0.8), 21: (6, 0.6),
    22: (32, 0.5), 23: (38, 0.5), 24: (35, 0.5), 25: (55, 0.35), 26: (42, 0.45), 27: (30, 0.5), 28: (60, 0.3),
    29: (20, 0.5), 30: (50, 0.3), 31: (65, 0.25), 32: (35, 0.45), 33: (8, 0.6), 34: (7, 0.55), 35: (15, 0.7),
    36: (6, 0.7), 37: (36, 0.85), 38: (12, 0.5), 39: (10, 0.55), 40: (30, 0.45), 41: (8, 0.6), 42: (45, 0.35),
    43: (25, 0.45), 44: (6, 0.5), 45: (10, 0.5), 46: (12, 0.45), 47: (14, 0.8), 48: (28, 0.85), 49: (18, 0.7),
    50: (24, 0.75), 51: (9, 0.6), 52: (25, 0.5), 53: (12, 0.75), 54: (50, 0.3), 55: (15, 0.55),
}


# chapters 1-4 and their missions, then the final mission
STORY_CHAIN = [3, 8, 4, 10, 5, 9, 6, 11, 12, 13, 7]


def hours(dt: datetime) -> float:
    return (dt - T0).total_seconds() / 3600


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def scenario(seed: int, base: dict) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    trophies = base["trophies"]
    ids = [t["id"] for t in trophies]
    k = len(ids)
    launch_rates = np.array(base["history"][-1]["rates"], dtype=float)
    early_counts = np.round(launch_rates * 638 / 100)

    # scenario-level knobs
    n_players = int(rng.uniform(15_000, 120_000))
    pace = np.exp(rng.normal(0, 0.35))  # >1: the game takes longer than the table says
    wave_share = rng.uniform(0.35, 0.7)
    tail_mean = rng.uniform(40, 120)
    wave_from, wave_width = rng.uniform(15, 19), rng.uniform(12, 30)  # the wave: from ~03:00-07:00Z on 11-19
    lag_mean = rng.uniform(1, 12)
    intensity = rng.uniform(0.15, 0.3)
    refresh_mean = rng.uniform(1.0, 6.0)

    # players: start, tracked time, intensity, skill, engagement latent
    wave = rng.random(n_players) < wave_share
    start = np.where(wave, wave_from + rng.uniform(0, wave_width, n_players),
                     wave_from + rng.exponential(tail_mean, n_players))
    tracked = start + rng.exponential(lag_mean, n_players)
    v = np.exp(rng.normal(np.log(intensity), 0.5, n_players)) * np.where(wave, 1.25, 0.85)
    v = np.minimum(v, 0.75)
    skill = np.exp(rng.normal(0, 0.35, n_players))
    g = rng.normal(0, 1, n_players)

    need = np.array([REQ[i][0] for i in ids]) * pace * np.exp(rng.normal(0, 0.35, k))
    # the story chain is played in order: a later chapter never needs fewer hours than an earlier one
    chain = [ids.index(i) for i in STORY_CHAIN if i in ids]
    need[chain] = np.maximum.accumulate(need[chain])
    p = np.clip(np.array([REQ[i][1] for i in ids]) * np.exp(rng.normal(0, 0.25, k)), 0.01, 1.0)

    # earn times among tracked players: max(tracked, start + hours needed / intensity), inf if not engaged
    earn = np.empty((k, n_players))
    for j in range(k):
        if need[j] == 0:
            earn[j] = tracked
            continue
        latent = 0.6 * g + 0.8 * rng.normal(0, 1, n_players)
        engaged = latent < NormalDist().inv_cdf(min(p[j], 0.9999))
        e = start + need[j] * skill / v
        earn[j] = np.where(engaged, np.maximum(e, tracked), np.inf)
    earn.sort(axis=1)
    tracked_sorted = np.sort(tracked)

    # refresh times after the launch-night page, up to the target
    t = hours(FIRST_REFRESH_AFTER) + rng.uniform(0.05, 0.5)
    refreshes = []
    while t <= hours(TARGET) + 1:
        refreshes.append(t)
        t += max(0.25, rng.exponential(refresh_mean))

    def page(at_h: float):
        """The page as of the last refresh before at_h: (players, earned, rates), or the launch page."""
        past = [r for r in refreshes if r <= at_h]
        if not past:
            return 638, int(base["history"][-1]["trophies_earned"]), [float(x) for x in launch_rates]
        r = past[-1]
        n = 638 + int(np.searchsorted(tracked_sorted, r, side="right"))
        counts = early_counts + np.array([np.searchsorted(earn[j], r, side="right") for j in range(k)])
        rates = [round(min(100.0, 100 * c / n), 2) for c in counts]
        return n, int(counts.sum()), rates

    # the house's reads
    history = []
    slots = [datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)]
    slot = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    while slot < DEADLINE:
        slots.append(slot)
        slot += timedelta(minutes=30 if slot < datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc) else 10)
    for slot in slots:
        if rng.random() > 0.03:
            at = slot + timedelta(seconds=float(rng.uniform(0, 2)))
            n, earned, rates = page(hours(at))
            history.append({"slot": iso(slot)[:-5] + "Z", "at": iso(at), "players_tracked": n,
                            "trophies_earned": earned, "rates": rates})

    case = dict(base)
    case.update({"id": f"s{seed}", "at": "2026-11-21T11:30:00Z", "horizon_minutes": 6510,
                 "target": "2026-11-26T00:00:00Z", "history": history})
    _, _, answer = page(hours(TARGET) + 0.01)
    meta = {"n_players": n_players, "pace": round(float(pace), 3), "refresh_mean": round(refresh_mean, 2),
            "tail_mean": round(tail_mean, 1), "lag_mean": round(lag_mean, 1)}
    return case, {"rates": {str(i): r for i, r in zip(ids, answer)}, "meta": meta}


def main() -> None:
    inp, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((inp / "p01.in").read_text())
    for seed in range(first, first + n):
        case, ans = scenario(seed, base)
        (out / f"s{seed}.in").write_text(json.dumps(case))
        (out / f"s{seed}.ans").write_text(json.dumps(ans))
        print(f"s{seed}: reads {len(case['history'])}, players {case['history'][-1]['players_tracked']}"
              f" -> answer {ans['meta']}")


if __name__ == "__main__":
    main()
