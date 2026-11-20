# mkfn-gta6_trophy_forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aditisstillalive/mkfn-gta6_trophy_forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aditisstillalive/mkfn-gta6_trophy_forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PS5 trophy's Exophase earn rate a week after launch (five quantiles a trophy) from
the house's reads of the trophy page, with a refresh-aware, pooled progression-velocity model.

How it works (`main.py`, Python 3.13, standard library only):

1. **Refreshes, not reads.** Exophase refreshes a game's counts on its own schedule, so the history is collapsed to the
   distinct pages it published; the answer is the page as of the last refresh before the target, so the target is
   moved back by half the typical refresh gap.
2. **Genuine earners.** The launch-night page (638 players, every rate near 1%) is a block of early accounts whose
   counts stay put while real players flood in; their counts are subtracted before modelling and added back at the end.
3. **Progression velocity.** On a log-time axis every trophy follows roughly the same progression curve shifted by how
   much play it needs, so in probit space its velocity depends mostly on its level. A velocity line is fitted across the
   well-measured trophies, with story/side-content offsets, each trophy keeping a shrunk share of its own deviation, and
   integrated to the target with a velocity that decays along log time. Platinum and 100%-completion trophies climb at
   half pace, and the platinum never exceeds the rarest other trophy.
4. **Self-backtest.** The decay and the spread scale are chosen per case by forecasting the last page from earlier cuts
   of the case's own history, scored with the contest's loss, and shrunk toward priors.
5. **Never invalid.** Any failure falls back, trophy by trophy, to a wide persistence forecast; output is clamped,
   ordered and deterministic.

Development tools in `tools/`: a local copy of the contest's scorer (`score.py`, checked against the rules' worked
example), approximate practice answers rebuilt from the refresh the rules describe (`practice_answers.py`), a
week-one simulator of Exophase's page (`sim.py`), a re-implementation of the house curve baseline (`house_trend.py`),
per-trophy breakdowns and an edge-case/format test.

| model (sim: 40 held-out simulated finals)  | loss  |
|--------------------------------------------|-------|
| house baseline (persistence)               | 1.909 |
| house curve baseline (re-implemented)      | 2.252 |
| v1 per-trophy logit trend                  | 1.218 |
| v2 pooled probit velocity                  | 0.490 |
| v3 + velocity decay, self-backtest         | 0.382 |
| v4/v5 + story/side offsets, completion cap | 0.334 |

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=hp5q8mh0&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
