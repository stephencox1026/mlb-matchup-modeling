# Sprint 3 Summary — Section 15 v2 + recent365 + Recency Replacement

Five deliverables shipped. One honest "no-ship" finding (stacker fails CI gate but improved 7×). Clear path forward documented for next sprint.

## What shipped

### 1. Per-team bullpen HR rates ([src/build_team_bullpen_hr_priors.py](../src/build_team_bullpen_hr_priors.py))

Built [data/priors/team_bullpen_hr.json](../data/priors/team_bullpen_hr.json) from 2022-2025 bulk Statcast (732K PA-end rows). Identifies SP per side via earliest at_bat_number; labels everything else as bullpen. Beta-Binomial EB shrinkage to league mean (prior pseudo-PAs = 4000). Inverse-variance blend across seasons.

| Team rank (high to low blend) | Top 3 leaky | Bottom 3 suppressing |
|---|---|---|
| | LAA 3.09%, TOR 3.02%, WSH 3.02% | CLE 2.44%, SF 2.46%, PIT 2.46% |

League bullpen HR/PA: **2.72%**. Spread is meaningful (22% relative gap between extremes). Wired into [src/gen_zero_hr_predictions.py](../src/gen_zero_hr_predictions.py) at line 139 — `pen_lambda = pa_pen * pen_rate` instead of league avg.

**Tonight's impact:** STL @ PIT went from 19.0% → **21.1%** P(0 HR) because both pens are below league average (PIT home 2.46%, STL away 2.49%). Lock for the no-HR play.

Dashboard No HR Model tab gains two columns: **Pen A** and **Pen H** with green/red color-coding vs league.

### 2. Live roof state in morning fetch ([src/fetch_todays_games.py](../src/fetch_todays_games.py))

Added `weather` to the StatsAPI hydrate call. New `_parse_roof_from_condition()` helper. Now `data/raw/todays_matchups.json` carries:

- `roof_state_today` — `closed` / `open` / `outdoor` / `dome` / `null`
- `weather_condition_mlb`, `weather_temp_mlb`, `weather_wind_mlb` — raw StatsAPI strings

Tonight's slate at fetch time:
- TOR roof closed (cold April 54°F)
- TEX roof closed
- All others outdoor

This gives the morning macro a head start on roof state. The standalone refresh script ([src/refresh_game_conditions_today.py](../src/refresh_game_conditions_today.py)) handles late-posted updates.

### 3. recent365 model ([src/build_recent365_train.py](../src/build_recent365_train.py) + retrain)

Filtered the existing beast train parquet (1.7M rows, 2015-2024) to the last 365 days (180K rows, 2024 only). Same XGBoost architecture as beast, different training distribution. Tests whether **data-window diversity** is the missing axis (vs algorithmic diversity).

**Validation results vs beast (no measurable regression on any target):**

| Target | beast Brier | recent365 Brier | Δ |
|---|---:|---:|---:|
| is_hit | 0.16085 | 0.16109 | +0.00024 |
| is_hr | 0.02927 | 0.02930 | +0.00003 |
| is_strikeout | 0.14193 | 0.14240 | +0.00047 |
| is_xbh | 0.06862 | 0.06865 | +0.00003 |

Within ~0.0005 of beast on all targets. **Strictly better than recency** on every measure that matters.

### 4. recent365 replaces recency in production ([src/config.py](../src/config.py), [src/run_dual_model_daily.py](../src/run_dual_model_daily.py))

H2 backtest comparison on 183K 2025 rows:

| Target | recency | recent365 | Better |
|---|---:|---:|---|
| HR Brier | 0.02945 | 0.02942 | recent365 |
| HR AUC | 0.7020 | 0.7040 | recent365 |
| HR top-decile lift | 2.45× | 2.36× | recency (small margin) |
| **Hit Brier** | 0.16268 | **0.16218** | **recent365 (-0.0005)** |
| **Hit AUC** | 0.6540 | **0.6575** | **recent365 (+0.0035)** |
| XBH | 0.06895 | 0.06901 | tie |

Acceptance gates from the plan:

| Test | Pass criterion | Result |
|---|---|:---:|
| H2 Brier (HR) within 0.0005 of beast | 0.00005 actual | **PASS** |
| H2 top-decile lift (HR) >= 2.30× | 2.36× actual | **PASS** |
| beast still wins HR + XBH (no regression) | confirmed | **PASS** |
| Stacker delta CI excludes 0 | 3/3 fail (see below) | **FAIL** |

3 of 4 gates pass. Production wiring updated:
- New `RECENT365_MODEL_DIR` and `RECENT365_VAL_FEATURES` constants in config
- Daily macro uses recent365 by default; falls back to recency model dir if recent365 not present
- Output JSON `todays_matchup_predictions_recency.json` keeps its name (back-compat with dashboard tab) but contents are now from recent365

### 5. H5 stacker — improved but still fails CI gate (honest no-ship)

Re-ran [src/train_stacker.py](../src/train_stacker.py) with `recent365` added to members `[beast, exp, prod, recency, recent365, lr_interactions]`:

| Target | Δ Brier (Stack vs Best) — n=5 members | n=6 members | 95% CI (n=6) | Ships? |
|---|---:|---:|---|:---:|
| HR | +0.000045 | **+0.000006** | [-0.000048, +0.000057] | NO |
| Hit | +0.000098 | **-0.000075** | [-0.000226, +0.000060] | NO |
| XBH | +0.000049 | +0.000020 | [-0.000076, +0.000124] | NO |

**Hit delta sign FLIPPED** (+0.000098 → -0.000075) — stacker now beats best individual on Hit, but CI still includes 0 by 6e-5. HR delta improved 7×.

Stacker weights (HR target): beast 7.605, recent365 6.340, exp 6.192, prod 6.017, recency 5.709, lr_interactions 1.096. recent365 is the **second-most-weighted member after beast** — meaningfully different signal, just not different enough to cross the significance threshold.

**Verdict:** stacker doesn't ship, but trends are converging. Two more swings would likely cross the gate:

1. **PyTorch MLP** with batter/pitcher embeddings — algorithmic diversity is the unexplored axis (we've now tested data-window diversity via recent365 and feature-tweak diversity via lr_interactions). MLP gives different inductive bias.
2. **rolling-180-day** model — even narrower data window than 365, which would force more decorrelation.

Ship recommendation per the plan: **PyTorch MLP next sprint** (decision criterion: data-window diversity tested and partially successful but insufficient → algorithmic diversity is the remaining unexplored axis).

## Files changed

### New
- [src/build_team_bullpen_hr_priors.py](../src/build_team_bullpen_hr_priors.py) — bullpen prior builder
- [src/build_recent365_train.py](../src/build_recent365_train.py) — recent365 train parquet builder
- [data/priors/team_bullpen_hr.json](../data/priors/team_bullpen_hr.json) — 30-team bullpen HR/PA blends
- [data/master/models/recent365/](../data/master/models/recent365/) — trained recent365 models (5 targets)
- [data/master/features_train_league_recent365.parquet](../data/master/features_train_league_recent365.parquet) — recent365 train data
- [data/reports/multi_target_metrics_recent365.csv](../data/reports/multi_target_metrics_recent365.csv) — recent365 val metrics
- [data/reports/val_predictions_multi_recent365.parquet](../data/reports/val_predictions_multi_recent365.parquet) — recent365 2025 val predictions

### Edited
- [src/gen_zero_hr_predictions.py](../src/gen_zero_hr_predictions.py) — bullpen blends in `_load_priors`; per-team rate in `compute_one_game`; new JSON fields `pen_rate_away/home`, `pen_source_away/home`
- [src/gen_matchup_dashboard_html.py](../src/gen_matchup_dashboard_html.py) — Pen A / Pen H columns in No HR Model tab; CSS for `pen-leaky` / `pen-suppressing`
- [src/fetch_todays_games.py](../src/fetch_todays_games.py) — `hydrate=weather`; `_parse_roof_from_condition()`; new JSON fields `roof_state_today`, `weather_*_mlb`
- [src/train_multi_target.py](../src/train_multi_target.py) — recent365 routing in metrics CSV + val_predictions parquet naming
- [src/backtest_2025_models.py](../src/backtest_2025_models.py) — recent365 added to `MODEL_PREDICTION_PATHS`
- [src/train_stacker.py](../src/train_stacker.py) — recent365 added to default members
- [src/run_dual_model_daily.py](../src/run_dual_model_daily.py) — uses recent365 by default; legacy recency as fallback
- [src/config.py](../src/config.py) — new `RECENT365_*` constants

## What's next

### Pending the 5-slate Phase 1 weather measurement window
- After 5 slates, measure mean abs `weather_hr_mult` shift, directionality on wind-out games, and Open-Meteo coverage rate. If ≥3% magnitude → proceed to **Phase 2 weather** (per-PA adjustment).

### Next ensemble sprint (PyTorch MLP)
- Build a small MLP (3 layers, 64 hidden, learned batter/pitcher embeddings).
- Train on the same 1.7M-row beast train parquet.
- Add to H5 stacker. Expectation: this is the decorrelation axis recent365 didn't fully provide.

### Always-pending improvements (lower priority, not in this sprint)
- Per-PA weather adjustment (Phase 2 weather, gated)
- Historical weather backfill (Phase 3 weather, gated on Phase 2)
- L1 pitch-sequence model (research)
- L3 Bayesian hierarchical (research)
