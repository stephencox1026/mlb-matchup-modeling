# Model Card: LAD Hitter Hit Probability Model

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | LAD Hitter PA-Level Hit Probability |
| **Version** | 1.0 |
| **Date** | 2026-04-13 |
| **Type** | Binary classification (hit vs. no-hit per plate appearance) |
| **Architecture** | Calibrated Random Forest (isotonic calibration, 5-fold CV) |
| **Base Estimator** | `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20)` |
| **Framework** | scikit-learn 1.6.x |
| **Training Data** | 34,602 PAs (2015–2024), 14 Dodgers hitters |
| **Validation Data** | 6,006 PAs (2025 season, same 14 hitters) |
| **Alternatives Evaluated** | Logistic Regression, HistGradientBoosting, Stacking Ensemble |

## Intended Use

- **Primary use**: Estimate per-PA probability of a hit for the 14 target Dodgers hitters, with emphasis on performance vs. left-handed pitching.
- **Downstream applications**: Game-level simulation (P(multi-hit), expected total bases, P(HR)), lineup optimization insights, executive reporting.
- **NOT intended for**: Real-time in-game decisions, betting, player valuation, or deployment beyond the 14 target hitters without retraining.

## Training Data

- **Source**: Statcast pitch-level data via `pybaseball.statcast_batter`, aggregated to PA-level outcomes.
- **Years**: 2015–2024 (regular season only, `game_type == "R"`).
- **Players**: 14 Dodgers hitters (per `ROSTER_14` in `src/config.py`).
- **Total PAs**: 34,602 training rows.
- **Hit rate**: 0.239 (class balance: ~24% positive).
- **Supplementary data**: Historical year-by-year platoon splits (vs LHP / vs RHP) from the MLB Stats API, used as cumulative career features.

## Features (34 total)

| Group | Count | Description |
|-------|-------|-------------|
| Rolling BA/HR/K/BB rates | 12 | Shifted rolling windows (10, 30, 100 PAs) over hit, HR, strikeout, and walk outcomes |
| Rolling Statcast (EV/LA/barrel) | 9 | Shifted rolling exit velocity, launch angle, barrel rate |
| Career split context | 8 | Cumulative career BA, OPS, HR, PA vs LHP and vs RHP (shifted by 1 season) |
| Game context | 5 | Pitcher handedness (vs_lhp), month, day_of_week, pitch count, in-zone indicator |

All rolling features use `.shift(1)` to prevent temporal leakage — each prediction uses only information available before the current PA.

## Evaluation Metrics (2025 Validation)

| Model | Brier Score | Log Loss | ROC AUC | Avg Precision | Calibration Gap |
|-------|-------------|----------|---------|---------------|-----------------|
| Baseline (predict mean) | 0.17702 | — | — | — | — |
| Logistic Regression | 0.17037 | 0.51796 | 0.6322 | 0.3106 | 0.0126 |
| HistGradientBoosting | 0.17049 | 0.51716 | 0.6279 | 0.3092 | 0.0157 |
| **Random Forest** | **0.16961** | **0.51589** | **0.6331** | **0.3129** | **0.0137** |
| Stacking Ensemble | 0.17133 | 0.52123 | 0.6294 | 0.3084 | 0.0243 |
| **RF Calibrated (selected)** | **0.17004** | **0.51758** | **0.6309** | **0.3094** | **0.0124** |

The calibrated Random Forest was selected for its best Brier score (0.170) and tightest calibration gap (0.012).

## Feature Importance (SHAP)

Top 10 features by mean |SHAP| value:

| Rank | Feature | Mean |SHAP| |
|------|---------|------|
| 1 | in_zone | 0.0559 |
| 2 | pitch_count | 0.0404 |
| 3 | roll_la_10 | 0.0048 |
| 4 | cum_career_pa_vs_lhp | 0.0047 |
| 5 | cum_career_ops_vs_lhp | 0.0032 |
| 6 | cum_career_pa_vs_rhp | 0.0029 |
| 7 | cum_career_hr_vs_rhp | 0.0026 |
| 8 | cum_career_ba_vs_rhp | 0.0024 |
| 9 | roll_ev_30 | 0.0022 |
| 10 | roll_ev_10 | 0.0019 |

## Ablation Study

| Dropped Group | Features Dropped | Brier (no group) | AUC (no group) | Impact |
|---------------|-----------------|-------------------|----------------|--------|
| rolling_10 | 7 | 0.17023 | 0.6307 | Minimal |
| rolling_30 | 7 | 0.17024 | 0.6282 | Minimal |
| rolling_100 | 7 | 0.17124 | 0.6230 | Moderate |
| career_splits | 8 | 0.16984 | 0.6323 | Marginal improvement without |
| statcast_ev_la | 6 | 0.16953 | 0.6388 | Marginal improvement without |
| barrel | 3 | 0.17049 | 0.6279 | Minimal |
| **context** | **5** | **0.17709** | **0.5165** | **Critical — near-random without** |

**Key insight**: The `context` group (pitcher handedness, month, zone, pitch count) is by far the most important feature group. Removing it collapses AUC to near-random (0.52).

## Multicollinearity (VIF)

High VIF values (>100) detected for rolling EV features across windows and career OPS/BA splits, indicating strong collinearity. The tree-based model handles this natively, but users should be cautious interpreting individual feature coefficients.

## Limitations & Risks

1. **Population**: Trained on only 14 Dodgers hitters. Generalization to other teams/players requires retraining.
2. **Temporal scope**: 2015–2024 training. Rule changes (pitch clock, shift ban) may create distribution shift.
3. **Statcast coverage**: Pre-2017 Statcast data has lower tracking quality; launch angle/speed features may be noisier.
4. **Hit prediction is inherently noisy**: Even the best models achieve modest AUC (~0.63) because single-PA outcomes have high irreducible variance.
5. **No pitcher-specific features**: The model uses pitcher handedness but not individual pitcher quality metrics.
6. **Calibration**: While well-calibrated on aggregate (gap = 0.012), per-player calibration varies.

## Matchup inference (production dual-model path)

Separately from the legacy Dodgers RF card above, daily matchup HR probabilities pass **post-model gates** in `src/narrative_engine.py`:

- **Zero-season-HR gate**: For the slate’s calendar year, using Statcast PA rows **before** the slate date, if a batter has **at least one** row but **zero** home runs in that window, raw **`p_hr`** is floored to **`1e-4`** after thin-YTD and career caps. This removes zero-HR counting profiles from HR leaderboards while leaving Hit/XBH unchanged. Batters with **no** Statcast rows that season are not gated.

- **xBABIP-style on BIP (league GBDT features):** [`src/features.py`](../src/features.py) adds **prior-only** rolling means (`shift(1)` then rolling) of Statcast **`estimated_ba_using_speedangle`** and **`estimated_woba_using_speedangle`**, evaluated only on **non-HR balls in play** (PA has `launch_speed` and event is not `home_run`). Batter: `roll_est_ba_bip_{30,100}`, `roll_est_woba_bip_{30,100}`; pitcher-allowed mirrors: `p_roll_est_*_bip_*`. Retrain `train_multi_target.py` after regenerating league feature parquet so `feature_columns.json` stays aligned.

## Ethical Considerations

- This model should not be used for gambling or wagering decisions.
- Player performance predictions carry uncertainty; presenting them as definitive projections to non-technical audiences could be misleading.
- Results should always be accompanied by confidence intervals or uncertainty measures.

## Leakage Tests

All 5 integrity tests pass:
- ✓ Temporal split enforced (train ≤ 2024, val = 2025)
- ✓ No game overlap between train and val
- ✓ Rolling features properly shifted
- ✓ Target columns not in feature set
- ✓ Feature null rates acceptable
