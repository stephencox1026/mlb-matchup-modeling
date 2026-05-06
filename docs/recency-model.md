# Recency model (third bundle)

This is an optional **third** model alongside production and experiment. It does two things:

1. **Feature columns** — Built with `python3 src/features.py --league --recency`:
   - **PA-window shorts:** `roll_*_3`, `roll_*_5` and pitcher `p_roll_*_3/5` are **last N plate appearances** (rows sorted by batter/pitcher, then time), all with `shift(1)` so the current PA is excluded.
   - **PA-window medium/long:** `roll_*_10`, `roll_*_30`, `roll_*_100` (same PA semantics).
   - **True game-window shorts:** `g_roll_*_{3,5,10}` (batter) and `p_g_roll_*_{3,5,10}` (pitcher) are **last K completed games** before the current game: stats are aggregated per `game_pk`, then shifted and rolled on the game timeline (no same-game leakage).
2. **Training** — Optional **exponential sample weights** by `game_date` so more recent plate appearances count more when you pass `--recency-sample-weights` to `train_multi_target.py` (half-life configurable; default 120 days).

Paths are centralized in `src/config.py`:

| Role | Path |
|------|------|
| Model directory | `data/master/models/exp_recency_l3l5/` |
| Val features (inference) | `data/master/features_val_league_recency.parquet` |

## 1. Build recency feature parquet

From the project root (venv active if you use one):

```bash
python3 src/features.py --league --recency
```

Writes:

- `data/master/features_train_league_recency.parquet`
- `data/master/features_val_league_recency.parquet`

Do **not** combine `--recency` and `--exp` in one `features.py` run; pick one suffix per build.

## 2. Train the recency bundle

Point `--model-dir` at the `exp_recency_l3l5` folder (name must contain `recency` so metrics default to `multi_target_metrics_recency.csv` / `val_predictions_multi_recency.parquet` under `data/reports/`).

**With** recent-PA sample weights (recommended for the “hot hand” story):

```bash
python3 src/train_multi_target.py \
  --train-path data/master/features_train_league_recency.parquet \
  --val-path data/master/features_val_league_recency.parquet \
  --model-dir data/master/models/exp_recency_l3l5 \
  --recency-sample-weights \
  --recency-weight-half-life-days 120
```

**Without** sample weights (only the L3/L5 feature signal):

```bash
python3 src/train_multi_target.py \
  --train-path data/master/features_train_league_recency.parquet \
  --val-path data/master/features_val_league_recency.parquet \
  --model-dir data/master/models/exp_recency_l3l5
```

Confirm `data/master/models/exp_recency_l3l5/feature_columns.json` exists after training.

## 2b. Evaluation protocol (2025 val, proper scores)

- **Primary scoreboard:** Out-of-time **2025** validation rows in `features_val_league_recency.parquet` (same split as production: train `game_year <= 2024`, val `game_year == 2025`).
- **Metrics files:** Each `train_multi_target.py` run writes a CSV of per-target metrics. Defaults:
  - Production: `data/reports/multi_target_metrics.csv`
  - Recency bundle dir name contains `recency`: `data/reports/multi_target_metrics_recency.csv`
  - Overrides: `--metrics-csv` / `--val-pred-parquet`
- **Compare runs:** Use calibrated rows for `is_hit`, `is_hr`, `is_xbh` and prefer **Brier** (and calibration gap) over raw AUC for rare events.

```bash
# Compare arbitrary saved metrics CSVs (negative delta = better than baseline)
python3 src/recency_metrics_tools.py compare \
  --baseline data/reports/multi_target_metrics.csv \
  --label recency_u=data/reports/recency_ablation_unweighted.csv \
  --label recency_w=data/reports/recency_ablation_weighted_hl120.csv
```

**Baseline ablation (unweighted vs weighted, same recency features):**

```bash
python3 src/recency_metrics_tools.py train-baseline
```

Writes models under `data/master/models/exp_recency_ablation_unweighted/` and `.../exp_recency_ablation_weighted_hl120/`, metrics under `data/reports/recency_ablation_*.csv`, then prints a comparison table.

**Grid half-life (days) on training sample weights** — trains separate dirs `data/master/models/recency_tune_hl{60,120,180}/`:

```bash
python3 src/recency_metrics_tools.py tune-half-life --halves 60,120,180
```

Pick the half-life with best **val Brier** per target (or compromise), then train the deploy bundle into `exp_recency_l3l5` with that `--recency-weight-half-life-days`.

**Apples-to-apples note:** `multi_target_metrics.csv` is from models trained on **production** feature columns. Comparing those Briers directly to recency-feature models is a **sanity check**, not strict; the `train-baseline` command compares two recency trains and still prints prod numbers for context.

## 3. Daily predictions and reports

`src/run_dual_model_daily.py` will, when the bundle and val parquet exist:

- Write `data/reports/todays_matchup_predictions_recency.json`
- Archive it under `data/reports/archive/<slate>/`
- Regenerate `matchup_dashboard.html`. When the recency JSON is present and its `slate_date` matches production, the **Matchups · recency** tab shows top-25 tables for hits, XBH, and HR (same slate picker as the main matchups tab when multiple archive slates are embedded).

## 4. Single matchup CLI

```bash
python3 src/predict_matchup.py "Shohei Ohtani" "Some Pitcher" --recency
```

Use only one of `--experiment` or `--recency`.

## 5. Inference details

- `narrative_engine.predict_matchups(..., model_source="recency")` passes `skip_rolling_dampen=True` into `build_feature_vector` so short-window rolls are not stability-dampened the same way as the main league model.

If anything is missing (no model dir, no val parquet), the daily script prints a warning and skips recency; prod + exp still run.
