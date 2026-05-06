# Matchup prediction tracking (Parquet)

Append-only history of **`predict_matchups`** outputs for calibration and audits. **Not** used for model training unless you explicitly adopt a Phase 2 workflow (see [../../docs/future_model_recency_and_tracking.md](../../docs/future_model_recency_and_tracking.md)).

## Files

| File | Description |
|------|-------------|
| `matchup_predictions_runs.parquet` | All slate rows; outcomes filled after Statcast updates. |
| `matchup_predictions_runs_high_conf.parquet` | Same schema; **`max(conf_hr, conf_hit, conf_xbh) >= 1.05`** (same threshold as “High” in `narrative_engine.confidence_label`). |
| `matchup_dual_model_predictions.parquet` | **Optional:** wide rows when you run `build_matchup_dashboard.py --dual-model` — production vs **experiment** (`exp_bpt_xwoba`) probabilities, deltas, and paths used. |

## Experiment bundle (`exp_bpt_xwoba`)

1. Rebuild batter pitch profiles (shrunk pitch-type barrel): `python3 src/build_batter_pitch_profiles.py`
2. Build **experiment** league features only (does **not** overwrite default league features):  
   `python3 src/features.py --league --exp`  
   → `data/master/features_train_league_exp.parquet`, `features_val_league_exp.parquet`
3. Train experiment models into a separate folder:  
   `python3 src/train_multi_target.py --train-path data/master/features_train_league_exp.parquet --val-path data/master/features_val_league_exp.parquet --model-dir data/master/models/exp_bpt_xwoba`
4. Single-matchup CLI (experiment):  
   `python3 src/predict_matchup.py --experiment "Batter Name" "Pitcher Name"`
5. Dashboard + prod + dual diff Parquet + summary:  
   `python3 src/build_matchup_dashboard.py --dual-model`  
   (requires step 3 artifacts.) Also writes `data/reports/dual_model_diff_summary.md` for the latest dual run.  
   Re-run summary only: `python3 src/compare_dual_model_predictions.py`

## Typical run order

1. **Slate predictions** — after `todays_matchups.json` is set for the calendar day:

   `python3 src/build_matchup_dashboard.py`

   This writes `data/reports/todays_matchup_predictions.json`, archives a copy under `data/reports/archive/{slate}/`, and **appends** this directory’s Parquet files (deduped on `slate_date`, `game_pk`, `batter_mlbam_id`, `pitcher_mlbam_id`).

2. **Statcast** — when pybaseball / Statcast has the games (often the next morning):

   `python3 src/append_statcast_day_to_league_pa.py`  
   or a date range: `python3 src/append_statcast_date_range.py --start YYYY-MM-DD --end YYYY-MM-DD`

3. **Fill outcomes** — join same-day PA / H / HR / XBH per batter (aligned with `audit_top10_predictions`):

   `python3 src/fill_matchup_prediction_outcomes.py`

   Optional: `--slate 2026-04-19` to refresh one day; `--dry-run` to preview counts.

4. **Analyze** — Brier, ROC, deciles, tier table:

   `python3 src/analyze_prediction_tracking.py`  
   `python3 src/analyze_prediction_tracking.py --high-conf`

   Outputs: `data/reports/prediction_calibration_summary.md` and `_high_conf.md`.

## Outcome columns

- `outcome_pa`, `outcome_h`, `outcome_hr`, `outcome_xbh` — Statcast totals for that **calendar date** and batter (all teams that day).
- `outcome_hit_flag` — 1 if `outcome_h > 0` (any hit that day).
- `outcome_hr_flag`, `outcome_xbh_flag` — analogous for HR / XBH.
- `outcome_filled_at` — UTC ISO timestamp when the fill script wrote the row.

Doubleheaders and multi-team same-day edge cases are rare; counts follow the same-day aggregate join used elsewhere in this project.
