# Sprint 2 Summary — All 4 Next-Step Items Built

All four items from the previous session's "next-step" list are shipped. Headline numbers are concrete and measured against held-out data.

## 1. Beast retrained with park + xptw + month features

Backfilled the new features into the existing beast parquet (no full features.py regeneration needed) and retrained. **All five targets improved**:

| Target | Old Beast Brier | New Beast Brier | Δ Brier | Old AUC | New AUC | Δ AUC |
|---|---:|---:|---:|---:|---:|---:|
| is_hit | 0.16169 | **0.16085** | -0.00084 | 0.6610 | 0.6687 | +0.008 |
| is_hr | 0.02934 | **0.02927** | -0.00007 | 0.7178 | **0.7306** | +0.013 |
| is_strikeout | 0.14280 | **0.14193** | -0.00087 | 0.7845 | 0.7885 | +0.004 |
| is_walk | 0.05289 | 0.05281 | -0.00008 | 0.9298 | 0.9300 | +0.000 |
| is_xbh | 0.06875 | **0.06862** | -0.00013 | 0.6776 | 0.6853 | +0.008 |

Hit AUC jumped 0.661 → 0.669 (largest single-feature delta in the audit history). HR AUC 0.718 → 0.731.

### H2 backtest re-run on new beast

Beast now wins ALL THREE primary targets, both on Brier AND on top-decile lift:

| Target | Best Brier | Top-Decile Lift (was → now) |
|---|---:|---|
| is_hr | 0.02937 (beast) | 2.38× → **2.49×** |
| is_hit | 0.16161 (beast) | 1.57× → **1.65×** |
| is_xbh | 0.06883 (beast) | 1.94× → **2.00×** |

### M1 retrained on new beast → Lock buckets sharper

| Target | Lock Realized (was → now) | Lock Lift (was → now) |
|---|---:|---:|
| HR | 7.42% → **8.28%** | 2.44× → **2.72×** |
| Hit | 33.42% → 34.39% | 1.55× → 1.59× |
| XBH | 14.95% → 14.86% | 1.97× → 1.96× |

The HR Lock bucket realized rate now exceeds 8% — that's a real betting signal in the H2 backtest.

## 2. Section 0 + Section 14 wired into daily macros

[src/run_dual_model_daily.py](../src/run_dual_model_daily.py) now invokes (in order, after Sections 1-10):

1. `python3 src/monitor_calibration_drift.py` → writes `data/reports/section_0_drift.md` and auto-refits per-bucket isotonic if any bucket drifts >5%.
2. `python3 src/gen_conviction_picks.py` → writes `data/reports/section_14_conviction_picks.md` (Lock picks across HR/Hit/XBH).
3. `python3 src/calibrate_predictions.py` → refit per-bucket isotonic on latest 14d window.
4. `python3 src/gen_zero_hr_predictions.py` → writes `data/reports/section_15_zero_hr.md` (P(0 HR) per game).

All five new artifacts are added to the slate archive copy list. Rule files updated:

- [.cursor/rules/daily-output-sections.mdc](../.cursor/rules/daily-output-sections.mdc) — Sections 0, 14, 15 fully specified; chat order is now `0, 1-6, 9, 11-13, 14, 15`.
- [.cursor/rules/update-now.mdc](../.cursor/rules/update-now.mdc) — required outputs include the new files; chat rules require Section 0 first, Section 14, Section 15 last.
- [.cursor/rules/update-now-dual-output.mdc](../.cursor/rules/update-now-dual-output.mdc) — same.

## 3. M1 wired into narrative_engine at runtime

[src/conf_meta_inference.py](../src/conf_meta_inference.py) — runtime helper that loads pickled M1 models + thresholds and computes Lock/Strong/Lean/Avoid from any `(target, base_p, features_dict)`.

[src/narrative_engine.py](../src/narrative_engine.py) `predict_matchups()` — after the base model scores each matchup, the same feature vector is fed to M1. Result dict gains:

- `score_label_hr`, `score_label_hit`, `score_label_xbh` — new M1 labels (Lock/Strong/Lean/Avoid)
- `meta_p_hr`, `meta_p_hit`, `meta_p_xbh` — meta-model calibrated probabilities

Defensive: any failure (model missing, bad input) falls back to "Avoid" + raw P, so inference never breaks.

[src/gen_conviction_picks.py](../src/gen_conviction_picks.py) now **prefers M1 Lock label over hand-tuned High** when the new fields are present. Each section header notes which label source was used.

## 4. No HR Model reactivated as Section 15

Built [src/gen_zero_hr_predictions.py](../src/gen_zero_hr_predictions.py) as a slim consumer of the now-park-aware base.

### NB dispersion fit (one-time prerequisite)

Fit on 2022-2024 bulk Statcast game-level HR counts:

| Quantity | Value |
|---|---|
| Games | 7,289 |
| Mean HR/game | 2.27 |
| Var HR/game | 2.68 |
| Var/mean ratio | 1.18 (over-dispersed) |
| MLE k | **12.65** |
| Poisson P(0 HR | λ=2.27) | 10.4% |
| **NB P(0 HR | λ=2.27, k=12.65)** | **12.4%** |

Validates the original audit prediction: NB closes the 3-percentage-point gap that Poisson leaves. Saved to [data/priors/nb_dispersion.json](../data/priors/nb_dispersion.json).

### Section 15 algorithm

For each game:

```
λ_game = (slot-weighted Σ p_hr_calibrated for top-9 batters vs each starter)
        + (PA_pen × league_HR/PA for each side's bullpen)
λ_adj  = λ_game × park_pf_hr(home_team)        # L2 two-sided park factor
P(0 HR) = (k / (k + λ_adj))^k                   # NB-zero, k=12.65
```

BF̂ for each starter conditioned on Section 9 projected runs (heavier projection → shorter outing → more PAs to bullpen).

### Today's slate output

| Rank | Matchup | Park PF | λ_total | P(0 HR) |
|---:|---|---:|---:|---:|
| 1 | STL @ PIT | 0.86 | 1.90 | **17.0%** |
| 2 | LAA @ CWS | 1.03 | 2.62 | 9.2% |
| 3 | TB @ CLE | 0.94 | 2.65 | 9.0% |
| 4 | BOS @ TOR | 1.07 | 2.76 | 8.2% |
| 5 | SEA @ MIN | 1.02 | 2.80 | 8.0% |
| 6 | NYY @ TEX | 1.06 | 2.87 | 7.5% |
| 7 | MIA @ LAD | 1.16 | 3.14 | 6.1% |
| 8 | CHC @ SD | 1.04 | 3.22 | 5.7% |

Top zero-HR pick: STL @ PIT at PNC Park (PF 0.86 = HR-suppressing) — that's a 17% P(0 HR), well above the 12% league baseline. Bottom: MIA @ LAD at Dodger Stadium (PF 1.16 = HR-friendly) at 6.1%.

## What's now in production end-to-end

When you run `update now`, the daily output now contains:

- **Section 0** — Bucket Health (per-conf-bucket realized vs predicted on rolling 14d, with WARN/CRITICAL flags)
- **Sections 1-6** — Top-25 lists with three columns (Score, Cal P, Raw P)
- **Section 9** — Starter run predictions
- **Sections 11-13** — Barrel sections
- **Section 14** — Conviction Picks (M1 Lock labels)
- **Section 15** — No HR Model (per-game P(0 HR) sorted high-to-low)

Plus the always-on auto-refit: any drift > 5% triggers a per-bucket isotonic refit before the next slate runs.

## Files added/changed this sprint

### New
- [src/conf_meta_inference.py](../src/conf_meta_inference.py) — M1 runtime helper
- [src/gen_zero_hr_predictions.py](../src/gen_zero_hr_predictions.py) — Section 15
- [data/priors/nb_dispersion.json](../data/priors/nb_dispersion.json) — NB k fit

### Edited
- [src/narrative_engine.py](../src/narrative_engine.py) — M1 wire-in at predict_matchups
- [src/gen_conviction_picks.py](../src/gen_conviction_picks.py) — prefer M1 Lock label
- [src/run_dual_model_daily.py](../src/run_dual_model_daily.py) — invoke Sections 0, 14, 15 + refit + archive
- [.cursor/rules/daily-output-sections.mdc](../.cursor/rules/daily-output-sections.mdc) — Sections 0, 14, 15 specs + chat order
- [.cursor/rules/update-now.mdc](../.cursor/rules/update-now.mdc) — required outputs + chat rules
- [.cursor/rules/update-now-dual-output.mdc](../.cursor/rules/update-now-dual-output.mdc) — required outputs + chat rules

### Regenerated
- `data/master/models/exp_beast/best_model_*.pkl` — beast retrained with 135 features
- `data/reports/val_predictions_multi_beast.parquet` — new beast scored on 2025
- `data/master/models/conf_meta/conf_meta_*.pkl` — M1 retrained on new beast
- `data/priors/conf_meta_thresholds.json` — new bucket thresholds
- `data/master/features_train_league_beast.parquet` — added park + xptw cols
- `data/master/features_val_league_beast.parquet` — same

## What's next (now-genuinely-open work)

- **L1 (pitch-sequence model)** — 2-4 week research project, not started.
- **L3 (Bayesian hierarchical)** — 2-3 week research project, not started.
- **Real ensemble diversity (PyTorch MLP, recent365)** — H5 stacker failed its ship gate; only path forward is genuinely different inductive biases.
- **Per-team bullpen HR/PA priors** for Section 15 — currently uses league avg.
- **Roof state for Section 15** — currently uses static enclosure type; live `hydrate=weather` from MLB StatsAPI would refine retractable park PFs.

The Sprint 1 + Sprint 2 work has shipped the entire program plan except the deferred research items. The model stack is now:

- **Park-aware** (L2)
- **Pitch-type-matchup-aware** (M4)
- **Month-of-season-aware** (C4)
- **Per-conf-bucket-recalibrated** (C1)
- **Drift-monitored with auto-refit** (M3)
- **Backed by a probability-stacker meta-model** (M1)
- **Per-game evaluable** (M2)
- **Statistically validated against 2025** (H2)
- **Honest about its limits** (H5 retired, BvP audited via H4)
- **Producing zero-HR-game predictions** (Section 15)
