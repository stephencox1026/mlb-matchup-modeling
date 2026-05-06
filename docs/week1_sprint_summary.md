# Week 1 sprint outcomes (model audit fix program)

Five tasks, all shipped: **CB2, CB1, C3, C2, H1**. The cleanup phase that unblocks everything that follows is done. Two material findings emerged that change the program direction.

## Tasks shipped

### CB2 — HR Medium non-monotonicity bug (fixed)

**Symptom:** HR Medium bucket realized 8.0% in live data, *below* Very Low's 9.9%. Non-monotonic confidence labels.

**Root cause:** [src/narrative_engine.py](../src/narrative_engine.py) `compute_confidence()` boosted `bvp_hr` to 1.15 (HR≥2) and 1.20 (HR≥3) **without checking BvP PA sample size**. Six picks with BvP PA ≤ 13 (Buxton 5 PA, Pasquantino 6 PA, Sosa 7 PA, Ohtani 8 PA, Jung 9 PA, Tovar 11 PA, Ohtani 13 PA) got the same boost as 60-PA samples. The 13 small-sample boosted picks went 0-for-13.

**Fix:** added PA-size gates (`bvp_pa >= 25` for the 1.20 boost, `bvp_pa >= 20` for the 1.15 boost) mirroring the existing pattern for `bvp_hr == 1`.

**Replay outcome:** monotonicity restored — Very Low 9.9% ≤ Low 10.7% = Medium 10.7% ≤ High 16.7%. 31 picks correctly demoted out of Medium.

Full diagnosis: [docs/conf_label_audit.md](conf_label_audit.md).

### CB1 — Per-conf-bucket reliability analyzer (new headline report)

Rewrote [src/analyze_prediction_tracking.py](../src/analyze_prediction_tracking.py). New primary output [data/reports/conf_bucket_calibration.md](../data/reports/conf_bucket_calibration.md) contains:

- Per-(target × bucket) realized rate with Wilson 95% CIs
- Lift vs baseline
- Monotonicity check with explicit FAIL flag and warning block
- Drift check: last 7d vs 8-30d per High bucket
- Conf factor means by bucket (which sub-factors actually shift across buckets)
- Top-decile lift and overall Brier/AUC/log-loss demoted to secondary diagnostics

Backward compat: legacy `prediction_calibration_summary.md` still emitted via `--legacy` flag.

CLI: `--target hr|hit|xbh`, `--high-only`, `--window-days N`, `--high-conf`.

### C3 — vs-SP outcome scoping (the eval cleanup)

[src/fill_matchup_prediction_outcomes.py](../src/fill_matchup_prediction_outcomes.py) now adds vs-SP columns (`outcome_*_vs_sp`) scoped to PAs vs the predicted starter only, alongside the existing whole-game `outcome_*` columns.

Backfilled 1666 rows: 1105 (66%) had at least one vs-SP PA. The other 34% were predictions where the batter never faced the predicted starter — bench players who didn't enter, late lineup substitutions, or starters pulled before that batter's first PA.

**Material finding revealed by C3:** the High HR bucket's lift evaporates when scoped to vs-SP outcomes.

| Bucket | Whole-game realized | vs-SP realized |
|---|---:|---:|
| HR Very Low | 9.1% | 9.4% |
| HR Low | 12.4% | 12.5% |
| HR Medium | 11.2% | 11.3% |
| **HR High** | **15.4%** (n=13) | **8.3%** (n=12) |

One of the High bucket's 2 HRs was off a relief pitcher — the predicted starter never gave it up. With vs-SP scoping, n=12 High picks produced 1 HR (8.3%, *below* baseline). Sample size is too small to conclude — but this is exactly the kind of contamination C3 was built to expose.

**Implication:** the previously-reported 1.81× HR High lift on whole-game outcomes was inflated. The H2 backtest (Phase 3) is now even more critical to get a real number with statistical power.

### C2 — Output schema cleanup (rename adj_p_* → score_*)

Three additive changes in [src/narrative_engine.py](../src/narrative_engine.py) (`compute_confidence` and `predict_matchup` returns):

- `score_hr`, `score_hit`, `score_xbh` — new fields, ranking score (= raw P × confidence factor). NOT a probability.
- `p_hr_calibrated`, `p_hit_calibrated`, `p_xbh_calibrated` — new fields, calibrated probability. Equal to raw P until C1 (isotonic recalibration) lands.
- `adj_p_*` kept as a deprecated alias for one release for back-compat.

Sections 1-6 in [src/gen_sections_1_6.py](../src/gen_sections_1_6.py) updated to print three columns: **Score | Cal P | Raw P** — replacing the misleading single "Adj P" column. Rule [.cursor/rules/daily-output-sections.mdc](../.cursor/rules/daily-output-sections.mdc) updated accordingly.

The Score column now uses absolute units (e.g. `6.92` not `6.9%`) so it cannot be misread as a probability.

### H1 — exp/beast duplicate metrics resolved

Cause of the byte-identical CSVs: prior runs both used the *default* train/val parquets (the 82-feature base) instead of the variant `exp` (88-feature) and `beast` (108-feature) parquets. Re-ran both with the correct `--train-path` / `--val-path` flags. New MD5 hashes:

- `multi_target_metrics_exp.csv` → `aeac7ed4...`
- `multi_target_metrics_beast.csv` → `46c4f206...`

But the audit memo's broader conclusion stands: the deltas are tiny.

| Target | exp Brier | beast Brier | Δ |
|---|---:|---:|---:|
| is_hit_raw | 0.16169 | 0.16169 | 0.00000 |
| is_hr_raw | 0.02934 | 0.02934 | 0.00000 |
| is_strikeout_raw | 0.14285 | 0.14280 | 0.00005 |
| is_walk_raw | 0.05289 | 0.05289 | 0.00000 |
| is_xbh_raw | 0.06876 | 0.06875 | 0.00001 |

Beast adds 20 features over exp; Brier improvement is in the 5th decimal place — within noise. **Marginal feature tweaks on the same XGBoost architecture are not real model diversity.** H5 (real ensemble diversity via algorithmic + data diversity) remains the right Phase 4 work.

## Material findings that change program direction

1. **The previously-celebrated 1.81× HR High bucket lift was inflated by whole-game contamination.** With vs-SP scoping (n=12 picks), the High bucket realizes 8.3% — below baseline. The conviction picks story is **less established than we thought**. M1 (the conf meta-model) becomes even more important; we may need to gate Section 14 (Conviction Picks) until we have a real signal validated on more data.

2. **The four model variants really are essentially one model.** Even with proper training, exp ↔ beast Brier deltas are 0.00005 max. H5 should not bother trying yet another tree-based variant; it must add a different inductive bias (sklearn LR with interactions, PyTorch MLP) or different data window (`recent365`).

## What's next (Week 2-3)

- **C1**: per-conf-bucket isotonic recalibration. With C3 now providing clean vs-SP outcomes and CB1 providing the eval lens, C1 has everything it needs.
- **M3**: drift monitor as Section 0 in chat output.
- **CB3**: Conviction Picks digest as Section 14. Because of the vs-SP finding above, we may want to start with a more conservative threshold (only emit Section 14 picks when bucket lift is statistically significant).

## What's after that (Weeks 4-5)

- **H2**: 2025 retrospective backtest. Now the highest-priority work in the program — we cannot trust the n=12 High-bucket finding without orders of magnitude more data.
