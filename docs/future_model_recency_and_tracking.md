# Future: tracking data in the model and recency vs history

This document is **design-only**. It does not change training or inference. Use it when you explicitly decide to incorporate **prediction tracking** (`data/tracking/` once built) or to **retune how recent Statcast is weighted** relative to career/long windows.

## Principles

1. **Tracking stays observational until you opt in** — use it first for calibration, Brier/deciles, and bucket errors; only then retrain or add post-hoc layers.
2. **Three clocks** (do not conflate them):
   - **Calendar** — same season vs prior seasons, league environment drift.
   - **Sample size** — 8 BvP PAs vs 400 career PAs (noise vs signal).
   - **Model snapshot** — weights trained through date X vs features built “as of” slate date Y.
3. **No leakage** — any feature derived from outcomes must use only information **strictly before** the prediction time (shifted rolls, prior slates for calibration layers). Same-day outcomes never feed the same slate’s input row.
4. **Align train and inference** — the recipe that defines `roll_*` at training time should match `build_feature_vector` at prediction time. A mismatch (e.g. median of last N rows from a frozen val parquet vs true shifted rolling as of `slate_date`) systematically under- or over-weights “recent” signal.

## Where the codebase stands today

| Piece | Role |
|--------|------|
| [`src/features.py`](../src/features.py) `ROLLING_WINDOWS = [10, 30, 100]` | Shifted rolling rates per batter at PA granularity (no leakage). |
| [`src/narrative_engine.py`](../src/narrative_engine.py) `build_feature_vector` | Last 20 rows in `features_val_league` → **median** per feature; then `_dampen_rolling_features` blends short windows toward `roll_*_100`. |
| [`src/train_multi_target.py`](../src/train_multi_target.py) | HGB + calibration; no sample-weight decay by date unless you add it. |

## Phased use of tracking (when Parquet exists)

| Phase | What you do | Risk |
|-------|-------------|------|
| **A — Calibration only** | Isotonic/Platt or bucket corrections on **held-out** slates; adjust `conf_*` or reporting, not raw `p_*` in the pickle without validation. | Low if fit only on past slates. |
| **B — Retrain / features** | Use tracking + outcomes to propose new features or hyperparameters; retrain in a **separate** experiment. | Medium — need walk-forward validation. |
| **C — Meta-layer** | Small second stage on **lagged** aggregates (e.g. calibration error by tier last 30 slates). | Overfit if buckets are too fine or data is thin. |

## Ways to work “recent” into the model (options + tradeoffs)

Use **one primary knob** first (e.g. shrinkage **or** EWMA **or** time-weighted loss), then combine cautiously after ablation.

### 1. Multi-window features + learner (current pattern)

Keep `roll_*_10`, `_30`, `_100`; let HGB combine them; regularize with `min_samples_leaf`, depth, and time-based validation.

- **Pros:** Simple; trees learn interactions. **Cons:** Noisy short windows; collinearity. **Over-weight risk:** medium. **Under-weight:** low for recency if long windows dominate.

### 2. Explicit shrinkage short → long (already at inference)

Blend `roll_10` / `roll_30` toward `roll_100` with weights **tuned** on forward validation (Brier/log loss), not only by intuition.

- **Pros:** Direct control. **Cons:** Must match train-time definitions. **Balance:** Best lever for “not too hot, not too cold.”

### 3. Reliability / empirical Bayes shrinkage

Shrink rate estimates toward league or personal prior with weight ∝ effective sample size (e.g. \( \frac{n}{n+k} \bar{y} + \frac{k}{n+k}\mu \)).

- **Pros:** Under-weights recency when n is tiny. **Cons:** More plumbing; tune \(k\). **Under-weight risk:** high if \(k\) too large.

### 4. EWMA (exponential half-life on outcomes)

Replace or supplement rectangular windows with exponentially weighted past PAs (still shifted).

- **Pros:** Smooth recency. **Cons:** Extra hyperparameter; correlates with long windows. Tune half-life on walk-forward CV.

### 5. “Form” features: recent minus baseline

e.g. `roll_30 - roll_100` or vs career — lets the model use **deltas** only when they generalize.

- **Pros:** Separates level vs change. **Cons:** Difference of noisy series is noisy; pair with dampening or wider `min_periods`.

### 6. Time-decayed **training** weights (`sample_weight`)

Up-weight recent seasons or dates in the loss.

- **Pros:** Adapts to league drift. **Cons:** Can chase noise; recalibrate probabilities after. **Over-weight risk:** high if weights are aggressive.

### 7. Hierarchical / partial pooling (player-season offsets)

Mixed or embedding-style intercepts shrunk toward zero.

- **Pros:** Good for real non-stationarity and small n. **Cons:** Heavier than current sklearn-only stack.

### 8. Two-stage: base model + calibration layer

Base model unchanged; second layer adjusts by segment using **lagged** OOS errors (optionally from tracking).

- **Pros:** Protects core from streak-chasing. **Cons:** Needs volume; smooth buckets.

## When you are ready — checklist

- [ ] Rebuild `features_*` with any new columns; freeze **as-of-slate_date** rolling logic for inference.
- [ ] Walk-forward or chained validation on PA- or slate-level Brier/log loss.
- [ ] Compare calibration **by recency deciles** (e.g. after cold 10-PA stretch).
- [ ] Document chosen shrinkage / half-life / weights in this file or in `data/tracking/README.md`.
- [ ] If using tracking-derived signals: **lag by at least one slate** from the row being scored.

## Related plan

Cursor plan **Prediction tracking store** (`prediction_tracking_store_636e0a02.plan.md`): Parquet store, outcome fill, high-conf slice, analysis scripts. This doc is the **Phase 2** companion for model integration when you choose to turn it on.
