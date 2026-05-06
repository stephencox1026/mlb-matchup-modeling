# Hit model — elite tuning roadmap

Canonical critique + **full implementation plan**.  
Inference stack reference: Beast GBDT (`data/master/models/exp_beast/`) → [`apply_vs_hand_and_bvp_posteriors`](src/narrative_engine.py) → BvP blend → [`platoon_raw_probability_shrink`](src/narrative_engine.py); row construction in [`build_feature_vector`](src/narrative_engine.py) (`_median_recent_features_aligned`, `_dampen_rolling_features`). Lineup rollup JSON: [`lineup_context`](src/lineup_context.py).

---

## Part A — Critique (what we have today)

### Three things included but redundant or low value (for hit)

**1. Platoon / vs-hand is triple-stacked**  
The Beast row already has **`cum_career_*_vs_lhp/rhp`**, **`platoon_eff_*`**, and **`platoon_matched_ba` / `platoon_matched_ops` / `platoon_ops_gap` / `platoon_other_*`** — mostly the same split story told several ways (see [`add_platoon_aligned_split_features`](src/features.py), decayed platoon in [`platoon_decayed_rates_for_batter_year`](src/features.py)). Inference then runs **`apply_vs_hand_and_bvp_posteriors`**, which mixes **`p_hit`** again using **`platoon_eff_h/ab`** plus **YTD vs-hand**. So the model sees platoon-like signal **once in the trees and again in closed form**.

**2. BvP for hits is used twice**  
**`bvp_ba`**, **`log_bvp_pa`**, rates, etc. are **inputs to the GBDT** ([`build_matchup_features`](src/build_matchup_features.py) / league PA merges), and **`BVP_PRIOR_PA_HIT`** blending **updates `is_hit` again** from raw hit counts vs that pitcher. Same head-to-head evidence drives **both** stages unless roles are deliberately separated.

**3. Overlapping short-window contact rolls (hitter + pitcher)**  
Batter: **`roll_ba_*`**, **`roll_est_woba_*`**, **`roll_est_woba_bip_*`**, **`roll_est_ba_bip_*`**, plus hard-hit / air / iso rolls. Pitcher: **`p_roll_hit_allowed_*`**, **`p_roll_est_ba_bip_*`**, **`p_roll_est_woba_bip_*`**, etc. Those families are **highly correlated**; trees can absorb it but you pay in **stability, calibration drift**, and **sensitivity to the median inference row**.

**(Minor)** **`day_of_week` / `month_sin` / `month_cos`** — usually small lift for single-PA hit props unless very large data and careful regularization.

---

### Three tweaks that tend to help

**1. Put lineup / opportunity inside the model**  
You already have **`lineup_context`** on matchup JSON and **`batter_lineup_spot_last10_vs_hand.parquet`** ([`refresh_batter_lineup_context.py`](src/refresh_batter_lineup_context.py)). Add **`median_slot`**, **`mode_slot`**, **`n_games`** (and later **`expected_pa_vs_starter`** if modeled) into **training parquet**, **`feature_columns.json`**, and **`build_feature_vector`**. Hit markets are **P(hit) × chances**; ignoring slot biases rankings toward players who get fewer ABs.

**2. Simplify platoon + hit posterior**  
Either **drop overlapping platoon columns** in training **or** replace the hand-tuned Beta-style **`p_hit`** blend with **one calibration layer** (e.g. isotonic or ridge on residuals) fit on **tracking** outcomes, so **`p_hit`** is not forced twice from the same information.

**3. Add explicit pitch-mix × batter skill interactions**  
Low-dimensional products, winsorized — e.g. **`bpt_ba_vs_fastball * p_pct_fastball`**, same for **breaking** / **offspeed**. Stabilizes the core matchup vs depthwise splits on noisy median features.

---

### Not in the model today (notable gaps)

| Gap | Why it matters |
|-----|----------------|
| **PA expectation vs this starter** | Lineup slot + short hooks / bullpen games change how much **volume** a prop gets. |
| **Conditional “damage × pitch type thrown”** | Mix % exists; elite setups use **usage × batter damage × pitcher quality on that pitch shape**. |
| **Real game state at first PA** | **`times_thru_order` / `pitch_count`** are largely **[`IN_GAME_DEFAULTS`](src/narrative_engine.py)** at inference, not live script. |
| **Weather / contact-friendly park for non-HR** | HR gets more park/weather elsewhere; **BABIP/contact** can still move with conditions. |
| **Catcher framing / umpire** | Shifts called strikes and marginal contact. |
| **Batter travel / fatigue / altitude** | Second-order but standard in org models. |

---

## Part B — Full implementation plan

### Phase 1 — Lineup features (fast win)

**Goal:** GBDT sees opportunity proxy; rankings align with AB expectation.

| Step | Action |
|------|--------|
| 1.1 | Define stable column names: `lineup_median_slot_vs_hand`, `lineup_mode_slot_vs_hand`, `lineup_n_games_vs_hand` (values depend on PA’s `vs_lhp` / opposing starter hand). |
| 1.2 | **Training:** Join [`batter_lineup_spot_last10_vs_hand.parquet`](data/raw/batter_lineup_spot_last10_vs_hand.parquet) onto league PA table by `(batter, game_date)` using **as-of** logic (only games strictly before that PA’s game). Align `opp_hand` with pitcher throw side for that PA. Document join in [`features.py`](src/features.py) or a small `src/join_lineup_features.py` used from league build CLI. |
| 1.3 | **Inference:** In [`build_feature_vector`](src/narrative_engine.py), fill the three columns from [`load_lineup_slot_lookup`](src/lineup_context.py) using `(batter_id, tonight_hand)`; impute league-neutral defaults when missing (`n_games=0`, median/mode NaN → sentinel). |
| 1.4 | Regenerate **`features_*_beast.parquet`**, extend [`data/master/models/exp_beast/feature_columns.json`](data/master/models/exp_beast/feature_columns.json), retrain **`best_model_is_hit.pkl`** (and full Beast bundle if policy is joint refresh). |
| 1.5 | Validate: leaderboard drift vs prior; calibration bucket report [`monitor_calibration_drift.py`](src/monitor_calibration_drift.py). |

**Dependencies:** Daily [`refresh_batter_lineup_context.py`](src/refresh_batter_lineup_context.py) must run before inference (already wired in [`run_dual_model_daily.py`](src/run_dual_model_daily.py)).

---

### Phase 2 — Pitch-mix × batter interactions

**Goal:** Explicit economic matchup terms; less reliance on median-row interactions.

| Step | Action |
|------|--------|
| 2.1 | In [`features.py`](src/features.py) (post pitcher merge), add winsorized products: `int_bpt_ba_fb_x_pct` = clip(`bpt_ba_vs_fastball * p_pct_fastball`), analogous for breaking/offspeed using `pct_breaking`, `pct_offspeed`. |
| 2.2 | Ensure inference row has same columns via `build_feature_vector`. |
| 2.3 | Retrain hit model; ablation without interactions on holdout slice. |

---

### Phase 3 — Platoon redundancy + hit posterior simplification

**Goal:** Single coherent use of vs-hand signal; less double-counting.

| Step | Action |
|------|--------|
| 3.1 | **Ablation study (offline):** Train hit model variants — (a) full platoon block, (b) drop `platoon_matched_*` / `platoon_ops_gap`, (c) drop `cum_career_*` kept only in decayed `platoon_eff_*`. Pick subset with best **out-of-time Brier / log-loss** on league val. |
| 3.2 | **Posterior:** Options — (i) **Skip `is_hit` update** inside `apply_vs_hand_and_bvp_posteriors` when engagement true but rely on calibration only; (ii) reduce `VS_HAND_PRIOR_PA_HIT` toward “calibration-only” after ablation; (iii) replace vs-hand Beta blend for **hit only** with **isotonic** on GBDT output fit from [`data/tracking/matchup_predictions_runs.parquet`](data/tracking/) (extend [`calibrate_predictions.py`](src/calibrate_predictions.py) or hit-specific module). |
| 3.3 | Document chosen policy in [`matchup_predictions_field_semantics`](src/narrative_engine.py) artifact. |

---

### Phase 4 — BvP double-use resolution

**Goal:** Head-to-head evidence informs **either** trees **or** posterior, not both at full strength.

| Step | Action |
|------|--------|
| 4.1 | **Option A:** Remove **`bvp_ba`**, **`log_bvp_pa`** (and redundant BvP rate cols) from **`feature_columns.json`** for hit; keep posterior BvP blend. |
| 4.2 | **Option B:** Zero out / skip **`BVP_PRIOR_PA_HIT`** step for `is_hit` when `bvp_pa_count > 0` (trees trusted); keep posterior only when `bvp_pa_count == 0`. |
| 4.3 | Compare calibration + tail leaderboard stability; pick one policy for prod. |

---

### Phase 5 — Rolling-feature consolidation

**Goal:** Fewer collinear columns; stabler median snapshots.

| Step | Action |
|------|--------|
| 5.1 | Compute pairwise correlation matrix on train sample for batter `roll_*` and pitcher `p_roll_*` hit-related families. |
| 5.2 | Drop one of each highly correlated pair (e.g. retain **`roll_est_woba_100`** + **`roll_ba_100`**, drop redundant 10/30 where r > 0.92) **or** replace group with **PCA / composite “contact_index”** (document weights). |
| 5.3 | Retrain; confirm Section 0 drift does not regress. |

---

### Phase 6 — Gap backlog (prioritized)

| Priority | Gap | Implementation sketch |
|----------|-----|------------------------|
| P1 | **PA vs starter** | Combine lineup slot with [`starter_run_expectancies`](src/starter_run_expectancy.py) or BF estimate → feature `expected_bf_vs_tonight_sp` or multiplicative weight on display scores only. |
| P2 | **Conditional damage × pitch type** | Features: `sum_g(bpt_ba_vs_g * p_pct_g)` for g in {fb, br, ofs}; optional pitcher quality-by-type from Statcast. |
| P3 | **Weather / contact park** | Pull stadium hourly conditions (reuse Section 15 weather fetch patterns); additive BABIP-style multiplier or bucket features — **hit-only** calibration. |
| P4 | **Game state** | When official lineup + first-pitch state available, override [`IN_GAME_DEFAULTS`](src/narrative_engine.py) for `times_thru_order` / `pitch_count` for **first PA vs SP** only. |
| P5 | **Framing / umpire** | Annual catcher + ump prior tables merged by `game_pk` if feed available. |
| P6 | **Travel / altitude** | Join team travel miles / altitude from schedule analytics CSV. |

---

## Part C — Suggested execution order

1. **Phase 1** (lineup) — user-visible ranking fix, low architectural risk.  
2. **Phase 2** (interactions) — cheap feature engineering + retrain.  
3. **Phase 4** (BvP) — quick A/B vs Phase 3 overlap.  
4. **Phase 3** (platoon + posterior) — requires careful calibration regression testing.  
5. **Phase 5** (rolls) — after stabilizing posteriors to avoid confounding.  
6. **Phase 6** — iterate by **P1 → P6**.

---

## Part D — Acceptance criteria (each phase)

- **Calibration:** No sustained **CRITICAL** bucket in [`section_0_drift`](data/reports/section_0_drift.md) after change; compare pre/post Brier on rolling tracking slice.  
- **Stability:** Top-25 Hit churn day-over-day on fixed slate &lt; agreed threshold (define numerically after baseline).  
- **Audit:** JSON rows retain **`posterior_audit`** / **`lineup_context`** consistency; semantics artifact updated.

---

*Last updated: mirrors chat critique + expanded implementation plan for repo use.*
