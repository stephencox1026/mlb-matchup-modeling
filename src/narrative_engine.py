"""
Narrative matchup engine: generates quantified insights for batter vs pitcher matchups.

For each matchup, produces:
- Model predictions (P(Hit), P(HR), P(K), P(BB), P(XBH))
- Narrative text with league avg comparison, career avg comparison,
  pitch-type matchup insights, exit velocity data, BvP history
- Confidence tier (Elite / Strong / Average / Below Average / Weak)
"""
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config import BVP_HR_BAYES_PRIOR_WEIGHT, RAW_DIR, MASTER_DIR, REPORTS_DIR
from lineup_context import load_lineup_slot_lookup
from features import platoon_decayed_rates_for_batter_year

# Default production bundle (override with MATCHUP_MODEL_DIR / MATCHUP_FEATURES_VAL).
MODEL_DIR = MASTER_DIR / "models"
VAL_FEATURES_LEAGUE_PATH = MASTER_DIR / "features_val_league.parquet"
TARGETS = ["is_hit", "is_hr", "is_strikeout", "is_walk", "is_xbh"]


def _resolved_model_dir(model_dir=None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    env = os.environ.get("MATCHUP_MODEL_DIR")
    return Path(env) if env else MODEL_DIR


def _resolved_val_features_path(val_features_path=None) -> Path:
    if val_features_path is not None:
        return Path(val_features_path)
    env = os.environ.get("MATCHUP_FEATURES_VAL")
    return Path(env) if env else VAL_FEATURES_LEAGUE_PATH

LEAGUE_AVG = {
    "hit": 0.222, "hr": 0.031, "strikeout": 0.222, "walk": 0.084, "xbh": 0.079,
}

# Bayesian vs-hand posteriors. Higher hit/XBH priors = trust GBDT anchor more vs platoon/YTD mass
# (reduces posterior dragging elite contact profiles when vs-hand book is cold).
# YTD boost trimmed + ramp/engagement tightened so thin YTD vs-hand samples move leaderboards less.
VS_HAND_PRIOR_PA_HIT = 45
VS_HAND_PRIOR_PA_HR = 90
VS_HAND_PRIOR_PA_XBH = 55
VS_HAND_YTD_BOOST = 1.45

# Engagement gate: a row only earns the YTD boost when the player has real evidence behind them.
# Rookies / call-ups (29 PA Hao-Yu Lee, 4 PA Will Wilson) failed to meet any of these and were
# dominating XBH rankings via the YTD lever alone — gate suppresses that.
VS_HAND_ENGAGE_CAREER_PA = 160
VS_HAND_ENGAGE_YTD_PA = 58
VS_HAND_ENGAGE_BVP_PA = 20
VS_HAND_YTD_BOOST_RAMP_PA = 60

# YTD HR shrinkage (Beta prior on HR rate) before adding to eff_hr.
# Raw ytd_hr on 10–20 PA is lethal for leaderboards (e.g. 2 HR / 16 PA vs RHP balloons HR posterior).
# Replace raw yhr with an effective HR *mass* post_rate × ypa where post_rate blends observed HR/PA
# toward league HR with pseudo-count HR_YTD_HR_BETA_PRIOR_PA.
HR_YTD_HR_BETA_PRIOR_PA = 175

# BvP HR: shrink observed HR count toward league when matchup PA is tiny (same pathology as YTD).
BVP_HR_BETA_PRIOR_PA = 220

# When YTD vs-hand PA is still thin, cap how far platoon+BvP can lift HR above the GBDT anchor.
# Otherwise decay-career HR vs same-handed pitching plus a lucky micro-sample BvP/YTD stack
# pins middling bats #1 on the slate (JJ Bleday archetype).
HR_THIN_YTD_VS_HAND_PA = 85
HR_THIN_YTD_MAX_LIFT_OVER_GBDT = 0.008

# Career HR/PA ceiling: an established hitter's `p_hr` cannot exceed K * career_hr_per_pa.
# Stops mid-tier power bats (Bleday archetype) from getting elevated to top-10 HR ranks
# when the GBDT or platoon stack inflates them past their actual career HR rate.
# Only engages when the hitter has enough career PA for the rate to be reliable.
HR_CAREER_CAP_MULTIPLIER = 1.30
HR_CAREER_CAP_MIN_PA = 500

# Calendar-year HR gate: batters with PA logged this season but zero HR get HR prob floored
# (leaderboard hygiene; GBDT can still love pitcher/contact vs tiny-sample counting stats).
HR_ZERO_SEASON_GATE_MIN_STATCAST_ROWS = 1
HR_ZERO_SEASON_GATE_FLOOR = 1e-4

# BvP posteriors on raw P (symmetric; replaces asymmetric conf factors).
BVP_PRIOR_PA_HIT = 80
BVP_PRIOR_PA_HR = 300
BVP_PRIOR_PA_XBH = 120

# Pre-roadmap Beast: keep vs-hand + BvP Beta blends on ``is_hit`` (even though GBDT also sees platoon/BvP).
VS_HAND_SKIP_HIT_POSTERIOR_WHEN_ENGAGED = False
BVP_SKIP_HIT_BLEND_WHEN_BVP_PA_POSITIVE = False


def season_ytd_hr_row_counts(pa_data: pd.DataFrame, slate_date) -> dict[int, tuple[int, int]]:
    """
    For the slate's calendar year: Statcast rows strictly before ``slate_date``.
    Returns ``batter_id -> (n_rows, sum(is_hr))`` using the same row-grain as ``pa_data``.

    ``n_rows`` matches ``career_pa_totals`` / ``groupby("batter").size()`` semantics.
    """
    out: dict[int, tuple[int, int]] = {}
    if pa_data is None or slate_date is None or getattr(pa_data, "empty", True):
        return out
    if "game_date" not in pa_data.columns or "is_hr" not in pa_data.columns:
        return out
    asof = pd.Timestamp(slate_date)
    year = int(asof.year)
    m = (pa_data["game_date"] < asof) & (pa_data["game_date"].dt.year == year)
    sub = pa_data.loc[m]
    if sub.empty:
        return out
    for bid, gr in sub.groupby("batter"):
        out[int(bid)] = (int(len(gr)), int(gr["is_hr"].sum()))
    return out


def compute_ytd_platoon_vs_hand(
    batter_id: int,
    pa_data: pd.DataFrame,
    asof_date,
    vs_lhp_tonight: int,
) -> dict:
    """YTD Statcast counts vs pitcher hand strictly before ``asof_date`` (same calendar year)."""
    out = {"ytd_pa": 0, "ytd_h": 0, "ytd_hr": 0, "ytd_xbh": 0, "ytd_ab": 0}
    if pa_data is None or asof_date is None or "game_date" not in pa_data.columns:
        return out
    asof = pd.Timestamp(asof_date)
    m = (
        (pa_data["batter"] == batter_id)
        & (pa_data["game_date"] < asof)
        & (pa_data["game_date"].dt.year == asof.year)
    )
    throws = pa_data.get("p_throws")
    if throws is None:
        return out
    if int(vs_lhp_tonight):
        m &= throws == "L"
    else:
        m &= throws == "R"
    sub = pa_data.loc[m]
    if sub.empty:
        return out
    ev = sub["events"].fillna("").str.lower()
    hits = int(ev.isin({"single", "double", "triple", "home_run"}).sum())
    hr = int((ev == "home_run").sum())
    xbh = int(ev.isin({"double", "triple", "home_run"}).sum())
    ab = int((~ev.isin({"walk", "intent_walk", "hit_by_pitch", "catcher_interf", "sac_bunt"})).sum())
    return {
        "ytd_pa": int(len(sub)),
        "ytd_h": hits,
        "ytd_hr": hr,
        "ytd_xbh": xbh,
        "ytd_ab": ab,
    }


def apply_vs_hand_and_bvp_posteriors(
    predictions: dict,
    feature_snapshot: dict | None,
    pitcher_throws: str,
    ytd_platoon: dict | None,
    bvp_row: dict | None,
    total_career_pa: int = 0,
) -> tuple[dict, dict]:
    """Blend GBDT raw P with decay-weighted career + boosted YTD vs-hand, then BvP counts.

    Engagement gate: only applies vs-hand posterior when the batter has real evidence:
        career_pa >= VS_HAND_ENGAGE_CAREER_PA OR ytd_pa >= VS_HAND_ENGAGE_YTD_PA OR bvp_pa >= VS_HAND_ENGAGE_BVP_PA.
    Below that, the YTD boost is suppressed (rookies stop ranking #1 on tiny samples).
    YTD boost ramps from 1.0× → VS_HAND_YTD_BOOST as ytd_pa goes 0 → VS_HAND_YTD_BOOST_RAMP_PA.

    Export diagnostics on each matchup row under ``posterior_audit`` (JSON): engagement_gate,
    ytd_boost_applied, ytd_pa, vs_hand_eff_h/ab/pa, career_vs_hand_*, gbdt_hr_anchor, caps, BvP fields.
    After inference, ``platoon_raw_shrink`` holds platoon_raw_probability_shrink multipliers/reason.
    """
    pred = dict(predictions)
    audit: dict = {}
    vs_L = str(pitcher_throws or "R").upper().startswith("L")
    suff = "lhp" if vs_L else "rhp"
    snap = feature_snapshot or {}
    ytd = ytd_platoon or {}

    eh = float(snap.get(f"platoon_eff_h_{suff}", 0) or 0)
    eab = float(snap.get(f"platoon_eff_ab_{suff}", 0) or 0)
    epa = float(snap.get(f"platoon_eff_pa_{suff}", 0) or 0)
    ehr0 = float(snap.get(f"platoon_eff_hr_{suff}", 0) or 0)
    exbh0 = float(snap.get(f"platoon_eff_xbh_{suff}", 0) or 0)

    ypa = float(ytd.get("ytd_pa", 0) or 0)
    yh = float(ytd.get("ytd_h", 0) or 0)
    yhr = float(ytd.get("ytd_hr", 0) or 0)
    yxbh = float(ytd.get("ytd_xbh", 0) or 0)
    yab = float(ytd.get("ytd_ab", 0) or 0)

    bvp_pa_count = int(bvp_row.get("bvp_pa", 0) or 0) if bvp_row else 0

    # Engagement gate — must clear ONE of the three thresholds for the vs-hand posterior to fire.
    # Below the gate, vs-hand evidence is too thin to update the prior reliably (rookies on 15-PA
    # samples can otherwise inflate league-avg seeds by tens of percentage points). BvP still
    # applies — direct head-to-head evidence is informative even on tiny career samples.
    engaged = bool(
        (int(total_career_pa) >= VS_HAND_ENGAGE_CAREER_PA)
        or (ypa >= VS_HAND_ENGAGE_YTD_PA)
        or (bvp_pa_count >= VS_HAND_ENGAGE_BVP_PA)
    )

    # PA-weighted YTD boost ramp: 1.0× at ypa=0 → VS_HAND_YTD_BOOST at ypa>=VS_HAND_YTD_BOOST_RAMP_PA.
    if engaged and VS_HAND_YTD_BOOST_RAMP_PA > 0:
        ramp = min(ypa / float(VS_HAND_YTD_BOOST_RAMP_PA), 1.0)
        ytd_boost = 1.0 + (VS_HAND_YTD_BOOST - 1.0) * ramp
    else:
        ytd_boost = 0.0

    eff_h = eh + ytd_boost * yh
    eff_ab = eab + ytd_boost * yab
    eff_pa = epa + ytd_boost * ypa
    # Shrink YTD HR toward league before mixing — stops 2-HR-in-16-PA from dominating HR rankings.
    if ypa > 1e-6:
        ah = float(LEAGUE_AVG["hr"]) * float(HR_YTD_HR_BETA_PRIOR_PA)
        bh = float(1.0 - LEAGUE_AVG["hr"]) * float(HR_YTD_HR_BETA_PRIOR_PA)
        post_hr_rate = float(yhr + ah) / float(ypa + ah + bh)
        yhr_effective = post_hr_rate * ypa
    else:
        yhr_effective = 0.0
    eff_hr = ehr0 + ytd_boost * yhr_effective
    eff_xbh = exbh0 + ytd_boost * yxbh

    p_hit = float(pred.get("is_hit", LEAGUE_AVG["hit"]))
    p_hr = float(pred.get("is_hr", LEAGUE_AVG["hr"]))
    p_xbh = float(pred.get("is_xbh", LEAGUE_AVG["xbh"]))

    if engaged:
        den_hit = VS_HAND_PRIOR_PA_HIT + (eff_ab if eff_ab > 1e-6 else eff_pa)
        if den_hit > 1e-6 and not VS_HAND_SKIP_HIT_POSTERIOR_WHEN_ENGAGED:
            pred["is_hit"] = float(np.clip(
                (VS_HAND_PRIOR_PA_HIT * p_hit + eff_h) / den_hit, 1e-4, 1.0 - 1e-4))
        elif VS_HAND_SKIP_HIT_POSTERIOR_WHEN_ENGAGED:
            audit["vs_hand_hit_posterior_skipped_engaged"] = 1

        den_hr = VS_HAND_PRIOR_PA_HR + eff_pa
        if den_hr > 1e-6:
            pred["is_hr"] = float(np.clip(
                (VS_HAND_PRIOR_PA_HR * p_hr + eff_hr) / den_hr, 1e-4, 1.0 - 1e-4))

        den_xbh = VS_HAND_PRIOR_PA_XBH + eff_pa
        if den_xbh > 1e-6:
            pred["is_xbh"] = float(np.clip(
                (VS_HAND_PRIOR_PA_XBH * p_xbh + eff_xbh) / den_xbh, 1e-4, 1.0 - 1e-4))

    audit["vs_hand_eff_h"] = round(eff_h, 4)
    audit["vs_hand_eff_ab"] = round(eff_ab, 4)
    audit["vs_hand_eff_pa"] = round(eff_pa, 4)
    audit["ytd_pa"] = int(ypa)
    audit["ytd_h"] = int(yh)
    audit["ytd_xbh"] = int(yxbh)
    audit["ytd_ab"] = int(yab)
    audit["ytd_hr_raw"] = int(yhr)
    audit["ytd_hr_effective"] = round(float(yhr_effective), 4)
    audit["total_career_pa"] = int(total_career_pa)
    audit["engagement_gate"] = engaged
    audit["ytd_boost_applied"] = round(ytd_boost, 3)
    # Dashboard Beast tables: show splits vs tonight's pitcher hand (LHP/RHP).
    audit["platoon_pitch_hand_label"] = "LHP" if vs_L else "RHP"
    _pa_eff_side = float(snap.get(f"platoon_eff_pa_{suff}", 0) or 0)
    _xbh_eff_side = float(snap.get(f"platoon_eff_xbh_{suff}", 0) or 0)
    audit["career_vs_hand_pa_eff"] = round(float(snap.get(f"cum_career_pa_vs_{suff}", 0) or 0), 2)
    audit["career_vs_hand_ba"] = round(float(snap.get(f"cum_career_ba_vs_{suff}", 0) or 0), 4)
    audit["career_vs_hand_ops"] = round(float(snap.get(f"cum_career_ops_vs_{suff}", 0) or 0), 4)
    audit["career_vs_hand_hr_rate"] = round(float(snap.get(f"cum_career_hr_vs_{suff}", 0) or 0), 5)
    audit["career_vs_hand_xbh_rate"] = (
        round(float(_xbh_eff_side / _pa_eff_side), 5) if _pa_eff_side > 1e-6 else 0.0
    )

    if bvp_row:
        pa = bvp_pa_count
        if pa > 0:
            hits = int(bvp_row.get("bvp_hits", 0) or 0)
            hr = int(bvp_row.get("bvp_hr", 0) or 0)
            xbh = int(bvp_row.get("bvp_xbh", 0) or 0)
            ph = float(pred["is_hit"])
            phr = float(pred["is_hr"])
            pxb = float(pred["is_xbh"])
            if not BVP_SKIP_HIT_BLEND_WHEN_BVP_PA_POSITIVE:
                pred["is_hit"] = float(np.clip(
                    (BVP_PRIOR_PA_HIT * ph + hits) / (BVP_PRIOR_PA_HIT + pa), 1e-4, 1.0 - 1e-4))
            else:
                audit["bvp_hit_blend_skipped_trees_have_bvp"] = 1
            # Shrink BvP HR observation toward league when PA count is low (1 HR / 8 PA vs Keller).
            ah_b = float(LEAGUE_AVG["hr"]) * float(BVP_HR_BETA_PRIOR_PA)
            bh_b = float(1.0 - LEAGUE_AVG["hr"]) * float(BVP_HR_BETA_PRIOR_PA)
            hr_post_rate = float(hr + ah_b) / float(float(pa) + ah_b + bh_b)
            hr_effective = hr_post_rate * float(pa)
            audit["bvp_hr_raw"] = hr
            audit["bvp_hr_effective"] = round(hr_effective, 4)
            pred["is_hr"] = float(np.clip(
                (BVP_PRIOR_PA_HR * phr + hr_effective) / (BVP_PRIOR_PA_HR + pa),
                1e-4, 1.0 - 1e-4))
            pred["is_xbh"] = float(np.clip(
                (BVP_PRIOR_PA_XBH * pxb + xbh) / (BVP_PRIOR_PA_XBH + pa), 1e-4, 1.0 - 1e-4))
            audit["bvp_pa"] = pa

    return pred, audit


def matchup_predictions_field_semantics() -> dict:
    """Explain exported matchup JSON keys (evidence multiplier vs ranking vs calibrated P)."""
    return {
        "schema_version": 1,
        "summary": (
            "`p_hit` / `p_hr` / `p_xbh` are per-PA probabilities after vs-hand + BvP Beta–Binomial posterior blend "
            "on top of the GBDT; slate tables rank by these raw fields. "
            "`is_hit` receives vs-hand + BvP Beta updates whenever engaged / applicable even though GBDT also sees platoon/BvP features. "
            "`conf_*` is an evidence multiplier (not P(outcome)). "
            "`score_*` / `adj_p_*` (= raw × conf) is a display ranking score only — not the sort key for Sections 1–3. "
            "`p_*_calibrated` (Cal P) is isotonic on tracking."
        ),
        "fields": {
            "p_hr": "Raw per-PA P(HR); classifier output before confidence weighting.",
            "p_hit": "Raw per-PA P(hit).",
            "p_xbh": "Raw per-PA P(extra-base hit).",
            "conf_hr": "Evidence multiplier for HR ranking (pitcher data, BvP, staleness, graded convergence × optional platoon).",
            "conf_hit": "Evidence multiplier for hit ranking.",
            "conf_xbh": "Evidence multiplier for XBH; blends HR/hit composites and XBH convergence, capped by HR/hit composites.",
            "score_hr": "Ranking score = p_hr × conf_hr — display only; Sections 1–3 sort by raw p_hr / p_hit / p_xbh.",
            "score_hit": "Ranking score = p_hit × conf_hit.",
            "score_xbh": "Ranking score = p_xbh × conf_xbh.",
            "adj_p_hr": "Deprecated alias for score_hr (back-compat).",
            "adj_p_hit": "Deprecated alias for score_hit.",
            "adj_p_xbh": "Deprecated alias for score_xbh.",
            "p_hr_calibrated": "Isotonic-calibrated P(HR) (C1).",
            "p_hit_calibrated": "Isotonic-calibrated P(hit).",
            "p_xbh_calibrated": "Isotonic-calibrated P(XBH).",
            "conf_hr_label": "Display tier from composite conf_hr (High/Medium/Low/Very Low) — heuristic band, same scale as drift Section 0.",
            "conf_hit_label": "Display tier from composite conf_hit.",
            "conf_xbh_label": "Display tier from composite conf_xbh.",
            "conf_factors": "Structured breakdown of multiplier inputs (BvP, convergence_*, staleness, platoon, etc.).",
            "lineup_context": "Optional: recent batting-order spot vs opposing starter hand (last ≤10 games), from batter_lineup_spot_last10_vs_hand.parquet — median_slot, mode_slot, n_games, split.",
        },
    }


def write_matchup_predictions_semantics_artifact(report_dir: Path | str | None = None) -> Path:
    """Writes data/reports/matchup_predictions_field_semantics.json (non-breaking beside flat matchup lists)."""
    base = Path(report_dir) if report_dir is not None else REPORTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / "matchup_predictions_field_semantics.json"
    blob = matchup_predictions_field_semantics()
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


PA_VS_STARTER_DEFAULT = 3


def load_models(model_dir=None):
    mdir = _resolved_model_dir(model_dir)
    models = {}
    for target in TARGETS:
        path = mdir / f"best_model_{target}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[target] = pickle.load(f)
    return models


def load_feature_cols(model_dir=None):
    path = _resolved_model_dir(model_dir) / "feature_columns.json"
    with open(path) as f:
        return json.load(f)


def load_pitcher_profiles():
    path = RAW_DIR / "pitcher_profiles_by_season.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_batter_pitch_profiles():
    path = RAW_DIR / "batter_pitch_profiles.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_bvp_history():
    pa = pd.read_parquet(RAW_DIR / "statcast_pa_level_league.parquet")
    hit_events = {"single", "double", "triple", "home_run"}
    pa["is_hit"] = pa["events"].isin(hit_events).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_xbh"] = pa["events"].isin({"double", "triple", "home_run"}).astype(int)
    pa["is_k"] = pa["events"].str.contains("strikeout", case=False, na=False).astype(int)
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk", "hit_by_pitch"}).astype(int)
    bvp = pa.groupby(["batter", "pitcher"]).agg(
        bvp_pa=("events", "count"),
        bvp_hits=("is_hit", "sum"),
        bvp_hr=("is_hr", "sum"),
        bvp_xbh=("is_xbh", "sum"),
        bvp_k=("is_k", "sum"),
        bvp_bb=("is_bb", "sum"),
    ).reset_index()
    bvp["bvp_ba"] = (bvp["bvp_hits"] / bvp["bvp_pa"]).round(3)
    return bvp


def _bayesian_shrink(observed_rate, n_obs, prior_rate, prior_weight=20):
    return (observed_rate * n_obs + prior_rate * prior_weight) / (n_obs + prior_weight)


def compute_bvp_for_pair(batter_id, pitcher_id, pa_data):
    """Compute correct BvP features for a specific batter-pitcher pair from raw PA data."""
    pair = pa_data[(pa_data["batter"] == batter_id) & (pa_data["pitcher"] == pitcher_id)]
    if pair.empty:
        return {
            "bvp_pa_count": 0,
            "bvp_ba": 0.243,
            "bvp_k_rate": 0.222,
            "bvp_bb_rate": 0.084,
            "bvp_hr_count": 0,
            "bvp_hr_rate": 0.031,
            "bvp_xbh_rate": 0.079,
            "log_bvp_pa": 0.0,
            "bvp_has_history": 0,
        }

    hit_events = {"single", "double", "triple", "home_run"}
    bb_events = {"walk", "intent_walk", "hit_by_pitch"}
    n_pa = len(pair)
    n_hits = pair["events"].isin(hit_events).sum()
    n_hr = (pair["events"] == "home_run").sum()
    n_xbh = pair["events"].isin({"double", "triple", "home_run"}).sum()
    n_k = pair["events"].str.contains("strikeout", case=False, na=False).sum()
    n_bb = pair["events"].isin(bb_events).sum()
    n_ab = (~pair["events"].isin(bb_events | {"catcher_interf", "sac_bunt"})).sum()

    raw_ba = n_hits / n_ab if n_ab > 0 else 0.0
    raw_k = n_k / n_pa if n_pa > 0 else 0.0
    raw_bb = n_bb / n_pa if n_pa > 0 else 0.0
    raw_hr = n_hr / n_pa if n_pa > 0 else 0.0
    raw_xbh = n_xbh / n_pa if n_pa > 0 else 0.0

    return {
        "bvp_pa_count": int(n_pa),
        "bvp_ba": _bayesian_shrink(raw_ba, n_ab, 0.243),
        "bvp_k_rate": _bayesian_shrink(raw_k, n_pa, 0.222),
        "bvp_bb_rate": _bayesian_shrink(raw_bb, n_pa, 0.084),
        "bvp_hr_count": int(n_hr),
        "bvp_hr_rate": _bayesian_shrink(
            raw_hr, n_pa, 0.031, prior_weight=BVP_HR_BAYES_PRIOR_WEIGHT
        ),
        "bvp_xbh_rate": _bayesian_shrink(raw_xbh, n_pa, 0.079),
        "log_bvp_pa": float(np.log1p(n_pa)),
        "bvp_has_history": 1,
    }


IN_GAME_DEFAULTS = {
    "pitch_count": 4.0,
    "in_zone": 1.0,
    "times_thru_order": 1.0,
    "p_roll_k_10": 0.20,
    "p_roll_k_30": 0.22,
    "p_roll_bb_10": 0.08,
    "p_roll_bb_30": 0.08,
    "p_roll_hit_allowed_10": 0.22,
    "p_roll_hit_allowed_30": 0.22,
}

MIN_CAREER_PA = 200

# Matchup inference: align recent rolling stats & vs_lhp with tonight's pitcher hand.
PLATOON_RECENT_MIN_MATCHING = 6
PLATOON_RECENT_WINDOW_MATCH = 48
PLATOON_RECENT_FALLBACK_TAIL = 40

# Confidence fade when matchup-side cumulative platoon BA is dreadful with thin PA evidence.
PLATOON_FADE_MAX_BA = 0.205
PLATOON_FADE_MIN_PA = 120
PLATOON_FADE_MULT = 0.94
PLATOON_FADE_MAX_BA_STRICT = 0.235
PLATOON_FADE_MIN_PA_STRICT = 60
PLATOON_FADE_MULT_STRICT = 0.90

# Inference-time raw P shrink: prefer full-sample stability before penalizing tepid splits.
PLATOON_PENALTY_MIN_PA = 50

# Below PLATOON_PENALTY_MIN_PA, still apply shrink when multi-year cumulative futility vs this arm
# is already unmistakable (closes the "Moniak hole": ~25 PA @ .096 / .254 OPS still ranked on raw HR).
PLATOON_EXTREME_ELIGIBLE_MIN_PA = 15

# Applied after model logits → multiplicative on is_hit / is_xbh / is_hr when matched-side slash is hopeless.
PLATOON_SHRINK_BA_CATASTROPHIC = 0.120
PLATOON_SHRINK_MULT_HIT_CAT = 0.60
PLATOON_SHRINK_MULT_XBH_CAT = 0.50
PLATOON_SHRINK_MULT_HR_CAT = 0.40

PLATOON_SHRINK_BA_VERY_BAD = 0.165
PLATOON_SHRINK_MULT_HIT_VBAD = 0.75
PLATOON_SHRINK_MULT_XBH_VBAD = 0.68
PLATOON_SHRINK_MULT_HR_VBAD = 0.58

PLATOON_SHRINK_BA_POOR = 0.195
PLATOON_SHRINK_MULT_HIT_POOR = 0.88
PLATOON_SHRINK_MULT_XBH_POOR = 0.82
PLATOON_SHRINK_MULT_HR_POOR = 0.75

PLATOON_SHRINK_OPS_HARD = 0.400
PLATOON_SHRINK_OPS_HR_STACK = 0.88
PLATOON_SHRINK_OPS_XBH_STACK = 0.92

PLATOON_SHRINK_OPS_SOFT = 0.500
PLATOON_SHRINK_OPS_HR_SOFT = 0.94
PLATOON_SHRINK_OPS_XBH_SOFT = 0.96

# Confidence: sustained weak platoon (high PA), complementing shrink above.
PLATOON_CONF_HI_PA_LARGE = 200
PLATOON_CONF_HI_PA_MULT_CAT_BA = 0.72
PLATOON_CONF_HI_PA_MULT_VBAD_BA = 0.82
PLATOON_CONF_HI_PA_MULT_WEAK_BA = 0.90

PLATOON_CONF_MED_PA_MULT_CAT_BA = 0.78
PLATOON_CONF_MED_PA_MULT_VBAD_BA = 0.88

PLATOON_CONF_OPS_HARD = 0.400
PLATOON_CONF_MULT_OPS_HARD = 0.85
PLATOON_CONF_OPS_SOFT = 0.500
PLATOON_CONF_MULT_OPS_SOFT = 0.92

# Confidence composition: mild full-agreement bonus (legacy 1.10 crowded Medium/High); BvP capped at neutral.
CONF_CONVERGENCE_AGREE_MULT = 1.04
# Graded disagreement endpoint (agree_frac=0 → this mult; agree_frac=1 → CONF_CONVERGENCE_AGREE_MULT).
CONF_CONVERGENCE_DISAGREE_MULT = 0.85
CONF_BVP_FACTOR_CEILING = 1.0

# Per-target label cutoffs vs composite confidence multiplier (rarer outcomes need higher multiplier for High/Medium).
CONF_LABEL_HIT = (1.05, 0.88, 0.75)
CONF_LABEL_XBH = (1.085, 0.90, 0.775)
CONF_LABEL_HR = (1.115, 0.94, 0.805)

# Convergence gates: league-relative model P(*) and career rate vs LEAGUE_AVG.
CONF_CONV_MODEL_K_HR = 1.62
CONF_CONV_MODEL_K_HIT = 1.07
CONF_CONV_MODEL_K_XBH = 1.10
CONF_CONV_CAREER_MULT_HR = 1.40
CONF_CONV_CAREER_MULT_HIT = 1.045
CONF_CONV_CAREER_MULT_XBH = 1.07
CONF_CONV_BVP_PA_XBH_SIGNAL = 12


def graded_convergence_factor(agree: int, n_signals: int) -> float:
    """Map count of agreeing strength signals → multiplier in [DISAGREE, AGREE].

    Smooth in agree_frac; avoids tri-state cliffs at thresholds.
    """
    if n_signals <= 0:
        return 1.0
    frac = agree / float(n_signals)
    lo = CONF_CONVERGENCE_DISAGREE_MULT
    hi = CONF_CONVERGENCE_AGREE_MULT
    return float(lo + (hi - lo) * frac)


N_SIMS = 50_000


def _snapshot_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _platoon_matched_splits(
    feature_snapshot: dict | None,
    pitcher_throws: str,
) -> tuple[float | None, float | None, float | None]:
    """BA / PA / OPS for the batter vs tonight's pitcher hand (from val row snapshot).

    Prefer ``platoon_matched_*``; fall back to ``cum_career_*_vs_lhp`` / ``_vs_rhp``.
    Returns (None, None, None) when snapshot missing, PA invalid, or BA unavailable for that arm.
    """
    if not feature_snapshot:
        return None, None, None
    vs_left = str(pitcher_throws or "R").upper().startswith("L")

    ba_pb = _snapshot_float(feature_snapshot.get("platoon_matched_ba"))
    ops_pb = _snapshot_float(feature_snapshot.get("platoon_matched_ops"))

    ba_cum = _snapshot_float(
        feature_snapshot.get("cum_career_ba_vs_lhp" if vs_left else "cum_career_ba_vs_rhp")
    )
    pa_cum = _snapshot_float(
        feature_snapshot.get("cum_career_pa_vs_lhp" if vs_left else "cum_career_pa_vs_rhp")
    )
    ops_cum = _snapshot_float(
        feature_snapshot.get("cum_career_ops_vs_lhp" if vs_left else "cum_career_ops_vs_rhp")
    )

    ba_side = ba_pb if ba_pb is not None else ba_cum
    pa_side = pa_cum
    ops_side = ops_pb if ops_pb is not None else ops_cum

    if pa_side is None or pa_side <= 0:
        return None, None, None
    if ba_side is None:
        ba_side = ba_cum
    if ba_side is None:
        return None, None, None

    return ba_side, pa_side, ops_side


def _platoon_extreme_weak_side_for_raw_shrink(
    ba: float,
    pa: float,
    ops: float | None,
) -> bool:
    """True when cumulative PA vs tonight's pitcher hand is thin but slash is disastrous.

    Keeps PA < PLATOON_EXTREME_ELIGIBLE_MIN_PA as mostly model-only (tiny sample noise),
    but between EXTREME and PLATOON_PENALTY_MIN_PA we enforce shrink on very bad slash.
    """
    if pa < float(PLATOON_EXTREME_ELIGIBLE_MIN_PA) or pa >= float(PLATOON_PENALTY_MIN_PA):
        return False
    if ba < PLATOON_SHRINK_BA_VERY_BAD:
        return True
    if ops is not None and ops < PLATOON_SHRINK_OPS_HARD:
        return True
    return False


def platoon_raw_probability_shrink(
    predictions: dict,
    feature_snapshot: dict | None,
    pitcher_throws: str,
) -> tuple[dict, dict]:
    """Down-weight raw hit / HR / XBH probabilities when cumulative platoon slash proves futile.

    Runs **after** model ``predict_proba`` and **before** HR⊂XBH⊂Hit clamp + narratives.
    See ``PLATOON_PENALTY_MIN_PA``, ``PLATOON_EXTREME_ELIGIBLE_MIN_PA``, and ``PLATOON_SHRINK_*`` tiers.
    """
    out = dict(predictions)
    audit = {
        "platoon_raw_hit_mult": 1.0,
        "platoon_raw_xbh_mult": 1.0,
        "platoon_raw_hr_mult": 1.0,
        "platoon_raw_reason": "none",
    }
    ba, pa, ops = _platoon_matched_splits(feature_snapshot, pitcher_throws)

    if ba is None or pa is None:
        return out, audit

    platoon_measured = pa >= PLATOON_PENALTY_MIN_PA or _platoon_extreme_weak_side_for_raw_shrink(
        ba, pa, ops
    )
    if not platoon_measured:
        return out, audit

    mh = mx = mhr = 1.0
    tags: list[str] = []

    if ba < PLATOON_SHRINK_BA_CATASTROPHIC:
        mh = PLATOON_SHRINK_MULT_HIT_CAT
        mx = PLATOON_SHRINK_MULT_XBH_CAT
        mhr = PLATOON_SHRINK_MULT_HR_CAT
        tags.append("ba_catastrophic")
    elif ba < PLATOON_SHRINK_BA_VERY_BAD:
        mh = PLATOON_SHRINK_MULT_HIT_VBAD
        mx = PLATOON_SHRINK_MULT_XBH_VBAD
        mhr = PLATOON_SHRINK_MULT_HR_VBAD
        tags.append("ba_very_bad")
    elif ba < PLATOON_SHRINK_BA_POOR:
        mh = PLATOON_SHRINK_MULT_HIT_POOR
        mx = PLATOON_SHRINK_MULT_XBH_POOR
        mhr = PLATOON_SHRINK_MULT_HR_POOR
        tags.append("ba_poor")

    if ops is not None:
        if ops < PLATOON_SHRINK_OPS_HARD:
            mhr *= PLATOON_SHRINK_OPS_HR_STACK
            mx *= PLATOON_SHRINK_OPS_XBH_STACK
            tags.append("ops_hard")
        elif ops < PLATOON_SHRINK_OPS_SOFT:
            mhr *= PLATOON_SHRINK_OPS_HR_SOFT
            mx *= PLATOON_SHRINK_OPS_XBH_SOFT
            tags.append("ops_soft")

    if not tags:
        return out, audit

    if "is_hit" in out:
        out["is_hit"] = float(np.clip(out["is_hit"] * mh, 1e-4, 1.0 - 1e-4))
    if "is_xbh" in out:
        out["is_xbh"] = float(np.clip(out["is_xbh"] * mx, 1e-4, 1.0 - 1e-4))
    if "is_hr" in out:
        out["is_hr"] = float(np.clip(out["is_hr"] * mhr, 1e-4, 1.0 - 1e-4))

    audit["platoon_raw_hit_mult"] = round(mh, 4)
    audit["platoon_raw_xbh_mult"] = round(mx, 4)
    audit["platoon_raw_hr_mult"] = round(mhr, 4)
    audit["platoon_raw_reason"] = "|".join(tags)
    return out, audit


def _enforce_hit_xbh_hr_order(predictions: dict) -> None:
    """In-place HR ⊂ XBH ⊂ Hit."""
    if "is_hr" in predictions and "is_xbh" in predictions:
        if predictions["is_hr"] > predictions["is_xbh"]:
            predictions["is_xbh"] = predictions["is_hr"]
    if "is_xbh" in predictions and "is_hit" in predictions:
        if predictions["is_xbh"] > predictions["is_hit"]:
            predictions["is_hit"] = predictions["is_xbh"]


def simulate_multi_hit(p_hit, p_k, p_bb, n_pa=3, n_sims=N_SIMS):
    """Multi-outcome PA simulation for multi-hit probability.

    Each PA is resolved via a multinomial draw over four disjoint outcomes:
    Hit, Strikeout, Walk, and Other Out.  This means a high-K batter genuinely
    has fewer "live" PAs available for hits, unlike the old pure-binomial
    approach where P(Multi-Hit) was a monotonic function of P(Hit).

    Returns (p_multi_hit, p_any_hit).
    """
    total = p_hit + p_k + p_bb
    if total > 1.0:
        scale = 1.0 / total
        p_hit, p_k, p_bb = p_hit * scale, p_k * scale, p_bb * scale
    p_other = 1.0 - p_hit - p_k - p_bb

    rng = np.random.default_rng(42)
    outcomes = rng.multinomial(n_pa, [p_hit, p_k, p_bb, p_other], size=n_sims)
    hits = outcomes[:, 0]
    return float((hits >= 2).mean()), float((hits >= 1).mean())


BPT_REQUIRED_FEATURES = [
    "bpt_ev_vs_fastball", "bpt_ev_vs_breaking", "bpt_ev_vs_offspeed",
]

# Map model BPT feature → batter_pitch_profiles raw column (for breakout/rookie fallback).
_BPT_TO_PROFILE_COL = {
    "bpt_ev_vs_fastball": "ev_vs_fastball",
    "bpt_ev_vs_breaking": "ev_vs_breaking",
    "bpt_ev_vs_offspeed": "ev_vs_offspeed",
}

# League-average exit velocity by pitch group (mph). Final fallback when the
# batter has no row in batter_pitch_profiles.parquet either (true rookies / late
# call-ups). Tuned from 2025 league bulk EV by pitch group.
_BPT_LEAGUE_DEFAULTS = {
    "bpt_ev_vs_fastball": 89.0,
    "bpt_ev_vs_breaking": 86.0,
    "bpt_ev_vs_offspeed": 85.0,
}


def _bpt_fallback_value(
    bpt_col: str,
    batter_pitch_profiles: pd.DataFrame,
    batter_id: int,
) -> tuple[float, str]:
    """Lookup bpt_* fallback: latest-season batter_pitch_profiles row → league default."""
    profile_col = _BPT_TO_PROFILE_COL.get(bpt_col)
    if profile_col and not batter_pitch_profiles.empty:
        rows = batter_pitch_profiles[batter_pitch_profiles["batter"] == batter_id]
        if not rows.empty and profile_col in rows.columns:
            latest = rows.sort_values("season").iloc[-1]
            v = latest.get(profile_col)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            if pd.notna(fv) and fv > 0:
                return fv, "profile"
    return _BPT_LEAGUE_DEFAULTS.get(bpt_col, 87.0), "league_default"

ROLLING_DAMPEN_PAIRS = [
    ("roll_ba_10", "roll_ba_100", 0.50),
    ("roll_ba_30", "roll_ba_100", 0.30),
    ("roll_k_rate_10", "roll_k_rate_100", 0.50),
    ("roll_k_rate_30", "roll_k_rate_100", 0.30),
    ("roll_ev_10", "roll_ev_100", 0.50),
    ("roll_ev_30", "roll_ev_100", 0.30),
    ("roll_bb_rate_10", "roll_bb_rate_100", 0.50),
    ("roll_bb_rate_30", "roll_bb_rate_100", 0.30),
    ("roll_hr_rate_10", "roll_hr_rate_100", 0.50),
    ("roll_hr_rate_30", "roll_hr_rate_100", 0.30),
    ("roll_est_woba_10", "roll_est_woba_100", 0.50),
    ("roll_est_woba_30", "roll_est_woba_100", 0.30),
]


def resolve_pitcher_throws(pitcher_id, pitcher_profiles, pitcher_throws_hint=None):
    """Return 'L' or 'R' for tonight's starter (profile first, optional schedule hint)."""
    if pitcher_throws_hint is not None:
        h = str(pitcher_throws_hint).strip().upper()
        if h.startswith("L"):
            return "L"
        if h.startswith("R"):
            return "R"
    pp = pitcher_profiles[pitcher_profiles["pitcher"] == pitcher_id]
    if pp.empty:
        return "R"
    t = pp.sort_values("season").iloc[-1].get("throws")
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return "R"
    s = str(t).strip().upper()
    return "L" if s.startswith("L") else "R"


def _median_recent_features_aligned(
    batter_rows: pd.DataFrame,
    feat_cols: list,
    vs_lhp_tonight: int,
) -> pd.Series:
    """Median of batter features prioritizing recent PAs vs same pitcher hand."""
    cols_present = [c for c in feat_cols if c in batter_rows.columns]
    if not cols_present:
        return pd.Series(index=feat_cols, dtype=float)

    br = batter_rows.sort_values("game_date")
    if "vs_lhp" not in br.columns:
        med = br.tail(PLATOON_RECENT_FALLBACK_TAIL)[cols_present].median()
        return med.reindex(feat_cols)

    mask = br["vs_lhp"] == vs_lhp_tonight
    matching = br.loc[mask]
    if len(matching) >= PLATOON_RECENT_MIN_MATCHING:
        window = matching.tail(PLATOON_RECENT_WINDOW_MATCH)
    else:
        window = br.tail(PLATOON_RECENT_FALLBACK_TAIL)
    med = window[cols_present].median()
    return med.reindex(feat_cols)


def platoon_quality_multiplier(feature_snapshot: dict | None,
                               pitcher_throws: str) -> tuple[float, dict]:
    """Post-model confidence multiplier from cumulative platoon slash vs tonight's arm.

    Combines legacy thin-sample fades with **high-PA sustained weak** platoon tiers
    and optional OPS gates (aligned with ``platoon_raw_probability_shrink``).
    """
    detail = {"platoon_matchup_quality": 1.0, "platoon_quality_reason": "none"}
    ba, pa, ops = _platoon_matched_splits(feature_snapshot, pitcher_throws)

    if ba is None or pa is None or pa <= 0:
        return 1.0, detail

    mult = 1.0
    tags: list[str] = []

    # Thin PA + weak BA (legacy fades)
    if pa < PLATOON_FADE_MIN_PA_STRICT and ba < PLATOON_FADE_MAX_BA_STRICT:
        mult = min(mult, PLATOON_FADE_MULT_STRICT)
        tags.append("fade_strict")
    elif pa < PLATOON_FADE_MIN_PA and ba < PLATOON_FADE_MAX_BA:
        mult = min(mult, PLATOON_FADE_MULT)
        tags.append("fade_soft")

    # High PA sustained weak platoon — confidence only (skipped on confidence when raw shrink already hit P).
    if pa >= PLATOON_CONF_HI_PA_LARGE:
        if ba < PLATOON_SHRINK_BA_CATASTROPHIC:
            mult = min(mult, PLATOON_CONF_HI_PA_MULT_CAT_BA)
            tags.append("hi_pa_cat_ba")
        elif ba < PLATOON_SHRINK_BA_VERY_BAD:
            mult = min(mult, PLATOON_CONF_HI_PA_MULT_VBAD_BA)
            tags.append("hi_pa_vbad_ba")
        elif ba < PLATOON_SHRINK_BA_POOR:
            mult = min(mult, PLATOON_CONF_HI_PA_MULT_WEAK_BA)
            tags.append("hi_pa_weak_ba")
    elif pa >= PLATOON_PENALTY_MIN_PA:
        if ba < PLATOON_SHRINK_BA_CATASTROPHIC:
            mult = min(mult, PLATOON_CONF_MED_PA_MULT_CAT_BA)
            tags.append("med_pa_cat_ba")
        elif ba < PLATOON_SHRINK_BA_VERY_BAD and pa >= PLATOON_FADE_MIN_PA_STRICT:
            mult = min(mult, PLATOON_CONF_MED_PA_MULT_VBAD_BA)
            tags.append("med_pa_vbad_ba")

    if ops is not None and pa >= PLATOON_PENALTY_MIN_PA:
        if ops < PLATOON_CONF_OPS_HARD:
            mult = min(mult, PLATOON_CONF_MULT_OPS_HARD)
            tags.append("conf_low_ops_hard")
        elif ops < PLATOON_CONF_OPS_SOFT:
            mult = min(mult, PLATOON_CONF_MULT_OPS_SOFT)
            tags.append("conf_low_ops_soft")

    if tags:
        detail["platoon_matchup_quality"] = mult
        detail["platoon_quality_reason"] = "|".join(tags)
    return mult, detail


def _dampen_rolling_features(feature_vec, feat_cols):
    """Shrink noisy short-window rolling stats toward the 100-PA window.

    Matchup inference medians prioritize recent PAs **vs the same pitcher
    handedness as tonight** (see PLATOON_RECENT_* constants), then this step
    shrinks short windows toward the longer 100-PA roll for stability.
    """
    for short_col, long_col, blend_weight in ROLLING_DAMPEN_PAIRS:
        if short_col in feat_cols and long_col in feat_cols:
            short_val = feature_vec.get(short_col, 0)
            long_val = feature_vec.get(long_col, 0)
            if pd.notna(short_val) and pd.notna(long_val):
                feature_vec[short_col] = (
                    (1 - blend_weight) * short_val + blend_weight * long_val
                )


def build_feature_vector(
    batter_id,
    pitcher_id,
    feat_cols,
    val_df,
    pitcher_profiles,
    batter_pitch_profiles,
    pa_data=None,
    *,
    skip_rolling_dampen: bool = False,
    game_context: dict | None = None,
    pitcher_throws_hint: str | None = None,
):
    """Build a feature vector for a specific batter-pitcher matchup.

    Returns None (excluded from predictions) when:
    - Batter has fewer than MIN_CAREER_PA plate appearances in data
    - Batter is missing pitch-type profile data (zero EV vs any pitch group)

    Uses batter-centric features from recent PAs preferentially **vs the same
    pitcher handedness as tonight**, then overlays pitcher profile / BvP.
    ``vs_lhp`` is forced to match tonight's starter hand (``pitcher_throws_hint``
    if provided, else pitcher profile).
    """
    batter_rows = val_df[val_df["batter"] == batter_id]
    if batter_rows.empty:
        return None, {}

    if pa_data is not None:
        career_pa = int((pa_data["batter"] == batter_id).sum())
    else:
        career_pa = len(batter_rows)
    if career_pa < MIN_CAREER_PA:
        return None, {}

    throws_live = resolve_pitcher_throws(
        pitcher_id, pitcher_profiles, pitcher_throws_hint
    )
    vs_lhp_tonight_int = int(str(throws_live).upper().startswith("L"))

    feature_vec = _median_recent_features_aligned(batter_rows, feat_cols, vs_lhp_tonight_int)

    if not skip_rolling_dampen:
        _dampen_rolling_features(feature_vec, feat_cols)

    bpt_imputed_sources: dict[str, str] = {}
    for bpt_col in BPT_REQUIRED_FEATURES:
        if bpt_col in feat_cols:
            val = feature_vec.get(bpt_col, 0)
            if pd.isna(val) or val == 0:
                fb_val, fb_src = _bpt_fallback_value(bpt_col, batter_pitch_profiles, batter_id)
                feature_vec[bpt_col] = fb_val
                bpt_imputed_sources[bpt_col] = fb_src

    for col, default in IN_GAME_DEFAULTS.items():
        if col in feat_cols:
            feature_vec[col] = default

    if game_context:
        if (
            game_context.get("times_thru_order") is not None
            and "times_thru_order" in feat_cols
        ):
            try:
                feature_vec["times_thru_order"] = int(game_context["times_thru_order"])
            except (TypeError, ValueError):
                pass
        if game_context.get("pitch_count") is not None and "pitch_count" in feat_cols:
            try:
                feature_vec["pitch_count"] = float(game_context["pitch_count"])
            except (TypeError, ValueError):
                pass

    if game_context and game_context.get("game_date") is not None:
        gdt = pd.Timestamp(game_context.get("game_date"))
        if "month" in feat_cols:
            feature_vec["month"] = int(gdt.month)
        if "day_of_week" in feat_cols:
            feature_vec["day_of_week"] = int(gdt.dayofweek)
    else:
        today = pd.Timestamp.now()
        if "month" in feat_cols:
            feature_vec["month"] = int(today.month)
        if "day_of_week" in feat_cols:
            feature_vec["day_of_week"] = int(today.dayofweek)

    pp = pitcher_profiles[pitcher_profiles["pitcher"] == pitcher_id]
    if not pp.empty:
        latest = pp.sort_values("season").iloc[-1]
        pitcher_feat_map = {
            "p_pct_fastball": "pct_fastball", "p_pct_breaking": "pct_breaking",
            "p_pct_offspeed": "pct_offspeed", "p_velo_fastball": "velo_fastball",
            "p_velo_breaking": "velo_breaking", "p_velo_overall": "velo_overall",
            "p_spin_fastball": "spin_fastball", "p_spin_breaking": "spin_breaking",
            "p_pfx_x_fastball": "pfx_x_fastball", "p_pfx_z_fastball": "pfx_z_fastball",
            "p_pfx_x_breaking": "pfx_x_breaking", "p_pfx_z_breaking": "pfx_z_breaking",
            "p_pct_in_zone": "pct_in_zone", "p_whiff_rate": "whiff_rate",
            "p_k_rate": "k_rate", "p_bb_rate": "bb_rate",
            "p_barrel_rate_allowed": "barrel_rate_allowed",
            "p_arm_angle": "arm_angle", "p_extension": "extension",
        }
        for feat_name, profile_col in pitcher_feat_map.items():
            if feat_name in feat_cols and profile_col in latest.index:
                feature_vec[feat_name] = latest[profile_col]

        if "p_roll_hit_allowed_10" in feat_cols:
            hit_rate_est = latest.get("hit_rate_allowed", None)
            if hit_rate_est is None or pd.isna(hit_rate_est):
                hit_rate_est = np.clip(
                    1.0 - latest.get("k_rate", 0.22) - latest.get("bb_rate", 0.08) - 0.40,
                    0.15, 0.30,
                )
            feature_vec["p_roll_hit_allowed_10"] = hit_rate_est
            feature_vec["p_roll_hit_allowed_30"] = hit_rate_est
        if "p_roll_k_10" in feat_cols and "k_rate" in latest.index:
            feature_vec["p_roll_k_10"] = latest["k_rate"]
            feature_vec["p_roll_k_30"] = latest["k_rate"]
        if "p_roll_bb_10" in feat_cols and "bb_rate" in latest.index:
            feature_vec["p_roll_bb_10"] = latest["bb_rate"]
            feature_vec["p_roll_bb_30"] = latest["bb_rate"]
        for suf in ("_3", "_5"):
            if f"p_roll_k{suf}" in feat_cols and "k_rate" in latest.index:
                feature_vec[f"p_roll_k{suf}"] = latest["k_rate"]
            if f"p_roll_bb{suf}" in feat_cols and "bb_rate" in latest.index:
                feature_vec[f"p_roll_bb{suf}"] = latest["bb_rate"]
            if f"p_roll_hit_allowed{suf}" in feat_cols:
                hit_rate_est = latest.get("hit_rate_allowed", None)
                if hit_rate_est is None or pd.isna(hit_rate_est):
                    hit_rate_est = np.clip(
                        1.0 - latest.get("k_rate", 0.22) - latest.get("bb_rate", 0.08) - 0.40,
                        0.15,
                        0.30,
                    )
                feature_vec[f"p_roll_hit_allowed{suf}"] = hit_rate_est

    # Critical: classify this PA vs tonight's actual starter hand (trees use vs_lhp
    # × cum_*_vs_lhp / cum_*_vs_rhp interactions).
    if "vs_lhp" in feat_cols:
        feature_vec["vs_lhp"] = float(vs_lhp_tonight_int)

    g_roll_imputed_count = 0
    for col in feat_cols:
        if not (col.startswith("g_roll_") or col.startswith("p_g_roll_")):
            continue
        v = feature_vec.get(col, np.nan)
        if pd.isna(v):
            if "_hr_" in col:
                feature_vec[col] = 0.031
            elif "_xbh_" in col:
                feature_vec[col] = 0.0765
            else:
                feature_vec[col] = 0.243
            g_roll_imputed_count += 1

    if pa_data is not None:
        bvp_vals = compute_bvp_for_pair(batter_id, pitcher_id, pa_data)
        for col, val in bvp_vals.items():
            if col in feat_cols:
                feature_vec[col] = val

    extra_platoon: dict = {}
    if bpt_imputed_sources:
        extra_platoon["bpt_imputed"] = 1
        extra_platoon["bpt_imputed_sources"] = dict(bpt_imputed_sources)
    if g_roll_imputed_count > 0:
        extra_platoon["g_roll_imputed_count"] = int(g_roll_imputed_count)
    if game_context and game_context.get("game_date") is not None:
        gy = int(pd.Timestamp(game_context["game_date"]).year)
    elif game_context and game_context.get("game_year") is not None:
        gy = int(game_context["game_year"])
    else:
        gy = int(pd.Timestamp.now().year)

    try:
        pr = platoon_decayed_rates_for_batter_year(int(batter_id), gy, None)
        for k, v in pr.items():
            fv = float(v) if v == v else 0.0
            if k in feat_cols:
                feature_vec[k] = fv
            # Mirror decay splits into extra_platoon so feat_snap + X-is-None paths can
            # populate posterior_audit career-vs-hand fields for Beast dashboard columns.
            if k.startswith("platoon_eff_") or k.startswith("cum_career_"):
                extra_platoon[k] = fv
    except Exception:
        pass

    if pa_data is not None and game_context and game_context.get("game_date") is not None:
        extra_platoon.update(
            compute_ytd_platoon_vs_hand(
                int(batter_id),
                pa_data,
                game_context["game_date"],
                vs_lhp_tonight_int,
            )
        )

    return pd.DataFrame([feature_vec])[feat_cols].fillna(0), extra_platoon


def pct(v):
    return f"{v:.1%}"


def diff_str(val, baseline):
    d = val - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1%}"


def tier_label(p_hit):
    if p_hit >= 0.30:
        return "Elite"
    elif p_hit >= 0.26:
        return "Strong"
    elif p_hit >= 0.22:
        return "Average"
    elif p_hit >= 0.18:
        return "Below Average"
    return "Weak"


def compute_confidence(predictions, pitcher_profile, bvp_row,
                       batter_career_rates, pitcher_pa_total,
                       val_df_last_date=None,
                       feature_snapshot=None,
                       pitcher_throws="R",
                       platoon_raw_shrink_applied: bool = False):
    """Post-model confidence multipliers for ranking score (raw P × conf).

    BvP and platoon-vs-hand are blended into raw ``p_*`` upstream via
    ``apply_vs_hand_and_bvp_posteriors``; this function keeps ``bvp_*`` and
    platoon conf factors at 1.0 and uses career-vs-model convergence only.
    """
    factors = {}

    # --- Pitcher data factor ---
    if pitcher_pa_total == 0:
        factors["pitcher_data"] = 0.70
    elif pitcher_pa_total < 200:
        factors["pitcher_data"] = 0.80
    elif pitcher_pa_total < 500:
        factors["pitcher_data"] = 0.90
    else:
        factors["pitcher_data"] = 1.0

    has_pitcher_profile = bool(pitcher_profile and pitcher_profile.get("k_rate"))
    if not has_pitcher_profile:
        factors["pitcher_data"] = min(factors["pitcher_data"], 0.75)

    # --- BvP: blended into raw P via apply_vs_hand_and_bvp_posteriors; conf stays neutral here.
    bvp_pa = int(bvp_row.get("bvp_pa", 0)) if bvp_row else 0
    factors["bvp_hr"] = 1.0
    factors["bvp_hit"] = 1.0

    # --- Staleness factor ---
    if val_df_last_date is not None:
        days_stale = (pd.Timestamp.now() - pd.Timestamp(val_df_last_date)).days
    else:
        days_stale = 200
    if days_stale <= 30:
        factors["staleness"] = 1.0
    elif days_stale <= 90:
        factors["staleness"] = 0.95
    elif days_stale <= 180:
        factors["staleness"] = 0.90
    else:
        factors["staleness"] = 0.85

    # --- Convergence factor (do career, model, and BvP agree?) — league-relative where possible ---
    p_hr = predictions.get("is_hr", LEAGUE_AVG["hr"])
    career_hr = batter_career_rates.get("hr", LEAGUE_AVG["hr"])
    p_hit = predictions.get("is_hit", LEAGUE_AVG["hit"])
    career_hit = batter_career_rates.get("hit", LEAGUE_AVG["hit"])
    p_xbh = predictions.get("is_xbh", LEAGUE_AVG["xbh"])
    career_xbh = batter_career_rates.get("xbh", LEAGUE_AVG["xbh"])

    lg_hr = LEAGUE_AVG["hr"]
    lg_hit = LEAGUE_AVG["hit"]
    lg_xbh = LEAGUE_AVG["xbh"]

    hr_signals = [
        career_hr >= CONF_CONV_CAREER_MULT_HR * lg_hr,
        p_hr >= CONF_CONV_MODEL_K_HR * lg_hr,
    ]
    n_hr_sig = len(hr_signals)
    factors["convergence_hr"] = graded_convergence_factor(sum(hr_signals), n_hr_sig)

    hit_signals = [
        career_hit >= CONF_CONV_CAREER_MULT_HIT * lg_hit,
        p_hit >= CONF_CONV_MODEL_K_HIT * lg_hit,
    ]
    n_hit_sig = len(hit_signals)
    factors["convergence_hit"] = graded_convergence_factor(sum(hit_signals), n_hit_sig)

    xbh_signals = [
        career_xbh >= CONF_CONV_CAREER_MULT_XBH * lg_xbh,
        p_xbh >= CONF_CONV_MODEL_K_XBH * lg_xbh,
    ]
    n_xbh_sig = len(xbh_signals)
    factors["convergence_xbh"] = graded_convergence_factor(sum(xbh_signals), n_xbh_sig)

    factors["platoon_matchup_quality"] = 1.0
    factors["platoon_quality_reason"] = "posterior_blend"
    plat_mult_eff = 1.0
    factors["platoon_conf_mult_skipped_duplicate_with_raw_shrink"] = bool(
        platoon_raw_shrink_applied)

    body_hr = (
        factors["pitcher_data"] * factors["bvp_hr"]
        * factors["staleness"] * factors["convergence_hr"]
    )
    body_hit = (
        factors["pitcher_data"] * factors["bvp_hit"]
        * factors["staleness"] * factors["convergence_hit"]
    )
    conf_hr = body_hr * plat_mult_eff
    conf_hit = body_hit * plat_mult_eff
    conf_xbh_pre = min(body_hr, body_hit) * factors["convergence_xbh"] * plat_mult_eff
    # Keep XBH composite weakly capped by HR/Hit composites; convergence_xbh can only pull down vs their min body.
    conf_xbh = min(conf_xbh_pre, conf_hr, conf_hit)
    score_hr = round(p_hr * conf_hr, 5)
    score_hit = round(p_hit * conf_hit, 5)
    score_xbh = round(p_xbh * conf_xbh, 5)

    # C1: isotonic calibration bucket prefers quantiles of numeric conf_* when trained;
    # else falls back to hand tier cutoffs via resolve_calibration_bucket.
    try:
        from calibrate_predictions import apply_calibration, resolve_calibration_bucket

        bucket_hr = resolve_calibration_bucket(conf_hr, "hr")
        bucket_hit = resolve_calibration_bucket(conf_hit, "hit")
        bucket_xbh = resolve_calibration_bucket(conf_xbh, "xbh")
        p_hr_cal = apply_calibration(p_hr, "hr", bucket_hr)
        p_hit_cal = apply_calibration(p_hit, "hit", bucket_hit)
        p_xbh_cal = apply_calibration(p_xbh, "xbh", bucket_xbh)
    except Exception:
        # Defensive: never let calibration loading break inference. Fall back to raw P.
        p_hr_cal, p_hit_cal, p_xbh_cal = p_hr, p_hit, p_xbh

    return {
        "conf_hr": round(conf_hr, 3),
        "conf_hit": round(conf_hit, 3),
        "conf_xbh": round(conf_xbh, 3),
        # NEW (C2): score_* is the ranking score (= raw P × conf). It is NOT a probability.
        "score_hr": score_hr,
        "score_hit": score_hit,
        "score_xbh": score_xbh,
        # C1: per-conf-bucket isotonic-recalibrated probability (= raw P if not yet trained).
        "p_hr_calibrated": round(p_hr_cal, 5),
        "p_hit_calibrated": round(p_hit_cal, 5),
        "p_xbh_calibrated": round(p_xbh_cal, 5),
        # DEPRECATED ALIAS: adj_p_* kept for back-compat (one release). New consumers use score_*.
        "adj_p_hr": score_hr,
        "adj_p_hit": score_hit,
        "adj_p_xbh": score_xbh,
        "factors": factors,
    }


def _json_safe_scalar(val):
    """JSON / Parquet friendly scalar from numpy/pandas types."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return val


def confidence_label_for_target(conf, target: str) -> str:
    """Map composite confidence multiplier to High/Medium/Low/Very Low; stricter tiers for HR, middle for XBH."""
    t = (target or "hit").lower()
    if t == "hr":
        hi, md, lo = CONF_LABEL_HR
    elif t == "xbh":
        hi, md, lo = CONF_LABEL_XBH
    else:
        hi, md, lo = CONF_LABEL_HIT
    if conf >= hi:
        return "High"
    if conf >= md:
        return "Medium"
    if conf >= lo:
        return "Low"
    return "Very Low"


def confidence_label(conf):
    """Legacy alias; Hit-target cutoffs (same as CONF_LABEL_HIT)."""
    return confidence_label_for_target(conf, "hit")


def generate_narrative(batter_name, batter_team, pitcher_name, pitcher_team,
                       predictions, pitcher_profile, batter_pitch_prof,
                       bvp_row, batter_career_rates,
                       pitcher_pa_total=0, val_df_last_date=None,
                       feature_snapshot=None, pitcher_throws=None,
                       platoon_raw_shrink_applied: bool = False):
    """Generate quantified narrative for a single batter-pitcher matchup."""
    lines = []

    p_hit = predictions.get("is_hit", LEAGUE_AVG["hit"])
    p_hr = predictions.get("is_hr", LEAGUE_AVG["hr"])
    p_k = predictions.get("is_strikeout", LEAGUE_AVG["strikeout"])
    p_bb = predictions.get("is_walk", LEAGUE_AVG["walk"])
    p_xbh = predictions.get("is_xbh", LEAGUE_AVG["xbh"])

    career_hit = batter_career_rates.get("hit", LEAGUE_AVG["hit"])
    career_hr = batter_career_rates.get("hr", LEAGUE_AVG["hr"])
    career_k = batter_career_rates.get("strikeout", LEAGUE_AVG["strikeout"])
    career_bb = batter_career_rates.get("walk", LEAGUE_AVG["walk"])

    prof_thr = pitcher_profile.get("throws", "R") if pitcher_profile else "R"
    if pitcher_throws is not None:
        h = str(pitcher_throws).strip().upper()
        throws = "L" if h.startswith("L") else "R"
    else:
        h2 = str(prof_thr).strip().upper()
        throws = "L" if h2.startswith("L") else "R"

    # P(Hit) narrative
    reasons = []
    if pitcher_profile:
        top_pitch_pct = pitcher_profile.get("pct_fastball", 0)
        top_pitch_type = "fastballs"
        if pitcher_profile.get("pct_breaking", 0) > top_pitch_pct:
            top_pitch_pct = pitcher_profile["pct_breaking"]
            top_pitch_type = "breaking balls"
        if top_pitch_pct > 0:
            batter_ba_key = f"ba_vs_{top_pitch_type.split()[0].lower() if top_pitch_type != 'breaking balls' else 'breaking'}"
            ba_vs_type = batter_pitch_prof.get(batter_ba_key, None)
            if ba_vs_type and ba_vs_type > 0:
                reasons.append(
                    f"Pitcher throws {top_pitch_pct:.0%} {top_pitch_type}; "
                    f"batter hits {top_pitch_type} at .{int(ba_vs_type*1000):03d}"
                )
        ev_key = f"ev_vs_{'lhp' if throws == 'L' else 'rhp'}"
        ev_val = batter_pitch_prof.get(ev_key)
        if ev_val and ev_val > 0:
            reasons.append(f"Exit velo vs {'LHP' if throws == 'L' else 'RHP'}: {ev_val:.1f} mph")

    hit_narrative = f"{diff_str(p_hit, career_hit)} vs career ({pct(career_hit)}), " \
                    f"{diff_str(p_hit, LEAGUE_AVG['hit'])} vs league ({pct(LEAGUE_AVG['hit'])})"
    if reasons:
        hit_narrative += ". " + ". ".join(reasons[:2])

    # P(K) narrative
    k_reasons = []
    if pitcher_profile.get("k_rate"):
        k_reasons.append(f"Pitcher K rate: {pitcher_profile['k_rate']:.0%}")
    if pitcher_profile.get("whiff_rate"):
        k_reasons.append(f"Whiff rate: {pitcher_profile['whiff_rate']:.0%}")
    k_narrative = f"{diff_str(p_k, career_k)} vs career ({pct(career_k)}), " \
                  f"{diff_str(p_k, LEAGUE_AVG['strikeout'])} vs league"
    if k_reasons:
        k_narrative += ". " + ". ".join(k_reasons[:2])

    # BvP narrative
    bvp_text = ""
    if bvp_row is not None and bvp_row.get("bvp_pa", 0) > 0:
        n = bvp_row["bvp_pa"]
        h = bvp_row["bvp_hits"]
        k = bvp_row["bvp_k"]
        ba = bvp_row["bvp_ba"]
        bvp_text = f"{n} career PAs vs this pitcher: {h}-for-{n} (.{int(ba*1000):03d}), {k} Ks"

    # Multi-outcome Monte Carlo: each PA resolves as Hit / K / BB / Other Out
    # independently, so high-K batters are properly penalized for multi-hit upside
    p_multi_hit, p_any_hit = simulate_multi_hit(p_hit, p_k, p_bb,
                                                 n_pa=PA_VS_STARTER_DEFAULT)

    tier = tier_label(p_hit)

    conf = compute_confidence(
        predictions, pitcher_profile, bvp_row, batter_career_rates,
        pitcher_pa_total=pitcher_pa_total,
        val_df_last_date=val_df_last_date,
        feature_snapshot=feature_snapshot,
        pitcher_throws=throws,
        platoon_raw_shrink_applied=platoon_raw_shrink_applied,
    )

    return {
        "batter_name": batter_name,
        "batter_team": batter_team,
        "pitcher_name": pitcher_name,
        "pitcher_team": pitcher_team,
        "pitcher_throws": throws,
        "tier": tier,
        "p_hit": p_hit,
        "p_hr": p_hr,
        "p_k": p_k,
        "p_bb": p_bb,
        "p_xbh": p_xbh,
        "p_multi_hit": p_multi_hit,
        "p_any_hit": p_any_hit,
        "career_hit": career_hit,
        "career_hr": career_hr,
        "career_k": career_k,
        "career_bb": career_bb,
        "hit_narrative": hit_narrative,
        "k_narrative": k_narrative,
        "bvp_text": bvp_text,
        "top_reasons": reasons[:3],
        "conf_hr": conf["conf_hr"],
        "conf_hit": conf["conf_hit"],
        "conf_xbh": conf["conf_xbh"],
        # NEW (C2): ranking scores (raw P × conf). NOT probabilities.
        "score_hr": conf["score_hr"],
        "score_hit": conf["score_hit"],
        "score_xbh": conf["score_xbh"],
        # NEW (C2): calibrated probabilities (= raw P until C1 ships).
        "p_hr_calibrated": conf["p_hr_calibrated"],
        "p_hit_calibrated": conf["p_hit_calibrated"],
        "p_xbh_calibrated": conf["p_xbh_calibrated"],
        # DEPRECATED ALIAS: adj_p_* mirrors score_* for one release.
        "adj_p_hr": conf["adj_p_hr"],
        "adj_p_hit": conf["adj_p_hit"],
        "adj_p_xbh": conf["adj_p_xbh"],
        "conf_hr_label": confidence_label_for_target(conf["conf_hr"], "hr"),
        "conf_hit_label": confidence_label_for_target(conf["conf_hit"], "hit"),
        "conf_xbh_label": confidence_label_for_target(conf["conf_xbh"], "xbh"),
        "conf_factors": conf["factors"],
    }


def _weather_hr_mult_by_game_pk() -> dict[int, float]:
    """Section 15 JSON sidecar: game-level HR carry multiplier → weak contact proxy."""
    path = REPORTS_DIR / "todays_zero_hr_predictions.json"
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, float] = {}
    for g in blob.get("games") or []:
        pk = g.get("game_pk")
        if pk is None:
            continue
        try:
            out[int(pk)] = float(g.get("weather_hr_mult") or 1.0)
        except (TypeError, ValueError):
            continue
    return out


def predict_matchups(
    matchups_json_path=None,
    model_dir=None,
    val_features_path=None,
    *,
    model_source: str = "prod",
):
    """Run predictions for all today's matchups.

    ``model_dir`` / ``val_features_path`` override defaults so an experiment bundle
    (e.g. ``exp_bpt_xwoba`` + ``features_val_league_exp.parquet``) can run alongside production.
    ``model_source`` is stored on each matchup dict for dual-model auditing (e.g. ``\"prod\"`` / ``\"exp\"``).
    """
    if matchups_json_path is None:
        matchups_json_path = RAW_DIR / "todays_matchups.json"

    with open(matchups_json_path) as f:
        games = json.load(f)

    models = load_models(model_dir)
    feat_cols = load_feature_cols(model_dir)
    pitcher_profiles = load_pitcher_profiles()
    batter_profiles = load_batter_pitch_profiles()
    bvp_history = load_bvp_history()

    val_df = pd.read_parquet(_resolved_val_features_path(val_features_path))

    pa_data = pd.read_parquet(RAW_DIR / "statcast_pa_level_league.parquet")
    hit_events = {"single", "double", "triple", "home_run"}
    pa_data["is_hit"] = pa_data["events"].isin(hit_events).astype(int)
    pa_data["is_hr"] = (pa_data["events"] == "home_run").astype(int)
    pa_data["is_k"] = pa_data["events"].str.contains("strikeout", case=False, na=False).astype(int)
    pa_data["is_bb"] = pa_data["events"].isin({"walk", "intent_walk"}).astype(int)
    pa_data["is_xbh"] = pa_data["events"].isin({"double", "triple", "home_run"}).astype(int)
    career_rates = pa_data.groupby("batter").agg(
        hit=("is_hit", "mean"), hr=("is_hr", "mean"), xbh=("is_xbh", "mean"),
        strikeout=("is_k", "mean"), walk=("is_bb", "mean"),
    ).to_dict("index")

    pitcher_pa_counts = pa_data.groupby("pitcher").size().to_dict()
    career_pa_totals = pa_data.groupby("batter").size().to_dict()
    pitcher_ids_in_profile = set(pitcher_profiles["pitcher"].unique()) if not pitcher_profiles.empty else set()

    slate_date_for_season = games[0].get("game_date") if games else None
    season_hr_row_counts = season_ytd_hr_row_counts(pa_data, slate_date_for_season)

    lineup_lookup = load_lineup_slot_lookup()
    wx_by_game_pk = _weather_hr_mult_by_game_pk()

    all_results = []

    for game in games:
        gdt = game.get("game_date")
        gpk = game.get("game_pk")
        wx_mult = None
        if gpk is not None:
            try:
                wx_mult = wx_by_game_pk.get(int(gpk))
            except (TypeError, ValueError):
                wx_mult = None
        game_context_vec = {
            "home_team": game.get("home_team"),
            "game_date": gdt,
            "game_year": int(pd.Timestamp(gdt).year) if gdt is not None else None,
            "weather_hr_mult": wx_mult,
            "times_thru_order": game.get("times_thru_order_first_pa"),
            "pitch_count": game.get("pitch_count_first_pa"),
        }
        game_results = {"game": game, "matchups": [], "park_context": None}

        for side in [("away", "home"), ("home", "away")]:
            batting_side, pitching_side = side
            pitcher_id = game[f"{pitching_side}_pitcher_id"]
            pitcher_name = game[f"{pitching_side}_pitcher_name"]
            pitcher_team = game[f"{pitching_side}_team"]
            batters = game[f"{batting_side}_batters"]
            batter_team = game[f"{batting_side}_team"]

            pp = pitcher_profiles[pitcher_profiles["pitcher"] == pitcher_id]
            pitcher_prof = pp.sort_values("season").iloc[-1].to_dict() if not pp.empty else {}
            pitcher_pa_total = pitcher_pa_counts.get(pitcher_id, 0)
            pitcher_throws_hint = game.get(f"{pitching_side}_pitcher_throws")
            throws_for_narrative = resolve_pitcher_throws(
                pitcher_id, pitcher_profiles, pitcher_throws_hint
            )

            for batter in batters:
                bid = batter["mlbam_id"]

                X, extra_platoon = build_feature_vector(
                    bid,
                    pitcher_id,
                    feat_cols,
                    val_df,
                    pitcher_profiles,
                    batter_profiles,
                    pa_data=pa_data,
                    skip_rolling_dampen=(model_source == "recency"),
                    game_context=game_context_vec,
                    pitcher_throws_hint=pitcher_throws_hint,
                )

                bpp_rows = batter_profiles[batter_profiles["batter"] == bid]
                bpp = bpp_rows.sort_values("season").iloc[-1].to_dict() if not bpp_rows.empty else {}

                bvp_match = bvp_history[(bvp_history["batter"] == bid) &
                                        (bvp_history["pitcher"] == pitcher_id)]
                bvp_row = bvp_match.iloc[0].to_dict() if not bvp_match.empty else None

                total_career_pa = int(career_pa_totals.get(bid, 0))
                pitcher_profile_missing = (int(pitcher_id) not in pitcher_ids_in_profile)
                pitcher_is_tbd = (
                    pitcher_id is None
                    or pitcher_name is None
                    or "TBD" in str(pitcher_name).upper()
                )

                if X is None:
                    gbdt_anchor_hr = float(LEAGUE_AVG["hr"])
                    predictions = {t: LEAGUE_AVG.get(t.replace("is_", ""), 0.2) for t in TARGETS}
                    feat_snap = None
                    extra_platoon = {}
                    try:
                        gy_none = (
                            int(pd.Timestamp(gdt).year)
                            if gdt is not None
                            else int(pd.Timestamp.now().year)
                        )
                        pr_none = platoon_decayed_rates_for_batter_year(int(bid), gy_none, None)
                        for k, v in pr_none.items():
                            try:
                                fv = float(v)
                            except (TypeError, ValueError):
                                continue
                            if pd.isna(fv):
                                continue
                            if k.startswith("platoon_eff_") or k.startswith("cum_career_"):
                                extra_platoon[k] = fv
                    except Exception:
                        pass
                    if pa_data is not None and gdt is not None:
                        vs_lhp_int = int(str(throws_for_narrative or "R").upper().startswith("L"))
                        try:
                            extra_platoon.update(
                                compute_ytd_platoon_vs_hand(
                                    int(bid), pa_data, gdt, vs_lhp_int
                                )
                            )
                        except Exception:
                            pass
                    predictions, post_audit = apply_vs_hand_and_bvp_posteriors(
                        predictions, extra_platoon, throws_for_narrative, extra_platoon, bvp_row,
                        total_career_pa=total_career_pa,
                    )
                    post_audit = dict(post_audit or {})
                    post_audit["x_is_none_path"] = True
                    data_quality = "x_is_none_league_avg"
                else:
                    predictions = {}
                    for target, model in models.items():
                        try:
                            predictions[target] = float(model.predict_proba(X)[:, 1][0])
                        except Exception:
                            predictions[target] = LEAGUE_AVG.get(target.replace("is_", ""), 0.2)

                    gbdt_anchor_hr = float(predictions.get("is_hr", LEAGUE_AVG["hr"]))

                    feat_snap = X.iloc[0].to_dict()
                    feat_snap.update(extra_platoon or {})
                    predictions, post_audit = apply_vs_hand_and_bvp_posteriors(
                        predictions, feat_snap, throws_for_narrative, extra_platoon, bvp_row,
                        total_career_pa=total_career_pa,
                    )
                    post_audit = dict(post_audit or {})
                    post_audit["x_is_none_path"] = False
                    data_quality = "full_gbdt"

                # Surface fall-back signals on every row (Phase B audit fields).
                post_audit["pitcher_profile_missing"] = bool(pitcher_profile_missing)
                post_audit["pitcher_is_tbd"] = bool(pitcher_is_tbd)
                if (extra_platoon or {}).get("bpt_imputed"):
                    post_audit["bpt_imputed"] = 1
                    sources = (extra_platoon or {}).get("bpt_imputed_sources") or {}
                    if sources:
                        post_audit["bpt_imputed_sources"] = dict(sources)
                if (extra_platoon or {}).get("g_roll_imputed_count"):
                    post_audit["g_roll_imputed_count"] = int(extra_platoon["g_roll_imputed_count"])

                # Hierarchy: TBD overrides all (you can't model what isn't named yet);
                # missing pitcher profile next; X-is-None already set above; else full_gbdt.
                if pitcher_is_tbd:
                    data_quality = "tbd_pitcher"
                elif pitcher_profile_missing and data_quality == "full_gbdt":
                    data_quality = "missing_pitcher_profile"
                post_audit["data_quality"] = data_quality

                # Thin early-season vs-hand PA: don't let stacked platoon+BvP blow HR past GBDT + small epsilon.
                post_audit["gbdt_hr_anchor"] = round(float(gbdt_anchor_hr), 5)
                ypa_gate = int((extra_platoon or {}).get("ytd_pa", 0) or 0)
                if ypa_gate < HR_THIN_YTD_VS_HAND_PA:
                    ceiling = float(
                        np.clip(
                            gbdt_anchor_hr + HR_THIN_YTD_MAX_LIFT_OVER_GBDT,
                            1e-4,
                            0.35,
                        )
                    )
                    phr_now = float(predictions["is_hr"])
                    if phr_now > ceiling:
                        predictions["is_hr"] = ceiling
                        post_audit["hr_thin_ytd_cap_to"] = round(ceiling, 5)

                cr = career_rates.get(bid, {})

                # Career HR/PA ceiling — a Top-10 HR pick must have a career HR rate that supports it.
                # Strict mode: cap = HR_CAREER_CAP_MULTIPLIER * career_hr_per_pa, no GBDT floor.
                # Disengaged for batters with < HR_CAREER_CAP_MIN_PA career PA so rookies/call-ups
                # aren't pinned to a tiny historical sample.
                career_hr_rate = float((cr or {}).get("hr", 0.0) or 0.0)
                post_audit["career_hr_rate"] = round(career_hr_rate, 5)
                if int(total_career_pa) >= HR_CAREER_CAP_MIN_PA and career_hr_rate > 0:
                    rate_cap = float(
                        np.clip(
                            HR_CAREER_CAP_MULTIPLIER * career_hr_rate,
                            1e-4,
                            0.35,
                        )
                    )
                    if predictions["is_hr"] > rate_cap:
                        predictions["is_hr"] = rate_cap
                        post_audit["hr_career_cap_to"] = round(rate_cap, 5)

                s_rows, s_hr = season_hr_row_counts.get(int(bid), (0, 0))
                post_audit["season_ytd_statcast_rows"] = int(s_rows)
                post_audit["season_ytd_hr"] = int(s_hr)
                if (
                    s_rows >= HR_ZERO_SEASON_GATE_MIN_STATCAST_ROWS
                    and s_hr == 0
                    and float(predictions["is_hr"]) > HR_ZERO_SEASON_GATE_FLOOR
                ):
                    post_audit["p_hr_pre_zero_season_gate"] = round(float(predictions["is_hr"]), 5)
                    predictions["is_hr"] = float(HR_ZERO_SEASON_GATE_FLOOR)
                    post_audit["hr_zero_season_suppressed"] = 1

                # Weak cumulative platoon vs tonight's arm → multiplicative down-weight on raw P
                # (after vs-hand posterior + HR gates; see platoon_raw_probability_shrink).
                shrink_snap = feat_snap if feat_snap is not None else extra_platoon
                predictions, plat_raw_audit = platoon_raw_probability_shrink(
                    predictions,
                    shrink_snap if shrink_snap else {},
                    throws_for_narrative,
                )
                plat_raw_pen = (
                    float(plat_raw_audit.get("platoon_raw_hit_mult") or 1.0) < 1.0
                    or float(plat_raw_audit.get("platoon_raw_xbh_mult") or 1.0) < 1.0
                    or float(plat_raw_audit.get("platoon_raw_hr_mult") or 1.0) < 1.0
                )
                _enforce_hit_xbh_hr_order(predictions)

                batter_rows = val_df[val_df["batter"] == bid]
                if not batter_rows.empty:
                    last_date = batter_rows["game_date"].max()
                else:
                    last_date = None

                result = generate_narrative(
                    batter["name"], batter_team,
                    pitcher_name, pitcher_team,
                    predictions, pitcher_prof, bpp, bvp_row, cr,
                    pitcher_pa_total=pitcher_pa_total,
                    val_df_last_date=last_date,
                    feature_snapshot=feat_snap,
                    pitcher_throws=throws_for_narrative,
                    platoon_raw_shrink_applied=plat_raw_pen,
                )
                result["platoon_raw_shrink"] = {
                    k: _json_safe_scalar(v) for k, v in plat_raw_audit.items()
                }
                result["posterior_audit"] = {
                    k: _json_safe_scalar(v) for k, v in post_audit.items()
                }
                result["data_quality"] = data_quality
                result["batter_mlbam_id"] = int(bid)
                result["pitcher_mlbam_id"] = int(pitcher_id)

                # M1: attach learned conf labels (Lock/Strong/Lean/Avoid) alongside hand-tuned ones.
                # Pulls features from the same feature vector X that scored the base model.
                # T2.5: pass cross-target predictions so M1 can use hr_xbh_consistency.
                if X is not None:
                    try:
                        from conf_meta_inference import compute_meta_label
                        # X may be either a numpy ndarray (1xN) or a pandas DataFrame.
                        # Convert to a plain {feat_name: scalar} dict regardless.
                        if hasattr(X, "iloc"):
                            x_dict = X.iloc[0].to_dict()
                        else:
                            x_dict = dict(zip(feat_cols, np.asarray(X).flatten().tolist()))
                        p_hr_now = predictions.get("is_hr", 0.0)
                        p_hit_now = predictions.get("is_hit", 0.0)
                        p_xbh_now = predictions.get("is_xbh", 0.0)
                        for tgt_short, tgt_long in [("hr", "is_hr"), ("hit", "is_hit"), ("xbh", "is_xbh")]:
                            label, meta_p = compute_meta_label(
                                tgt_short,
                                predictions.get(tgt_long, 0.0),
                                x_dict,
                                p_hr=p_hr_now, p_xbh=p_xbh_now, p_hit=p_hit_now,
                            )
                            result[f"score_label_{tgt_short}"] = label
                            result[f"meta_p_{tgt_short}"] = round(meta_p, 5)
                    except Exception as _meta_exc:
                        import os as _os
                        if _os.environ.get("META_DEBUG"):
                            import traceback as _tb
                            try:
                                _bn = result.get("batter_name", "?")
                            except Exception:
                                _bn = "?"
                            print(f"[META_DEBUG] {_bn}: {_meta_exc}")
                            _tb.print_exc()
                        # Defensive: never let meta-model failure break inference
                        for tgt_short in ("hr", "hit", "xbh"):
                            result.setdefault(f"score_label_{tgt_short}", "Avoid")
                            result.setdefault(f"meta_p_{tgt_short}", float(predictions.get(f"is_{tgt_short}", 0.0)))
                # Recent lineup spot vs opposing starter hand (Layer 2 rollup; optional).
                oh_lc = "L" if str(throws_for_narrative).upper().startswith("L") else "R"
                lc_row = lineup_lookup.get((int(bid), oh_lc))
                result["lineup_context"] = (
                    {k: _json_safe_scalar(v) for k, v in lc_row.items()}
                    if lc_row
                    else None
                )

                # Slate metadata for post-hoc audits only (additive; does not affect model I/O).
                result["slate_date"] = str(game.get("game_date", ""))
                result["game_pk"] = int(game["game_pk"]) if game.get("game_pk") is not None else None
                # Structured BvP for tracking Parquet / JSON (same inputs the model uses).
                if bvp_row is not None:
                    result["bvp_career_vs_pitcher"] = {
                        k: _json_safe_scalar(v) for k, v in bvp_row.items()
                    }
                else:
                    result["bvp_career_vs_pitcher"] = None
                result["bvp_model_features"] = compute_bvp_for_pair(bid, pitcher_id, pa_data)
                result["model_source"] = model_source
                result["park_context"] = None
                game_results["matchups"].append(result)

        all_results.append(game_results)

    return all_results


if __name__ == "__main__":
    results = predict_matchups()
    for game_result in results:
        g = game_result["game"]
        print(f"\n{'='*70}")
        print(f"  {g['away_team']} @ {g['home_team']}")
        print(f"  {g['away_pitcher_name']} vs {g['home_pitcher_name']}")
        print(f"{'='*70}")
        for m in sorted(game_result["matchups"], key=lambda x: x["p_hit"], reverse=True)[:5]:
            print(f"\n  {m['batter_name']} ({m['batter_team']}) vs {m['pitcher_name']} [{m['tier']}]")
            print(f"    Hit: {pct(m['p_hit'])} | HR: {pct(m['p_hr'])} | K: {pct(m['p_k'])} | BB: {pct(m['p_bb'])}")
            print(f"    Multi-Hit: {pct(m['p_multi_hit'])} | Any Hit: {pct(m['p_any_hit'])}")
            if m["hit_narrative"]:
                print(f"    → {m['hit_narrative']}")
            if m["bvp_text"]:
                print(f"    → BvP: {m['bvp_text']}")
