"""
Append-only Parquet store for slate matchup predictions (analysis only; not used in training).

Called from build_matchup_dashboard after predict_matchups. High-confidence slice
prefers any target with conf_*_label == High; falls back on legacy composite max(conf_*) ≥ 1.05.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from config import TRACKING_DIR

TRACKING_MAIN = TRACKING_DIR / "matchup_predictions_runs.parquet"
TRACKING_HIGH_CONF = TRACKING_DIR / "matchup_predictions_runs_high_conf.parquet"
DUAL_MODEL_PARQUET = TRACKING_DIR / "matchup_dual_model_predictions.parquet"
HIGH_CONF_LEGACY_COMPOSITE_MIN = 1.05  # pre per-target tiers: coarse max(conf_*)


def _scalar(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return v


def flatten_matchup(m: dict, run_timestamp: str) -> dict:
    """One row dict for Parquet; nested structures JSON-encoded where needed."""
    career = m.get("bvp_career_vs_pitcher") or {}
    model_bvp = m.get("bvp_model_features") or {}
    factors = m.get("conf_factors") or {}

    row: dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "slate_date": str(m.get("slate_date") or ""),
        "game_pk": _scalar(m.get("game_pk")),
        "batter_mlbam_id": _scalar(m.get("batter_mlbam_id")),
        "pitcher_mlbam_id": _scalar(m.get("pitcher_mlbam_id")),
        "batter_team": m.get("batter_team"),
        "pitcher_team": m.get("pitcher_team"),
        "batter_name": m.get("batter_name"),
        "pitcher_name": m.get("pitcher_name"),
        "pitcher_throws": m.get("pitcher_throws"),
        "tier": m.get("tier"),
        "p_hit": _scalar(m.get("p_hit")),
        "p_hr": _scalar(m.get("p_hr")),
        "p_k": _scalar(m.get("p_k")),
        "p_bb": _scalar(m.get("p_bb")),
        "p_xbh": _scalar(m.get("p_xbh")),
        "p_multi_hit": _scalar(m.get("p_multi_hit")),
        "p_any_hit": _scalar(m.get("p_any_hit")),
        "adj_p_hit": _scalar(m.get("adj_p_hit")),
        "adj_p_hr": _scalar(m.get("adj_p_hr")),
        "adj_p_xbh": _scalar(m.get("adj_p_xbh")),
        "conf_hr": _scalar(m.get("conf_hr")),
        "conf_hit": _scalar(m.get("conf_hit")),
        "conf_xbh": _scalar(m.get("conf_xbh")),
        "conf_hr_label": m.get("conf_hr_label"),
        "conf_hit_label": m.get("conf_hit_label"),
        "conf_xbh_label": m.get("conf_xbh_label"),
        "career_hit": _scalar(m.get("career_hit")),
        "career_hr": _scalar(m.get("career_hr")),
        "career_k": _scalar(m.get("career_k")),
        "career_bb": _scalar(m.get("career_bb")),
        "hit_narrative": (m.get("hit_narrative") or "")[:2000],
        "k_narrative": (m.get("k_narrative") or "")[:2000],
        "bvp_text": m.get("bvp_text"),
        "top_reasons_json": json.dumps(m.get("top_reasons") or []),
        "conf_factors_json": json.dumps(factors, default=str),
        "career_bvp_pa": _scalar(career.get("bvp_pa")),
        "career_bvp_hits": _scalar(career.get("bvp_hits")),
        "career_bvp_hr": _scalar(career.get("bvp_hr")),
        "career_bvp_k": _scalar(career.get("bvp_k")),
        "career_bvp_bb": _scalar(career.get("bvp_bb")),
        "career_bvp_ba": _scalar(career.get("bvp_ba")),
        "model_bvp_pa_count": _scalar(model_bvp.get("bvp_pa_count")),
        "model_bvp_ba": _scalar(model_bvp.get("bvp_ba")),
        "model_bvp_k_rate": _scalar(model_bvp.get("bvp_k_rate")),
        "model_bvp_bb_rate": _scalar(model_bvp.get("bvp_bb_rate")),
        "model_bvp_hr_count": _scalar(model_bvp.get("bvp_hr_count")),
        "model_bvp_hr_rate": _scalar(model_bvp.get("bvp_hr_rate")),
        "model_bvp_xbh_rate": _scalar(model_bvp.get("bvp_xbh_rate")),
        "model_log_bvp_pa": _scalar(model_bvp.get("log_bvp_pa")),
        "model_bvp_has_history": _scalar(model_bvp.get("bvp_has_history")),
        "outcome_pa": _scalar(m.get("outcome_pa")),
        "outcome_h": _scalar(m.get("outcome_h")),
        "outcome_hr": _scalar(m.get("outcome_hr")),
        "outcome_xbh": _scalar(m.get("outcome_xbh")),
        "outcome_hit_flag": m.get("outcome_hit_flag"),
        "outcome_hr_flag": m.get("outcome_hr_flag"),
        "outcome_xbh_flag": m.get("outcome_xbh_flag"),
        "outcome_filled_at": m.get("outcome_filled_at"),
    }
    return row


def append_matchup_tracking_rows(matchups: list[dict], run_timestamp: str) -> int:
    """Append flattened rows; dedupe on (slate_date, game_pk, batter, pitcher), keep last."""
    if not matchups:
        return 0
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    rows = [flatten_matchup(m, run_timestamp) for m in matchups]
    new_df = pd.DataFrame(rows)
    if TRACKING_MAIN.exists():
        old = pd.read_parquet(TRACKING_MAIN)
        combined = pd.concat([old, new_df], ignore_index=True)
        keys = ["slate_date", "game_pk", "batter_mlbam_id", "pitcher_mlbam_id"]
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = new_df
    combined.to_parquet(TRACKING_MAIN, index=False)
    return len(new_df)


def _matchup_join_key(m: dict) -> tuple:
    return (
        str(m.get("slate_date") or ""),
        _scalar(m.get("game_pk")),
        _scalar(m.get("batter_mlbam_id")),
        _scalar(m.get("pitcher_mlbam_id")),
    )


def append_dual_model_predictions(
    prod_flat: list[dict],
    exp_flat: list[dict],
    run_timestamp: str,
    *,
    prod_model_dir: str,
    exp_model_dir: str,
    prod_val_path: str,
    exp_val_path: str,
) -> int:
    """Wide rows: production vs experiment probabilities for the same slate keys."""
    if not prod_flat or not exp_flat:
        return 0
    exp_map = {_matchup_join_key(m): m for m in exp_flat}
    rows: list[dict[str, Any]] = []
    for m in prod_flat:
        e = exp_map.get(_matchup_join_key(m))
        if e is None:
            continue
        row: dict[str, Any] = {
            "run_timestamp": run_timestamp,
            "slate_date": m.get("slate_date"),
            "game_pk": _scalar(m.get("game_pk")),
            "batter_mlbam_id": _scalar(m.get("batter_mlbam_id")),
            "pitcher_mlbam_id": _scalar(m.get("pitcher_mlbam_id")),
            "batter_team": m.get("batter_team"),
            "pitcher_team": m.get("pitcher_team"),
            "batter_name": m.get("batter_name"),
            "pitcher_name": m.get("pitcher_name"),
            "tier_prod": m.get("tier"),
            "tier_exp": e.get("tier"),
            "prod_model_dir": prod_model_dir,
            "exp_model_dir": exp_model_dir,
            "prod_val_features_path": prod_val_path,
            "exp_val_features_path": exp_val_path,
        }
        for key, short in [
            ("p_hit", "hit"),
            ("p_hr", "hr"),
            ("p_xbh", "xbh"),
            ("p_k", "k"),
            ("p_bb", "bb"),
        ]:
            vp = _scalar(m.get(key))
            ve = _scalar(e.get(key))
            row[f"{key}_prod"] = vp
            row[f"{key}_exp"] = ve
            row[f"d_{short}"] = (ve - vp) if vp is not None and ve is not None else None
        for key, dcol in [
            ("adj_p_hit", "d_adj_p_hit"),
            ("adj_p_hr", "d_adj_p_hr"),
            ("adj_p_xbh", "d_adj_p_xbh"),
        ]:
            vp = _scalar(m.get(key))
            ve = _scalar(e.get(key))
            row[f"{key}_prod"] = vp
            row[f"{key}_exp"] = ve
            row[dcol] = (ve - vp) if vp is not None and ve is not None else None
        rows.append(row)

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if DUAL_MODEL_PARQUET.exists():
        old = pd.read_parquet(DUAL_MODEL_PARQUET)
        combined = pd.concat([old, new_df], ignore_index=True)
        keys = ["run_timestamp", "slate_date", "game_pk", "batter_mlbam_id", "pitcher_mlbam_id"]
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = new_df
    combined.to_parquet(DUAL_MODEL_PARQUET, index=False)
    return len(new_df)


def materialize_high_conf_tracking(min_conf: float | None = None) -> int:
    """Rewrite high-conf Parquet as a filtered view of the main tracking file.

    Prefers semantic High on any target (per-target calibrated cutoffs).
    Fallback for legacy rows without any label present: max(conf_*) >= threshold.
    """
    if not TRACKING_MAIN.exists():
        return 0
    df = pd.read_parquet(TRACKING_MAIN)
    if df.empty:
        TRACKING_HIGH_CONF.unlink(missing_ok=True)
        return 0

    semantic = pd.Series(False, index=df.index)
    labeled_row = pd.Series(False, index=df.index)
    for col in ("conf_hr_label", "conf_hit_label", "conf_xbh_label"):
        if col not in df.columns:
            continue
        semantic |= df[col] == "High"
        labeled_row |= df[col].notna()

    mn = HIGH_CONF_LEGACY_COMPOSITE_MIN if min_conf is None else min_conf
    mx = df[["conf_hr", "conf_hit", "conf_xbh"]].max(axis=1, skipna=True)
    legacy_fallback = (~labeled_row) & (mx >= mn)
    sub = df[(semantic) | legacy_fallback].copy()

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(TRACKING_HIGH_CONF, index=False)
    return len(sub)
