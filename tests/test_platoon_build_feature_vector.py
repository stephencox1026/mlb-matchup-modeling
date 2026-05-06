"""Tests for platoon-aligned recent medians, throws resolution, confidence fade."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narrative_engine import (  # noqa: E402
    _median_recent_features_aligned,
    platoon_quality_multiplier,
    resolve_pitcher_throws,
)


def test_median_recent_prefers_matching_platoon_segment():
    rows = []
    for i in range(20):
        rows.append({
            "game_date": pd.Timestamp(f"2024-06-{i + 1:02d}"),
            "vs_lhp": 1,
            "roll_ba_10": 0.50,
        })
    for i in range(20):
        rows.append({
            "game_date": pd.Timestamp(f"2024-07-{i + 1:02d}"),
            "vs_lhp": 0,
            "roll_ba_10": 0.10,
        })
    df = pd.DataFrame(rows)
    feat_cols = ["roll_ba_10", "vs_lhp"]
    med_l = _median_recent_features_aligned(df, feat_cols, 1)
    assert abs(float(med_l["roll_ba_10"]) - 0.50) < 1e-9

    med_r = _median_recent_features_aligned(df, feat_cols, 0)
    assert abs(float(med_r["roll_ba_10"]) - 0.10) < 1e-9


def test_resolve_pitcher_throws_schedule_hint_wins_profile():
    pitchers = pd.DataFrame([{"pitcher": 9001, "season": 2025, "throws": "R"}])
    assert resolve_pitcher_throws(9001, pitchers, "L") == "L"
    assert resolve_pitcher_throws(9001, pitchers, None) == "R"
    empty = pitchers.iloc[:0].copy()
    assert resolve_pitcher_throws(9999, empty, None) == "R"


def test_platoon_quality_multiplier_sparse_bad_side_l():
    snap = {
        "platoon_matched_ba": 0.18,
        "cum_career_pa_vs_lhp": 55.0,
        "cum_career_pa_vs_rhp": 900.0,
        "cum_career_ba_vs_lhp": 0.18,
        "cum_career_ba_vs_rhp": 0.260,
    }
    mult, detail = platoon_quality_multiplier(snap, "L")
    assert mult < 1.0
    assert detail["platoon_quality_reason"] != "none"


def test_platoon_quality_multiplier_no_penalty_when_rhp_line_strong():
    """No platoon_matched_* so RHP side uses cumulative RHP splits only."""
    snap = {
        "cum_career_pa_vs_lhp": 55.0,
        "cum_career_pa_vs_rhp": 900.0,
        "cum_career_ba_vs_lhp": 0.18,
        "cum_career_ba_vs_rhp": 0.260,
    }
    mult, detail = platoon_quality_multiplier(snap, "R")
    assert mult == 1.0
    assert detail["platoon_quality_reason"] == "none"


def test_platoon_quality_high_pa_weak_platoon_applies():
    snap = {
        "cum_career_ba_vs_lhp": 0.155,
        "cum_career_pa_vs_lhp": 220,
    }
    mult, detail = platoon_quality_multiplier(snap, "L")
    assert mult <= 0.82
    assert "hi_pa" in detail["platoon_quality_reason"] or "vbad" in detail["platoon_quality_reason"]
