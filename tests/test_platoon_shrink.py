"""Tests for platoon_raw_probability_shrink and HR/XBH/Hit ordering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narrative_engine import (  # noqa: E402
    PLATOON_EXTREME_ELIGIBLE_MIN_PA,
    PLATOON_PENALTY_MIN_PA,
    _enforce_hit_xbh_hr_order,
    platoon_raw_probability_shrink,
)


def _base_preds(hit=0.33, hr=0.08, xbh=0.12):
    return {
        "is_hit": hit,
        "is_hr": hr,
        "is_xbh": xbh,
        "is_strikeout": 0.22,
        "is_walk": 0.08,
    }


def test_platoon_raw_shrink_skips_when_pa_too_thin():
    snap = {
        "platoon_matched_ba": 0.096,
        "cum_career_pa_vs_lhp": PLATOON_EXTREME_ELIGIBLE_MIN_PA - 1,
        "platoon_matched_ops": 0.35,
    }
    preds = _base_preds()
    adj, audit = platoon_raw_probability_shrink(preds, snap, "L")
    assert adj["is_hit"] == preds["is_hit"]
    assert audit["platoon_raw_reason"] == "none"


def test_platoon_raw_shrink_extreme_weak_side_mid_pa():
    """PA between EXTREME and full min: dreadful BA still triggers catastrophic shrink."""
    snap = {
        "platoon_matched_ba": 0.096,
        "cum_career_pa_vs_lhp": 25.2,
        "platoon_matched_ops": 0.254,
    }
    preds = _base_preds(hit=0.35, hr=0.08, xbh=0.12)
    adj, audit = platoon_raw_probability_shrink(preds, snap, "L")
    assert "ba_catastrophic" in audit["platoon_raw_reason"]
    assert adj["is_hr"] < preds["is_hr"]
    assert audit["platoon_raw_hr_mult"] < 1.0


def test_platoon_raw_shrink_below_full_min_but_marginal_ba_skips():
    snap = {
        "platoon_matched_ba": 0.220,
        "cum_career_pa_vs_lhp": 40.0,
        "platoon_matched_ops": 0.70,
    }
    preds = _base_preds()
    adj, audit = platoon_raw_probability_shrink(preds, snap, "L")
    assert audit["platoon_raw_reason"] == "none"


def test_platoon_raw_shrink_catastrophic_vs_lhp_high_pa():
    snap = {
        "platoon_matched_ba": 0.096,
        "cum_career_pa_vs_lhp": 250.0,
        "cum_career_pa_vs_rhp": 800.0,
        "platoon_matched_ops": 0.35,
    }
    preds = _base_preds(hit=0.35, hr=0.08, xbh=0.12)
    adj, audit = platoon_raw_probability_shrink(preds, snap, "L")
    assert audit["platoon_raw_reason"] != "none"
    assert "ba_catastrophic" in audit["platoon_raw_reason"]
    assert adj["is_hit"] < preds["is_hit"]
    assert adj["is_xbh"] < preds["is_xbh"]
    assert adj["is_hr"] < preds["is_hr"]
    assert adj["is_hr"] < adj["is_xbh"] < adj["is_hit"]


def test_enforce_order_after_extreme_shrink():
    preds = {"is_hit": 0.1, "is_hr": 0.05, "is_xbh": 0.02}
    _enforce_hit_xbh_hr_order(preds)
    assert preds["is_hr"] <= preds["is_xbh"] <= preds["is_hit"]
    assert preds["is_xbh"] == preds["is_hr"]


def test_ops_only_stack_when_ba_marginal_but_ops_awful():
    snap = {
        "platoon_matched_ba": 0.21,
        "cum_career_pa_vs_lhp": 80.0,
        "platoon_matched_ops": 0.35,
    }
    preds = _base_preds()
    adj, audit = platoon_raw_probability_shrink(preds, snap, "L")
    assert "ops_hard" in audit["platoon_raw_reason"]
    assert adj["is_hit"] == preds["is_hit"]
    assert adj["is_xbh"] < preds["is_xbh"]
    assert adj["is_hr"] < preds["is_hr"]
    _enforce_hit_xbh_hr_order(adj)
    assert adj["is_hr"] <= adj["is_xbh"] <= adj["is_hit"]
