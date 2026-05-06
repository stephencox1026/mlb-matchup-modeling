"""Unit tests for per-target confidence tiers, XBH blend, and convergence flags."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narrative_engine import (  # noqa: E402
    CONF_CONV_MODEL_K_HR,
    CONF_LABEL_HR,
    compute_confidence,
    confidence_label_for_target,
    graded_convergence_factor,
    LEAGUE_AVG,
)


def test_confidence_label_for_target_triples():
    assert confidence_label_for_target(1.12, "hit") == "High"
    assert confidence_label_for_target(1.10, "hr") == "Medium"  # below CONF_LABEL_HR[0]
    assert confidence_label_for_target(CONF_LABEL_HR[0], "hr") == "High"
    assert confidence_label_for_target(1.05, "xbh") == "Medium"


def test_conv_model_floor_hr_stricter_than_legacy():
    assert CONF_CONV_MODEL_K_HR * LEAGUE_AVG["hr"] >= 0.049


def test_conf_xbh_is_bounded_by_hr_hit_composites():
    """conf_xbh ≤ min(conf_hr, conf_hit); uses XBH convergence in the composite."""
    pred = {"is_hr": 0.055, "is_hit": 0.28, "is_xbh": 0.11}
    cr = {"hr": 0.05, "hit": 0.26, "xbh": 0.09}
    bvp = {
        "bvp_pa": 30,
        "bvp_hr": 2,
        "bvp_hits": 10,
        "bvp_xbh": 3,
        "bvp_ba": 0.30,
    }
    out = compute_confidence(
        pred,
        {"k_rate": 0.22},
        bvp,
        cr,
        pitcher_pa_total=2000,
        val_df_last_date=pd.Timestamp.now(),
        feature_snapshot=None,
        pitcher_throws="R",
    )
    assert out["conf_xbh"] <= out["conf_hr"] + 1e-6
    assert out["conf_xbh"] <= out["conf_hit"] + 1e-6
    assert "convergence_xbh" in out["factors"]


def test_graded_convergence_endpoints_match_legacy_extremes():
    n = 3
    lo = graded_convergence_factor(0, n)
    hi = graded_convergence_factor(n, n)
    assert abs(lo - 0.85) < 1e-9
    assert abs(hi - 1.04) < 1e-9


def test_graded_convergence_midpoint_between_extremes():
    n = 2
    mid = graded_convergence_factor(1, n)
    assert mid > graded_convergence_factor(0, n)
    assert mid < graded_convergence_factor(n, n)


def test_convergence_xbh_in_factors():
    pred = {"is_hr": 0.04, "is_hit": 0.23, "is_xbh": 0.08}
    cr = {"hr": 0.04, "hit": 0.23, "xbh": 0.078}
    out = compute_confidence(
        pred,
        {"k_rate": 0.22},
        None,
        cr,
        pitcher_pa_total=500,
        val_df_last_date=pd.Timestamp.now(),
        feature_snapshot=None,
        pitcher_throws="R",
    )
    assert "convergence_xbh" in out["factors"]
