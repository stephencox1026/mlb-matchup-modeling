"""Tests for C1 calibration bucket helpers (quantile vs label fallback)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibrate_predictions import quantile_calibration_bucket  # noqa: E402


def test_quantile_bucket_labels_align_with_four_quartiles():
    edges = [0.76, 0.88, 0.93]
    assert quantile_calibration_bucket(0.50, edges) == "Very Low"
    assert quantile_calibration_bucket(0.77, edges) == "Low"
    assert quantile_calibration_bucket(0.90, edges) == "Medium"
    assert quantile_calibration_bucket(0.93, edges) == "Medium"
    assert quantile_calibration_bucket(0.9301, edges) == "High"
