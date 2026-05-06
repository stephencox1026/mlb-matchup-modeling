"""Season zero-HR gate: season_ytd_hr_row_counts and floor behavior."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narrative_engine import (  # noqa: E402
    HR_ZERO_SEASON_GATE_FLOOR,
    season_ytd_hr_row_counts,
)


def test_season_counts_basic():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2026-04-01", "2026-04-02", "2026-04-02"]
            ),
            "batter": [1, 1, 2],
            "is_hr": [0, 0, 1],
        }
    )
    # Slate 2026-05-03: year 2026 before slate → bid 1: 2 rows 0 HR; bid 2: 1 row 1 HR
    out = season_ytd_hr_row_counts(df, "2026-05-03")
    assert out[1] == (2, 0)
    assert out[2] == (1, 1)


def test_season_counts_empty_year():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2025-06-01"]),
            "batter": [99],
            "is_hr": [1],
        }
    )
    out = season_ytd_hr_row_counts(df, "2026-05-03")
    assert out == {}


def test_floor_constant():
    assert HR_ZERO_SEASON_GATE_FLOOR == 1e-4
