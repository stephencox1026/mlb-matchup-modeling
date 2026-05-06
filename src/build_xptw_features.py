#!/usr/bin/env python3
"""
M4: Pitch-type-weighted batter feature builder (xptw_*).

For each (batter, pitcher, season) triple, compute pitcher-pitch-usage-weighted
batter performance per pitch class:

  xptw_p_barrel = sum over pitch_class (pct_pitcher_uses * barrel_vs_class)
  xptw_p_ba     = sum over pitch_class (pct_pitcher_uses * ba_vs_class)
  xptw_p_ev     = sum over pitch_class (pct_pitcher_uses * ev_vs_class)
  xptw_p_whiff  = sum over pitch_class (pct_pitcher_uses * whiff_vs_class)

These features capture the matchup-specific expected batter outcome by mixing
the pitcher's arsenal with the batter's per-pitch-type performance. A batter
who slugs sinkers facing a sinker-heavy pitcher should score high on
xptw_p_barrel.

Output: data/raw/xptw_features.parquet keyed on (batter, pitcher, season).

Usage:
  python3 src/build_xptw_features.py
  python3 src/build_xptw_features.py --season 2025

Next step (documented, not done here):
  Wire `xptw_*` columns into src/features.py via a left-join on
  (batter, pitcher, season). Add `xptw_` to FEATURE_PREFIXES in
  src/train_multi_target.py and retrain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR  # noqa: E402

OUT_PATH = RAW_DIR / "xptw_features.parquet"

PITCH_CLASSES = ["fastball", "breaking", "offspeed"]

# (batter side stat, output column suffix)
# T2.1: hr_rate_vs_* added; barrel kept as a secondary signal for back-compat.
BATTER_STAT_COLS = [
    ("hr_rate", "xptw_p_hr_rate"),
    ("barrel", "xptw_p_barrel"),
    ("ba", "xptw_p_ba"),
    ("ev", "xptw_p_ev"),
    ("whiff", "xptw_p_whiff"),
]


def _norm_pcts(p: pd.Series, b: pd.Series, o: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Normalize pct_* columns so they sum to 1.0 per pitcher row.

    Pitcher profiles sometimes have <100% sum because of "other" pitch types
    (eephus, knuckle). Normalize to make weights well-defined.
    """
    p = p.fillna(0.0)
    b = b.fillna(0.0)
    o = o.fillna(0.0)
    s = p + b + o
    s = s.where(s > 0, other=1.0)  # avoid divide-by-zero
    return p / s, b / s, o / s


def build(seasons: list[int] | None = None) -> pd.DataFrame:
    bpt = pd.read_parquet(RAW_DIR / "batter_pitch_profiles.parquet")
    pp = pd.read_parquet(RAW_DIR / "pitcher_profiles_by_season.parquet")

    if seasons:
        bpt = bpt[bpt["season"].isin(seasons)]
        pp = pp[pp["season"].isin(seasons)]

    p_ff, p_br, p_off = _norm_pcts(pp["pct_fastball"], pp["pct_breaking"], pp["pct_offspeed"])
    pp = pp.copy()
    pp["_w_ff"] = p_ff
    pp["_w_br"] = p_br
    pp["_w_off"] = p_off

    # Cross-join on season; restrict to overlapping (batter, pitcher) pairs would be
    # huge. Instead build the cross-product per season then join callers separately.
    out_rows = []
    for season in sorted(set(bpt["season"]).intersection(set(pp["season"]))):
        b = bpt[bpt["season"] == season].copy()
        p = pp[pp["season"] == season].copy()
        # Cross join (batter × pitcher in this season). Drop overlapping `season`
        # column from the pitcher side before merge so the result has a single one.
        b["_key"] = 1
        p_renamed = p.drop(columns=["season"]).copy()
        p_renamed["_key"] = 1
        cj = b.merge(p_renamed, on="_key", suffixes=("_bat", "_pit")).drop(columns=["_key"])

        for stat, out_col in BATTER_STAT_COLS:
            ff = cj[f"{stat}_vs_fastball"].fillna(0.0)
            br = cj[f"{stat}_vs_breaking"].fillna(0.0)
            of = cj[f"{stat}_vs_offspeed"].fillna(0.0)
            cj[out_col] = cj["_w_ff"] * ff + cj["_w_br"] * br + cj["_w_off"] * of

        # Keep only the join keys + xptw cols; drop big intermediate columns.
        keep = ["batter", "pitcher", "season"] + [c for _, c in BATTER_STAT_COLS]
        out_rows.append(cj[keep])

    if not out_rows:
        return pd.DataFrame(columns=["batter", "pitcher", "season"] +
                                       [c for _, c in BATTER_STAT_COLS])
    return pd.concat(out_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pitch-type-weighted batter features.")
    parser.add_argument("--season", type=int, action="append",
                        help="Restrict to specific season(s); pass multiple times.")
    args = parser.parse_args()

    df = build(seasons=args.season)
    print(f"xptw rows: {len(df):,}")
    print(df.head(5))
    print()
    print("Stats:")
    print(df.describe().round(4))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
