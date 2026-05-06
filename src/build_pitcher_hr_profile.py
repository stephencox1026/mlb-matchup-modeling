#!/usr/bin/env python3
"""
T2.2: build per-pitcher-per-season HR profile features.

For each (pitcher, season), compute the characteristics of HRs allowed:
  p_hr_la_mean             — mean launch angle on HRs allowed
  p_hr_ev_mean             — mean exit velocity on HRs allowed
  p_hr_pulled_pct          — % of HRs allowed that were pulled
  p_hr_zone_rate           — % of HRs allowed on in-zone pitches (mistake rate)
  p_hr_count_advantage_pct — % of HRs allowed when pitcher was ahead in count

These let the model learn that some pitchers give up flyball-only HRs (Wrigley
wind dependent) vs line-drive HRs (less park-sensitive), pull-side bombs (high
wind dependence) vs spread (low wind dependence), mistake HRs vs chase HRs.

Empirical-Bayes shrinkage to league mean per quantity using inverse-variance
weighting with a prior pseudo-HR count of 30 (so a pitcher needs ~10 HRs allowed
before their personal rate dominates).

Output: data/raw/pitcher_hr_profile.parquet keyed on (pitcher, season).

Usage:
  python3 src/build_pitcher_hr_profile.py
  python3 src/build_pitcher_hr_profile.py --years 2022 2023 2024 2025
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

OUT_PATH = RAW_DIR / "pitcher_hr_profile.parquet"
BULK_DIR = RAW_DIR / "statcast_bulk"

PRIOR_HR = 30.0  # pseudo-HR count for EB shrinkage


def _load_year_hrs(year: int) -> pd.DataFrame:
    p = BULK_DIR / f"statcast_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p, columns=[
        "pitcher", "events",
        "launch_speed", "launch_angle", "zone",
        "balls", "strikes", "hc_x", "hc_y", "stand",
    ])
    hrs = df[df["events"] == "home_run"].copy()
    hrs["season"] = year
    return hrs


def _is_pulled(row) -> int:
    """Determine if a HR was pulled given hc_x, hc_y, batter handedness.

    Statcast hc_x/hc_y are in the batting-coordinate frame:
      - home plate is at approximately (125, 200) in standard Statcast coords
      - higher hc_y = farther from home plate
      - hc_x: to the right of home plate (3B side) is HIGHER for RH batters
        meaning a pulled ball for a righty has hc_x < ~125
      - For LH batters, pulled means hc_x > ~125 (RF side)

    Use the angle from home plate; classify pull as the third of the field
    nearest the batter's pull side.
    """
    hc_x = row.get("hc_x")
    hc_y = row.get("hc_y")
    stand = row.get("stand")
    if pd.isna(hc_x) or pd.isna(hc_y) or pd.isna(stand):
        return 0
    dx = float(hc_x) - 125.42  # approximate home plate x
    dy = 200.0 - float(hc_y)   # invert; positive = into the field
    if dy <= 0:
        return 0
    angle = np.degrees(np.arctan2(dx, dy))  # 0 = CF, +ve = RF, -ve = LF
    # Right field: angle > +15deg; Left field: angle < -15deg; CF in between.
    if stand == "L":
        return 1 if angle > 15 else 0  # lefty pulls to RF
    else:  # RH
        return 1 if angle < -15 else 0  # righty pulls to LF


def _is_pitcher_ahead(row) -> int:
    """Pitcher-ahead counts: 0-1, 0-2, 1-2."""
    b = row.get("balls")
    s = row.get("strikes")
    if pd.isna(b) or pd.isna(s):
        return 0
    return 1 if (s > b) else 0


def _is_in_zone(row) -> int:
    z = row.get("zone")
    if pd.isna(z):
        return 0
    return 1 if 1 <= int(z) <= 9 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-pitcher HR profile features.")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025])
    args = parser.parse_args()

    frames = []
    for y in args.years:
        f = _load_year_hrs(y)
        if not f.empty:
            print(f"  {y}: {len(f):,} HRs allowed")
            frames.append(f)
    if not frames:
        print("No bulk parquets; nothing to do.", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)

    print("Computing per-HR row tags ...")
    df["pulled"] = df.apply(_is_pulled, axis=1).astype(int)
    df["pitcher_ahead"] = df.apply(_is_pitcher_ahead, axis=1).astype(int)
    df["in_zone"] = df.apply(_is_in_zone, axis=1).astype(int)

    # League means for EB shrinkage
    league = {
        "la_mean": float(df["launch_angle"].dropna().mean()),
        "ev_mean": float(df["launch_speed"].dropna().mean()),
        "pulled_pct": float(df["pulled"].mean()),
        "zone_rate": float(df["in_zone"].mean()),
        "count_advantage_pct": float(df["pitcher_ahead"].mean()),
    }
    print(f"League HR profile (means): {league}")

    rows = []
    for (pid, season), g in df.groupby(["pitcher", "season"]):
        n = len(g)
        # EB shrinkage to league mean for each quantity
        def shrink(value, prior):
            return (value * n + prior * PRIOR_HR) / (n + PRIOR_HR)

        la_raw = float(g["launch_angle"].dropna().mean()) if g["launch_angle"].notna().any() else league["la_mean"]
        ev_raw = float(g["launch_speed"].dropna().mean()) if g["launch_speed"].notna().any() else league["ev_mean"]
        pulled_raw = float(g["pulled"].mean())
        zone_raw = float(g["in_zone"].mean())
        ahead_raw = float(g["pitcher_ahead"].mean())

        rows.append({
            "pitcher": int(pid),
            "season": int(season),
            "n_hr_allowed": int(n),
            "p_hr_la_mean": round(shrink(la_raw, league["la_mean"]), 3),
            "p_hr_ev_mean": round(shrink(ev_raw, league["ev_mean"]), 3),
            "p_hr_pulled_pct": round(shrink(pulled_raw, league["pulled_pct"]), 4),
            "p_hr_zone_rate": round(shrink(zone_raw, league["zone_rate"]), 4),
            "p_hr_count_advantage_pct": round(shrink(ahead_raw, league["count_advantage_pct"]), 4),
        })

    out = pd.DataFrame(rows).sort_values(["pitcher", "season"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n=== Sample (high HR-allowed pitchers from latest season) ===")
    latest = out[out["season"] == out["season"].max()].sort_values("n_hr_allowed", ascending=False).head(10)
    print(latest.to_string(index=False))
    print(f"\nWrote {OUT_PATH} ({len(out):,} pitcher-season rows)")


if __name__ == "__main__":
    main()
