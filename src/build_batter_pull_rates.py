#!/usr/bin/env python3
"""
T2.3 prereq: build per-batter pull rate on HR contact + handedness lookup.

For each batter, compute the fraction of their HRs that were pulled (using
hc_x / hc_y / stand). Schwarber and other pull-power LHB will land near 0.75;
spread-field bats like Aaron Judge land near 0.45.

Output: data/raw/batter_pull_rates.parquet
  batter, name, stand, n_hr, pull_rate
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

OUT_PATH = RAW_DIR / "batter_pull_rates.parquet"
BULK_DIR = RAW_DIR / "statcast_bulk"

LEAGUE_PULL_RATE = 0.66
PRIOR_HR = 30.0  # pseudo-HR count for EB shrinkage


def _is_pulled(row) -> int:
    hc_x = row.get("hc_x")
    hc_y = row.get("hc_y")
    stand = row.get("stand")
    if pd.isna(hc_x) or pd.isna(hc_y) or pd.isna(stand):
        return 0
    dx = float(hc_x) - 125.42
    dy = 200.0 - float(hc_y)
    if dy <= 0:
        return 0
    angle = np.degrees(np.arctan2(dx, dy))
    if stand == "L":
        return 1 if angle > 15 else 0
    return 1 if angle < -15 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-batter HR pull rates.")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025])
    args = parser.parse_args()

    frames = []
    for y in args.years:
        p = BULK_DIR / f"statcast_{y}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["batter", "events", "stand",
                                            "hc_x", "hc_y", "player_name"])
        hrs = df[df["events"] == "home_run"].copy()
        if not hrs.empty:
            print(f"  {y}: {len(hrs):,} HRs")
            frames.append(hrs)
    if not frames:
        sys.exit("No bulk parquets")
    df = pd.concat(frames, ignore_index=True)
    df["pulled"] = df.apply(_is_pulled, axis=1).astype(int)

    rows = []
    for bid, g in df.groupby("batter"):
        n = len(g)
        raw = float(g["pulled"].mean())
        shrunk = (raw * n + LEAGUE_PULL_RATE * PRIOR_HR) / (n + PRIOR_HR)
        stand_mode = g["stand"].mode()
        stand = str(stand_mode.iloc[0]) if not stand_mode.empty else "R"
        name_mode = g["player_name"].mode() if "player_name" in g.columns else pd.Series(dtype=str)
        name = str(name_mode.iloc[0]) if not name_mode.empty else ""
        rows.append({
            "batter": int(bid),
            "name": name,
            "stand": stand,
            "n_hr": int(n),
            "pull_rate_raw": round(raw, 4),
            "pull_rate": round(shrunk, 4),
        })

    out = pd.DataFrame(rows).sort_values("n_hr", ascending=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n=== Top 10 pull-power bats ===")
    print(out.head(10).to_string(index=False))
    print(f"\nWrote {OUT_PATH} ({len(out):,} batters)")


if __name__ == "__main__":
    main()
