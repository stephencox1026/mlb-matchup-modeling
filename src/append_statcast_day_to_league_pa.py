#!/usr/bin/env python3
"""Fetch one calendar day of Statcast (all teams) and append to statcast_pa_level_league.parquet.

If Baseball Savant returns no rows for that date, falls back to MLB Stats API play-by-play
for Final games (same PA keys for dashboard / outcome fills).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate_bulk_pa import pitches_to_pa  # noqa: E402

RAW = ROOT / "data" / "raw"
LEAGUE_PA = RAW / "statcast_pa_level_league.parquet"


def _append_pa_new_to_league(pa_new: pd.DataFrame, day_s: str) -> int:
    pa_new = pa_new.copy()
    pa_new["game_date"] = pd.to_datetime(pa_new["game_date"], errors="coerce")
    if "game_year" not in pa_new.columns or pa_new["game_year"].isna().all():
        pa_new["game_year"] = pa_new["game_date"].dt.year

    base = pd.read_parquet(LEAGUE_PA)
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")

    for c in base.columns:
        if c not in pa_new.columns:
            pa_new[c] = pd.NA
    pa_new = pa_new[base.columns]

    merged = pd.concat([base, pa_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["game_pk", "at_bat_number", "batter"], keep="last")
    merged = merged.sort_values(["batter", "game_date", "at_bat_number"], kind="mergesort").reset_index(drop=True)

    merged.to_parquet(LEAGUE_PA, index=False)
    n = len(pa_new)
    print(f"  Appended {n:,} PAs from {day_s}. Total rows: {len(merged):,}")
    print(f"  Wrote → {LEAGUE_PA}")
    return n


def append_statcast_for_date(d: date) -> int:
    """
    Pull Statcast for one calendar day (regular season only), append to league PA parquet.
    Falls back to MLB Stats API when Savant returns no data for that calendar day.
    Returns number of PA rows appended (0 if none / skip).
    """
    day_s = d.isoformat()
    print(f"Fetching Statcast for {day_s} ...")
    pa_new = pd.DataFrame()

    try:
        from pybaseball import statcast
    except ImportError as e:
        raise SystemExit(f"pybaseball required: {e}") from e

    raw = statcast(start_dt=day_s, end_dt=day_s)
    if raw is not None and not raw.empty:
        if "game_type" in raw.columns:
            before = len(raw)
            raw = raw[raw["game_type"] == "R"].copy()
            print(f"  Filtered to regular season: {before:,} → {len(raw):,} pitches")

        if not raw.empty:
            pa_new = pitches_to_pa(raw)
            if pa_new.empty:
                print(f"  No PA-ending Statcast rows for {day_s}.")
        else:
            print(f"  No regular-season Statcast pitches for {day_s}.")
    else:
        print(f"  No Statcast pitch rows for {day_s}.")

    if pa_new.empty:
        print(f"  Trying MLB Stats API play-by-play for {day_s} ...")
        from mlb_statsapi_pa_day import fetch_pa_rows_from_statsapi

        pa_new = fetch_pa_rows_from_statsapi(d)
        if not pa_new.empty:
            print(f"  Stats API: {len(pa_new):,} plate appearances from Final games.")

    if pa_new.empty:
        print(f"  No PA rows for {day_s} — skip.")
        return 0

    return _append_pa_new_to_league(pa_new, day_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="YYYY-MM-DD (default: yesterday UTC/local calendar)",
    )
    args = p.parse_args()
    if args.date:
        d = date.fromisoformat(args.date)
    else:
        d = date.today() - timedelta(days=1)

    append_statcast_for_date(d)


if __name__ == "__main__":
    main()
