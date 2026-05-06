#!/usr/bin/env python3
"""
Append Statcast PA rows for each calendar day in [start, end] into statcast_pa_level_league.parquet.

Uses pybaseball statcast per day (same as append_statcast_day_to_league_pa.py) with a pause
between requests to reduce throttling.

Example (2026 season through 4/19):
  python3 src/append_statcast_date_range.py --start 2026-03-25 --end 2026-04-19
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import append_statcast_day_to_league_pa as _sc_day  # noqa: E402


def daterange_inclusive(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--sleep", type=float, default=2.5, help="Seconds between Statcast calls (default 2.5)")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        print("ERROR: --end must be >= --start", file=sys.stderr)
        return 1

    total = 0
    days = list(daterange_inclusive(start, end))
    print(f"Backfill {len(days)} day(s): {start} .. {end}\n")

    for i, d in enumerate(days):
        print(f"[{i + 1}/{len(days)}] ", end="")
        try:
            n = _sc_day.append_statcast_for_date(d)
            total += n
        except Exception as e:
            print(f"  ERROR for {d}: {e}", file=sys.stderr)
            raise
        if i < len(days) - 1 and args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\nDone. Total new PA rows this run (sum per day): {total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
