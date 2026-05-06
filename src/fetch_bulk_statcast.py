"""
Bulk Statcast download for league-wide training data (2015-2025).

Uses pybaseball.statcast() to pull ALL pitches for ALL players per season,
cached as parquet files. Much more efficient than per-player pulls for 400+ batters.

Output: data/raw/statcast_bulk/statcast_{year}.parquet (one file per season)
"""
import time
import sys, os
from pathlib import Path
import pandas as pd
from pybaseball import statcast, cache

cache.enable()

sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DIR

BULK_DIR = RAW_DIR / "statcast_bulk"
BULK_DIR.mkdir(parents=True, exist_ok=True)

SEASON_WINDOWS = {
    2015: ("2015-04-05", "2015-11-05"),
    2016: ("2016-04-03", "2016-11-05"),
    2017: ("2017-04-02", "2017-11-05"),
    2018: ("2018-03-29", "2018-11-01"),
    2019: ("2019-03-20", "2019-11-01"),
    2020: ("2020-07-23", "2020-10-28"),
    2021: ("2021-04-01", "2021-11-03"),
    2022: ("2022-04-07", "2022-11-06"),
    2023: ("2023-03-30", "2023-11-02"),
    2024: ("2024-03-20", "2024-11-03"),
    2025: ("2025-03-18", "2025-11-02"),
    2026: ("2026-03-26", "2026-11-02"),
}


def download_season(year: int) -> pd.DataFrame | None:
    cache_path = BULK_DIR / f"statcast_{year}.parquet"
    if cache_path.exists():
        print(f"  {year}: loading from cache ({cache_path.stat().st_size / 1e6:.0f} MB)")
        return pd.read_parquet(cache_path)

    if year not in SEASON_WINDOWS:
        print(f"  {year}: no season window defined, skipping")
        return None

    start, end = SEASON_WINDOWS[year]
    print(f"  {year}: downloading {start} → {end} (this may take 10-30 min)...")

    try:
        df = statcast(start_dt=start, end_dt=end, verbose=True)
    except Exception as e:
        print(f"  {year}: ERROR - {e}")
        return None

    if df is None or df.empty:
        print(f"  {year}: no data returned")
        return None

    if "game_type" in df.columns:
        before = len(df)
        df = df[df["game_type"] == "R"].copy()
        print(f"  {year}: filtered to regular season: {before:,} → {len(df):,} pitches")

    df.to_parquet(cache_path, index=False)
    print(f"  {year}: saved {len(df):,} pitches ({cache_path.stat().st_size / 1e6:.0f} MB)")
    return df


def download_all(years: list[int] | None = None):
    if years is None:
        years = sorted(SEASON_WINDOWS.keys())

    print("=" * 60)
    print("  BULK STATCAST DOWNLOAD (League-Wide)")
    print("=" * 60)

    total_pitches = 0
    for yr in years:
        df = download_season(yr)
        if df is not None:
            total_pitches += len(df)
            print(f"  Running total: {total_pitches:,} pitches\n")
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE: {total_pitches:,} total pitches across {len(years)} seasons")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="Specific years to download (default: all 2015-2025)")
    args = parser.parse_args()
    download_all(args.years)
