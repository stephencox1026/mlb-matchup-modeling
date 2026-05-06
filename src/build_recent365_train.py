#!/usr/bin/env python3
"""
Build the recent365 training parquet by filtering the beast training set to
the last 365 days before train cutoff.

This is the foundation for the H5 ensemble re-test: same XGBoost architecture
as beast, but trained on a different data window. Tests whether data-window
diversity is the missing axis (vs the algorithmic diversity PyTorch would add).

Source: data/master/features_train_league_beast.parquet (1.7M rows, 2015-2024)
Output: data/master/features_train_league_recent365.parquet (~360K rows, 2024 only)

Val parquet stays the same (features_val_league.parquet covers 2025).

Usage:
  python3 src/build_recent365_train.py
  python3 src/build_recent365_train.py --days 365 --base beast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MASTER_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recent365 train parquet.")
    parser.add_argument("--base", default="beast",
                        help="Base parquet variant (beast / exp / league). Default: beast.")
    parser.add_argument("--days", type=int, default=365,
                        help="Days back from train cutoff to keep. Default: 365.")
    args = parser.parse_args()

    src = MASTER_DIR / f"features_train_league_{args.base}.parquet"
    if not src.exists():
        print(f"Source parquet missing: {src}", file=sys.stderr)
        sys.exit(1)
    out = MASTER_DIR / "features_train_league_recent365.parquet"

    print(f"Reading {src} ...")
    df = pd.read_parquet(src)
    if "game_date" not in df.columns:
        print("Missing game_date column; cannot filter.", file=sys.stderr)
        sys.exit(1)
    df["game_date"] = pd.to_datetime(df["game_date"])
    train_max = df["game_date"].max()
    cutoff = train_max - pd.Timedelta(days=args.days - 1)
    pre_n = len(df)
    df_recent = df[df["game_date"] >= cutoff].copy()
    print(f"Train cutoff: {train_max.date()}")
    print(f"Filter: game_date >= {cutoff.date()} ({args.days} days)")
    print(f"Rows: {pre_n:,} -> {len(df_recent):,} ({100*len(df_recent)/pre_n:.1f}% retained)")
    print(f"Date range retained: {df_recent['game_date'].min().date()} -> "
          f"{df_recent['game_date'].max().date()}")

    df_recent.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
