"""
Aggregate bulk Statcast pitch-level data into PA-level rows for league-wide training.

Reads cached season parquets from data/raw/statcast_bulk/,
applies pitches_to_pa() logic, and outputs a single league-wide PA file.

Output: data/raw/statcast_pa_level_league.parquet
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from pathlib import Path
from config import RAW_DIR

BULK_DIR = RAW_DIR / "statcast_bulk"

HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "hit_by_pitch", "intent_walk"}


def pitches_to_pa(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    pa_pitches = df[df["events"].notna() & (df["events"] != "")].copy()
    if pa_pitches.empty:
        return pd.DataFrame()

    keep_cols = [
        "game_pk", "game_date", "game_year", "at_bat_number", "batter",
        "pitcher", "events", "description", "p_throws", "stand",
        "home_team", "away_team", "venue",
        "launch_speed", "launch_angle", "launch_speed_angle",
        "hit_distance_sc", "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom",
        "babip_value", "iso_value", "barrel",
        "pitch_number", "zone", "plate_x", "plate_z",
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
        "player_name",
        "pitch_type", "pitch_name", "effective_speed",
        "release_extension", "arm_angle", "spin_axis",
        "n_thruorder_pitcher", "pitcher_days_since_prev_game", "age_pit",
    ]
    available = [c for c in keep_cols if c in pa_pitches.columns]
    pa = pa_pitches[available].copy()

    pitch_summary = df.groupby(["game_pk", "at_bat_number", "batter"]).agg(
        pa_mean_velo=("release_speed", "mean"),
        pa_max_velo=("release_speed", "max"),
        pa_pitch_count=("pitch_number", "max"),
    ).reset_index()
    pa = pa.merge(pitch_summary, on=["game_pk", "at_bat_number", "batter"], how="left")

    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_ab"] = (~pa["events"].isin(WALK_EVENTS | {"catcher_interf", "sac_bunt"})).astype(int)
    pa["is_strikeout"] = pa["events"].str.contains("strikeout", case=False, na=False).astype(int)
    pa["is_walk"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    pa["is_hbp"] = (pa["events"] == "hit_by_pitch").astype(int)
    pa["is_xbh"] = pa["events"].isin({"double", "triple", "home_run"}).astype(int)
    pa["vs_lhp"] = (pa["p_throws"] == "L").astype(int) if "p_throws" in pa.columns else 0
    # Statcast often leaves `barrel` empty on PA-ending rows; LSA bucket 6 is the barrel classification.
    if "launch_speed_angle" in pa.columns:
        # Nullable Int64: (lsa == 6).astype(int) raises on NA; use fillna after eq.
        lsa_barrel = pa["launch_speed_angle"].eq(6).fillna(False).astype(int)
    else:
        lsa_barrel = pd.Series(0, index=pa.index, dtype=int)
    if "barrel" in pa.columns:
        bcol = pa["barrel"].fillna(0).astype(int)
        pa["barrel"] = ((bcol > 0) | (lsa_barrel > 0)).astype(int)
    else:
        pa["barrel"] = lsa_barrel.astype(int)

    if "player_name" in pa.columns:
        pa = pa.rename(columns={"player_name": "player_name_clean"})

    return pa


def main():
    print("=" * 60)
    print("  AGGREGATE BULK STATCAST → PA-LEVEL")
    print("=" * 60)

    parquet_files = sorted(BULK_DIR.glob("statcast_*.parquet"))
    if not parquet_files:
        print("ERROR: No bulk statcast parquet files found")
        return

    print(f"Found {len(parquet_files)} season files")

    all_pa = []
    for f in parquet_files:
        year = f.stem.replace("statcast_", "")
        print(f"\n  Processing {year}...")
        df = pd.read_parquet(f)
        print(f"    {len(df):,} pitches loaded")

        pa = pitches_to_pa(df)
        if not pa.empty:
            all_pa.append(pa)
            print(f"    → {len(pa):,} PAs")
        else:
            print(f"    → 0 PAs")

    if not all_pa:
        print("\nERROR: No PAs extracted")
        return

    combined = pd.concat(all_pa, ignore_index=True)
    combined = combined.sort_values(["batter", "game_date", "at_bat_number"]).reset_index(drop=True)

    out = RAW_DIR / "statcast_pa_level_league.parquet"
    combined.to_parquet(out, index=False)

    print(f"\n{'='*60}")
    print(f"  PA AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total PAs: {len(combined):,}")
    print(f"Unique batters: {combined['batter'].nunique():,}")

    if "game_year" in combined.columns:
        print(f"\nPAs by year:")
        print(combined.groupby("game_year").size().to_string())
        train = combined[combined["game_year"] <= 2024]
        val = combined[combined["game_year"] == 2025]
        print(f"\nTrain (2015-2024): {len(train):,} PAs")
        print(f"Val   (2025):      {len(val):,} PAs")

    print(f"\nHit rate:  {combined['is_hit'].mean():.3f}")
    print(f"HR rate:   {combined['is_hr'].mean():.4f}")
    print(f"K rate:    {combined['is_strikeout'].mean():.3f}")
    print(f"BB rate:   {combined['is_walk'].mean():.3f}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
