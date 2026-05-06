"""
Build per-pitcher-per-season profiles from cached Statcast bulk data.
No new downloads needed — reads from data/raw/statcast_bulk/.

Output: data/raw/pitcher_profiles_by_season.parquet
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config import RAW_DIR

BULK_DIR = RAW_DIR / "statcast_bulk"

FASTBALL = {"FF", "SI", "FC", "FA"}
BREAKING = {"SL", "ST", "CU", "SV", "KC", "CS"}
OFFSPEED = {"CH", "FS"}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWING_DESC = WHIFF_DESC | {"foul", "foul_bunt", "hit_into_play", "hit_into_play_no_out",
                            "hit_into_play_score"}


def classify_pitch_group(pitch_type):
    if pitch_type in FASTBALL:
        return "fastball"
    elif pitch_type in BREAKING:
        return "breaking"
    elif pitch_type in OFFSPEED:
        return "offspeed"
    return "other"


def build_profiles_for_season(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["pitch_group"] = df["pitch_type"].map(classify_pitch_group)
    df["is_whiff"] = df["description"].isin(WHIFF_DESC).astype(int)
    df["is_swing"] = df["description"].isin(SWING_DESC).astype(int)
    df["is_chase"] = ((~df["zone"].fillna(0).between(1, 9)) & (df["is_swing"] == 1)).astype(int)

    pa_pitches = df[df["events"].notna() & (df["events"] != "")]
    pa_pitches = pa_pitches.copy()
    pa_pitches["is_hit"] = pa_pitches["events"].isin(HIT_EVENTS).astype(int)
    pa_pitches["is_k"] = pa_pitches["events"].str.contains("strikeout", case=False, na=False).astype(int)
    pa_pitches["is_bb"] = pa_pitches["events"].isin(BB_EVENTS).astype(int)
    pa_pitches["is_hr"] = (pa_pitches["events"] == "home_run").astype(int)
    pa_pitches["is_barrel"] = pa_pitches["barrel"].fillna(0).astype(int) if "barrel" in pa_pitches.columns else 0

    rows = []
    for pid, grp in df.groupby("pitcher"):
        pa_grp = pa_pitches[pa_pitches["pitcher"] == pid]
        n_pitches = len(grp)
        n_pa = len(pa_grp)
        if n_pa < 10:
            continue

        row = {"pitcher": pid, "season": year, "n_pitches": n_pitches, "n_pa": n_pa}

        for pg in ["fastball", "breaking", "offspeed"]:
            mask = grp["pitch_group"] == pg
            row[f"pct_{pg}"] = mask.mean() if n_pitches > 0 else 0.0
            subset = grp[mask]
            row[f"velo_{pg}"] = subset["release_speed"].mean() if len(subset) > 0 else np.nan
            row[f"spin_{pg}"] = subset["release_spin_rate"].mean() if len(subset) > 0 else np.nan
            row[f"pfx_x_{pg}"] = subset["pfx_x"].mean() if len(subset) > 0 else np.nan
            row[f"pfx_z_{pg}"] = subset["pfx_z"].mean() if len(subset) > 0 else np.nan

        row["velo_overall"] = grp["release_speed"].mean()
        row["spin_overall"] = grp["release_spin_rate"].mean()
        row["arm_angle"] = grp["arm_angle"].mean() if "arm_angle" in grp.columns else np.nan
        row["extension"] = grp["release_extension"].mean() if "release_extension" in grp.columns else np.nan

        zone_mask = grp["zone"].fillna(0).between(1, 9)
        row["pct_in_zone"] = zone_mask.mean() if n_pitches > 0 else 0.0
        chase_mask = grp["zone"].isin([11, 12, 13, 14])
        row["pct_chase_zone"] = (chase_mask & (grp["is_swing"] == 1)).sum() / max(chase_mask.sum(), 1)

        row["whiff_rate"] = grp["is_whiff"].sum() / max(grp["is_swing"].sum(), 1)

        row["k_rate"] = pa_grp["is_k"].mean()
        row["bb_rate"] = pa_grp["is_bb"].mean()
        row["hit_rate_allowed"] = pa_grp["is_hit"].mean()
        row["hr_rate_allowed"] = pa_grp["is_hr"].mean()
        row["barrel_rate_allowed"] = pa_grp["is_barrel"].mean()

        if "estimated_woba_using_speedangle" in pa_grp.columns:
            row["xwoba_allowed"] = pa_grp["estimated_woba_using_speedangle"].mean()
        else:
            row["xwoba_allowed"] = np.nan

        if "delta_pitcher_run_exp" in grp.columns:
            row["run_value_per_100"] = grp["delta_pitcher_run_exp"].sum() / max(n_pitches, 1) * 100

        if "p_throws" in grp.columns:
            row["throws"] = grp["p_throws"].mode().iloc[0] if not grp["p_throws"].mode().empty else "R"

        if "age_pit" in grp.columns:
            row["age"] = grp["age_pit"].mode().iloc[0] if not grp["age_pit"].mode().empty else np.nan

        if "n_thruorder_pitcher" in grp.columns:
            row["avg_times_thru_order"] = grp.groupby(["game_pk", "batter"])["n_thruorder_pitcher"].first().mean()

        if "pitcher_days_since_prev_game" in grp.columns:
            row["avg_rest_days"] = grp.groupby("game_pk")["pitcher_days_since_prev_game"].first().mean()

        top_types = grp["pitch_type"].value_counts(normalize=True).head(5)
        for i, (pt, pct) in enumerate(top_types.items()):
            row[f"top_pitch_{i+1}"] = pt
            row[f"top_pitch_{i+1}_pct"] = round(pct, 3)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  BUILD PITCHER PROFILES (from cached Statcast)")
    print("=" * 60)

    parquet_files = sorted(BULK_DIR.glob("statcast_*.parquet"))
    print(f"Found {len(parquet_files)} season files\n")

    all_profiles = []
    for f in parquet_files:
        year = int(f.stem.replace("statcast_", ""))
        print(f"  {year}: loading...", end=" ", flush=True)
        df = pd.read_parquet(f)
        print(f"{len(df):,} pitches →", end=" ", flush=True)
        profiles = build_profiles_for_season(df, year)
        all_profiles.append(profiles)
        print(f"{len(profiles)} pitcher profiles")

    combined = pd.concat(all_profiles, ignore_index=True)
    combined = combined.sort_values(["pitcher", "season"]).reset_index(drop=True)

    out = RAW_DIR / "pitcher_profiles_by_season.parquet"
    combined.to_parquet(out, index=False)

    print(f"\n{'='*60}")
    print(f"  PITCHER PROFILES COMPLETE")
    print(f"{'='*60}")
    print(f"Total profiles: {len(combined):,}")
    print(f"Unique pitchers: {combined['pitcher'].nunique():,}")
    print(f"Seasons: {sorted(combined['season'].unique())}")
    print(f"\nSample columns: {list(combined.columns[:20])}")
    print(f"Total columns: {len(combined.columns)}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
