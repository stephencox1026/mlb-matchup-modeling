"""
Build per-batter-per-season profiles BY PITCH TYPE GROUP from cached Statcast.
Powers insights like "Ohtani hits sliders at .312 with 91.2 mph exit velo".

Output: data/raw/batter_pitch_profiles.parquet
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
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWING_DESC = WHIFF_DESC | {"foul", "foul_bunt", "hit_into_play", "hit_into_play_no_out",
                            "hit_into_play_score"}


def _barrel_indicator(pa_subset: pd.DataFrame) -> pd.Series:
    """1 if barrel from Statcast `barrel` or LSA bucket 6 (bulk files often omit `barrel`)."""
    if "barrel" in pa_subset.columns:
        bcol = pa_subset["barrel"].fillna(0).astype(int)
    else:
        bcol = pd.Series(0, index=pa_subset.index, dtype=int)
    if "launch_speed_angle" in pa_subset.columns:
        lsa_barrel = pa_subset["launch_speed_angle"].eq(6).fillna(False).astype(int)
    else:
        lsa_barrel = pd.Series(0, index=pa_subset.index, dtype=int)
    return ((bcol > 0) | (lsa_barrel > 0)).astype(float)


def classify_pitch_group(pitch_type):
    if pitch_type in FASTBALL:
        return "fastball"
    elif pitch_type in BREAKING:
        return "breaking"
    elif pitch_type in OFFSPEED:
        return "offspeed"
    return "other"


def build_batter_profiles_for_season(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["pitch_group"] = df["pitch_type"].map(classify_pitch_group)
    df["is_whiff"] = df["description"].isin(WHIFF_DESC).astype(int)
    df["is_swing"] = df["description"].isin(SWING_DESC).astype(int)
    df["is_chase"] = ((~df["zone"].fillna(0).between(1, 9)) & (df["is_swing"] == 1)).astype(int)

    pa_pitches = df[df["events"].notna() & (df["events"] != "")].copy()
    pa_pitches["is_hit"] = pa_pitches["events"].isin(HIT_EVENTS).astype(int)
    pa_pitches["is_hr"] = (pa_pitches["events"] == "home_run").astype(int)
    pa_pitches["is_ab"] = (~pa_pitches["events"].isin({"walk", "intent_walk", "hit_by_pitch",
                                                         "catcher_interf", "sac_bunt"})).astype(int)

    # League-wide barrel rate for shrinkage (season slice).
    league_prior_barrel = 0.08
    if len(pa_pitches) > 200:
        league_prior_barrel = float(_barrel_indicator(pa_pitches).mean())

    BARREL_SHRINK_K = 40.0
    MIN_PA_BARREL_GROUP = 5

    rows = []
    for bid, batter_pitches in df.groupby("batter"):
        batter_pas = pa_pitches[pa_pitches["batter"] == bid]
        n_total_pa = len(batter_pas)
        if n_total_pa < 10:
            continue

        row = {"batter": bid, "season": year, "total_pa": n_total_pa}

        if "player_name" in batter_pitches.columns:
            mode = batter_pitches["player_name"].mode()
            row["batter_name"] = mode.iloc[0] if not mode.empty else ""

        for pg in ["fastball", "breaking", "offspeed"]:
            all_mask = batter_pitches["pitch_group"] == pg
            subset = batter_pitches[all_mask]

            pa_mask = batter_pas["pitch_group"] == pg
            pa_subset = batter_pas[pa_mask]

            n_swings = subset["is_swing"].sum()
            n_whiffs = subset["is_whiff"].sum()

            row[f"whiff_vs_{pg}"] = n_whiffs / max(n_swings, 1)
            row[f"chase_vs_{pg}"] = subset["is_chase"].sum() / max(len(subset[~subset["zone"].fillna(0).between(1, 9)]), 1)

            ab_subset = pa_subset[pa_subset["is_ab"] == 1] if "is_ab" in pa_subset.columns else pa_subset
            row[f"ba_vs_{pg}"] = ab_subset["is_hit"].mean() if len(ab_subset) >= 5 else np.nan

            if "launch_speed" in pa_subset.columns:
                ev_vals = pa_subset["launch_speed"].dropna()
                row[f"ev_vs_{pg}"] = ev_vals.mean() if len(ev_vals) >= 5 else np.nan
            else:
                row[f"ev_vs_{pg}"] = np.nan

            if "launch_angle" in pa_subset.columns:
                la_vals = pa_subset["launch_angle"].dropna()
                row[f"la_vs_{pg}"] = la_vals.mean() if len(la_vals) >= 5 else np.nan
            else:
                row[f"la_vs_{pg}"] = np.nan

            if len(pa_subset) >= MIN_PA_BARREL_GROUP:
                n_g = len(pa_subset)
                br = _barrel_indicator(pa_subset)
                raw_mean = float(br.mean())
                row[f"barrel_vs_{pg}"] = (
                    (raw_mean * n_g + league_prior_barrel * BARREL_SHRINK_K) / (n_g + BARREL_SHRINK_K)
                )
            else:
                row[f"barrel_vs_{pg}"] = np.nan

            # T2.1: actual HR rate per pitch type (replaces barrel-as-proxy in xptw).
            # Same shrinkage approach as barrel rate so small samples regress to league HR rate.
            league_prior_hr = 0.031  # league HR/PA baseline
            HR_SHRINK_K = 60.0
            if len(pa_subset) >= MIN_PA_BARREL_GROUP:
                n_g = len(pa_subset)
                raw_hr = float(pa_subset["is_hr"].mean())
                row[f"hr_rate_vs_{pg}"] = (
                    (raw_hr * n_g + league_prior_hr * HR_SHRINK_K) / (n_g + HR_SHRINK_K)
                )
            else:
                row[f"hr_rate_vs_{pg}"] = np.nan

        for hand in ["L", "R"]:
            hand_mask = batter_pitches["p_throws"] == hand
            hand_pitches = batter_pitches[hand_mask]
            hand_pas = batter_pas[batter_pas["p_throws"] == hand]

            suffix = "lhp" if hand == "L" else "rhp"
            if "launch_speed" in hand_pas.columns:
                ev = hand_pas["launch_speed"].dropna()
                row[f"ev_vs_{suffix}"] = ev.mean() if len(ev) >= 5 else np.nan
            else:
                row[f"ev_vs_{suffix}"] = np.nan

            ab_hand = hand_pas[hand_pas["is_ab"] == 1] if "is_ab" in hand_pas.columns else hand_pas
            row[f"ba_vs_{suffix}"] = ab_hand["is_hit"].mean() if len(ab_hand) >= 10 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  BUILD BATTER-BY-PITCH-TYPE PROFILES")
    print("=" * 60)

    parquet_files = sorted(BULK_DIR.glob("statcast_*.parquet"))
    print(f"Found {len(parquet_files)} season files\n")

    all_profiles = []
    for f in parquet_files:
        year = int(f.stem.replace("statcast_", ""))
        print(f"  {year}: loading...", end=" ", flush=True)
        df = pd.read_parquet(f)
        print(f"{len(df):,} pitches →", end=" ", flush=True)
        profiles = build_batter_profiles_for_season(df, year)
        all_profiles.append(profiles)
        print(f"{len(profiles)} batter profiles")

    combined = pd.concat(all_profiles, ignore_index=True)
    combined = combined.sort_values(["batter", "season"]).reset_index(drop=True)

    out = RAW_DIR / "batter_pitch_profiles.parquet"
    combined.to_parquet(out, index=False)

    print(f"\n{'='*60}")
    print(f"  BATTER PITCH-TYPE PROFILES COMPLETE")
    print(f"{'='*60}")
    print(f"Total profiles: {len(combined):,}")
    print(f"Unique batters: {combined['batter'].nunique():,}")
    print(f"Columns: {list(combined.columns)}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
