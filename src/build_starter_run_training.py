"""
Build game-level training set for the starter runs-allowed model.

For each bulk Statcast season file (2015-2025):
  1. Identify starters (first pitcher per game-side).
  2. Compute actual_runs = max(post_bat_score) - min(bat_score) while starter pitched.
  3. Record opposing lineup batter IDs.

Then engineer game-level features:
  A. Pitcher features from pitcher_profiles_by_season (shifted: prior season).
  B. Opposing lineup features aggregated from PA-level feature vectors.
  C. Context features (month, day_of_week, is_home).

Outputs:
  data/master/starter_runs_train.parquet  (2015-2023)
  data/master/starter_runs_val.parquet    (2024)
  data/master/starter_runs_test.parquet   (2025)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, MASTER_DIR

BULK_DIR = RAW_DIR / "statcast_bulk"
HIT_EVENTS = {"single", "double", "triple", "home_run"}
BB_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}

BATTER_AGG_FEATURES = [
    "roll_ba_30", "roll_ba_100",
    "roll_hr_rate_30", "roll_hr_rate_100",
    "roll_k_rate_30", "roll_k_rate_100",
    "roll_bb_rate_30", "roll_bb_rate_100",
    "roll_ev_30", "roll_ev_100",
    "roll_la_30",
    "bpt_ba_vs_fastball", "bpt_ba_vs_breaking", "bpt_ba_vs_offspeed",
    "bpt_ev_vs_fastball", "bpt_ev_vs_breaking",
    "bpt_whiff_vs_fastball", "bpt_whiff_vs_breaking",
    "bvp_ba", "bvp_hr_rate", "bvp_pa_count", "bvp_has_history",
]

PITCHER_PROFILE_COLS = [
    "pct_fastball", "velo_fastball", "spin_fastball",
    "pct_breaking", "velo_breaking", "spin_breaking",
    "pct_offspeed",
    "k_rate", "bb_rate", "hr_rate_allowed", "whiff_rate",
    "barrel_rate_allowed", "arm_angle", "extension", "pct_in_zone",
    "velo_overall",
]


def extract_starter_outings_from_bulk(filepath: Path) -> pd.DataFrame:
    """Extract starter outings with actual runs from a single bulk Statcast parquet."""
    df = pd.read_parquet(filepath)
    pa_events = df[df["events"].notna() & (df["events"] != "")].copy()

    rows = []
    for gp, game in pa_events.groupby("game_pk"):
        game_date = game["game_date"].iloc[0]
        game_year = game["game_year"].iloc[0] if "game_year" in game.columns else None

        for topbot, side_label in [("Top", "home"), ("Bot", "away")]:
            side = game[game["inning_topbot"] == topbot]
            if len(side) == 0:
                continue

            first = side.sort_values(["inning", "at_bat_number"]).iloc[0]
            starter_id = first["pitcher"]
            starter_pas = side[side["pitcher"] == starter_id]

            n_pa = len(starter_pas)
            if n_pa < 5:
                continue

            actual_runs = int(
                starter_pas["post_bat_score"].max() - starter_pas["bat_score"].min()
            )

            opposing_batters = starter_pas["batter"].unique().tolist()

            n_hits = int(starter_pas["events"].isin(HIT_EVENTS).sum())
            n_hr = int((starter_pas["events"] == "home_run").sum())
            n_bb = int(starter_pas["events"].isin(BB_EVENTS).sum())
            n_k = int(
                starter_pas["events"]
                .str.contains("strikeout", case=False, na=False)
                .sum()
            )
            n_xbh = int(
                starter_pas["events"]
                .isin({"double", "triple", "home_run"})
                .sum()
            )
            max_thru = (
                starter_pas["n_thruorder_pitcher"].max()
                if "n_thruorder_pitcher" in starter_pas.columns
                else None
            )
            p_throws = first.get("p_throws", None)
            stand_mode = starter_pas["stand"].mode()
            stand_mode = stand_mode.iloc[0] if len(stand_mode) else None

            rows.append({
                "game_pk": gp,
                "game_date": game_date,
                "game_year": game_year,
                "pitcher": int(starter_id),
                "p_throws": p_throws,
                "fielding_side": side_label,
                "n_pa_faced": n_pa,
                "actual_runs": actual_runs,
                "n_hits": n_hits,
                "n_hr": n_hr,
                "n_bb": n_bb,
                "n_k": n_k,
                "n_xbh": n_xbh,
                "max_thru_order": max_thru,
                "opposing_batters": opposing_batters,
            })

    return pd.DataFrame(rows)


def build_pitcher_feature_map(profiles: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """Build {(pitcher_id, season): feature_dict} using the PRIOR season's profile."""
    feature_map = {}
    for pid, grp in profiles.groupby("pitcher"):
        grp = grp.sort_values("season")
        for i in range(1, len(grp)):
            prior = grp.iloc[i - 1]
            current_season = int(grp.iloc[i]["season"])
            feature_map[(int(pid), current_season)] = {
                f"sp_{c}": prior[c] for c in PITCHER_PROFILE_COLS if c in prior.index
            }
    return feature_map


def build_pitcher_rolling_from_outings(outings: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling starter-level stats from the outings themselves (shifted)."""
    outings = outings.sort_values(["pitcher", "game_date"]).copy()
    outings["hit_rate_outing"] = outings["n_hits"] / outings["n_pa_faced"].clip(lower=1)
    outings["k_rate_outing"] = outings["n_k"] / outings["n_pa_faced"].clip(lower=1)
    outings["bb_rate_outing"] = outings["n_bb"] / outings["n_pa_faced"].clip(lower=1)
    outings["hr_rate_outing"] = outings["n_hr"] / outings["n_pa_faced"].clip(lower=1)
    outings["runs_per_pa_outing"] = outings["actual_runs"] / outings["n_pa_faced"].clip(lower=1)

    for col in ["hit_rate_outing", "k_rate_outing", "bb_rate_outing",
                "hr_rate_outing", "runs_per_pa_outing"]:
        for w in [3, 10]:
            outings[f"sp_roll_{col}_{w}"] = (
                outings.groupby("pitcher")[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    return outings


def _build_batter_latest_lookup(pa_features: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Pre-index: for each batter, keep their rows sorted by game_date for fast bisect."""
    available_agg = [c for c in BATTER_AGG_FEATURES if c in pa_features.columns]
    keep = ["batter", "game_date"] + available_agg
    slim = pa_features[[c for c in keep if c in pa_features.columns]].copy()
    slim = slim.sort_values(["batter", "game_date"])
    return {int(bid): grp.reset_index(drop=True) for bid, grp in slim.groupby("batter")}


def aggregate_lineup_features_fast(
    outing_row: pd.Series,
    batter_lookup: dict[int, pd.DataFrame],
) -> dict:
    """Aggregate batter features for the opposing lineup using pre-indexed lookup."""
    opposing = outing_row["opposing_batters"]
    game_date = pd.Timestamp(outing_row["game_date"])

    lineup_feats = []
    for bid in opposing:
        grp = batter_lookup.get(int(bid))
        if grp is None or grp.empty:
            continue
        prior = grp[grp["game_date"] < game_date]
        if prior.empty:
            continue
        last_row = prior.iloc[-1]
        available = {
            c: last_row[c]
            for c in BATTER_AGG_FEATURES
            if c in last_row.index and pd.notna(last_row[c])
        }
        if available:
            lineup_feats.append(available)

    if not lineup_feats:
        return {}

    lf = pd.DataFrame(lineup_feats)
    result = {}
    for col in lf.columns:
        vals = lf[col].dropna()
        if vals.empty:
            continue
        result[f"lineup_mean_{col}"] = float(vals.mean())
        result[f"lineup_max_{col}"] = float(vals.max())
        if len(vals) >= 3:
            result[f"lineup_std_{col}"] = float(vals.std())

    result["lineup_n_batters"] = len(lineup_feats)
    result["lineup_n_with_bvp"] = int(lf["bvp_has_history"].sum()) if "bvp_has_history" in lf.columns else 0
    result["lineup_mean_bvp_pa"] = float(lf["bvp_pa_count"].mean()) if "bvp_pa_count" in lf.columns else 0.0

    return result


def add_context_features(outings: pd.DataFrame) -> pd.DataFrame:
    outings = outings.copy()
    outings["game_date"] = pd.to_datetime(outings["game_date"])
    outings["month"] = outings["game_date"].dt.month
    outings["day_of_week"] = outings["game_date"].dt.dayofweek
    outings["is_home"] = (outings["fielding_side"] == "away").astype(int)
    return outings


def main():
    print("=" * 60)
    print("  BUILD STARTER RUNS TRAINING DATA")
    print("=" * 60)

    bulk_files = sorted(BULK_DIR.glob("statcast_*.parquet"))
    if not bulk_files:
        print("ERROR: No bulk Statcast files found")
        return

    print(f"Found {len(bulk_files)} season files\n")

    all_outings = []
    for bf in bulk_files:
        year = bf.stem.replace("statcast_", "")
        print(f"  Extracting starters from {year}...")
        outings = extract_starter_outings_from_bulk(bf)
        all_outings.append(outings)
        print(f"    → {len(outings)} starter outings")

    outings = pd.concat(all_outings, ignore_index=True)
    outings["game_date"] = pd.to_datetime(outings["game_date"])
    print(f"\nTotal starter outings: {len(outings):,}")
    print(f"Actual runs: mean={outings['actual_runs'].mean():.2f} "
          f"med={outings['actual_runs'].median():.0f} "
          f"std={outings['actual_runs'].std():.2f}")

    print("\nComputing pitcher rolling features...")
    outings = build_pitcher_rolling_from_outings(outings)

    print("Loading pitcher profiles...")
    pp_path = RAW_DIR / "pitcher_profiles_by_season.parquet"
    if pp_path.exists():
        pp = pd.read_parquet(pp_path)
        pitcher_map = build_pitcher_feature_map(pp)
        print(f"  {len(pitcher_map)} (pitcher, season) profiles loaded")
    else:
        pitcher_map = {}
        print("  WARNING: pitcher_profiles_by_season.parquet not found")

    pitcher_rows = []
    for _, row in outings.iterrows():
        key = (int(row["pitcher"]), int(row["game_year"]))
        pitcher_rows.append(pitcher_map.get(key, {}))
    pitcher_df = pd.DataFrame(pitcher_rows)
    outings = pd.concat([outings.reset_index(drop=True), pitcher_df], axis=1)

    print("Loading PA-level features for lineup aggregation...")
    pa_feat_path = MASTER_DIR / "features_train_league.parquet"
    pa_feat_val_path = MASTER_DIR / "features_val_league.parquet"
    pa_features_parts = []
    for p in [pa_feat_path, pa_feat_val_path]:
        if p.exists():
            df = pd.read_parquet(p)
            df["game_date"] = pd.to_datetime(df["game_date"])
            pa_features_parts.append(df)
            print(f"  Loaded {len(df):,} rows from {p.name}")
    if pa_features_parts:
        pa_features = pd.concat(pa_features_parts, ignore_index=True)
    else:
        pa_features = pd.DataFrame()
        print("  WARNING: No PA feature files found")

    if not pa_features.empty:
        print("Building batter lookup index...")
        batter_lookup = _build_batter_latest_lookup(pa_features)
        print(f"  {len(batter_lookup)} batters indexed")

        print("Aggregating lineup features...")
        lineup_rows = []
        total = len(outings)
        for i, (_, row) in enumerate(outings.iterrows()):
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{total}...")
            lineup_rows.append(aggregate_lineup_features_fast(row, batter_lookup))
        lineup_df = pd.DataFrame(lineup_rows)
        outings = pd.concat([outings.reset_index(drop=True), lineup_df], axis=1)
        del batter_lookup

    outings = add_context_features(outings)

    outings = outings.drop(columns=["opposing_batters"], errors="ignore")

    feature_cols = [c for c in outings.columns if c.startswith((
        "sp_", "lineup_", "month", "day_of_week", "is_home",
    )) and c not in {"sp_roll_hit_rate_outing_3", "sp_roll_hit_rate_outing_10"}
       or c.startswith("sp_roll_")]
    id_cols = ["game_pk", "game_date", "game_year", "pitcher", "p_throws",
               "fielding_side", "n_pa_faced"]
    target_col = "actual_runs"

    keep = [c for c in id_cols if c in outings.columns]
    keep += [target_col]
    keep += sorted(set(c for c in feature_cols if c in outings.columns))

    outings = outings[keep].copy()

    train = outings[outings["game_year"] <= 2023].copy()
    val = outings[outings["game_year"] == 2024].copy()
    test = outings[outings["game_year"] == 2025].copy()

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(MASTER_DIR / "starter_runs_train.parquet", index=False)
    val.to_parquet(MASTER_DIR / "starter_runs_val.parquet", index=False)
    test.to_parquet(MASTER_DIR / "starter_runs_test.parquet", index=False)

    print(f"\n{'=' * 60}")
    print(f"  STARTER RUNS TRAINING DATA COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train):,} outings (2015-2023)")
    print(f"  Val:   {len(val):,} outings (2024)")
    print(f"  Test:  {len(test):,} outings (2025)")
    feat_count = len([c for c in train.columns if c not in id_cols + [target_col]])
    print(f"  Features: {feat_count}")
    print(f"  Saved → {MASTER_DIR}")


if __name__ == "__main__":
    main()
