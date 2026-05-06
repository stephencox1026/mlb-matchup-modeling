"""
Build historical Batter vs Pitcher (BvP) matchup features.
For each (batter, pitcher) pair, compute cumulative stats shifted to avoid leakage.

Output: data/raw/bvp_matchup_features.parquet
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config import BVP_HR_BAYES_PRIOR_WEIGHT, RAW_DIR

HIT_EVENTS = {"single", "double", "triple", "home_run"}
K_EVENTS_PAT = "strikeout"
BB_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}

LEAGUE_BA = 0.243
LEAGUE_K_RATE = 0.222
LEAGUE_BB_RATE = 0.084
LEAGUE_HR_RATE = 0.031
LEAGUE_XBH_RATE = 0.079


def bayesian_shrink(observed_rate, n_obs, prior_rate, prior_weight=20):
    """Shrink observed rate toward prior using Beta-Binomial-style weighting."""
    return (observed_rate * n_obs + prior_rate * prior_weight) / (n_obs + prior_weight)


def main():
    print("=" * 60)
    print("  BUILD BvP MATCHUP FEATURES")
    print("=" * 60)

    pa_path = RAW_DIR / "statcast_pa_level_league.parquet"
    pa = pd.read_parquet(pa_path)
    print(f"  Loaded {len(pa):,} PAs")

    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_k"] = pa["events"].str.contains(K_EVENTS_PAT, case=False, na=False).astype(int)
    pa["is_bb"] = pa["events"].isin(BB_EVENTS).astype(int)
    pa["is_ab"] = (~pa["events"].isin(BB_EVENTS | {"catcher_interf", "sac_bunt"})).astype(int)
    pa["is_xbh"] = pa["events"].isin({"double", "triple", "home_run"}).astype(int)

    pa = pa.sort_values(["batter", "pitcher", "game_date", "at_bat_number"]).reset_index(drop=True)

    print("  Computing cumulative BvP stats (shifted)...")

    bvp_rows = []
    for (bid, pid), grp in pa.groupby(["batter", "pitcher"]):
        if len(grp) < 1:
            continue

        cum_pa = grp["events"].expanding().count().shift(1).fillna(0).astype(int)
        cum_h = grp["is_hit"].expanding().sum().shift(1).fillna(0)
        cum_hr = grp["is_hr"].expanding().sum().shift(1).fillna(0)
        cum_k = grp["is_k"].expanding().sum().shift(1).fillna(0)
        cum_bb = grp["is_bb"].expanding().sum().shift(1).fillna(0)
        cum_ab = grp["is_ab"].expanding().sum().shift(1).fillna(0)
        cum_xbh = grp["is_xbh"].expanding().sum().shift(1).fillna(0)

        raw_ba = cum_h / cum_ab.replace(0, np.nan)
        raw_k_rate = cum_k / cum_pa.replace(0, np.nan)
        raw_bb_rate = cum_bb / cum_pa.replace(0, np.nan)
        raw_hr_rate = cum_hr / cum_pa.replace(0, np.nan)
        raw_xbh_rate = cum_xbh / cum_pa.replace(0, np.nan)

        shrunk_ba = [bayesian_shrink(r, n, LEAGUE_BA) if pd.notna(r) else LEAGUE_BA
                     for r, n in zip(raw_ba, cum_ab)]
        shrunk_k = [bayesian_shrink(r, n, LEAGUE_K_RATE) if pd.notna(r) else LEAGUE_K_RATE
                    for r, n in zip(raw_k_rate, cum_pa)]
        shrunk_bb = [bayesian_shrink(r, n, LEAGUE_BB_RATE) if pd.notna(r) else LEAGUE_BB_RATE
                     for r, n in zip(raw_bb_rate, cum_pa)]
        shrunk_hr = [
            bayesian_shrink(r, n, LEAGUE_HR_RATE, BVP_HR_BAYES_PRIOR_WEIGHT)
            if pd.notna(r)
            else LEAGUE_HR_RATE
            for r, n in zip(raw_hr_rate, cum_pa)
        ]
        shrunk_xbh = [bayesian_shrink(r, n, LEAGUE_XBH_RATE) if pd.notna(r) else LEAGUE_XBH_RATE
                      for r, n in zip(raw_xbh_rate, cum_pa)]

        for i, idx in enumerate(grp.index):
            pa_count = int(cum_pa.iloc[i])
            bvp_rows.append({
                "pa_index": idx,
                "bvp_pa_count": pa_count,
                "bvp_ba": shrunk_ba[i],
                "bvp_k_rate": shrunk_k[i],
                "bvp_bb_rate": shrunk_bb[i],
                "bvp_hr_count": int(cum_hr.iloc[i]),
                "bvp_hr_rate": shrunk_hr[i],
                "bvp_xbh_rate": shrunk_xbh[i],
                "log_bvp_pa": np.log1p(pa_count),
                "bvp_has_history": int(pa_count > 0),
            })

        if len(bvp_rows) % 500000 == 0:
            print(f"    Processed {len(bvp_rows):,} PA rows...")

    bvp_df = pd.DataFrame(bvp_rows)
    bvp_df = bvp_df.set_index("pa_index")

    out = RAW_DIR / "bvp_matchup_features.parquet"
    bvp_df.to_parquet(out)

    print(f"\n{'='*60}")
    print(f"  BvP MATCHUP FEATURES COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows: {len(bvp_df):,}")
    print(f"Columns: {list(bvp_df.columns)}")
    print(f"Non-zero BvP history: {(bvp_df['bvp_pa_count'] > 0).sum():,} ({(bvp_df['bvp_pa_count'] > 0).mean():.1%})")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
