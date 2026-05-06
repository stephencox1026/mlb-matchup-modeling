#!/usr/bin/env python3
"""
Build per-team bullpen HR/PA priors from bulk Statcast.

For each game and each defensive team (the team currently pitching), identify
the starting pitcher (the first pitcher to record a PA on that side) and label
every other PA on that side as a bullpen PA. Aggregate PA + HR counts per
(team, season). Empirical-Bayes shrink per season toward league mean using
Beta-Binomial fit. Inverse-variance blend across seasons (recent seasons
weighted by sample size).

Output: data/priors/team_bullpen_hr.json

Usage:
  python3 src/build_team_bullpen_hr_priors.py
  python3 src/build_team_bullpen_hr_priors.py --years 2022 2023 2024 2025 2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR  # noqa: E402

OUT_PATH = ROOT / "data" / "priors" / "team_bullpen_hr.json"
BULK_DIR = RAW_DIR / "statcast_bulk"
PA_LEAGUE = RAW_DIR / "statcast_pa_level_league.parquet"  # 2026 YTD source

# Heuristic pseudo-PA prior strength for EB shrinkage. ~5000 reliever PAs is
# enough to trust a team's blend; below that the league mean dominates.
PRIOR_PA_WEIGHT = 4000


def _load_year(year: int) -> pd.DataFrame:
    p = BULK_DIR / f"statcast_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(
        p, columns=["game_pk", "at_bat_number", "pitcher", "events",
                    "home_team", "away_team", "inning_topbot"],
    )
    df = df[df["events"].notna()].copy()  # PA-end rows only
    df["season"] = year
    return df


def _load_2026_ytd() -> pd.DataFrame:
    """The PA-level league parquet is our 2026 YTD source — it lacks home_team/
    away_team but has game_pk and pitcher. We join home_team via the bulk
    files when available; for 2026, assume the matchup file's home_team is
    canonical (skip 2026 if no parsed home_team source exists)."""
    if not PA_LEAGUE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(
        PA_LEAGUE,
        columns=["game_pk", "game_date", "at_bat_number", "pitcher", "events"],
    )
    df["season"] = pd.to_datetime(df["game_date"]).dt.year
    df = df[df["season"] == 2026].copy()
    df = df[df["events"].notna()]
    return df


def _identify_starters(df: pd.DataFrame) -> pd.DataFrame:
    """Per (game_pk, inning_topbot), the starting pitcher is the pitcher with
    the smallest at_bat_number. We label that pitcher_id as the SP for that
    side; everyone else is bullpen."""
    if df.empty:
        return df
    df = df.sort_values(["game_pk", "inning_topbot", "at_bat_number"])
    sp = (df.groupby(["game_pk", "inning_topbot"])
            .first()
            .reset_index()[["game_pk", "inning_topbot", "pitcher"]]
            .rename(columns={"pitcher": "starter_pitcher"}))
    df = df.merge(sp, on=["game_pk", "inning_topbot"], how="left")
    df["is_pen"] = (df["pitcher"] != df["starter_pitcher"]).astype(int)
    return df


def _aggregate_per_team_season(df: pd.DataFrame) -> pd.DataFrame:
    """Per (defensive team, season) PA + HR counts on bullpen PAs only."""
    if df.empty:
        return pd.DataFrame()
    df = df[df["is_pen"] == 1].copy()
    if "home_team" in df.columns and "away_team" in df.columns:
        df["def_team"] = np.where(df["inning_topbot"] == "Top",
                                   df["home_team"], df["away_team"])
    else:
        # 2026 PA-level league lacks home/away — skip those rows for now.
        return pd.DataFrame()
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    g = df.groupby(["def_team", "season"]).agg(
        pa=("events", "size"),
        hr=("is_hr", "sum"),
    ).reset_index()
    return g


def _eb_shrink_season(group: pd.DataFrame, league_p: float, prior_pa: float) -> pd.DataFrame:
    """Beta-Binomial-style shrinkage to league mean."""
    g = group.copy()
    g["raw_p"] = g["hr"] / g["pa"].clip(lower=1)
    # Equivalent Bayesian: posterior = (HR + alpha) / (PA + alpha + beta)
    # alpha = prior_pa * league_p, beta = prior_pa * (1 - league_p)
    alpha = prior_pa * league_p
    beta = prior_pa * (1 - league_p)
    g["shrunk_p"] = (g["hr"] + alpha) / (g["pa"] + alpha + beta)
    return g


def _blend_seasons(per_season: pd.DataFrame, league_p: float, prior_pa: float) -> pd.DataFrame:
    """Inverse-variance blend across seasons within a team. Weight = pa per
    season (which is proportional to 1/variance for a Bernoulli rate)."""
    if per_season.empty:
        return pd.DataFrame()
    rows = []
    for team, sub in per_season.groupby("def_team"):
        weights = sub["pa"].astype(float)
        if weights.sum() <= 0:
            continue
        blended_raw = float((sub["raw_p"] * weights).sum() / weights.sum())
        # Re-shrink the blended raw rate using the total pooled sample
        total_pa = float(sub["pa"].sum())
        total_hr = float(sub["hr"].sum())
        alpha = prior_pa * league_p
        beta = prior_pa * (1 - league_p)
        blended_shrunk = (total_hr + alpha) / (total_pa + alpha + beta)
        by_season = {
            int(r.season): {"pa": int(r.pa), "hr": int(r.hr),
                             "raw_p": round(float(r.raw_p), 5),
                             "shrunk_p": round(float(r.shrunk_p), 5)}
            for r in sub.itertuples(index=False)
        }
        rows.append({
            "team": team,
            "n_pa": int(total_pa),
            "n_hr": int(total_hr),
            "raw_p_blended": round(blended_raw, 5),
            "blend": round(blended_shrunk, 5),
            "by_season": by_season,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-team bullpen HR/PA priors.")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--prior-pa", type=float, default=PRIOR_PA_WEIGHT)
    args = parser.parse_args()

    frames = []
    for y in args.years:
        if y == 2026:
            df_y = _load_2026_ytd()
            # 2026 lacks home_team — skip aggregation step (handled by _aggregate_per_team_season)
            if df_y.empty:
                print(f"  {y}: skip (no rows)")
                continue
            print(f"  {y}: {len(df_y):,} PA rows from PA-level league file (2026 YTD; "
                  "home_team not joined here -> seasonal blend will exclude 2026)")
            continue
        df_y = _load_year(y)
        if df_y.empty:
            print(f"  {y}: skip (no parquet)")
            continue
        print(f"  {y}: {len(df_y):,} PA-end rows")
        frames.append(df_y)

    if not frames:
        print("No bulk parquets found; nothing to do.", file=sys.stderr)
        sys.exit(1)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = _identify_starters(df_all)
    print(f"Total PA rows after starter labelling: {len(df_all):,} "
          f"(bullpen: {df_all.is_pen.sum():,})")

    per_season_raw = _aggregate_per_team_season(df_all)
    if per_season_raw.empty:
        print("Aggregation produced no rows.", file=sys.stderr)
        sys.exit(1)

    league_p = float(per_season_raw["hr"].sum() / per_season_raw["pa"].sum())
    print(f"League bullpen HR/PA across {args.years}: {league_p:.4f}")

    per_season_shrunk = _eb_shrink_season(per_season_raw, league_p, args.prior_pa)
    blended = _blend_seasons(per_season_shrunk, league_p, args.prior_pa)
    blended = blended.sort_values("blend", ascending=False).reset_index(drop=True)

    print()
    print("=== Per-team bullpen HR/PA (sorted high to low) ===")
    print(blended[["team", "n_pa", "n_hr", "raw_p_blended", "blend"]]
          .to_string(index=False))

    out = {
        "_meta": {
            "league_hr_pa_pen": round(league_p, 5),
            "years": args.years,
            "prior_pa_weight": args.prior_pa,
            "method": "Beta-Binomial EB shrinkage to league mean; "
                      "inverse-variance blend across seasons via shared posterior",
            "n_teams": int(len(blended)),
        },
        "teams": {r["team"]: {
            "blend": r["blend"],
            "raw_p_blended": r["raw_p_blended"],
            "n_pa": r["n_pa"],
            "n_hr": r["n_hr"],
            "by_season": r["by_season"],
        } for r in blended.to_dict(orient="records")},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
