#!/usr/bin/env python3
"""
L2 part 1: build per-park HR factor features.

For each park (home_team), compute the long-run HR rate at that park vs the
league HR rate (two-sided park factor). Output is keyed on home_team and stored
as data/priors/park_hr_factors.json.

Two-sided PF formula (FanGraphs / Baseball Reference style):
  PF_HR(park) = (HR/PA at park, both teams) / (HR/PA in road games for the same teams)

Empirical-Bayes shrinkage toward 1.0 by inverse-variance.

Usage:
  python3 src/build_park_hr_factors.py
  python3 src/build_park_hr_factors.py --years 2022 2023 2024 2025
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

OUT_PATH = ROOT / "data" / "priors" / "park_hr_factors.json"
BULK_DIR = RAW_DIR / "statcast_bulk"

# Heuristic shrinkage: ~3 seasons of full data per park is enough to trust the cell.
PRIOR_PA_WEIGHT = 5000  # pseudo-PAs of league-mean prior strength


def _load_year(year: int) -> pd.DataFrame:
    p = BULK_DIR / f"statcast_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p, columns=["game_pk", "home_team", "away_team",
                                       "inning_topbot", "events"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two-sided park HR factors.")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025])
    args = parser.parse_args()

    frames = []
    for y in args.years:
        f = _load_year(y)
        if not f.empty:
            print(f"  {y}: {len(f):,} rows")
            frames.append(f)
    if not frames:
        print("No bulk parquets found; nothing to do.", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["events"].notna()].copy()
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    # PAs only — events column is set per-PA only on the last pitch of the PA in
    # baseball savant. Most rows have NaN events; keep only PA-end rows.
    df["is_pa"] = 1

    # Defensive team determines who's pitching: if Top of inning, home team pitches; else away.
    df["def_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    df["bat_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])

    # Per-park aggregate (PAs at each home park, both teams batting)
    park_agg = df.groupby("home_team").agg(pa=("is_pa", "sum"),
                                             hr=("is_hr", "sum")).reset_index()
    park_agg["raw_hr_pa"] = park_agg["hr"] / park_agg["pa"]

    # Two-sided PF: control for which teams play at each park
    # Each team's road HR/PA — used as the comparison baseline for that team's home park.
    # Compute PA + HR for each (bat_team, home_team) cell; the "road games" for team T
    # are all PAs where T bats AND home_team != T.
    bat_park = df.groupby(["bat_team", "home_team"]).agg(
        pa=("is_pa", "sum"), hr=("is_hr", "sum")
    ).reset_index()

    # For each park, compute team-quality-adjusted expected HR rate:
    # For each team that batted at this park, look up that team's HR/PA on the road
    # (not at this specific park), weight by their PAs at this park, sum.
    rows = []
    league_hr_pa = float(df["is_hr"].sum() / len(df))

    for park, g in bat_park.groupby("home_team"):
        total_pa = float(g["pa"].sum())
        total_hr = float(g["hr"].sum())
        observed_hr_pa = total_hr / total_pa if total_pa else float("nan")

        # Expected HR/PA at this park (controlling for batting teams) =
        # weighted avg of each batting team's road-HR/PA
        expected_pa_weighted = 0.0
        weight_sum = 0.0
        for _, r in g.iterrows():
            bt = r["bat_team"]
            road_pa = float(bat_park[(bat_park["bat_team"] == bt)
                                     & (bat_park["home_team"] != park)]["pa"].sum())
            road_hr = float(bat_park[(bat_park["bat_team"] == bt)
                                     & (bat_park["home_team"] != park)]["hr"].sum())
            road_rate = road_hr / road_pa if road_pa else league_hr_pa
            expected_pa_weighted += r["pa"] * road_rate
            weight_sum += r["pa"]
        expected_hr_pa = expected_pa_weighted / weight_sum if weight_sum else league_hr_pa

        pf_two_sided = observed_hr_pa / expected_hr_pa if expected_hr_pa > 0 else 1.0

        # Empirical-Bayes shrinkage toward 1.0:
        # equivalent strength of prior = PRIOR_PA_WEIGHT pseudo-PAs at PF=1.0
        shrunk = ((total_pa * pf_two_sided + PRIOR_PA_WEIGHT * 1.0) /
                  (total_pa + PRIOR_PA_WEIGHT))

        rows.append({
            "home_team": park,
            "n_pa": int(total_pa),
            "n_hr": int(total_hr),
            "observed_hr_pa": round(observed_hr_pa, 5),
            "expected_hr_pa": round(expected_hr_pa, 5),
            "pf_two_sided_raw": round(pf_two_sided, 4),
            "pf_two_sided_shrunk": round(shrunk, 4),
        })

    out = {
        "_meta": {
            "league_hr_pa": round(league_hr_pa, 5),
            "years": args.years,
            "n_pa_total": int(len(df)),
            "shrinkage_pseudo_pa": PRIOR_PA_WEIGHT,
            "method": "two-sided park factor with EB shrinkage to 1.0",
        },
        "parks": {r["home_team"]: r for r in rows},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    summary = pd.DataFrame(rows).sort_values("pf_two_sided_shrunk", ascending=False)
    print("\n=== Two-sided park HR factors (sorted high to low) ===")
    print(summary[["home_team", "n_pa", "observed_hr_pa", "expected_hr_pa",
                   "pf_two_sided_raw", "pf_two_sided_shrunk"]].to_string(index=False))
    print(f"\nLeague HR/PA: {league_hr_pa:.4f}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
