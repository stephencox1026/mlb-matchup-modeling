#!/usr/bin/env python3
"""
Join tracking Parquet to same-day Statcast PA aggregates per batter; set outcome_* columns.

Run after Statcast league PA is updated for the slate calendar day(s), e.g. the morning after games:

  python3 src/fill_matchup_prediction_outcomes.py
  python3 src/fill_matchup_prediction_outcomes.py --slate 2026-04-19

Then refresh the high-confidence slice:

  python3 -c \"from matchup_tracking import materialize_high_conf_tracking; materialize_high_conf_tracking()\"
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR  # noqa: E402
from matchup_tracking import (  # noqa: E402
    TRACKING_MAIN,
    materialize_high_conf_tracking,
)

PA_PATH = RAW_DIR / "statcast_pa_level_league.parquet"
HIT_EVENTS = {"single", "double", "triple", "home_run"}
XBH_EVENTS = {"double", "triple", "home_run"}


def load_pa_frame() -> pd.DataFrame:
    """Single PA-level load used by both whole-game and vs-SP aggregations."""
    cols = ["game_date", "batter", "pitcher", "events", "is_strikeout", "is_walk"]
    pa = pd.read_parquet(PA_PATH, columns=cols)
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce").dt.normalize()
    pa = pa.dropna(subset=["game_date", "batter", "pitcher"])
    pa["batter"] = pa["batter"].astype(int)
    pa["pitcher"] = pa["pitcher"].astype(int)
    ev = pa["events"].fillna("").str.lower()
    pa["is_hit"] = ev.isin(HIT_EVENTS).astype(int)
    pa["is_hr"] = (ev == "home_run").astype(int)
    pa["is_xbh"] = ev.isin(XBH_EVENTS).astype(int)
    # is_strikeout/is_walk are already int columns in the PA file but coerce safely
    pa["is_strikeout"] = pa["is_strikeout"].fillna(0).astype(int)
    pa["is_walk"] = pa["is_walk"].fillna(0).astype(int)
    return pa


def load_batter_day_stats(pa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Whole-game per-(date, batter) aggregate (kept for backward compat)."""
    if pa is None:
        pa = load_pa_frame()
    return pa.groupby(["game_date", "batter"], as_index=False).agg(
        PA=("events", "count"),
        H=("is_hit", "sum"),
        HR=("is_hr", "sum"),
        XBH=("is_xbh", "sum"),
    )


def load_batter_vs_pitcher_day_stats(pa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-(date, batter, pitcher) aggregate — used for outcome_*_vs_sp columns.

    This is the eval-cleanup C3 aggregation: we score predictions only against
    the PAs where the batter actually faced the predicted starting pitcher,
    not the whole-game outcomes (which include bullpen PAs we never predicted).
    """
    if pa is None:
        pa = load_pa_frame()
    return pa.groupby(["game_date", "batter", "pitcher"], as_index=False).agg(
        PA=("events", "count"),
        H=("is_hit", "sum"),
        HR=("is_hr", "sum"),
        XBH=("is_xbh", "sum"),
        K=("is_strikeout", "sum"),
        BB=("is_walk", "sum"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill tracking Parquet outcomes from Statcast PA.")
    parser.add_argument(
        "--slate",
        type=str,
        default=None,
        help="Only fill rows for this YYYY-MM-DD (default: all slates ≤ latest Statcast day).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print row counts that would update; do not write Parquet.",
    )
    args = parser.parse_args()

    if not TRACKING_MAIN.exists():
        print(f"No tracking file at {TRACKING_MAIN}; run build_matchup_dashboard first.")
        sys.exit(1)

    df = pd.read_parquet(TRACKING_MAIN)
    pa = load_pa_frame()
    stats = load_batter_day_stats(pa)
    vs_sp_stats = load_batter_vs_pitcher_day_stats(pa)
    max_obs = stats["game_date"].max()
    if pd.isna(max_obs):
        print("No Statcast dates in league PA file.")
        sys.exit(1)

    df["_slate"] = pd.to_datetime(df["slate_date"], errors="coerce").dt.normalize()

    # Whole-game aggregate (back-compat columns)
    st = stats.rename(
        columns={"PA": "_st_pa", "H": "_st_h", "HR": "_st_hr", "XBH": "_st_xbh"},
    )
    merged = df.merge(
        st,
        left_on=["_slate", "batter_mlbam_id"],
        right_on=["game_date", "batter"],
        how="left",
    )
    merged = merged.drop(columns=["game_date", "batter"], errors="ignore")

    # vs-SP aggregate (C3 cleanup): join on (date, batter, predicted starter)
    vs = vs_sp_stats.rename(
        columns={
            "PA": "_vs_pa", "H": "_vs_h", "HR": "_vs_hr",
            "XBH": "_vs_xbh", "K": "_vs_k", "BB": "_vs_bb",
        }
    )
    merged = merged.merge(
        vs,
        left_on=["_slate", "batter_mlbam_id", "pitcher_mlbam_id"],
        right_on=["game_date", "batter", "pitcher"],
        how="left",
    )
    merged = merged.drop(columns=["game_date", "batter", "pitcher"], errors="ignore")

    update = merged["_slate"] <= max_obs
    if args.slate:
        sdt = pd.Timestamp(args.slate).normalize()
        update &= merged["_slate"] == sdt

    n = int(update.sum())
    print(f"Statcast max game_date: {max_obs.date()} | rows to fill: {n}")
    if n == 0 and not args.slate:
        print(
            "Hint: if your slate is after this date, append Statcast first "
            "(append_statcast_day_to_league_pa / append_statcast_date_range)."
        )

    if args.dry_run:
        # Show vs-SP fill coverage as part of the dry-run report
        if n > 0:
            vs_filled = merged.loc[update, "_vs_pa"].notna().sum()
            print(f"  whole-game outcome rows would fill: {n}")
            print(f"  vs-SP rows where batter faced predicted starter at all: {int(vs_filled)} "
                  f"({100*vs_filled/n:.1f}%)")
        return

    now = datetime.now(timezone.utc).isoformat()

    # Whole-game (back-compat)
    pa_c = merged.loc[update, "_st_pa"].fillna(0).astype(int)
    h_c = merged.loc[update, "_st_h"].fillna(0).astype(int)
    hr_c = merged.loc[update, "_st_hr"].fillna(0).astype(int)
    xbh_c = merged.loc[update, "_st_xbh"].fillna(0).astype(int)

    merged.loc[update, "outcome_pa"] = pa_c.values
    merged.loc[update, "outcome_h"] = h_c.values
    merged.loc[update, "outcome_hr"] = hr_c.values
    merged.loc[update, "outcome_xbh"] = xbh_c.values
    merged.loc[update, "outcome_hit_flag"] = (h_c > 0).astype(int).values
    merged.loc[update, "outcome_hr_flag"] = (hr_c > 0).astype(int).values
    merged.loc[update, "outcome_xbh_flag"] = (xbh_c > 0).astype(int).values
    merged.loc[update, "outcome_filled_at"] = now

    # vs-SP (C3 cleanup) — NaN means batter didn't face this pitcher at all that day
    # (e.g., starter was pulled before the batter's first PA, or batter was a late sub).
    # Distinguish "didn't face SP" (NaN) from "faced SP but went 0-fer" (0) downstream.
    vs_pa = merged.loc[update, "_vs_pa"]
    vs_h = merged.loc[update, "_vs_h"]
    vs_hr = merged.loc[update, "_vs_hr"]
    vs_xbh = merged.loc[update, "_vs_xbh"]
    vs_k = merged.loc[update, "_vs_k"]
    vs_bb = merged.loc[update, "_vs_bb"]

    merged.loc[update, "outcome_pa_vs_sp"] = vs_pa.values
    merged.loc[update, "outcome_h_vs_sp"] = vs_h.values
    merged.loc[update, "outcome_hr_vs_sp"] = vs_hr.values
    merged.loc[update, "outcome_xbh_vs_sp"] = vs_xbh.values
    merged.loc[update, "outcome_k_vs_sp"] = vs_k.values
    merged.loc[update, "outcome_bb_vs_sp"] = vs_bb.values
    # Flag columns: 1 if had at least one (faced SP AND got the outcome), 0 if faced SP but 0-fer,
    # NaN if never faced SP. Using float NaN to preserve the "did not face" signal.
    merged.loc[update, "outcome_hit_flag_vs_sp"] = (vs_h > 0).astype(float).where(vs_pa.notna()).values
    merged.loc[update, "outcome_hr_flag_vs_sp"] = (vs_hr > 0).astype(float).where(vs_pa.notna()).values
    merged.loc[update, "outcome_xbh_flag_vs_sp"] = (vs_xbh > 0).astype(float).where(vs_pa.notna()).values

    drop_tmp = [c for c in merged.columns
                if c.startswith("_st_") or c.startswith("_vs_") or c == "_slate"]
    out = merged.drop(columns=drop_tmp)
    TRACKING_MAIN.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TRACKING_MAIN, index=False)
    hc = materialize_high_conf_tracking()
    n_vs = int(merged.loc[update, "outcome_pa_vs_sp"].notna().sum())
    print(f"Wrote {TRACKING_MAIN} | filled {n} (whole-game), {n_vs} (vs-SP) | high_conf rows: {hc}")


if __name__ == "__main__":
    main()
