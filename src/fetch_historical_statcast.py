"""
Fetch historical Statcast pitch-level data for the 14 target hitters (2015–2025),
then aggregate to PA-level rows with outcome labels (hit/out/walk/etc.).

This is the training data for the ML pipeline:
  - 2015–2024: training set
  - 2025: validation set

Output:
  data/raw/statcast_historical/  (per-player parquet caches)
  data/raw/statcast_pa_level.parquet  (aggregated PA rows with labels)
"""
import time
from pathlib import Path
import pandas as pd
from pybaseball import statcast_batter, cache
from config import RAW_DIR

cache.enable()

IDS_CSV = RAW_DIR / "player_ids.csv"
BASE = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = RAW_DIR / "statcast_historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HIT_EVENTS = {"single", "double", "triple", "home_run"}
OUT_EVENTS = {
    "field_out", "strikeout", "grounded_into_double_play",
    "double_play", "force_out", "fielders_choice",
    "fielders_choice_out", "strikeout_double_play",
    "triple_play", "sac_fly", "sac_bunt", "sac_fly_double_play",
    "field_error",
}
WALK_EVENTS = {"walk", "hit_by_pitch", "intent_walk"}

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
}


def _get_debut_year(player_id: int) -> int | None:
    import requests
    r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}", timeout=15)
    r.raise_for_status()
    info = r.json().get("people", [{}])[0]
    debut = info.get("mlbDebutDate", "")
    return int(debut[:4]) if debut else None


def fetch_player_statcast(player_id: int, name: str, start_year: int) -> pd.DataFrame:
    """Fetch and cache Statcast data for one player across multiple seasons."""
    cache_path = CACHE_DIR / f"{player_id}_{name.replace(' ', '_')}.parquet"

    if cache_path.exists():
        print(f"  {name}: loading from cache")
        return pd.read_parquet(cache_path)

    frames = []
    for yr in range(start_year, 2026):
        if yr not in SEASON_WINDOWS:
            continue
        sd, ed = SEASON_WINDOWS[yr]
        try:
            df = statcast_batter(sd, ed, player_id)
            if df is not None and len(df) > 0:
                df = df[df["game_type"] == "R"].copy() if "game_type" in df.columns else df
                frames.append(df)
                print(f"    {name} {yr}: {len(df)} pitches (regular season)")
            else:
                print(f"    {name} {yr}: no data")
        except Exception as e:
            print(f"    {name} {yr}: ERROR - {e}")
        time.sleep(2.0)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["player_name_clean"] = name
    combined.to_parquet(cache_path, index=False)
    return combined


def pitches_to_pa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level Statcast rows into PA-level rows.
    Each PA is identified by (game_pk, at_bat_number, batter).
    The final pitch of each PA carries the 'events' column with the outcome.
    """
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
        "player_name_clean",
    ]
    available = [c for c in keep_cols if c in pa_pitches.columns]
    pa = pa_pitches[available].copy()

    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_ab"] = (~pa["events"].isin(WALK_EVENTS | {"catcher_interf", "sac_bunt"})).astype(int)
    pa["is_strikeout"] = pa["events"].str.contains("strikeout", case=False, na=False).astype(int)
    pa["is_walk"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    pa["is_hbp"] = (pa["events"] == "hit_by_pitch").astype(int)

    pa["vs_lhp"] = (pa["p_throws"] == "L").astype(int) if "p_throws" in pa.columns else 0
    pa["barrel"] = pa["barrel"].fillna(0).astype(int) if "barrel" in pa.columns else 0

    return pa.sort_values(["batter", "game_date", "at_bat_number"]).reset_index(drop=True)


def fetch_all_historical_statcast() -> pd.DataFrame:
    """Main orchestrator: fetch pitch-level, convert to PA-level for all players."""
    ids = pd.read_csv(IDS_CSV)
    all_pa = []

    for _, player in ids.iterrows():
        pid = int(player["mlbam_id"])
        name = player["name"]

        debut = _get_debut_year(pid)
        if debut is None:
            print(f"  {name}: no debut date, skipping")
            continue

        start_year = max(debut, 2015)
        print(f"\n{'='*50}")
        print(f"  {name} (id={pid}): debut={debut}, fetching {start_year}–2025")
        print(f"{'='*50}")

        sc = fetch_player_statcast(pid, name, start_year)
        if sc.empty:
            print(f"  {name}: no Statcast data found")
            continue

        pa = pitches_to_pa(sc)
        if not pa.empty:
            all_pa.append(pa)
            print(f"  {name}: {len(pa)} PAs extracted")

    if not all_pa:
        return pd.DataFrame()

    return pd.concat(all_pa, ignore_index=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  HISTORICAL STATCAST INGEST (2015–2025)")
    print("  14 Dodgers Hitters → PA-Level Training Data")
    print("=" * 60)

    pa_df = fetch_all_historical_statcast()

    if pa_df.empty:
        print("\nERROR: No PA data collected")
        exit(1)

    out_path = RAW_DIR / "statcast_pa_level.parquet"
    pa_df.to_parquet(out_path, index=False)
    pa_df.to_csv(RAW_DIR / "statcast_pa_level.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  STATCAST INGEST COMPLETE")
    print(f"{'='*60}")
    print(f"Total PAs: {len(pa_df):,}")
    print(f"Players:   {pa_df['player_name_clean'].nunique() if 'player_name_clean' in pa_df.columns else pa_df['batter'].nunique()}")

    if "game_year" in pa_df.columns:
        print(f"\nPAs by year:")
        print(pa_df.groupby("game_year").size().to_string())

        train = pa_df[pa_df["game_year"] <= 2024]
        val = pa_df[pa_df["game_year"] == 2025]
        print(f"\nTrain (2015–2024): {len(train):,} PAs")
        print(f"Val   (2025):      {len(val):,} PAs")

    print(f"\nHit rate:  {pa_df['is_hit'].mean():.3f}")
    print(f"HR rate:   {pa_df['is_hr'].mean():.4f}")
    print(f"K rate:    {pa_df['is_strikeout'].mean():.3f}")
    print(f"BB rate:   {pa_df['is_walk'].mean():.3f}")

    print(f"\nSaved to {out_path}")
