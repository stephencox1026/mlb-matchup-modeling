"""
Fetch year-by-year platoon splits (vs LHP / vs RHP) for all qualifying 2026 batters.
Uses the MLB Stats API. Covers each player from max(debut, 2015) through 2025.

Output: data/raw/historical_splits_league.parquet
"""
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import requests
import pandas as pd
from config import RAW_DIR

QUALIFYING_CSV = RAW_DIR / "qualifying_batters_2026.csv"
BASE = "https://statsapi.mlb.com/api/v1"

RENAME = {
    "gamesPlayed": "G", "plateAppearances": "PA", "atBats": "AB",
    "hits": "H", "doubles": "2B", "triples": "3B", "homeRuns": "HR",
    "rbi": "RBI", "baseOnBalls": "BB", "strikeOuts": "SO",
    "stolenBases": "SB", "avg": "BA", "obp": "OBP", "slg": "SLG",
    "ops": "OPS", "babip": "BABIP", "totalBases": "TB",
    "sacFlies": "SF", "sacBunts": "SH", "hitByPitch": "HBP",
    "intentionalWalks": "IBB",
}
FLOAT_COLS = {"BA", "OBP", "SLG", "OPS", "BABIP"}
STAT_COLS = list(RENAME.keys())


def _get_debut_year(player_id: int) -> int | None:
    try:
        r = requests.get(f"{BASE}/people/{player_id}", timeout=10)
        r.raise_for_status()
        info = r.json().get("people", [{}])[0]
        debut = info.get("mlbDebutDate", "")
        return int(debut[:4]) if debut else None
    except Exception:
        return None


def _fetch_platoon_season(player_id: int, season: int) -> list[dict]:
    params = {
        "stats": "statSplits", "group": "hitting",
        "season": season, "sitCodes": "vl,vr", "gameType": "R",
    }
    try:
        resp = requests.get(f"{BASE}/people/{player_id}/stats", params=params, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    rows = []
    for sg in resp.json().get("stats", []):
        for sp in sg.get("splits", []):
            desc = sp.get("split", {}).get("description", "")
            stat = sp.get("stat", {})
            row = {"season": season}
            for api_key in STAT_COLS:
                val = stat.get(api_key, 0)
                col = RENAME[api_key]
                if col in FLOAT_COLS:
                    try:
                        row[col] = float(val) if val and str(val).replace('.','',1).replace('-','').isdigit() else 0.0
                    except (ValueError, TypeError):
                        row[col] = 0.0
                else:
                    try:
                        row[col] = int(val) if val else 0
                    except (ValueError, TypeError):
                        row[col] = 0
            row["split"] = "vs LHP" if "Left" in desc else "vs RHP"
            rows.append(row)
    return rows


def main():
    print("=" * 60)
    print("  LEAGUE-WIDE HISTORICAL SPLITS (2015-2025)")
    print("=" * 60)

    players = pd.read_csv(QUALIFYING_CSV)
    print(f"  {len(players)} qualifying batters loaded")

    all_rows = []
    for idx, player in players.iterrows():
        pid = int(player["mlbam_id"])
        name = player["name"]

        debut = _get_debut_year(pid)
        if debut is None:
            continue

        start_year = max(debut, 2015)
        for yr in range(start_year, 2026):
            splits = _fetch_platoon_season(pid, yr)
            for s in splits:
                s["name"] = name
                s["mlbam_id"] = pid
                s["team"] = player.get("team", "")
                all_rows.append(s)
            time.sleep(0.05)

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(players)} players processed, {len(all_rows)} split-rows so far")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("ERROR: No split data collected")
        return

    col_order = ["name", "mlbam_id", "team", "season", "split",
                 "G", "PA", "AB", "H", "2B", "3B", "HR", "RBI",
                 "BB", "SO", "SB", "BA", "OBP", "SLG", "OPS",
                 "BABIP", "TB", "SF", "SH", "HBP", "IBB"]
    for c in col_order:
        if c not in df.columns:
            df[c] = 0
    df = df[col_order].sort_values(["name", "season", "split"]).reset_index(drop=True)

    out = RAW_DIR / "historical_splits_league.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(RAW_DIR / "historical_splits_league.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  LEAGUE SPLITS COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows: {len(df):,}")
    print(f"Players:    {df['name'].nunique()}")
    print(f"Seasons:    {sorted(df['season'].unique())}")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
