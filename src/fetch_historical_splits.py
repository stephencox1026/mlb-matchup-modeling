"""
Fetch year-by-year platoon splits (vs LHP / vs RHP) from the MLB Stats API
for each of the 14 target hitters, covering all available seasons through 2025.

Output: data/raw/historical_splits.parquet  (one row per player × year × split)
"""
import time
import requests
import pandas as pd
from config import RAW_DIR

IDS_CSV = RAW_DIR / "player_ids.csv"
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
    """Return MLB debut year from the people endpoint."""
    r = requests.get(f"{BASE}/people/{player_id}", timeout=15)
    r.raise_for_status()
    info = r.json().get("people", [{}])[0]
    debut = info.get("mlbDebutDate", "")
    if debut:
        return int(debut[:4])
    return None


def _fetch_platoon_season(player_id: int, season: int) -> list[dict]:
    """Fetch vs-L and vs-R splits for one player in one season."""
    params = {
        "stats": "statSplits",
        "group": "hitting",
        "season": season,
        "sitCodes": "vl,vr",
        "gameType": "R",
    }
    url = f"{BASE}/people/{player_id}/stats"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for sg in data.get("stats", []):
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


def fetch_all_historical_splits() -> pd.DataFrame:
    """Pull year-by-year platoon splits for every player from debut through 2025."""
    ids = pd.read_csv(IDS_CSV)
    all_rows = []

    for _, player in ids.iterrows():
        pid = int(player["mlbam_id"])
        name = player["name"]
        pos = player["pos"]

        debut = _get_debut_year(pid)
        if debut is None:
            print(f"  {name}: no debut date found, skipping historical")
            continue

        start_year = max(debut, 2015)
        print(f"  {name} (id={pid}): debut={debut}, pulling {start_year}–2025")

        for yr in range(start_year, 2026):
            splits = _fetch_platoon_season(pid, yr)
            for s in splits:
                s["name"] = name
                s["pos"] = pos
                s["mlbam_id"] = pid
                all_rows.append(s)
            time.sleep(0.15)

        print(f"    → {sum(1 for r in all_rows if r['name'] == name)} split-rows")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    col_order = ["name", "pos", "mlbam_id", "season", "split",
                 "G", "PA", "AB", "H", "2B", "3B", "HR", "RBI",
                 "BB", "SO", "SB", "BA", "OBP", "SLG", "OPS",
                 "BABIP", "TB", "SF", "SH", "HBP", "IBB"]
    for c in col_order:
        if c not in df.columns:
            df[c] = 0
    return df[col_order].sort_values(["name", "season", "split"]).reset_index(drop=True)


if __name__ == "__main__":
    print("=== Historical Platoon Splits Backfill (2015–2025) ===\n")
    df = fetch_all_historical_splits()

    out_pq = RAW_DIR / "historical_splits.parquet"
    out_csv = RAW_DIR / "historical_splits.csv"
    df.to_parquet(out_pq, index=False)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved {len(df)} rows to {out_pq}")
    print(f"Players: {df['name'].nunique()}")
    print(f"Seasons covered: {sorted(df['season'].unique())}")
    print(f"\nRows per player:")
    print(df.groupby("name")["season"].agg(["min", "max", "count"]).to_string())
