"""Build full master table: 2026 overall + platoon splits + career splits + vs Peterson."""
import time
import requests
import pandas as pd
from config import SEASON, RAW_DIR, MASTER_DIR, REPORTS_DIR

IDS_CSV = RAW_DIR / "player_ids.csv"
BASE = "https://statsapi.mlb.com/api/v1"
PETERSON_ID = 656849

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


def _parse_stat(stat: dict) -> dict:
    row = {}
    for api_key, col in RENAME.items():
        val = stat.get(api_key, 0)
        if col in FLOAT_COLS:
            row[col] = float(val) if val else 0.0
        else:
            row[col] = int(val) if val else 0
    return row


def fetch_season_overall(player_id: int, season: int) -> dict | None:
    url = f"{BASE}/people/{player_id}/stats"
    r = requests.get(url, params={"stats": "season", "group": "hitting", "season": season}, timeout=30)
    r.raise_for_status()
    data = r.json()
    for sg in data.get("stats", []):
        for sp in sg.get("splits", []):
            return _parse_stat(sp.get("stat", {}))
    return None


def fetch_career_overall(player_id: int) -> dict | None:
    url = f"{BASE}/people/{player_id}/stats"
    r = requests.get(url, params={"stats": "career", "group": "hitting"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    for sg in data.get("stats", []):
        for sp in sg.get("splits", []):
            return _parse_stat(sp.get("stat", {}))
    return None


def fetch_vs_pitcher(batter_id: int, pitcher_id: int) -> dict | None:
    """Career batter vs specific pitcher from MLB Stats API."""
    url = f"{BASE}/people/{batter_id}/stats"
    params = {
        "stats": "vsPlayer",
        "group": "hitting",
        "opposingPlayerId": pitcher_id,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    for sg in data.get("stats", []):
        for sp in sg.get("splits", []):
            s = sp.get("stat", {})
            if s.get("plateAppearances", 0) > 0 or s.get("atBats", 0) > 0:
                return _parse_stat(s)
    return None


def build_master() -> pd.DataFrame:
    ids = pd.read_csv(IDS_CSV)
    splits_df = pd.read_parquet(MASTER_DIR / "splits_mlb_api.parquet")

    all_rows = []

    for _, player in ids.iterrows():
        pid = int(player["mlbam_id"])
        name = player["name"]
        pos = player["pos"]

        base = {"name": name, "pos": pos, "mlbam_id": pid}

        # 2026 overall
        overall = fetch_season_overall(pid, SEASON)
        if overall:
            all_rows.append({**base, "scope": "2026", "split": "Overall", **overall})

        # Career overall
        career = fetch_career_overall(pid)
        if career:
            all_rows.append({**base, "scope": "Career", "split": "Overall", **career})

        # vs Peterson (career)
        vs_pet = fetch_vs_pitcher(pid, PETERSON_ID)
        if vs_pet:
            all_rows.append({**base, "scope": "Career", "split": "vs Peterson", **vs_pet})
        else:
            empty = {col: (0.0 if col in FLOAT_COLS else 0) for col in RENAME.values()}
            all_rows.append({**base, "scope": "Career", "split": "vs Peterson", **empty})

        time.sleep(0.2)
        print(f"  {name}: overall={'Y' if overall else 'N'} career={'Y' if career else 'N'} peterson={'Y' if vs_pet else 'n=0'}")

    new_df = pd.DataFrame(all_rows)
    master = pd.concat([splits_df, new_df], ignore_index=True)

    col_order = ["name", "pos", "mlbam_id", "scope", "split",
                 "G", "PA", "AB", "H", "2B", "3B", "HR", "RBI",
                 "BB", "SO", "SB", "BA", "OBP", "SLG", "OPS",
                 "BABIP", "TB", "SF", "SH", "HBP", "IBB"]
    for c in col_order:
        if c not in master.columns:
            master[c] = 0
    master = master[col_order].sort_values(["name", "scope", "split"]).reset_index(drop=True)
    return master


if __name__ == "__main__":
    print("Building master table...")
    master = build_master()

    master.to_parquet(MASTER_DIR / "lad_hitters_sprint.parquet", index=False)
    master.to_csv(MASTER_DIR / "lad_hitters_sprint.csv", index=False)
    print(f"\nMaster table: {len(master)} rows, {master['split'].nunique()} split types")
    print(f"Splits: {master['split'].unique().tolist()}")
    print(f"Scopes: {master['scope'].unique().tolist()}")
    print(f"Players: {master['name'].nunique()}")

    # Export analysis-specific CSVs
    for split_name, fname in [
        ("vs LHP", "analysis_vs_lhp_2026.csv"),
        ("vs RHP", "analysis_vs_rhp_2026.csv"),
        ("vs Peterson", "analysis_peterson.csv"),
    ]:
        subset = master[master["split"] == split_name].copy()
        subset.to_csv(REPORTS_DIR / fname, index=False)
        print(f"  Exported {len(subset)} rows -> {fname}")

    # Print a nice 2026 vs LHP table
    print("\n=== 2026 vs LHP ===")
    lhp_2026 = master[(master["scope"] == "2026") & (master["split"] == "vs LHP")]
    print(lhp_2026[["name", "G", "PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "BA", "OBP", "SLG", "OPS"]].to_string(index=False))

    print("\n=== Career vs Peterson ===")
    pet = master[master["split"] == "vs Peterson"]
    print(pet[["name", "scope", "PA", "AB", "H", "HR", "BA", "OBP", "SLG", "OPS"]].to_string(index=False))
