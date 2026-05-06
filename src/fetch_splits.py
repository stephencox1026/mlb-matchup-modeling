"""Fetch platoon splits (vs LHP / vs RHP) from MLB Stats API for 2026 + career."""
import time
import requests
import pandas as pd
from config import SEASON, RAW_DIR, MASTER_DIR, DATA_DIR

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

STAT_COLS = list(RENAME.keys())
FLOAT_COLS = {"BA", "OBP", "SLG", "OPS", "BABIP"}


def fetch_platoon(player_id: int, season: int | None = None) -> list[dict]:
    """Fetch vs-L and vs-R splits for one player. season=None → career."""
    params = {
        "group": "hitting",
        "sitCodes": "vl,vr",
        "gameType": "R",
    }
    if season:
        params["stats"] = "statSplits"
        params["season"] = season
    else:
        params["stats"] = "careerStatSplits"

    url = f"{BASE}/people/{player_id}/stats"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for sg in data.get("stats", []):
        for sp in sg.get("splits", []):
            desc = sp.get("split", {}).get("description", "")
            stat = sp.get("stat", {})
            row = {}
            for api_key in STAT_COLS:
                val = stat.get(api_key, 0)
                col = RENAME[api_key]
                if col in FLOAT_COLS:
                    row[col] = float(val) if val else 0.0
                else:
                    row[col] = int(val) if val else 0
            split_label = "vs LHP" if "Left" in desc else "vs RHP"
            row["split"] = split_label
            rows.append(row)
    return rows


def build_splits_table() -> pd.DataFrame:
    ids = pd.read_csv(IDS_CSV)
    all_rows = []
    for _, player in ids.iterrows():
        pid = int(player["mlbam_id"])
        name = player["name"]
        pos = player["pos"]

        for scope, season_arg in [("2026", SEASON), ("Career", None)]:
            splits = fetch_platoon(pid, season=season_arg)
            if splits:
                for s in splits:
                    s["name"] = name
                    s["pos"] = pos
                    s["mlbam_id"] = pid
                    s["scope"] = scope
                    all_rows.append(s)
            else:
                for label in ["vs LHP", "vs RHP"]:
                    all_rows.append({
                        "name": name, "pos": pos, "mlbam_id": pid,
                        "scope": scope, "split": label,
                        **{RENAME[k]: (0.0 if RENAME[k] in FLOAT_COLS else 0)
                           for k in STAT_COLS},
                    })
        time.sleep(0.25)

    df = pd.DataFrame(all_rows)
    col_order = ["name", "pos", "mlbam_id", "scope", "split",
                 "G", "PA", "AB", "H", "2B", "3B", "HR", "RBI",
                 "BB", "SO", "SB", "BA", "OBP", "SLG", "OPS",
                 "BABIP", "TB", "SF", "SH", "HBP", "IBB"]
    for c in col_order:
        if c not in df.columns:
            df[c] = 0
    return df[col_order]


def golden_assert(df: pd.DataFrame, golden_path: str) -> pd.DataFrame:
    """Compare pulled 2026 vs LHP against golden reference on AB-level fields."""
    golden = pd.read_csv(golden_path)
    pulled = df[(df["scope"] == "2026") & (df["split"] == "vs LHP")].copy()

    checks = []
    for _, grow in golden.iterrows():
        gname = grow["Player"]
        prow = pulled[pulled["name"] == gname]
        if prow.empty:
            checks.append({"player": gname, "match": False, "diffs": "NOT FOUND"})
            continue
        prow = prow.iloc[0]
        diffs = {}
        for col in ["G", "AB", "H", "2B", "3B", "HR", "RBI", "BB", "SO",
                     "BA", "OBP", "SLG", "OPS"]:
            gv = grow.get(col)
            pv = prow.get(col)
            if pd.isna(gv):
                continue
            if col in ("BA", "OBP", "SLG", "OPS"):
                if abs(float(gv) - float(pv)) > 0.002:
                    diffs[col] = f"gold={gv} pull={pv}"
            else:
                if int(gv) != int(pv):
                    diffs[col] = f"gold={gv} pull={pv}"
        checks.append({
            "player": gname,
            "match": len(diffs) == 0,
            "diffs": diffs if diffs else "EXACT MATCH",
        })
    return pd.DataFrame(checks)


if __name__ == "__main__":
    print("Fetching splits from MLB Stats API (gameType=R, sitCodes=vl,vr)...")
    df = build_splits_table()

    out_parquet = MASTER_DIR / "splits_mlb_api.parquet"
    out_csv = MASTER_DIR / "splits_mlb_api.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_parquet}")
    print()
    print("2026 vs LHP (from API, gameType=R):")
    print(df[(df["scope"] == "2026") & (df["split"] == "vs LHP")][
        ["name", "G", "PA", "AB", "H", "HR", "BA", "OBP", "SLG", "OPS"]
    ].to_string(index=False))

    golden_path = DATA_DIR / "golden_vs_lhp_2026.csv"
    if golden_path.exists():
        print("\n--- Golden validation (AB-level) ---")
        result = golden_assert(df, golden_path)
        print(result.to_string(index=False))
        n_match = result["match"].sum()
        n_total = len(result)
        print(f"\n{n_match}/{n_total} players match golden reference EXACTLY")
        if n_match < n_total:
            misses = result[~result["match"]]
            print("Mismatches (likely games played since golden snapshot):")
            for _, row in misses.iterrows():
                print(f"  {row['player']}: {row['diffs']}")
    else:
        print("No golden file found, skipping validation")
