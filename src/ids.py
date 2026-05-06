"""Resolve MLBAM IDs for the 14 Dodgers hitters using the team roster API."""
import requests
import pandas as pd
from config import ROSTER_14, SEASON, RAW_DIR

BASE = "https://statsapi.mlb.com/api/v1"
TEAM_ID = 119  # LAD


def lookup_roster_ids() -> pd.DataFrame:
    """Pull active + 40-man roster from the MLB API and match to our 14 hitters."""
    all_players = {}
    for rtype in ["active", "fullSeason", "40Man"]:
        r = requests.get(f"{BASE}/teams/{TEAM_ID}/roster",
                         params={"rosterType": rtype, "season": SEASON}, timeout=15)
        r.raise_for_status()
        for p in r.json().get("roster", []):
            pid = p["person"]["id"]
            name = p["person"]["fullName"]
            pos = p["position"]["abbreviation"]
            if pid not in all_players:
                all_players[pid] = {"roster_name": name, "pos_roster": pos, "mlbam_id": pid}

    target_names = {p["name"] for p in ROSTER_14}
    matched = []
    for p in ROSTER_14:
        target = p["name"]
        found = False
        for pid, info in all_players.items():
            if _names_match(target, info["roster_name"]):
                matched.append({
                    "name": target,
                    "pos": p["pos"],
                    "roster_name": info["roster_name"],
                    "mlbam_id": info["mlbam_id"],
                })
                found = True
                break
        if not found:
            matched.append({
                "name": target,
                "pos": p["pos"],
                "roster_name": target,
                "mlbam_id": None,
            })
            print(f"  WARNING: {target} not found on any LAD roster — manual ID needed")

    return pd.DataFrame(matched)


def _names_match(target: str, roster_name: str) -> bool:
    t = target.lower().replace("é", "e").replace("á", "a")
    r = roster_name.lower().replace("é", "e").replace("á", "a")
    if t == r:
        return True
    t_parts = t.split()
    r_parts = r.split()
    if len(t_parts) >= 2 and len(r_parts) >= 2:
        if t_parts[-1] == r_parts[-1] and t_parts[0] == r_parts[0]:
            return True
    return False


if __name__ == "__main__":
    df = lookup_roster_ids()
    out = RAW_DIR / "player_ids.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")
