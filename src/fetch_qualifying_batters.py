"""
Fetch all MLB batters with 10+ ABs in 2026 from the MLB Stats API.
Output: data/raw/qualifying_batters_2026.csv
"""
import requests
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DIR, SEASON

BASE = "https://statsapi.mlb.com/api/v1"
MIN_AB = 10


def fetch_all_teams() -> list[dict]:
    r = requests.get(f"{BASE}/teams", params={"sportId": 1, "season": SEASON}, timeout=15)
    r.raise_for_status()
    return [{"id": t["id"], "abbr": t["abbreviation"], "name": t["name"]}
            for t in r.json().get("teams", [])]


def fetch_team_roster(team_id: int) -> list[dict]:
    players = {}
    for rtype in ["active", "fullSeason", "40Man"]:
        r = requests.get(f"{BASE}/teams/{team_id}/roster",
                         params={"rosterType": rtype, "season": SEASON}, timeout=15)
        r.raise_for_status()
        for p in r.json().get("roster", []):
            pid = p["person"]["id"]
            if pid not in players:
                players[pid] = {
                    "mlbam_id": pid,
                    "name": p["person"]["fullName"],
                    "pos": p["position"]["abbreviation"],
                }
    return list(players.values())


def fetch_season_stats(player_id: int) -> dict | None:
    r = requests.get(f"{BASE}/people/{player_id}/stats",
                     params={"stats": "season", "group": "hitting", "season": SEASON,
                             "gameType": "R"},
                     timeout=15)
    r.raise_for_status()
    for sg in r.json().get("stats", []):
        for sp in sg.get("splits", []):
            stat = sp.get("stat", {})
            ab = stat.get("atBats", 0)
            if ab and int(ab) >= MIN_AB:
                return {
                    "AB": int(ab),
                    "PA": int(stat.get("plateAppearances", 0)),
                    "H": int(stat.get("hits", 0)),
                    "HR": int(stat.get("homeRuns", 0)),
                    "BA": stat.get("avg", ".000"),
                    "OPS": stat.get("ops", ".000"),
                }
    return None


def main():
    print(f"Fetching all MLB teams for {SEASON}...")
    teams = fetch_all_teams()
    print(f"  Found {len(teams)} teams")

    all_players = {}
    for t in teams:
        roster = fetch_team_roster(t["id"])
        for p in roster:
            if p["mlbam_id"] not in all_players:
                all_players[p["mlbam_id"]] = {**p, "team": t["abbr"]}
    print(f"  Found {len(all_players)} unique rostered players across MLB")

    print(f"\nChecking 2026 stats (min {MIN_AB} AB)...")
    qualifying = []
    checked = 0
    for pid, info in all_players.items():
        stats = fetch_season_stats(pid)
        checked += 1
        if checked % 100 == 0:
            print(f"  Checked {checked}/{len(all_players)}, qualifying so far: {len(qualifying)}")
        if stats:
            qualifying.append({**info, **stats})

    df = pd.DataFrame(qualifying)
    df = df.sort_values("AB", ascending=False).reset_index(drop=True)

    out = RAW_DIR / "qualifying_batters_2026.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"  {len(df)} players with {MIN_AB}+ ABs in {SEASON}")
    print(f"  Saved → {out}")
    print(f"{'='*60}")
    print(f"\nTop 20 by AB:")
    print(df.head(20)[["name", "team", "pos", "AB", "PA", "H", "HR", "BA", "OPS"]].to_string(index=False))
    print(f"\nTeam breakdown:")
    print(df["team"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
