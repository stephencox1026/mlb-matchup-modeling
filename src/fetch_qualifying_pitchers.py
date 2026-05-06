"""
Fetch all MLB starting pitchers with 10+ IP in 2026 from the MLB Stats API.
Output: data/raw/qualifying_pitchers_2026.csv
"""
import requests
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DIR, SEASON

BASE = "https://statsapi.mlb.com/api/v1"
MIN_IP = 10.0


def fetch_all_teams() -> list[dict]:
    r = requests.get(f"{BASE}/teams", params={"sportId": 1, "season": SEASON}, timeout=15)
    r.raise_for_status()
    return [{"id": t["id"], "abbr": t["abbreviation"], "name": t["name"]}
            for t in r.json().get("teams", [])]


def fetch_team_pitchers(team_id: int) -> list[dict]:
    players = {}
    for rtype in ["active", "fullSeason", "40Man"]:
        r = requests.get(f"{BASE}/teams/{team_id}/roster",
                         params={"rosterType": rtype, "season": SEASON}, timeout=15)
        r.raise_for_status()
        for p in r.json().get("roster", []):
            pid = p["person"]["id"]
            pos = p["position"]["abbreviation"]
            if pos == "P" and pid not in players:
                players[pid] = {
                    "mlbam_id": pid,
                    "name": p["person"]["fullName"],
                    "pos": pos,
                }
    return list(players.values())


def fetch_pitching_stats(player_id: int) -> dict | None:
    r = requests.get(f"{BASE}/people/{player_id}/stats",
                     params={"stats": "season", "group": "pitching", "season": SEASON,
                             "gameType": "R"},
                     timeout=15)
    r.raise_for_status()
    for sg in r.json().get("stats", []):
        for sp in sg.get("splits", []):
            stat = sp.get("stat", {})
            ip_str = str(stat.get("inningsPitched", "0"))
            try:
                whole, frac = ip_str.split(".") if "." in ip_str else (ip_str, "0")
                ip = int(whole) + int(frac) / 3.0
            except (ValueError, TypeError):
                ip = 0.0
            if ip >= MIN_IP:
                gs = int(stat.get("gamesStarted", 0))
                return {
                    "IP": round(ip, 1),
                    "GS": gs,
                    "G": int(stat.get("gamesPlayed", 0)),
                    "W": int(stat.get("wins", 0)),
                    "L": int(stat.get("losses", 0)),
                    "ERA": stat.get("era", "0.00"),
                    "WHIP": stat.get("whip", "0.00"),
                    "K9": stat.get("strikeoutsPer9Inn", "0.00"),
                    "BB9": stat.get("walksPer9Inn", "0.00"),
                    "K": int(stat.get("strikeOuts", 0)),
                    "BB": int(stat.get("baseOnBalls", 0)),
                    "H_allowed": int(stat.get("hits", 0)),
                    "HR_allowed": int(stat.get("homeRuns", 0)),
                    "role": "SP" if gs > 0 else "RP",
                }
    return None


def main():
    print(f"Fetching all MLB pitchers for {SEASON}...")
    teams = fetch_all_teams()
    print(f"  Found {len(teams)} teams")

    all_pitchers = {}
    for t in teams:
        roster = fetch_team_pitchers(t["id"])
        for p in roster:
            if p["mlbam_id"] not in all_pitchers:
                all_pitchers[p["mlbam_id"]] = {**p, "team": t["abbr"]}
    print(f"  Found {len(all_pitchers)} unique rostered pitchers")

    print(f"\nChecking {SEASON} stats (min {MIN_IP} IP)...")
    qualifying = []
    checked = 0
    for pid, info in all_pitchers.items():
        stats = fetch_pitching_stats(pid)
        checked += 1
        if checked % 100 == 0:
            print(f"  Checked {checked}/{len(all_pitchers)}, qualifying: {len(qualifying)}")
        if stats:
            qualifying.append({**info, **stats})

    df = pd.DataFrame(qualifying)
    starters = df[df["role"] == "SP"].copy()
    relievers = df[df["role"] == "RP"].copy()

    df = df.sort_values("IP", ascending=False).reset_index(drop=True)
    starters = starters.sort_values("IP", ascending=False).reset_index(drop=True)

    df.to_csv(RAW_DIR / "qualifying_pitchers_2026.csv", index=False)
    starters.to_csv(RAW_DIR / "qualifying_starters_2026.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  {len(df)} pitchers with {MIN_IP}+ IP in {SEASON}")
    print(f"  {len(starters)} starting pitchers (GS > 0)")
    print(f"  {len(relievers)} relief pitchers (GS = 0)")
    print(f"{'='*60}")
    print(f"\nTop 20 starters by IP:")
    print(starters.head(20)[["name", "team", "IP", "GS", "ERA", "K9", "BB9", "WHIP"]].to_string(index=False))


if __name__ == "__main__":
    main()
