"""
Fetch today's MLB schedule with probable starting pitchers and team rosters.

Output: data/raw/todays_matchups.json
"""
import json
import requests
import sys, os
from datetime import date
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from config import RAW_DIR, SEASON

BASE = "https://statsapi.mlb.com/api/v1"

# Full active roster can include bench; cap to the N batters most likely to appear
# (YTD PA from qualifying_batters_2026.csv, API roster order as tiebreaker).
SLATE_BATTER_CAP = 13


def _load_ytd_pa_map() -> dict[int, int]:
    path = RAW_DIR / "qualifying_batters_2026.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "mlbam_id" not in df.columns or "PA" not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            pa = float(r["PA"])
        except (TypeError, ValueError):
            pa = 0.0
        if pd.isna(pa):
            pa = 0.0
        out[int(r["mlbam_id"])] = max(0, int(round(pa)))
    return out


def trim_batters_for_slate(batters: list[dict], pa_map: dict[int, int]) -> list[dict]:
    """Subset active position players to SLATE_BATTER_CAP by descending YTD PA."""
    if len(batters) <= SLATE_BATTER_CAP:
        return batters
    indexed = [(i, b) for i, b in enumerate(batters)]
    indexed.sort(
        key=lambda ib: (-pa_map.get(int(ib[1]["mlbam_id"]), 0), ib[0]),
    )
    kept = [b for _, b in indexed[:SLATE_BATTER_CAP]]
    id_order = {int(b["mlbam_id"]): j for j, b in enumerate(batters)}
    kept.sort(key=lambda b: id_order[int(b["mlbam_id"])])
    return kept


def _parse_roof_from_condition(cond: str | None) -> str | None:
    """Map MLB StatsAPI weather condition string to roof_state."""
    if not cond:
        return None
    c = cond.strip().lower()
    if "roof closed" in c or c == "dome" or c == "indoors":
        return "closed"
    if "roof open" in c:
        return "open"
    if c in {"sunny", "cloudy", "partly cloudy", "clear", "overcast",
             "rain", "drizzle", "snow", "fair"}:
        return "outdoor"
    return None


def fetch_schedule(game_date: str | None = None) -> list[dict]:
    if game_date is None:
        game_date = date.today().isoformat()

    # `weather` hydrate adds the StatsAPI roof + weather strings (often null
    # this early, populated within ~3 hours of first pitch).
    r = requests.get(f"{BASE}/schedule", params={
        "sportId": 1, "date": game_date,
        "hydrate": "probablePitcher,team,weather",
        "gameType": "R",
    }, timeout=15)
    r.raise_for_status()

    games = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            away = g.get("teams", {}).get("away", {})
            home = g.get("teams", {}).get("home", {})

            away_pitcher = away.get("probablePitcher", {})
            home_pitcher = home.get("probablePitcher", {})

            weather = g.get("weather") or {}
            condition = weather.get("condition")
            roof_state = _parse_roof_from_condition(condition)

            games.append({
                "game_pk": g["gamePk"],
                "game_date": game_date,
                "game_datetime_utc": g.get("gameDate"),
                "status": g.get("status", {}).get("abstractGameState", ""),
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "away_team_name": away.get("team", {}).get("name", ""),
                "away_team_id": away.get("team", {}).get("id", 0),
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "home_team_name": home.get("team", {}).get("name", ""),
                "home_team_id": home.get("team", {}).get("id", 0),
                "away_pitcher_id": away_pitcher.get("id", 0),
                "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                "home_pitcher_id": home_pitcher.get("id", 0),
                "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                # Phase 1 weather wiring: live MLB roof state at fetch time
                # (often null this early; refresh via src/refresh_game_conditions_today.py
                #  ~90 minutes before first pitch to catch the late post).
                "roof_state_today": roof_state,
                "weather_condition_mlb": condition,
                "weather_temp_mlb": weather.get("temp"),
                "weather_wind_mlb": weather.get("wind"),
            })
    return games


def fetch_team_batters(team_id: int) -> list[dict]:
    r = requests.get(f"{BASE}/teams/{team_id}/roster",
                     params={"rosterType": "active", "season": SEASON}, timeout=15)
    r.raise_for_status()

    batters = []
    for p in r.json().get("roster", []):
        pos = p["position"]["abbreviation"]
        if pos != "P":
            batters.append({
                "mlbam_id": p["person"]["id"],
                "name": p["person"]["fullName"],
                "pos": pos,
            })
    return batters


def main(game_date=None):
    print("=" * 60)
    print("  FETCH TODAY'S GAMES + PROBABLE PITCHERS")
    print("=" * 60)

    games = fetch_schedule(game_date)
    print(f"\n  Found {len(games)} games scheduled\n")

    pa_map = _load_ytd_pa_map()
    if pa_map:
        print(f"  Loaded YTD PA map for {len(pa_map)} batters (qualifying_batters_2026.csv)")
        print(f"  Trimming each side to top {SLATE_BATTER_CAP} by PA among active roster\n")
    else:
        print("  (No qualifying_batters_2026.csv — using full active rosters)\n")

    matchups = []
    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']}: "
              f"{g['away_pitcher_name']} vs {g['home_pitcher_name']}")

        away_batters = trim_batters_for_slate(fetch_team_batters(g["away_team_id"]), pa_map)
        home_batters = trim_batters_for_slate(fetch_team_batters(g["home_team_id"]), pa_map)

        matchups.append({
            **g,
            "away_batters": away_batters,
            "home_batters": home_batters,
        })

    out = RAW_DIR / "todays_matchups.json"
    with open(out, "w") as f:
        json.dump(matchups, f, indent=2)

    total_matchups = sum(
        len(m["away_batters"]) + len(m["home_batters"]) for m in matchups
    )
    print(f"\n{'='*60}")
    print(f"  {len(matchups)} games, {total_matchups} batter-pitcher matchups")
    print(f"  Saved → {out}")
    print(f"{'='*60}")

    return matchups


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD), default=today")
    args = parser.parse_args()
    main(args.date)
