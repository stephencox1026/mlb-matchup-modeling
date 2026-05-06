#!/usr/bin/env python3
"""
Refresh weather (and roof state) for tonight's games close to first pitch,
then regenerate ONLY Section 15 + the dashboard No HR Model tab.

Run manually ~90 minutes before first pitch (or schedule via cron) to catch
late-posted MLB roof states and the freshest Open-Meteo forecast horizon.

Does NOT re-run the heavy prediction pipeline. Bullpen/starter lambdas don't
change — only the weather multiplier + park PF (if roof state arrives).

Usage:
  python3 src/refresh_game_conditions_today.py
  python3 src/refresh_game_conditions_today.py --dry-run    # show diff, do not write
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR, REPORTS_DIR  # noqa: E402
from weather_fetch import _fetch_open_meteo_cached, forecast_at_first_pitch  # noqa: E402

MATCHUPS_JSON = RAW_DIR / "todays_matchups.json"
SECTION_15_JSON = REPORTS_DIR / "todays_zero_hr_predictions.json"

STATS_API = "https://statsapi.mlb.com/api/v1"


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


def _fetch_mlb_weather(date_iso: str) -> dict[int, dict]:
    """Returns game_pk -> {condition, temp, wind, roof_state}."""
    try:
        r = requests.get(
            f"{STATS_API}/schedule",
            params={"sportId": 1, "date": date_iso, "hydrate": "weather"},
            timeout=15,
        )
        r.raise_for_status()
    except Exception as ex:
        print(f"WARN: MLB schedule fetch failed: {ex}", file=sys.stderr)
        return {}
    out = {}
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            gpk = int(g.get("gamePk", 0))
            if gpk <= 0:
                continue
            w = g.get("weather") or {}
            cond = w.get("condition")
            out[gpk] = {
                "weather_condition": cond,
                "weather_temp": w.get("temp"),
                "weather_wind": w.get("wind"),
                "roof_state_today": _parse_roof_from_condition(cond),
            }
    return out


def _clear_open_meteo_cache() -> None:
    """Force Open-Meteo refetch (cache is process-local, but we guard anyway)."""
    _fetch_open_meteo_cached.cache_clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh game conditions (weather + roof) for tonight.")
    parser.add_argument("--dry-run", action="store_true", help="Show diff; do not write files.")
    parser.add_argument("--no-regen", action="store_true", help="Skip Section 15 + dashboard regen.")
    args = parser.parse_args()

    if not MATCHUPS_JSON.exists():
        print(f"Missing {MATCHUPS_JSON}; nothing to refresh.", file=sys.stderr)
        sys.exit(1)
    games = json.loads(MATCHUPS_JSON.read_text())
    if not games:
        print("No games in matchups JSON.")
        return

    # Capture pre-state for diff
    pre_section15 = {}
    if SECTION_15_JSON.exists():
        try:
            payload = json.loads(SECTION_15_JSON.read_text())
            for r in payload.get("games", []):
                pre_section15[int(r.get("game_pk") or 0)] = {
                    "p_zero_hr": r.get("p_zero_hr"),
                    "lambda_total_adj": r.get("lambda_total_adj"),
                    "weather_hr_mult": r.get("weather_hr_mult"),
                    "roof_state_today": r.get("roof_state_today"),
                }
        except Exception:
            pass

    date_iso = games[0].get("game_date")
    if not date_iso:
        print("No game_date in matchups[0]; refusing to fetch.", file=sys.stderr)
        sys.exit(1)

    # Refresh roof state via MLB weather hydrate
    mlb_weather = _fetch_mlb_weather(date_iso)
    print(f"MLB weather hydrate: {sum(1 for w in mlb_weather.values() if w.get('roof_state_today')):,}/{len(games)} games have a parseable roof_state")

    # Re-fetch Open-Meteo per stadium (clear cache so we always hit fresh)
    _clear_open_meteo_cache()
    fresh_weather = {}
    for g in games:
        gpk = int(g.get("game_pk") or 0)
        wx = forecast_at_first_pitch(g.get("home_team"), g.get("game_datetime_utc"))
        fresh_weather[gpk] = wx

    # Patch the matchups JSON
    n_changed = 0
    for g in games:
        gpk = int(g.get("game_pk") or 0)
        # Roof state from MLB (only meaningful for retractable parks)
        if gpk in mlb_weather:
            mw = mlb_weather[gpk]
            new_roof = mw.get("roof_state_today")
            if new_roof and g.get("roof_state_today") != new_roof:
                n_changed += 1
            g["roof_state_today"] = new_roof
            g["weather_condition_mlb"] = mw.get("weather_condition")
            g["weather_temp_mlb"] = mw.get("weather_temp")
            g["weather_wind_mlb"] = mw.get("weather_wind")

    if not args.dry_run:
        MATCHUPS_JSON.write_text(json.dumps(games, indent=2))
        print(f"Patched {MATCHUPS_JSON} with refreshed roof + MLB weather strings ({n_changed} roof_state changes)")

    if args.no_regen:
        print("--no-regen: skipping Section 15 + dashboard regen.")
        return

    # Re-run Section 15 only (it'll pull the freshest Open-Meteo via the wired call)
    res = subprocess.run(
        [sys.executable, str(ROOT / "src" / "gen_zero_hr_predictions.py")],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("WARN: Section 15 regen failed:", res.stderr, file=sys.stderr)
    else:
        print("Section 15 regenerated.")

    # Diff vs pre-state
    if pre_section15 and SECTION_15_JSON.exists():
        try:
            payload = json.loads(SECTION_15_JSON.read_text())
            print()
            print("=== Diff vs pre-refresh ===")
            for r in payload.get("games", []):
                gpk = int(r.get("game_pk") or 0)
                old = pre_section15.get(gpk)
                if not old:
                    continue
                old_p = (old.get("p_zero_hr") or 0.0) * 100
                new_p = (r.get("p_zero_hr") or 0.0) * 100
                old_mult = old.get("weather_hr_mult") or 1.0
                new_mult = r.get("weather_hr_mult") or 1.0
                old_roof = old.get("roof_state_today") or "—"
                new_roof = r.get("roof_state_today") or "—"
                tag = ""
                if abs(new_p - old_p) >= 0.5 or old_roof != new_roof:
                    tag = "  <-- CHANGED"
                print(
                    f"  {r.get('matchup','?'):<14} roof: {str(old_roof):<8} -> {str(new_roof):<8}  "
                    f"wx_mult: {old_mult:.3f} -> {new_mult:.3f}  "
                    f"P(0 HR): {old_p:5.2f}% -> {new_p:5.2f}%{tag}"
                )
        except Exception as ex:
            print(f"WARN: diff failed: {ex}", file=sys.stderr)

    # Regenerate dashboard
    print("\nRegenerating dashboard...")
    res2 = subprocess.run(
        [sys.executable, str(ROOT / "src" / "gen_matchup_dashboard_html.py"),
         "--date", date_iso],
        capture_output=True, text=True,
    )
    if res2.returncode != 0:
        print("WARN: dashboard regen failed:", res2.stderr, file=sys.stderr)
    else:
        print("Dashboard regenerated.")

    print(f"\nRefresh complete at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
