"""
Build PA-level rows for one calendar day from MLB Stats API play-by-play.

Used when Baseball Savant / pybaseball returns no pitch rows for that date but
games are Final (common lag or alternate data availability).
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
WALK_EVENTS = frozenset({"walk", "hit_by_pitch", "intent_walk"})

# MLB Stats API eventType -> Statcast-style `events` string used in league parquet.
_EVENT_TYPE_TO_STATCAST: dict[str, str] = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    "walk": "walk",
    "intent_walk": "intent_walk",
    "hit_by_pitch": "hit_by_pitch",
    "field_out": "field_out",
    "strikeout": "strikeout",
    "force_out": "force_out",
    "grounded_into_double_play": "grounded_into_double_play",
    "field_error": "field_error",
    "sac_fly": "sac_fly",
    "sac_bunt": "sac_bunt",
    "double_play": "double_play",
    "fielders_choice": "fielders_choice",
    "catcher_interf": "catcher_interf",
}


def _map_event_type(event_type: str | None) -> str:
    et = (event_type or "").strip()
    if et in _EVENT_TYPE_TO_STATCAST:
        return _EVENT_TYPE_TO_STATCAST[et]
    # Rare / non-Statcast labels — keep scoring & hit logic conservative.
    if "sac" in et:
        return "sac_fly" if "fly" in et else "sac_bunt"
    if "pickoff" in et or "stealing" in et or "caught_stealing" in et:
        return "field_out"
    return et if et else "field_out"


def _apply_pa_flags(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    ev = pa["events"].fillna("").str.lower()
    pa["is_hit"] = ev.isin(HIT_EVENTS).astype(int)
    pa["is_hr"] = (ev == "home_run").astype(int)
    pa["is_ab"] = (~ev.isin(WALK_EVENTS | {"catcher_interf", "sac_bunt"})).astype(int)
    pa["is_strikeout"] = ev.str.contains("strikeout", case=False, na=False).astype(int)
    pa["is_walk"] = ev.isin({"walk", "intent_walk"}).astype(int)
    pa["is_hbp"] = (ev == "hit_by_pitch").astype(int)
    pa["is_xbh"] = ev.isin({"double", "triple", "home_run"}).astype(int)
    pa["vs_lhp"] = (pa["p_throws"] == "L").astype(int) if "p_throws" in pa.columns else 0
    pa["barrel"] = 0
    return pa


def _schedule_final_pks_and_home_map(
    d: date,
) -> tuple[list[int], dict[int, str]]:
    """One schedule fetch: final game pks and gamePk → home team abbrev (ballpark key)."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d.isoformat()}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    pks: list[int] = []
    m: dict[int, str] = {}
    for block in data.get("dates", []):
        for g in block.get("games", []):
            try:
                pk = int(g["gamePk"])
            except (KeyError, TypeError, ValueError):
                continue
            h = (g.get("teams") or {}).get("home") or {}
            t = h.get("team") or {}
            ab = (t.get("abbreviation") or t.get("fileCode") or "")[:3]
            if ab:
                m[pk] = ab.upper()
            if g.get("status", {}).get("detailedState") == "Final":
                pks.append(pk)
    return pks, m


def _plays_to_rows(game_pk: int, game_date: pd.Timestamp, plays: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for play in plays:
        res = play.get("result") or {}
        if res.get("type") != "atBat":
            continue
        mu = play.get("matchup") or {}
        batter = mu.get("batter") or {}
        pitcher = mu.get("pitcher") or {}
        bid = batter.get("id")
        pid = pitcher.get("id")
        if bid is None or pid is None:
            continue
        about = play.get("about") or {}
        at_idx = about.get("atBatIndex")
        if at_idx is None:
            continue
        et = res.get("eventType")
        events = _map_event_type(et)
        stand = (mu.get("batSide") or {}).get("code") or ""
        p_throws = (mu.get("pitchHand") or {}).get("code") or ""
        name = batter.get("fullName") or ""
        desc = res.get("description") or ""

        rows.append(
            {
                "game_pk": int(game_pk),
                "game_date": game_date,
                "game_year": int(game_date.year),
                "at_bat_number": int(at_idx),
                "batter": int(bid),
                "pitcher": int(pid),
                "events": events,
                "description": desc[:500],
                "p_throws": str(p_throws)[:1] or "R",
                "stand": str(stand)[:1] or "R",
                "player_name_clean": str(name)[:120],
            }
        )
    return rows


def fetch_pa_rows_from_statsapi(d: date, *, sleep_s: float = 0.15) -> pd.DataFrame:
    """Return PA-level rows aligned with `statcast_pa_level_league.parquet` columns (sparse)."""
    day_ts = pd.Timestamp(d.isoformat())
    pks, home_by_pk = _schedule_final_pks_and_home_map(d)
    if not pks:
        return pd.DataFrame()

    all_rows: list[dict] = []
    for i, pk in enumerate(pks):
        url = f"https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        pb = r.json()
        all_rows.extend(_plays_to_rows(pk, day_ts, pb.get("allPlays") or []))
        if sleep_s > 0 and i + 1 < len(pks):
            time.sleep(sleep_s)

    if not all_rows:
        return pd.DataFrame()

    pa = pd.DataFrame(all_rows)
    pa["home_team"] = pa["game_pk"].map(lambda gpk: home_by_pk.get(int(gpk), pd.NA))
    pa = _apply_pa_flags(pa)
    pa["pa_mean_velo"] = pd.NA
    pa["pa_max_velo"] = pd.NA
    pa["pa_pitch_count"] = pd.NA
    pa["pitch_number"] = pd.NA
    for c in (
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "hit_distance_sc",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value",
        "zone",
        "plate_x",
        "plate_z",
        "release_speed",
        "release_spin_rate",
        "pfx_x",
        "pfx_z",
        "pitch_type",
        "pitch_name",
        "effective_speed",
        "release_extension",
        "arm_angle",
        "spin_axis",
        "n_thruorder_pitcher",
        "pitcher_days_since_prev_game",
        "age_pit",
    ):
        pa[c] = pd.NA
    return pa
