"""
Open-Meteo weather fetcher for live MLB stadium forecasts.

Public API:
  forecast_at_first_pitch(home_team, game_datetime_utc) -> dict
      Returns the hourly forecast at the stadium for the hour matching first pitch.
      Output dict has temp_f, wind_mph, wind_dir_deg, humidity_pct, source, fetched_at.

  hr_carry_multiplier(weather, park_lookup_row) -> dict
      Returns {hr_mult, breakdown} where hr_mult is roughly 0.85..1.15.
      Indoor games (fixed_dome OR retractable+closed) -> hr_mult = 1.0.

Open-Meteo: free, no API key, 5km grid, ECMWF + GFS ensemble.
  https://open-meteo.com/en/docs

Caching: in-memory dict keyed on (lat, lon, hour_iso) to avoid duplicate calls
within a single dashboard build. Persistent disk cache is overkill for the
current usage.

Optional `data/priors/weather_slope_override.json` (from
`calibrate_section15_weather_slopes.py --write`) can shrink combined weather
multipliers toward 1.0 and optionally override TEMP/WIND slopes at runtime.
"""
from __future__ import annotations

import functools
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR  # noqa: E402

PARK_LOOKUP_CSV = RAW_DIR / "park_lookup.csv"

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10  # seconds

# Multiplier slopes (conservative defaults; refined after Phase 1 measurement)
TEMP_BASE_F = 70.0
TEMP_PER_F = 0.004           # +0.4% per F above 70
TEMP_CAP_PCT = 0.08          # max +/-8% from temp alone
WIND_PER_MPH_OUT = 0.010     # +1.0% per mph blowing OUT to CF
WIND_PER_MPH_IN = 0.010      # -1.0% per mph blowing IN from CF
WIND_CAP_PCT = 0.12          # max +/-12% from wind alone
GLOBAL_MULT_FLOOR = 0.80     # absolute floor on hr_mult (defensive)
GLOBAL_MULT_CEIL = 1.20      # absolute ceiling on hr_mult (defensive)

_WX_OVERRIDE_PATH = ROOT / "data" / "priors" / "weather_slope_override.json"
_WX_OVERRIDE_MT: float | None = None
_WX_OVERRIDE_CACHE: dict = {}


def _wx_override_blob() -> dict:
    """Optional JSON overrides from calibrate_section15_weather_slopes --write."""
    global _WX_OVERRIDE_MT, _WX_OVERRIDE_CACHE
    if not _WX_OVERRIDE_PATH.exists():
        return {}
    try:
        mt = _WX_OVERRIDE_PATH.stat().st_mtime
    except OSError:
        return {}
    if _WX_OVERRIDE_MT != mt:
        try:
            _WX_OVERRIDE_CACHE = json.loads(_WX_OVERRIDE_PATH.read_text())
        except Exception:
            _WX_OVERRIDE_CACHE = {}
        _WX_OVERRIDE_MT = mt
    return _WX_OVERRIDE_CACHE


def _wx_float(key: str, default: float) -> float:
    blob = _wx_override_blob()
    if key not in blob:
        return default
    try:
        return float(blob[key])
    except (TypeError, ValueError):
        return default


def _apply_weather_mult_shrink(raw_mult: float) -> float:
    """Shrink combined temp×wind multiplier toward 1.0 (Priority 3 knob)."""
    s = _wx_float("WEATHER_MULT_SHRINK", 1.0)
    s = max(0.0, min(1.0, s))
    return float(1.0 + s * (float(raw_mult) - 1.0))


@functools.lru_cache(maxsize=1)
def _park_lookup_df() -> pd.DataFrame:
    df = pd.read_csv(PARK_LOOKUP_CSV)
    return df


def _park_row(home_team: str) -> dict | None:
    if not home_team:
        return None
    df = _park_lookup_df()
    sub = df[df["home_team"].astype(str).str.upper() == str(home_team).upper()]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _hour_floor_iso(dt_utc: datetime) -> str:
    return dt_utc.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")


def _project_wind_to_cf(wind_dir_deg: float, wind_mph: float, cf_bearing_deg: float) -> float:
    """Signed projection of wind onto stadium's CF bearing, in mph.

    Open-Meteo `wind_direction_10m` follows meteorological convention: the
    direction the wind is COMING FROM. To find "blowing toward CF", compare
    the wind's coming-from bearing to (cf_bearing + 180) mod 360, the
    bearing wind would come FROM if blowing toward CF. The cosine of the
    angular difference gives the projection.

    Returns positive mph for wind blowing OUT toward CF, negative for IN.
    """
    if wind_mph <= 0 or wind_dir_deg is None:
        return 0.0
    coming_from_for_out = (cf_bearing_deg + 180.0) % 360.0
    delta = math.radians(wind_dir_deg - coming_from_for_out)
    return float(wind_mph * math.cos(delta))


def _wind_orientation_label(out_mph: float) -> str:
    if abs(out_mph) < 3.0:
        return "calm" if abs(out_mph) < 1.0 else "cross"
    return "out_to_cf" if out_mph > 0 else "in_from_cf"


@functools.lru_cache(maxsize=512)
def _fetch_open_meteo_cached(lat: float, lon: float, date_iso: str) -> dict:
    """One HTTP call returns 24 hourly rows for the given date at lat/lon."""
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m,precipitation_probability",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": date_iso,
        "end_date": date_iso,
    }
    try:
        r = requests.get(OPEN_METEO_BASE, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as ex:
        return {"_error": str(ex)}


def fetch_open_meteo(lat: float, lon: float, dt_utc: datetime) -> dict | None:
    """Returns the single-hour forecast row matching dt_utc, or None on failure."""
    date_iso = dt_utc.strftime("%Y-%m-%d")
    payload = _fetch_open_meteo_cached(round(lat, 4), round(lon, 4), date_iso)
    if "_error" in payload:
        return None
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    target = _hour_floor_iso(dt_utc)
    if target not in times:
        # Pick nearest hour as fallback
        if not times:
            return None
        try:
            target_dt = datetime.strptime(target, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            best_idx, best_diff = 0, None
            for i, t in enumerate(times):
                tdt = datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                diff = abs((tdt - target_dt).total_seconds())
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_idx = i
            idx = best_idx
        except Exception:
            idx = 0
    else:
        idx = times.index(target)

    def _at(name):
        arr = hourly.get(name) or []
        return arr[idx] if idx < len(arr) else None

    return {
        "time_utc": times[idx] if idx < len(times) else target,
        "temp_f": _at("temperature_2m"),
        "wind_mph": _at("wind_speed_10m"),
        "wind_dir_deg": _at("wind_direction_10m"),
        "humidity_pct": _at("relative_humidity_2m"),
        "precip_prob_pct": _at("precipitation_probability"),
    }


def forecast_at_first_pitch(home_team: str, game_datetime_utc: str | None) -> dict:
    """Returns weather at the stadium for the hour of first pitch.

    Always returns a dict with `source` field; on any failure, source is
    `"unavailable"` and numeric fields are None.
    """
    out = {
        "home_team": home_team,
        "source": "unavailable",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temp_f": None,
        "wind_mph": None,
        "wind_dir_deg": None,
        "humidity_pct": None,
        "precip_prob_pct": None,
        "time_utc": None,
    }
    park = _park_row(home_team)
    if not park or pd.isna(park.get("lat")) or pd.isna(park.get("lon")):
        out["source"] = "unavailable_no_coords"
        return out
    if not game_datetime_utc:
        out["source"] = "unavailable_no_game_time"
        return out
    try:
        dt = pd.to_datetime(game_datetime_utc, utc=True).to_pydatetime()
    except Exception:
        out["source"] = "unavailable_bad_game_time"
        return out

    fc = fetch_open_meteo(float(park["lat"]), float(park["lon"]), dt)
    if fc is None:
        return out
    out.update(fc)
    out["source"] = "open_meteo"
    return out


@functools.lru_cache(maxsize=1)
def _pull_rates_df() -> pd.DataFrame:
    p = ROOT / "data" / "raw" / "batter_pull_rates.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _batter_pull_info(batter_id: int) -> tuple[float, str]:
    """Returns (pull_rate, stand). Defaults to (0.66, 'R') if not found."""
    df = _pull_rates_df()
    if df.empty:
        return (0.66, "R")
    row = df[df["batter"] == int(batter_id)]
    if row.empty:
        return (0.66, "R")
    r = row.iloc[0]
    return (float(r["pull_rate"]), str(r["stand"]))


def per_batter_hr_multiplier(weather: dict, park_row: dict | None,
                              batter_id: int,
                              roof_state: str | None = None) -> dict:
    """T2.3: per-batter weather multiplier accounting for batter pull rate.

    A wind-out-RF day at Wrigley benefits a high-pull lefty (Schwarber) far
    more than a spread-field righty (Aaron Judge). This helper projects wind
    onto each batter's primary HR direction and weights by pull rate.

    Returns same dict shape as hr_carry_multiplier() but with hr_mult adjusted
    for batter handedness × pull rate × wind direction.
    """
    pull_rate, stand = _batter_pull_info(batter_id)
    enclosure = (park_row or {}).get("enclosure", "outdoor")
    indoor = (
        enclosure == "fixed_dome"
        or (enclosure == "retractable" and (roof_state or "").lower() == "closed")
    )
    breakdown = {
        "indoor": indoor,
        "batter_id": int(batter_id),
        "pull_rate": pull_rate,
        "stand": stand,
        "temp_mult": 1.0,
        "wind_mult": 1.0,
        "wind_to_pullside_mph": 0.0,
        "wind_orientation": "indoor" if indoor else None,
    }
    if indoor:
        return {"hr_mult": 1.0, "breakdown": breakdown}

    # Temperature contribution (same as game-level)
    temp_mult = 1.0
    temp_f = weather.get("temp_f")
    _tpf = _wx_float("TEMP_PER_F", TEMP_PER_F)
    _tcp = _wx_float("TEMP_CAP_PCT", TEMP_CAP_PCT)
    if temp_f is not None:
        delta_f = float(temp_f) - TEMP_BASE_F
        adjustment = max(-_tcp, min(_tcp, delta_f * _tpf))
        temp_mult = 1.0 + adjustment
    breakdown["temp_mult"] = round(temp_mult, 4)

    # Wind contribution: project onto BATTER's primary HR direction.
    # Lefty pull = RF (~+45deg from CF bearing). Righty pull = LF (~-45deg).
    wind_mult = 1.0
    wind_mph = weather.get("wind_mph")
    wind_dir = weather.get("wind_dir_deg")
    cf_bearing = (park_row or {}).get("wind_orientation_cf_deg", 0)
    out_to_pullside = 0.0
    if wind_mph is not None and wind_dir is not None and cf_bearing is not None:
        # Pull-side bearing offset from CF: +45 for LH, -45 for RH
        pull_offset = 45.0 if stand == "L" else -45.0
        pull_bearing = (float(cf_bearing) + pull_offset) % 360.0
        out_to_pullside = _project_wind_to_cf(float(wind_dir), float(wind_mph), pull_bearing)
        # Pull-rate-weighted: pure pull batter feels full pull-side wind;
        # spread batter feels closer to dead-CF wind.
        cf_proj = _project_wind_to_cf(float(wind_dir), float(wind_mph), float(cf_bearing))
        effective = pull_rate * out_to_pullside + (1.0 - pull_rate) * cf_proj
        _wcap = _wx_float("WIND_CAP_PCT", WIND_CAP_PCT)
        _wout = _wx_float("WIND_PER_MPH_OUT", WIND_PER_MPH_OUT)
        _win = _wx_float("WIND_PER_MPH_IN", WIND_PER_MPH_IN)
        if effective >= 0:
            adj = min(_wcap, effective * _wout)
        else:
            adj = max(-_wcap, effective * _win)
        wind_mult = 1.0 + adj
    breakdown["wind_mult"] = round(wind_mult, 4)
    breakdown["wind_to_pullside_mph"] = round(out_to_pullside, 2)
    breakdown["wind_orientation"] = _wind_orientation_label(out_to_pullside)

    hr_mult = float(temp_mult * wind_mult)
    hr_mult = _apply_weather_mult_shrink(hr_mult)
    hr_mult = max(GLOBAL_MULT_FLOOR, min(GLOBAL_MULT_CEIL, hr_mult))
    return {"hr_mult": round(hr_mult, 4), "breakdown": breakdown}


def hr_carry_multiplier(weather: dict, park_row: dict | None,
                        roof_state: str | None = None) -> dict:
    """Returns {hr_mult, breakdown} where hr_mult is in approximately [0.80, 1.20].

    Indoor games (fixed_dome OR retractable+closed roof) -> hr_mult = 1.0
    (weather doesn't matter inside).

    Outdoor temp slope: +/-0.4% per 1F off 70F (cap +/-8%).
    Wind slope: +/-1.0% per mph projected onto stadium's CF bearing (cap +/-12%).
    """
    enclosure = (park_row or {}).get("enclosure", "outdoor")
    indoor = (
        enclosure == "fixed_dome"
        or (enclosure == "retractable" and (roof_state or "").lower() == "closed")
    )
    breakdown = {
        "indoor": indoor,
        "enclosure": enclosure,
        "roof_state": roof_state,
        "temp_mult": 1.0,
        "wind_mult": 1.0,
        "wind_out_mph": 0.0,
        "wind_orientation": "indoor" if indoor else None,
    }
    if indoor:
        return {"hr_mult": 1.0, "breakdown": breakdown}

    # Temperature contribution
    temp_mult = 1.0
    temp_f = weather.get("temp_f")
    _tpf = _wx_float("TEMP_PER_F", TEMP_PER_F)
    _tcp = _wx_float("TEMP_CAP_PCT", TEMP_CAP_PCT)
    if temp_f is not None:
        delta_f = float(temp_f) - TEMP_BASE_F
        adjustment = max(-_tcp, min(_tcp, delta_f * _tpf))
        temp_mult = 1.0 + adjustment
    breakdown["temp_mult"] = round(temp_mult, 4)

    # Wind contribution
    wind_mult = 1.0
    wind_mph = weather.get("wind_mph")
    wind_dir = weather.get("wind_dir_deg")
    cf_bearing = (park_row or {}).get("wind_orientation_cf_deg", 0)
    out_mph = 0.0
    _wcap = _wx_float("WIND_CAP_PCT", WIND_CAP_PCT)
    _wout = _wx_float("WIND_PER_MPH_OUT", WIND_PER_MPH_OUT)
    _win = _wx_float("WIND_PER_MPH_IN", WIND_PER_MPH_IN)
    if wind_mph is not None and wind_dir is not None and cf_bearing is not None:
        out_mph = _project_wind_to_cf(float(wind_dir), float(wind_mph), float(cf_bearing))
        if out_mph >= 0:
            adj = min(_wcap, out_mph * _wout)
        else:
            adj = max(-_wcap, out_mph * _win)
        wind_mult = 1.0 + adj
    breakdown["wind_mult"] = round(wind_mult, 4)
    breakdown["wind_out_mph"] = round(out_mph, 2)
    breakdown["wind_orientation"] = _wind_orientation_label(out_mph)

    hr_mult = float(temp_mult * wind_mult)
    hr_mult = _apply_weather_mult_shrink(hr_mult)
    hr_mult = max(GLOBAL_MULT_FLOOR, min(GLOBAL_MULT_CEIL, hr_mult))
    return {"hr_mult": round(hr_mult, 4), "breakdown": breakdown}


# ---- CLI for manual sanity checks ----
def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Open-Meteo weather fetch for an MLB stadium.")
    p.add_argument("home_team", help="3-letter abbreviation, e.g. CHC")
    p.add_argument("--datetime", required=True,
                   help="UTC ISO datetime, e.g. 2026-04-28T22:10:00Z")
    p.add_argument("--roof", default=None, help="Override live roof state (open/closed)")
    args = p.parse_args()
    park = _park_row(args.home_team)
    if not park:
        print(f"Unknown home_team: {args.home_team}")
        sys.exit(1)
    fc = forecast_at_first_pitch(args.home_team, args.datetime)
    mult = hr_carry_multiplier(fc, park, roof_state=args.roof)
    print(f"Stadium: {park.get('venue_name')} ({park.get('lat')}, {park.get('lon')})")
    print(f"Forecast hour: {fc.get('time_utc')}")
    print(f"  Temp: {fc.get('temp_f')} F   Wind: {fc.get('wind_mph')} mph @ {fc.get('wind_dir_deg')} deg")
    print(f"  Humidity: {fc.get('humidity_pct')}%   Precip prob: {fc.get('precip_prob_pct')}%")
    print(f"  Source: {fc.get('source')}")
    print()
    bd = mult["breakdown"]
    print(f"hr_mult = {mult['hr_mult']}")
    print(f"  temp_mult = {bd['temp_mult']}, wind_mult = {bd['wind_mult']}")
    print(f"  wind projected onto CF bearing: {bd['wind_out_mph']} mph ({bd['wind_orientation']})")
    if bd["indoor"]:
        print("  indoor=True (weather ignored)")


if __name__ == "__main__":
    _cli()
