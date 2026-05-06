"""
Ballpark static metrics: stadium exposure (outdoor / fixed dome / retractable) and
seasonal P(roof open) priors for retractable sites.

See README_park_lookup.md. Column park_hook_reserved is a zero placeholder (schema stability).

Join key: home_team (3-letter, normalized) + calendar month for retractable priors.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from config import RAW_DIR

PARK_LOOKUP_CSV = RAW_DIR / "park_lookup.csv"
STATCAST_HOME_TEAM_ALIASES: dict[str, str] = {
    "ANA": "LAA",
    "WAS": "WSH",
    "AZ": "ARI",
    "ATH": "OAK",  # rare
    "KCR": "KC",  # some feeds
    "KCA": "KC",
    "SFG": "SF",
    "SDP": "SD",
    "TBR": "TB",
    "TBA": "TB",
}

# Feature names produced by add_park_features / inference helper (must match model column names)
PARK_FEATURE_NAMES: tuple[str, ...] = (
    "park_stadium_type_outdoor",
    "park_stadium_type_fixed_dome",
    "park_stadium_type_retractable",
    "park_p_roof_open",
    "park_p_roof_closed",
    "park_p_effective_plays_outside",
    "park_p_effective_plays_enclosed",
    "park_p_unknown_stadium",  # 1 when no join (missing home_team / unknown abbrev)
    "park_hook_reserved",
)


def normalize_mlb_home_team(abbr: str | None) -> str | None:
    if abbr is None or (isinstance(abbr, float) and math.isnan(abbr)):
        return None
    s = str(abbr).strip().upper()[:3]
    if not s or s in {"NAN", "NAT"}:
        return None
    return STATCAST_HOME_TEAM_ALIASES.get(s, s)


def _month_index(ts: Any) -> int:
    if ts is None:
        return 6
    t = pd.Timestamp(ts)
    m = int(t.month)
    return max(1, min(12, m))


def _p_open_for_row(enclosure: str, months: list[float] | None, m: int) -> float:
    en = (enclosure or "outdoor").strip().lower()
    if en == "outdoor":
        return 1.0
    if en == "fixed_dome":
        return 0.0
    if en == "retractable":
        if not months or len(months) != 12:
            return 0.55
        return float(max(0.0, min(1.0, months[m - 1])))
    return 0.5


def load_park_lookup() -> pd.DataFrame:
    if not PARK_LOOKUP_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(PARK_LOOKUP_CSV)


_park_cache: pd.DataFrame | None = None


def get_park_table() -> pd.DataFrame:
    global _park_cache
    if _park_cache is None or _park_cache.empty:
        _park_cache = load_park_lookup()
    return _park_cache


def _month_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i in range(1, 13):
        col = f"p_open_{i:02d}"
        if col not in out.columns:
            out[col] = 0.0
    return out


def _row_to_month_list(row: pd.Series) -> list[float]:
    return [float(row.get(f"p_open_{i:02d}", 0) or 0) for i in range(1, 13)]


def _compute_park_from_enclosure(
    enclosure: str,
    months: list[float] | None,
    month: int,
) -> dict[str, float]:
    en = (enclosure or "outdoor").strip().lower()
    p_open = _p_open_for_row(en, months, month)
    p_closed = 1.0 - p_open

    type_out = 1.0 if en == "outdoor" else 0.0
    type_fd = 1.0 if en == "fixed_dome" else 0.0
    type_re = 1.0 if en == "retractable" else 0.0

    if en == "outdoor":
        p_roof_open, p_roof_closed = 1.0, 0.0
        p_out, p_in = 1.0, 0.0
    elif en == "fixed_dome":
        p_roof_open, p_roof_closed = 0.0, 1.0
        p_out, p_in = 0.0, 1.0
    else:  # retractable
        p_roof_open, p_roof_closed = p_open, p_closed
        p_out, p_in = p_open, p_closed

    return {
        "park_stadium_type_outdoor": type_out,
        "park_stadium_type_fixed_dome": type_fd,
        "park_stadium_type_retractable": type_re,
        "park_p_roof_open": p_roof_open,
        "park_p_roof_closed": p_roof_closed,
        "park_p_effective_plays_outside": p_out,
        "park_p_effective_plays_enclosed": p_in,
        "park_p_unknown_stadium": 0.0,
        "park_hook_reserved": 0.0,
    }


def _unknown_park_dict() -> dict[str, float]:
    return {
        "park_stadium_type_outdoor": 0.0,
        "park_stadium_type_fixed_dome": 0.0,
        "park_stadium_type_retractable": 0.0,
        "park_p_roof_open": 0.5,
        "park_p_roof_closed": 0.5,
        "park_p_effective_plays_outside": 0.5,
        "park_p_effective_plays_enclosed": 0.5,
        "park_p_unknown_stadium": 1.0,
        "park_hook_reserved": 0.0,
    }


def get_park_features_for_inference(
    home_team: str | None,
    game_date: Any,
) -> dict[str, float]:
    """
    Return park_* feature dict for a scheduled game. Uses same logic as add_park_features
    (monthly priors for retractable; outdoor vs inside-enclosed for fixed dome).
    """
    tid = normalize_mlb_home_team(home_team)
    m = _month_index(game_date)
    tbl = get_park_table()
    if tid is None or tbl.empty or tid not in set(tbl["home_team"].astype(str).str.upper().str.strip()):
        return _unknown_park_dict()
    row = tbl.loc[tbl["home_team"].astype(str).str.upper().str.strip() == tid].iloc[0]
    en = str(row.get("enclosure", "outdoor"))
    months = _row_to_month_list(row)
    d = _compute_park_from_enclosure(en, months, m)
    hook = row.get("park_hook_reserved", row.get("weather_hook_reserved"))
    if hook is not None and pd.notna(hook):
        d["park_hook_reserved"] = float(hook or 0)
    return d


def add_park_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join park metrics to PA-level frame. Expects `home_team` and `game_date` when available.
    """
    if df.empty:
        return df
    out = df.copy()
    if "home_team" not in out.columns:
        for k, v in _unknown_park_dict().items():
            out[k] = v
        return out

    tbl = get_park_table()
    if tbl.empty:
        for k, v in _unknown_park_dict().items():
            out[k] = v
        return out
    tbl = _month_columns_df(tbl)
    key = out["home_team"].map(
        lambda x: normalize_mlb_home_team(x) if pd.notna(x) else None
    )
    out["__park_key__"] = key
    out["__month__"] = pd.to_datetime(out["game_date"], errors="coerce").dt.month.fillna(6).astype(int).clip(1, 12)

    d_rows: list[dict[str, float]] = []
    for _, r in out.iterrows():
        tid = r["__park_key__"]
        mo = int(r["__month__"])
        if not tid or tid not in set(tbl["home_team"].astype(str).str.upper().str.strip()):
            d_rows.append(_unknown_park_dict())
            continue
        prow = tbl.loc[tbl["home_team"].astype(str).str.upper().str.strip() == tid].iloc[0]
        en = str(prow.get("enclosure", "outdoor"))
        months = _row_to_month_list(prow)
        d = _compute_park_from_enclosure(en, months, mo)
        phook = prow.get("park_hook_reserved", prow.get("weather_hook_reserved"))
        if phook is not None and pd.notna(phook):
            d["park_hook_reserved"] = float(phook or 0)
        d_rows.append(d)

    for fn in PARK_FEATURE_NAMES:
        out[fn] = [d[fn] for d in d_rows]

    out = out.drop(columns=["__park_key__", "__month__"], errors="ignore")
    return out


def park_context_narration(home_team: str | None, game_date: Any) -> dict[str, Any]:
    """Readable bundle for JSON / UI (separate from model feature vector)."""
    d = get_park_features_for_inference(home_team, game_date)
    tid = normalize_mlb_home_team(home_team)
    tbl = get_park_table()
    name = None
    enc = None
    if tid and not tbl.empty and tid in set(tbl["home_team"].astype(str).str.upper().str.strip()):
        r = tbl.loc[tbl["home_team"].astype(str).str.upper().str.strip() == tid].iloc[0]
        name = r.get("venue_name")
        enc = r.get("enclosure")

    p_open = d["park_p_roof_open"]
    p_cl = d["park_p_roof_closed"]
    is_ret = enc == "retractable"
    is_out = enc == "outdoor"
    is_fd = enc == "fixed_dome"
    return {
        "home_team_abbrev": tid,
        "venue_name": name,
        "enclosure": enc,
        "p_roof_open": round(p_open, 4) if is_ret else (1.0 if is_out else 0.0 if is_fd else None),
        "p_roof_closed": round(p_cl, 4) if is_ret else (0.0 if is_out else 1.0 if is_fd else None),
        "p_effective_plays_outside": round(d["park_p_effective_plays_outside"], 4),
        "p_effective_plays_enclosed": round(d["park_p_effective_plays_enclosed"], 4),
        "stadium_type_one_hot": {
            "outdoor": d["park_stadium_type_outdoor"] > 0.5,
            "fixed_dome": d["park_stadium_type_fixed_dome"] > 0.5,
            "retractable": d["park_stadium_type_retractable"] > 0.5,
        },
        "narration": (
            f"Stadium: {name or tid or 'Unknown'} — {enc or 'unknown'}; "
            f"effective P(plays as open air / to outside) = {d['park_p_effective_plays_outside']:.2f}, "
            f"P(plays as enclosed/inside) = {d['park_p_effective_plays_enclosed']:.2f}. "
            + (
                f" Retractable: P(roof open) = {p_open:.2f}, P(closed) = {p_cl:.2f} (seasonal prior)."
                if is_ret
                else (
                    " Open-air: always outside the fixed building shell for roof purposes."
                    if is_out
                    else (
                        " Fixed roof: play is always inside the enclosed (non-retractable) shell."
                        if is_fd
                        else " Ballpark not found in park_lookup; features use unknown-stadium imputation."
                    )
                )
            )
        ),
    }
