#!/usr/bin/env python3
"""
Build a self-contained multi-model HTML dashboard: tabbed UI, side-by-side
top-25 model rankings (CSV export), raw BvP leaderboards (hit-type split),
barrel and laser (100+ mph EV) outcomes from Statcast, optional recency and Beast
tabs (Beast includes three collapsed prose panels — Hits / XBH / HR — compact write-ups +
leaderboard blurbs), and longform sections 7/8/10 from markdown on Matchups · Long.

Reads:
  data/reports/todays_matchup_predictions.json
  data/reports/todays_matchup_predictions_exp.json
  data/reports/todays_matchup_predictions_recency.json (optional; same slate_date as prod when present)
  data/reports/archive/YYYY-MM-DD/todays_matchup_predictions*.json (optional; embedded when multiple slates exist)
  data/raw/statcast_pa_level_league.parquet (BvP XBH counts; also calendar-day batter actuals for Results tab)
  data/reports/section_11.md
  data/reports/section_7_prod.md … section_10_prod.md (model 1 tab; fallback: section_*.md)
  data/reports/section_7_exp.md … section_10_exp.md (model 2 tab)
  data/tracking/matchup_predictions_runs.parquet (optional outcome overrides)

Writes:
  data/reports/matchup_dashboard.html

Results tab: keep Statcast PA parquet updated through the prior calendar day before
regenerating HTML so recent slates show ✓/✗ vs real plate appearances (dual-model
Parquet rows no longer rely on embedded outcome_* columns only).
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from collections.abc import Iterable

import pandas as pd

from residual_longshots import (
    ADJ_HR_MAX,
    ADJ_HR_MIN,
    compute_residual_longshots,
    evaluate_residual_history,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
RAW = ROOT / "data" / "raw"
TRACKING = ROOT / "data" / "tracking"
sys.path.insert(0, str(ROOT / "src"))
from gen_conviction_picks import SEC14_MAX_PICKS_PER_TARGET, SEC14_MIN_HAND_LABEL  # noqa: E402

_HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
_XBH_EVENTS = frozenset({"double", "triple", "home_run"})

_PITCH_PROFILE_DF_CACHE: pd.DataFrame | None = None
_BATTER_PROFILE_DF_CACHE: pd.DataFrame | None = None
_BEAST_RECENT_BIP_EV: dict[int, dict[str, float | int | None]] | None = None

# Rough Statcast/contact baselines used only for conversational context in Beast takeaways (not calibrated model inputs).
_BEAST_CTX_LEAGUE_EV_BIP = 88.2  # Typical MLB avg exit velo among PA with tracked launch_speed (bulk Statcast-era).
_BEAST_CTX_LEAGUE_HIT = 0.243
_BEAST_CTX_LEAGUE_HR = 0.031
_BEAST_CTX_LEAGUE_XBH = 0.079
_BEAST_CTX_LEAGUE_BATTER_K = 0.215  # ~21.5% PA strikeouts for hitters broadly (order-of-magnitude).
_BEAST_TAIL_BIP = 72  # Last N tracked BIP EV rows per hitter for "recent loud/soft contact" wording.
_BEAST_TAIL_BIP_HAND = 40


def _clear_beast_recent_bip_ev_cache() -> None:
    global _BEAST_RECENT_BIP_EV
    _BEAST_RECENT_BIP_EV = None


def _clear_beast_profile_cache() -> None:
    global _PITCH_PROFILE_DF_CACHE, _BATTER_PROFILE_DF_CACHE
    _PITCH_PROFILE_DF_CACHE = None
    _BATTER_PROFILE_DF_CACHE = None
    _clear_beast_recent_bip_ev_cache()


def _beast_profile_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pitcher/batter seasonal Statcast-shape profiles once per HTML build."""
    global _PITCH_PROFILE_DF_CACHE, _BATTER_PROFILE_DF_CACHE
    if _PITCH_PROFILE_DF_CACHE is None:
        p_path = RAW / "pitcher_profiles_by_season.parquet"
        b_path = RAW / "batter_pitch_profiles.parquet"
        try:
            _PITCH_PROFILE_DF_CACHE = pd.read_parquet(p_path) if p_path.is_file() else pd.DataFrame()
        except Exception:
            _PITCH_PROFILE_DF_CACHE = pd.DataFrame()
        try:
            _BATTER_PROFILE_DF_CACHE = pd.read_parquet(b_path) if b_path.is_file() else pd.DataFrame()
        except Exception:
            _BATTER_PROFILE_DF_CACHE = pd.DataFrame()
    return _PITCH_PROFILE_DF_CACHE, _BATTER_PROFILE_DF_CACHE


def _pitcher_throws_letter_row(r: dict | None) -> str:
    if not r:
        return ""
    raw = str(r.get("pitcher_throws") or "").strip().upper()
    return raw[:1] if raw[:1] in {"R", "L"} else ""


def _median_latest_pitcher_profiles(pp_df: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, float | None]:
    """Median of each numeric mix/rate column using each pitcher's newest season row (MLB-shape context bands)."""
    out: dict[str, float | None] = {c: None for c in cols}
    if pp_df is None or pp_df.empty or "pitcher" not in pp_df.columns:
        return out
    if "season" in pp_df.columns:
        sdf = pp_df.assign(_s=pd.to_numeric(pp_df["season"], errors="coerce"))
        latest = sdf.sort_values(["pitcher", "_s"]).groupby("pitcher", sort=False).tail(1)
    else:
        latest = pp_df.drop_duplicates("pitcher", keep="last")
    for c in cols:
        if c not in latest.columns:
            continue
        ser = pd.to_numeric(latest[c], errors="coerce").dropna()
        if ser.empty:
            continue
        v = float(ser.median())
        if 0 <= v <= 1.06:
            v *= 100.0
        out[c] = v
    return out


def _beast_warm_recent_bip_ev(batter_ids: Iterable[object]) -> None:
    """Pre-scan Statcast league PA parquet for Beast takeaway context (recent BIP loudness vs MLB ~88 mph band)."""
    global _BEAST_RECENT_BIP_EV
    bids = sorted({int(b) for b in batter_ids if b is not None})
    if not bids:
        _BEAST_RECENT_BIP_EV = {}
        return
    path = RAW / "statcast_pa_level_league.parquet"
    if not path.is_file():
        _BEAST_RECENT_BIP_EV = {}
        return
    try:
        df = pd.read_parquet(path, columns=["batter", "game_date", "launch_speed", "p_throws"])
        df = df.dropna(subset=["launch_speed", "batter"])
        df["batter"] = df["batter"].astype(int)
        df = df.loc[df["batter"].isin(bids)]
        if df.empty:
            _BEAST_RECENT_BIP_EV = {b: {} for b in bids}
            return
        df["_ph"] = df["p_throws"].astype(str).str.strip().str.upper().str.slice(0, 1)
        df = df.loc[df["_ph"].isin(["R", "L"])]
        df["_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values(["batter", "_dt"])
        out: dict[int, dict[str, float | int | None]] = {}
        for bid, grp in df.groupby("batter", sort=False):
            grp = grp.sort_values("_dt")
            tail = grp.tail(_BEAST_TAIL_BIP)
            tr = grp[grp["_ph"] == "R"].tail(_BEAST_TAIL_BIP_HAND)
            tl = grp[grp["_ph"] == "L"].tail(_BEAST_TAIL_BIP_HAND)
            out[int(bid)] = {
                "n_recent": int(len(tail)),
                "mean_recent": float(tail["launch_speed"].mean()) if len(tail) else None,
                "n_vs_rhp": int(len(tr)),
                "mean_vs_rhp": float(tr["launch_speed"].mean()) if len(tr) else None,
                "n_vs_lhp": int(len(tl)),
                "mean_vs_lhp": float(tl["launch_speed"].mean()) if len(tl) else None,
            }
        for b in bids:
            out.setdefault(b, {})
        _BEAST_RECENT_BIP_EV = out
    except Exception:
        _BEAST_RECENT_BIP_EV = {}


def _xbh_lookup() -> dict[tuple[int, int], int]:
    pa_path = RAW / "statcast_pa_level_league.parquet"
    if not pa_path.is_file():
        return {}
    df = pd.read_parquet(pa_path, columns=["batter", "pitcher", "events"])
    df["is_xbh"] = df["events"].isin({"double", "triple", "home_run"})
    n_xbh = (
        df.groupby(["batter", "pitcher"], as_index=False)["is_xbh"].sum().rename(columns={"is_xbh": "n_xbh"})
    )
    return {(int(r.batter), int(r.pitcher)): int(r.n_xbh) for r in n_xbh.itertuples()}


def _bvp_raw_parts(r: dict, xbh_lookup: dict[tuple[int, int], int]) -> dict | None:
    bvp = r.get("bvp_career_vs_pitcher") or {}
    pa = int(bvp.get("bvp_pa") or 0)
    if pa <= 0:
        return None
    h = int(bvp.get("bvp_hits") or 0)
    hr = int(bvp.get("bvp_hr") or 0)
    k = int(bvp.get("bvp_k") or 0)
    bb = int(bvp.get("bvp_bb") or 0)
    bid, pid = r.get("batter_mlbam_id"), r.get("pitcher_mlbam_id")
    nx = int(xbh_lookup.get((int(bid), int(pid)), 0))
    ba = float(bvp.get("bvp_ba") or (h / pa if pa else 0.0))
    one_b = max(0, h - nx)
    xbh_nohr = max(0, nx - hr)
    return {"pa": pa, "h": h, "xbh": nx, "hr": hr, "k": k, "bb": bb, "ba": ba, "1b": one_b, "xbh_nohr": xbh_nohr}


def _bvp_hit_types_cell(p: dict) -> str:
    return html.escape(f"1B {p['1b']} · 2B+3B {p['xbh_nohr']} · HR {p['hr']}")


def _is_barrel_mask(df: pd.DataFrame) -> pd.Series:
    lsa = df["launch_speed_angle"] if "launch_speed_angle" in df.columns else None
    br = df["barrel"] if "barrel" in df.columns else None
    out = pd.Series(False, index=df.index, dtype=bool)
    if lsa is not None:
        out = out | lsa.eq(6)
    if br is not None:
        out = out | br.fillna(0).astype(int).gt(0)
    return out


def _barrel_outcome_strings(pair_set: set[tuple[int, int]]) -> dict[tuple[int, int], str]:
    """Per (batter, pitcher), count Statcast outcomes on barrel-classified batted balls."""
    if not pair_set:
        return {}
    pa_path = RAW / "statcast_pa_level_league.parquet"
    if not pa_path.is_file():
        return {}
    mi = pd.MultiIndex.from_tuples(list(pair_set), names=["batter", "pitcher"])
    df = pd.read_parquet(
        pa_path,
        columns=["batter", "pitcher", "launch_speed_angle", "barrel", "events"],
    )
    df = df.dropna(subset=["batter", "pitcher"])
    df["batter"] = df["batter"].astype(int)
    df["pitcher"] = df["pitcher"].astype(int)
    df = df.set_index(["batter", "pitcher"])
    sub = df.loc[df.index.intersection(mi)].reset_index()
    if sub.empty:
        return {}
    sub = sub.loc[_is_barrel_mask(sub)].copy()
    if sub.empty:
        return {}
    out: dict[tuple[int, int], str] = {}
    for (bid, pid), g in sub.groupby(["batter", "pitcher"]):
        e = g["events"].fillna("").astype(str).str.lower()
        n1b = int(e.eq("single").sum())
        n2b = int(e.eq("double").sum())
        n3b = int(e.eq("triple").sum())
        nhr = int(e.eq("home_run").sum())
        nhit = n1b + n2b + n3b + nhr
        nother = int(len(g) - nhit)
        parts = [f"1B {n1b}", f"2B {n2b}", f"3B {n3b}", f"HR {nhr}"]
        if nother > 0:
            parts.append(f"non-hit {nother}")
        out[(int(bid), int(pid))] = " · ".join(parts)
    return out


_LASER_EV_MIN_MPH = 100.0


def _compress_statcast_dates(dates: pd.Series) -> str:
    if dates is None or len(dates) == 0:
        return "—"
    by: dict[str, int] = defaultdict(int)
    for d in dates.dropna():
        k = pd.Timestamp(d).strftime("%Y-%m-%d")
        by[k] += 1
    parts = [f"{k}×{v}" if v > 1 else k for k, v in sorted(by.items())]
    return "; ".join(parts) if parts else "—"


def _event_mix_string(events: pd.Series) -> str:
    e = events.fillna("").astype(str).str.lower()
    n1b = int(e.eq("single").sum())
    n2b = int(e.eq("double").sum())
    n3b = int(e.eq("triple").sum())
    nhr = int(e.eq("home_run").sum())
    nhit = n1b + n2b + n3b + nhr
    nother = int(len(e) - nhit)
    parts = [f"1B {n1b}", f"2B {n2b}", f"3B {n3b}", f"HR {nhr}"]
    if nother > 0:
        parts.append(f"other/out {nother}")
    return " · ".join(parts)


def _laser_hard_bvp_by_pair(
    pair_set: set[tuple[int, int]],
    slate_str: str | None,
) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int], dict]]:
    """Career BvP in Statcast feed: laser HRs (HR + EV>=100) and hard non-HR (EV>=100, not HR)."""
    empty: tuple[dict[tuple[int, int], dict], dict[tuple[int, int], dict]] = ({}, {})
    if not pair_set:
        return empty
    pa_path = RAW / "statcast_pa_level_league.parquet"
    if not pa_path.is_file():
        return empty
    slate_day: pd.Timestamp | None = None
    if slate_str:
        try:
            slate_day = pd.Timestamp(str(slate_str)).normalize()
        except (ValueError, TypeError):
            slate_day = None

    mi = pd.MultiIndex.from_tuples(list(pair_set), names=["batter", "pitcher"])
    df = pd.read_parquet(
        pa_path,
        columns=["batter", "pitcher", "launch_speed", "events", "game_date"],
    )
    df = df.dropna(subset=["batter", "pitcher"])
    df["batter"] = df["batter"].astype(int)
    df["pitcher"] = df["pitcher"].astype(int)
    df = df.set_index(["batter", "pitcher"])
    sub = df.loc[df.index.intersection(mi)].reset_index()
    if sub.empty:
        return empty

    ls = pd.to_numeric(sub["launch_speed"], errors="coerce")
    ev = sub["events"].fillna("").astype(str).str.lower()
    gdt = pd.to_datetime(sub["game_date"], errors="coerce").dt.normalize()
    sub = sub.assign(_ls=ls, _ev=ev, _gd=gdt)
    sub = sub.loc[sub["_ls"].notna() & sub["_ls"].ge(_LASER_EV_MIN_MPH)].copy()
    if sub.empty:
        return empty

    on_slate = sub["_gd"].eq(slate_day) if slate_day is not None else pd.Series(False, index=sub.index)

    laser_map: dict[tuple[int, int], dict] = {}
    hard_map: dict[tuple[int, int], dict] = {}

    for (bid, pid), g in sub.groupby(["batter", "pitcher"]):
        bp = (int(bid), int(pid))
        gl = g[g["_ev"].eq("home_run")]
        if not gl.empty:
            laser_map[bp] = {
                "n": int(len(gl)),
                "n_slate": int(on_slate.loc[gl.index].sum()),
                "dates": _compress_statcast_dates(gl["game_date"]),
                "max_ev": float(gl["_ls"].max()),
            }
        gh = g[~g["_ev"].eq("home_run")]
        if not gh.empty:
            hard_map[bp] = {
                "n": int(len(gh)),
                "n_slate": int(on_slate.loc[gh.index].sum()),
                "dates": _compress_statcast_dates(gh["game_date"]),
                "outcomes": _event_mix_string(gh["events"]),
            }

    return laser_map, hard_map


def _laser_hr_table_html(
    label_to_bp: dict[str, tuple[int, int]],
    laser_map: dict[tuple[int, int], dict],
    slate_str: str,
    *,
    table_id: str = "tbl-lasers-hr",
) -> str:
    rows: list[tuple[str, dict]] = []
    for label, bp in label_to_bp.items():
        st = laser_map.get(bp)
        if not st or (st["n"] == 0 and st["n_slate"] == 0):
            continue
        rows.append((label, st))
    rows.sort(key=lambda x: (-x[1]["n"], x[0]))
    if not rows:
        return '<p class="empty">No laser HRs (HR at ≥100 mph exit velo) vs these listed pitchers in this Statcast feed.</p>'
    trs = []
    for label, st in rows:
        mx = st.get("max_ev")
        mx_s = f"{mx:.1f}" if mx is not None else "—"
        trs.append(
            "<tr"
            + _data_teams_attr_from_label(label)
            + ">"
            f"<td>{html.escape(label)}</td>"
            f'<td class="num">{st["n"]}</td>'
            f'<td class="num">{st["n_slate"]}</td>'
            f'<td class="num">{html.escape(mx_s)}</td>'
            f"<td>{html.escape(st['dates'])}</td>"
            "</tr>"
        )
    slate_esc = html.escape(str(slate_str))
    id_esc = html.escape(str(table_id), quote=True)
    return (
        f'<table class="grid sortable" id="{id_esc}"><thead><tr>'
        "<th>Batter vs pitcher</th>"
        "<th>Career laser HRs</th>"
        f"<th>On slate ({slate_esc})</th>"
        "<th>Max EV (mph)</th>"
        "<th>Dates (YYYY-MM-DD, ×n if &gt;1 that day)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(trs)}</tbody></table>"
    )


def _hard_ev_non_hr_table_html(
    label_to_bp: dict[str, tuple[int, int]],
    hard_map: dict[tuple[int, int], dict],
    slate_str: str,
    *,
    table_id: str = "tbl-lasers-hard",
) -> str:
    rows: list[tuple[str, dict]] = []
    for label, bp in label_to_bp.items():
        st = hard_map.get(bp)
        if not st or (st["n"] == 0 and st["n_slate"] == 0):
            continue
        rows.append((label, st))
    rows.sort(key=lambda x: (-x[1]["n"], x[0]))
    if not rows:
        return (
            '<p class="empty">No ≥100 mph batted balls that were not home runs vs these listed pitchers '
            "in this Statcast feed.</p>"
        )
    trs = []
    for label, st in rows:
        oc = html.escape(st["outcomes"])
        trs.append(
            "<tr"
            + _data_teams_attr_from_label(label)
            + ">"
            f"<td>{html.escape(label)}</td>"
            f'<td class="num">{st["n"]}</td>'
            f'<td class="num">{st["n_slate"]}</td>'
            f"<td>{oc}</td>"
            f"<td>{html.escape(st['dates'])}</td>"
            "</tr>"
        )
    slate_esc = html.escape(str(slate_str))
    id_esc = html.escape(str(table_id), quote=True)
    return (
        f'<table class="grid sortable" id="{id_esc}"><thead><tr>'
        "<th>Batter vs pitcher</th>"
        "<th>Career ≥100 mph (non-HR)</th>"
        f"<th>On slate ({slate_esc})</th>"
        "<th>Outcome mix (1B·2B·3B·HR·other)</th>"
        "<th>Dates (YYYY-MM-DD, ×n if &gt;1 that day)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(trs)}</tbody></table>"
    )


def _three_pa(p: float) -> float:
    return 100.0 * (1.0 - (1.0 - float(p)) ** 3)


def _pitcher_hand_phrase(r: dict) -> str:
    """Short tag for proof copy (fits inside parentheses)."""
    t = str(r.get("pitcher_throws") or "").strip().upper()
    if t == "L":
        return "LHP"
    if t == "R":
        return "RHP"
    return "hand N/A"


def _career_rel_word(adj_pct: float, career_pct: float) -> str:
    """How adjusted % relates to career baseline (same units: 0–100)."""
    d = adj_pct - career_pct
    thr = 0.08 if career_pct < 10 else 0.22
    if d > thr:
        return "above"
    if d < -thr:
        return "below"
    return "roughly in line with"


def _first_sentence_from_text(text: str, max_len: int = 170) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if ". " in t:
        one = t.split(". ", 1)[0].strip()
        if not one.endswith((".", "?", "!")):
            one += "."
        out = one
    else:
        out = t
    if len(out) > max_len:
        return out[: max_len - 1].rstrip() + "…"
    return out


def _bvp_proof_snippet_html(r: dict, xbh_lookup: dict[tuple[int, int], int]) -> str | None:
    p = _bvp_raw_parts(r, xbh_lookup)
    if not p:
        return None
    return (
        f"There is Statcast BvP in this feed at <strong>{p['h']}</strong> hits in "
        f"<strong>{p['pa']}</strong> PA (<strong>{p['ba']:.3f}</strong> AVG)—treat tiny samples as directional only."
    )


def _proof_compact_html(r: dict, kind: str, xbh_lookup: dict[tuple[int, int], int]) -> str:
    """Two-sentence summary: headline case (stats + context), then supporting detail."""
    pn = html.escape(str(r.get("pitcher_name") or "this pitcher"))
    hand = html.escape(_pitcher_hand_phrase(r))
    tier = str(r.get("tier") or "").strip()
    tier_bit = f' (<strong>{html.escape(tier)}</strong> matchup tag)' if tier else ""

    reasons = r.get("top_reasons") or []
    reasons_txt = ""
    if isinstance(reasons, list) and reasons:
        reasons_txt = "; ".join(html.escape(str(x)) for x in reasons[:2])

    s1 = ""
    s2_inner = ""

    if kind == "hit":
        adj = float(r["adj_p_hit"]) * 100
        career = float(r.get("career_hit") or 0) * 100
        raw = float(r["p_hit"]) * 100
        tpa = _three_pa(float(r["p_hit"]))
        rel = html.escape(_career_rel_word(adj, career))
        s1 = (
            f"The model prices a hit at <strong>{adj:.1f}%</strong> this PA after matchup shrinkage—"
            f"<strong>{rel}</strong> a <strong>{career:.1f}%</strong> career hit baseline vs <strong>{pn}</strong> "
            f"({hand}){tier_bit}."
        )
        nar = _first_sentence_from_text(str(r.get("hit_narrative") or ""))
        if nar:
            s2_inner = html.escape(nar)
        elif reasons_txt:
            s2_inner = f"Supporting factors the model highlights: {reasons_txt}."
        elif (bv := _bvp_proof_snippet_html(r, xbh_lookup)) is not None:
            s2_inner = bv
        else:
            s2_inner = (
                f"At the <strong>{raw:.1f}%</strong> raw per-PA rate, about <strong>{tpa:.0f}%</strong> to log "
                "at least one hit over three independent tries—useful if you expect multiple PA."
            )
    elif kind == "xbh":
        adj = float(r["adj_p_xbh"]) * 100
        raw = float(r["p_xbh"]) * 100
        tpa = _three_pa(float(r["p_xbh"]))
        s1 = (
            f"The model prices an extra-base hit at <strong>{adj:.1f}%</strong> this PA (adjusted), "
            f"from a <strong>{raw:.1f}%</strong> raw XBH rate vs <strong>{pn}</strong> ({hand}){tier_bit}."
        )
        nar = _first_sentence_from_text(str(r.get("hit_narrative") or ""))
        if nar:
            s2_inner = html.escape(nar)
        elif reasons_txt:
            s2_inner = f"Supporting factors the model highlights: {reasons_txt}."
        elif (bv := _bvp_proof_snippet_html(r, xbh_lookup)) is not None:
            s2_inner = bv
        else:
            s2_inner = (
                f"About <strong>{tpa:.0f}%</strong> to record at least one XBH over three tries at the "
                f"<strong>{raw:.1f}%</strong> raw rate—extra context for multi-PA props."
            )
    else:
        adj = float(r["adj_p_hr"]) * 100
        career = float(r.get("career_hr") or 0) * 100
        raw = float(r["p_hr"]) * 100
        tpa = _three_pa(float(r["p_hr"]))
        rel = html.escape(_career_rel_word(adj, career))
        s1 = (
            f"The model prices a home run at <strong>{adj:.1f}%</strong> this PA after shrinkage—"
            f"<strong>{rel}</strong> a <strong>{career:.1f}%</strong> career HR baseline vs <strong>{pn}</strong> "
            f"({hand}){tier_bit}."
        )
        hr_hint = ", ".join(str(x) for x in reasons[:2]) if isinstance(reasons, list) and reasons else ""
        nar = _first_sentence_from_text(hr_hint or str(r.get("hit_narrative") or ""))
        if nar:
            s2_inner = html.escape(nar)
        elif reasons_txt:
            s2_inner = f"Supporting factors the model highlights: {reasons_txt}."
        elif (bv := _bvp_proof_snippet_html(r, xbh_lookup)) is not None:
            s2_inner = bv
        else:
            s2_inner = (
                f"About <strong>{tpa:.1f}%</strong> to see at least one HR over three tries at the "
                f"<strong>{raw:.1f}%</strong> raw HR rate."
            )

    return (
        f'<div class="proof-block">'
        f'<p class="proof-s">{s1}</p>'
        f'<p class="proof-s">{s2_inner}</p>'
        f"</div>"
    )


def _bet_score_label_kind(r: dict, kind: str) -> str:
    k = {"hit": "score_label_hit", "xbh": "score_label_xbh", "hr": "score_label_hr"}[kind]
    return str(r.get(k) or "").strip()


def _bet_conf_label_kind(r: dict, kind: str) -> str:
    k = {"hit": "conf_hit_label", "xbh": "conf_xbh_label", "hr": "conf_hr_label"}[kind]
    return str(r.get(k) or "").strip()


def _bet_take_sentences_plain(r: dict, kind: str, xbh_lookup: dict[tuple[int, int], int]) -> tuple[str, str, str]:
    """Three short prose lines for wagering context (minimal numbers)."""

    score_l = (_bet_score_label_kind(r, kind)).strip().lower()
    conf_l = (_bet_conf_label_kind(r, kind)).strip().lower()
    tier = str(r.get("tier") or "").strip()
    tier_use = tier and tier.lower() not in {"", "average"}
    tier_seg = (
        f" The matchup tier reads {tier} — that pushes the upside case beyond a coin flip label."
        if tier_use
        else ""
    )
    prop = {
        "hit": "hits or on-base angles",
        "xbh": "extra-base swings",
        "hr": "long-ball markets",
    }[kind]

    if score_l == "lock":
        s1 = (
            f"This is flagged as premium conviction territory on the board for {prop} — rare air if you trust the night's data pull."
            + tier_seg
        )
    elif score_l == "strong":
        s1 = (
            f"The conviction tag is plainly bullish here for {prop} — you're leaning into the matchup thesis, not a lottery ticket scrape."
            + tier_seg
        )
    elif score_l == "lean":
        s1 = (
            f"It lands as an honest lean for {prop}: interesting if the odds cooperate, nothing that demands a billboard."
            + tier_seg
        )
    elif score_l == "avoid":
        s1 = (
            f"Conviction tooling is steering away here for {prop} — coherent to pass unless your live reads scream otherwise."
            + ("" if tier_use else " Treat any micro-edge as discretionary, not automatic.")
        )
    else:
        if conf_l in ("high", "medium"):
            tail = tier_seg if tier_use else " Ingredients behind the headline look reasonably filled-in."
            s1 = (
                f"Rankings like this matchup for {prop}, even without a flashy conviction stamp."
                + tail
            )
        elif conf_l in ("very low", "low"):
            s1 = (
                f"This row still shows up hot on the leaderboard, but the evidence bucket for {prop} is thin behind the headline."
                + tier_seg
            )
        else:
            s1 = (
                f"Skew is toward playing {prop}, with room for both cheer and skepticism depending on lineup role and bullpen timelines."
                + tier_seg
            )

    bvp = _bvp_raw_parts(r, xbh_lookup)
    if bvp is None:
        s2 = (
            "BvP history is practically blank in our Statcast scrape — tonight is matchup math and scouting, not nostalgic boxscore hunting."
        )
    else:
        pa, ba = int(bvp["pa"]), float(bvp["ba"])
        hr_b = int(bvp["hr"])
        if pa <= 4:
            s2 = (
                "There are a handful of past meetings — enough to skim, not enough to preach a sermon. "
                "Let tonight's shape of the arsenal matter more."
            )
        elif kind == "hr" and hr_b >= 1:
            s2 = (
                "The head-to-head file already carries thunder against this pitcher — that narrative travels with betting markets even when counts stay small."
            )
        elif pa >= 8 and ba < 0.220:
            s2 = (
                "Career overlaps lean cold for this hitter-pitcher pairing; either the hitter adjusted later or those meetings were miserable — price that honestly."
            )
        elif ba >= 0.285:
            s2 = (
                "They've barreled or blooped their way around on this pitcher when they've squared off — reassuring if every ticket needs a scouting hook."
            )
        else:
            s2 = (
                f"They've swapped {pa}+ PA in Statcast-era meetings with mixed receipts — seasoning for the handicap, not the whole marinade."
            )

    factors = r.get("conf_factors") if isinstance(r.get("conf_factors"), dict) else {}
    stal = factors.get("staleness") if factors else None
    stale_low = False
    if stal is not None:
        try:
            stale_low = float(stal) < 0.88
        except (TypeError, ValueError):
            stale_low = False

    s3_candidates: list[str] = []

    pq = str(factors.get("platoon_quality_reason") or "").strip().lower()

    try:
        ck = float(r.get("career_k") or 0)
        if ck > 0.28:
            s3_candidates.append(
                "Swing-and-miss is still baked into his long-term profile — one ugly sequence nukes softer contact ladders."
            )
        elif ck > 0.23:
            s3_candidates.append(
                "Strikeouts remain part of the identity book — correlate your ticket with role and matchup length so one bad inning doesn't crater you."
            )
    except (TypeError, ValueError):
        pass

    if stale_low:
        s3_candidates.append(
            "Rolling inputs trail real life modestly — late scratches, tweaked arsenals, or weather can swamp what the nightly snapshot guessed."
        )

    if pq and pq not in ("none", "", "neutral"):
        s3_candidates.append(
            "Platoon scouting nudges the matchup grading — double-check batting order certainty before you overweight the model read."
        )

    hit_low = str(r.get("hit_narrative") or "").lower()
    if "exit velo" in hit_low:
        s3_candidates.append(
            "Ball-tracking still likes how aggressively he squares firm stuff this pitcher type invites — softer tailwind behind the flashy rank."
        )

    if kind == "hr":
        rs = r.get("top_reasons") or []
        lump = " ".join(str(x) for x in rs[:3]).lower()
        if any(w in lump for w in ("barrel", "launch", "hard", "elevat", "fly")):
            s3_candidates.append(
                "The feature stew points at loud air angles more than rollover grounders — temperamentally aligns with swinging for fences."
            )

    if not s3_candidates:
        s3 = (
            "Board rank is one cue; umpire tendencies, inning leverage, and the books' hangover still referee the wager."
        )
    else:
        s3 = s3_candidates[0]

    return s1.strip(), s2.strip(), s3.strip()


def _bet_takeaway_html(r: dict | None, kind: str, xbh_lookup: dict[tuple[int, int], int]) -> str:
    """Three plain sentences between Proof and Confidence; empty row returns em dash."""
    if not r:
        return '<div class="take-block muted">—</div>'
    a, b, c = _bet_take_sentences_plain(r, kind, xbh_lookup)
    parts = "".join(f'<p class="take-s">{html.escape(seg)}</p>' for seg in (a, b, c))
    return f'<div class="take-block take-col">{parts}</div>'


def _confidence_cell_html(r: dict, kind: str) -> str:
    if kind == "hit":
        lab = str(r.get("conf_hit_label") or "").strip() or "—"
        conf = r.get("conf_hit")
    elif kind == "xbh":
        lab = str(r.get("conf_xbh_label") or "").strip() or "—"
        conf = r.get("conf_xbh")
    else:
        lab = str(r.get("conf_hr_label") or "").strip() or "—"
        conf = r.get("conf_hr")
    lab_esc = html.escape(lab)
    try:
        num_s = f"{float(conf):.2f}" if conf is not None else "—"
    except (TypeError, ValueError):
        num_s = "—"
    return (
        f'<div class="conflev">{lab_esc}</div>'
        f'<div class="confnum proof-muted" title="Evidence multiplier (~1=data-rich); not calibrated P nor an uncertainty band. Ranking score uses raw × this. Cal P buckets use numeric conf quartiles when calibration JSON includes edges.">'
        f"{html.escape(num_s)}</div>"
    )


def _pitcher_hand_abbr(r: dict | None) -> str:
    """Pitcher handedness tag for HTML; empty if unknown. Expects pitcher_throws 'L' / 'R' (or synonyms)."""
    if not r:
        return ""
    t = r.get("pitcher_throws")
    if t is None or str(t).strip() == "":
        return ""
    s = str(t).strip().upper()
    if s.startswith("L"):
        return "LHP"
    if s.startswith("R"):
        return "RHP"
    return ""


def _pitcher_hand_paren(r: dict | None) -> str:
    h = _pitcher_hand_abbr(r)
    return f" ({h})" if h else ""


_lineup_lookup_dashboard_cache: dict[tuple[int, str], dict] | None = None


def _reset_lineup_lookup_dashboard_cache() -> None:
    global _lineup_lookup_dashboard_cache
    _lineup_lookup_dashboard_cache = None


def _lineup_lookup_for_dashboard() -> dict[tuple[int, str], dict]:
    """Layer 2 rollup (last ≤10 vs starter hand); same source as inference ``lineup_context``."""
    global _lineup_lookup_dashboard_cache
    if _lineup_lookup_dashboard_cache is None:
        try:
            from lineup_context import load_lineup_slot_lookup

            _lineup_lookup_dashboard_cache = load_lineup_slot_lookup()
        except Exception:
            _lineup_lookup_dashboard_cache = {}
    return _lineup_lookup_dashboard_cache


def _median_slot_string(ms: object) -> str:
    try:
        slot = int(ms)
    except (TypeError, ValueError):
        return ""
    if not 1 <= slot <= 9:
        return ""
    return str(slot)


def _lineup_spot_median_vs_sp_hand(r: dict | None) -> str:
    """Median batting-order slot (last ≤10 games vs opposing starter hand); empty if unknown."""
    if not r:
        return ""

    lc = r.get("lineup_context")
    if lc and isinstance(lc, dict):
        s = _median_slot_string(lc.get("median_slot"))
        if s:
            return s

    # Predictions JSON may omit ``lineup_context`` on older runs — join Layer 2 parquet here.
    bid = r.get("batter_mlbam_id")
    pt = r.get("pitcher_throws")
    if bid is None or pt is None or str(pt).strip() == "":
        return ""
    try:
        bid_i = int(bid)
    except (TypeError, ValueError):
        return ""
    oh = "L" if str(pt).strip().upper().startswith("L") else "R"
    row = _lineup_lookup_for_dashboard().get((bid_i, oh))
    if not row:
        return ""
    return _median_slot_string(row.get("median_slot"))


MatchupKey = tuple[str, str, str, str, str]


def _matchup_key(r: dict) -> MatchupKey:
    return (
        str(r.get("batter_name") or ""),
        str(r.get("pitcher_name") or ""),
        str(r.get("batter_team") or ""),
        str(r.get("pitcher_team") or ""),
        _pitcher_hand_abbr(r),
    )


def _label_from_matchup_key(k: tuple[str, ...]) -> str:
    if len(k) >= 5:
        bn, pn, bt, hand = k[0], k[1], k[2], k[4]
    elif len(k) == 4:
        bn, pn, bt, hand = k[0], k[1], k[2], ""
    else:
        return ""
    suf = f" ({hand})" if hand else ""
    return f"{bn} ({bt}) vs {pn}{suf}"


def _matchup_line(r: dict) -> str:
    suf = _pitcher_hand_paren(r)
    spot = _lineup_spot_median_vs_sp_hand(r)
    spot_s = f" {spot}" if spot else ""
    s = f"{r['batter_name']} ({r['batter_team']}){spot_s} vs {r['pitcher_name']}{suf}"
    return html.escape(s)


def _data_teams_attr(*row_dicts: dict | None) -> str:
    """HTML data-teams: batter_team only (sorted), for filtering to hitters on a club (excludes pitcher org)."""
    teams: set[str] = set()
    for r in row_dicts:
        if not r:
            continue
        v = r.get("batter_team")
        if isinstance(v, str) and v.strip():
            teams.add(v.strip().upper())
    if not teams:
        return ""
    sl = ",".join(sorted(teams))
    return f' data-teams="{html.escape(sl, quote=True)}"'


def _data_teams_attr_from_label(label: str) -> str:
    """Batter's (TEAM) from matchup label text — text before ' vs ' only, so pitcher names cannot add teams."""
    if not label:
        return ""
    batter_side = label.split(" vs ", 1)[0] if " vs " in label else label
    found = re.findall(r"\(([A-Za-z]{2,4})\)", batter_side)
    if not found:
        return ""
    teams = sorted({x.upper() for x in found if 2 <= len(x) <= 4})
    if not teams:
        return ""
    sl = ",".join(teams)
    return f' data-teams="{html.escape(sl, quote=True)}"'


def _compare_models_summary_html(rows_m1: list[dict], rows_m2: list[dict], kind: str) -> str:
    """Readable divergence summary + conservative model pick for one outcome type."""
    adj_k = {"hit": "adj_p_hit", "xbh": "adj_p_xbh", "hr": "adj_p_hr"}[kind]
    conf_k = {"hit": "conf_hit", "xbh": "conf_xbh", "hr": "conf_hr"}[kind]
    title = {"hit": "Hits", "xbh": "XBH", "hr": "Home runs"}[kind]
    if not rows_m1 or not rows_m2:
        return (
            f'<aside class="divergence-summary" aria-label="{html.escape(title)} summary">'
            "<p class=\"empty\">Not enough rows to summarize.</p></aside>"
        )

    m1 = {_matchup_key(r): (i + 1, float(r[adj_k]), float(r.get(conf_k) or 0)) for i, r in enumerate(rows_m1[:25])}
    m2 = {_matchup_key(r): (i + 1, float(r[adj_k]), float(r.get(conf_k) or 0)) for i, r in enumerate(rows_m2[:25])}
    k1, k2 = set(m1), set(m2)
    both = k1 & k2
    only1, only2 = k1 - k2, k2 - k1
    overlap_n = len(both)

    t1_k1 = _matchup_key(rows_m1[0])
    t1_k2 = _matchup_key(rows_m2[0])
    same_top = t1_k1 == t1_k2

    rank_diffs: list[int] = []
    adj_diff_pp: list[float] = []
    for k in both:
        r_a, a1, _c1 = m1[k]
        r_b, a2, _c2 = m2[k]
        rank_diffs.append(r_b - r_a)
        adj_diff_pp.append((a2 - a1) * 100.0)
    mean_rd = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0.0
    mean_abs_rd = sum(abs(x) for x in rank_diffs) / len(rank_diffs) if rank_diffs else 0.0
    mean_adj_d = sum(adj_diff_pp) / len(adj_diff_pp) if adj_diff_pp else 0.0
    mean_c1 = sum(m1[k][2] for k in both) / len(both) if both else 0.0
    mean_c2 = sum(m2[k][2] for k in both) / len(both) if both else 0.0

    moves = sorted(
        ((abs(m2[k][0] - m1[k][0]), m2[k][0] - m1[k][0], k) for k in both),
        key=lambda x: (-x[0], x[2]),
    )[:4]

    def esc_samples(keys: set[MatchupKey], rank_map: dict, n: int = 4) -> str:
        ordered = sorted(keys, key=lambda kk: rank_map[kk][0])[:n]
        parts = [html.escape(_label_from_matchup_key(kk)) for kk in ordered]
        if not parts:
            return "—"
        suf = " …" if len(keys) > n else ""
        return ", ".join(parts) + suf

    bullets: list[str] = []
    bullets.append(
        f"<strong>Overlap:</strong> {overlap_n} of 25 matchups appear on <em>both</em> top-25 boards "
        f"({len(only1)} only in model 1, {len(only2)} only in model 2)."
    )
    if same_top:
        a1, a2 = m1[t1_k1][1] * 100.0, m2[t1_k1][1] * 100.0
        bullets.append(
            "<strong>#1 pick agrees:</strong> "
            f"{html.escape(_label_from_matchup_key(t1_k1))} "
            f"(model 1 <strong>{a1:.1f}%</strong> adj vs model 2 <strong>{a2:.1f}%</strong> adj)."
        )
    else:
        a1 = m1[t1_k1][1] * 100.0
        a2 = m2[t1_k2][1] * 100.0
        bullets.append(
            "<strong>#1 picks differ.</strong> Model 1 leads with "
            f"{html.escape(_label_from_matchup_key(t1_k1))} (<strong>{a1:.1f}%</strong> adj). "
            "Model 2 leads with "
            f"{html.escape(_label_from_matchup_key(t1_k2))} (<strong>{a2:.1f}%</strong> adj)."
        )

    bullets.append(
        "<strong>Rank movement on shared names:</strong> average shift "
        f"(model 2 rank − model 1 rank) is <strong>{mean_rd:+.1f}</strong> spots "
        f"(positive means experiment ranks that matchup lower on its list). "
        f"Mean absolute move: <strong>{mean_abs_rd:.1f}</strong> spots."
    )
    if both:
        bullets.append(
            "<strong>Level shift on overlap (model 2 − model 1):</strong> "
            f"<strong>{mean_adj_d:+.2f}</strong> percentage points on adjusted probability, averaged over shared rows. "
            f"Mean confidence on those rows: model 1 <strong>{mean_c1:.2f}</strong> vs model 2 <strong>{mean_c2:.2f}</strong>."
        )

    if moves:
        bits = []
        for _absd, _d, k in moves[:3]:
            bits.append(
                f"{html.escape(_label_from_matchup_key(k))} "
                f"(#<strong>{m1[k][0]}</strong> → #<strong>{m2[k][0]}</strong>)"
            )
        bullets.append("<strong>Largest rank swings (shared names):</strong> " + "; ".join(bits) + ".")

    if only1:
        bullets.append(
            f"<strong>Model 1 exclusives ({len(only1)}):</strong> {esc_samples(only1, m1)}"
        )
    if only2:
        bullets.append(
            f"<strong>Model 2 exclusives ({len(only2)}):</strong> {esc_samples(only2, m2)}"
        )

    strong: str | None = None
    if overlap_n >= 22 and same_top:
        rec = (
            "The two boards are tightly aligned: same leader and almost the entire list in common. "
            "Impact of switching models here is mostly second-order (small probability nudges)."
        )
    elif overlap_n < 12:
        rec = (
            "The models disagree sharply on who belongs in the top 25 — many one-sided names. "
            "Impact is high: your short list changes materially depending on which model you trust."
        )
    elif mean_adj_d > 1.0 and mean_c2 + 0.02 >= mean_c1:
        strong = "experiment"
        rec = (
            "On names both models like, experiment is materially more bullish while confidence stays in the same ballpark. "
            "That pushes more ceiling into the same spots — worth a tilt toward model 2 if you buy the feature stack for this outcome."
        )
    elif mean_adj_d < -1.0:
        strong = "production"
        rec = (
            "Experiment is meaningfully lower than production on overlapping rows — rankings may reshuffle but the experiment is pulling this prop type downward. "
            "Production is the safer anchor unless you have a specific reason to trust the experiment shrinkage here."
        )
    elif mean_c1 > mean_c2 + 0.08:
        strong = "production"
        rec = (
            "Production carries noticeably higher confidence on the shared board; the experiment looks less sure about the same picks. "
            "For bankroll-sensitive use of this metric, that is a modest nod toward model 1."
        )
    elif mean_c2 > mean_c1 + 0.08:
        strong = "experiment"
        rec = (
            "Experiment shows higher confidence on overlapping picks — the new feature path may be more decisive for this prop tonight. "
            "Slight lean toward model 2 if the research narrative matches what you see in the table."
        )
    elif abs(mean_adj_d) < 0.4 and mean_abs_rd < 4.0:
        rec = (
            "Levels and ordering are close on overlap; neither model is screaming a different story. "
            "Impact of choosing one over the other is limited to a few spots unless you dig into individual matchups."
        )
    else:
        rec = (
            "Mixed signals: partial overlap and neither model clearly wins on both probability lift and confidence. "
            "Default to production for conservative staking, or split attention across both columns for this metric."
        )
        if mean_c1 >= mean_c2 + 0.03:
            strong = "production"

    if strong is None and overlap_n >= 18 and abs(mean_adj_d) < 0.25:
        rec += " Neither model dominates; either ranking is defensible for this slate slice."

    if strong == "production":
        pick_line = '<p class="rec-pick">Use for this metric: <strong>Model 1 (production)</strong> — slightly better alignment of confidence and/or levels for tonight.</p>'
    elif strong == "experiment":
        pick_line = '<p class="rec-pick">Use for this metric: <strong>Model 2 (experiment)</strong> — slightly stronger signal on the evidence above.</p>'
    else:
        pick_line = (
            '<p class="rec-pick">Use for this metric: <strong>No strong favorite</strong> — '
            "treat the two columns as complementary unless you have an outside view on the experiment features.</p>"
        )

    ul = '<ul class="diverge-ul">' + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    return (
        f'<aside class="divergence-summary" aria-label="{html.escape(title)} model comparison summary">'
        f'<h4 class="diverge-h">Summary · {html.escape(title)} (model 1 vs model 2)</h4>'
        f"{ul}"
        f'<p class="rec-p">{html.escape(rec)}</p>'
        f"{pick_line}"
        "<p class=\"rec-note\">Auto-generated from this slate’s JSON predictions (overlap, ranks, adj probabilities, confidence). "
        "Not a guarantee of out-of-sample model quality.</p>"
        "</aside>"
    )


def _bvp_cell_detailed(r: dict, xbh_lookup: dict[tuple[int, int], int]) -> str:
    p = _bvp_raw_parts(r, xbh_lookup)
    if not p:
        return "—"
    avg = f"{p['ba']:.3f}"
    s = f"{p['h']}/{p['pa']} · {avg} AVG · XBH {p['xbh']} · HR {p['hr']} · K {p['k']} · BB {p['bb']}"
    return html.escape(s)


def _rows_compare_top25(
    rows_m1: list[dict],
    rows_m2: list[dict],
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
) -> str:
    out = []
    for i, (r1, r2) in enumerate(zip_longest(rows_m1[:25], rows_m2[:25], fillvalue=None), 1):
        mu1 = _matchup_line(r1) if r1 else "—"
        pr1 = _proof_compact_html(r1, kind, xbh_lookup) if r1 else "—"
        tk1 = _bet_takeaway_html(r1, kind, xbh_lookup) if r1 else "—"
        cf1 = _confidence_cell_html(r1, kind) if r1 else "—"
        bv1 = _bvp_cell_detailed(r1, xbh_lookup) if r1 else "—"
        mu2 = _matchup_line(r2) if r2 else "—"
        pr2 = _proof_compact_html(r2, kind, xbh_lookup) if r2 else "—"
        tk2 = _bet_takeaway_html(r2, kind, xbh_lookup) if r2 else "—"
        cf2 = _confidence_cell_html(r2, kind) if r2 else "—"
        bv2 = _bvp_cell_detailed(r2, xbh_lookup) if r2 else "—"
        out.append(
            "<tr"
            + _data_teams_attr(r1, r2)
            + ">"
            f'<td class="rn">{i}</td>'
            f'<td class="mu1">{mu1}</td>'
            f'<td class="pr1">{pr1}</td>'
            f'<td class="tk1">{tk1}</td>'
            f'<td class="cf1">{cf1}</td>'
            f'<td class="bvp1">{bv1}</td>'
            f'<td class="mu2">{mu2}</td>'
            f'<td class="pr2">{pr2}</td>'
            f'<td class="tk2">{tk2}</td>'
            f'<td class="cf2">{cf2}</td>'
            f'<td class="bvp2">{bv2}</td>'
            "</tr>"
        )
    return "\n".join(out)


def _rows_recency_top25(
    rows_r: list[dict],
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
) -> str:
    """Single-model rows for optional recency bundle (caller sorts by adj_*)."""
    out: list[str] = []
    for i, r in enumerate(rows_r[:25], 1):
        if not r:
            continue
        mu = _matchup_line(r)
        pr = _proof_compact_html(r, kind, xbh_lookup)
        tk = _bet_takeaway_html(r, kind, xbh_lookup)
        cf = _confidence_cell_html(r, kind)
        bv = _bvp_cell_detailed(r, xbh_lookup)
        out.append(
            "<tr"
            + _data_teams_attr(r, None)
            + ">"
            f'<td class="rn">{i}</td>'
            f'<td class="mu1">{mu}</td>'
            f'<td class="pr1">{pr}</td>'
            f'<td class="tk1">{tk}</td>'
            f'<td class="cf1">{cf}</td>'
            f'<td class="bvp1">{bv}</td>'
            "</tr>"
        )
    return "\n".join(out)


def _beast_confidence_tier_map(
    rows_top25: list[dict],
    kind: str,
) -> dict[MatchupKey, int]:
    """Quintiles of numeric conf in this top-25 slice: 5 = highest evidence multiplier down to 1."""
    ck = {"hit": "conf_hit", "xbh": "conf_xbh", "hr": "conf_hr"}[kind]
    chunk = [r for r in (rows_top25 or [])[:25] if r]
    if not chunk:
        return {}
    pairs: list[tuple[float, MatchupKey]] = []
    for r in chunk:
        try:
            confv = float(r.get(ck) or 0.0)
        except (TypeError, ValueError):
            confv = 0.0
        pairs.append((confv, _matchup_key(r)))
    pairs.sort(key=lambda t: (-t[0], t[1]))
    n = len(pairs)
    denom = max(n, 1)
    out: dict[MatchupKey, int] = {}
    for rank, (_c, mk) in enumerate(pairs):
        bucket = min(4, (rank * 5) // denom)
        out[mk] = 5 - bucket
    return out


def _beast_conf_rel_cell_html(tier: int) -> str:
    title = (
        "Relative confidence in this top 25: "
        "5 = among the strongest evidence multipliers here; 1 = weakest in this slice vs the other rows."
    )
    return (
        f'<td class="beast-conf-rel tier-{int(tier)}" '
        f'title="{html.escape(title, quote=True)}">'
        f'<span class="beast-conf-rel-num">{int(tier)}</span></td>'
    )


def _beast_numeric_conf(r: dict, kind: str) -> float | None:
    ck = {"hit": "conf_hit", "xbh": "conf_xbh", "hr": "conf_hr"}[kind]
    return _beast_as_float(r.get(ck))


def _beast_last_name_for_label(name: str) -> str:
    """Short label for scatter annotations (prefer family name; strip suffix)."""
    parts = (name or "").strip().split()
    if not parts:
        return "?"
    last_norm = parts[-1].upper().replace(".", "")
    suffixes = {"JR", "SR", "II", "III", "IV", "V"}
    if last_norm in suffixes and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def _beast_scatter_label_bbox(
    tx: float, ty: float, text_len: int, fs: float = 7.0
) -> tuple[float, float, float, float]:
    """Rough SVG text bounds (start anchor, Latin-ish names); y is text baseline."""
    w = max(fs * 0.52 * float(max(text_len, 1)), fs * 1.8)
    return (tx, ty - fs * 0.82, tx + w, ty + fs * 0.28)


def _beast_scatter_bboxes_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 2.0
) -> bool:
    return not (
        a[2] + pad < b[0]
        or a[0] - pad > b[2]
        or a[3] + pad < b[1]
        or a[1] - pad > b[3]
    )


def _beast_scatter_pick_ranks(rows: list[dict], kind: str, top_n: int = 5) -> set[int]:
    """Ranks (1-based, among printed top 25) with best blend of Beast rank + numeric confidence."""
    scored: list[tuple[float, int]] = []
    for i, r in enumerate(rows[:25], 1):
        if not r:
            continue
        cv = _beast_numeric_conf(r, kind)
        if cv is None:
            continue
        score = cv * (26.0 - float(i)) / 25.0
        scored.append((score, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return {rank for _s, rank in scored[:top_n]}


def _beast_scatter_svg(rows: list[dict], kind: str, caption: str, dom_id: str) -> str:
    """SVG scatter: x = Beast rank (1 = best), y = numeric confidence multiplier."""
    W, H = 480, 228
    pad_l, pad_r, pad_t, pad_b = 48, 16, 26, 46
    pw = W - pad_l - pad_r
    ph = H - pad_t - pad_b

    pick_ranks = _beast_scatter_pick_ranks(rows, kind, top_n=5)

    pts: list[tuple[int, float, str, str]] = []
    for i, r in enumerate(rows[:25], 1):
        if not r:
            continue
        cv = _beast_numeric_conf(r, kind)
        if cv is None:
            continue
        bn = str(r.get("batter_name") or "?").strip()
        pn = str(r.get("pitcher_name") or "?").strip()
        tip = f"{bn} vs {pn} · rank #{i} · conf {cv:.2f}"
        pts.append((i, cv, tip, bn))

    y_vals = [p[1] for p in pts]
    if y_vals:
        y_lo = min(y_vals) - 0.04
        y_hi = max(y_vals) + 0.04
        y_lo = max(0.45, y_lo)
        y_hi = min(1.08, y_hi)
    else:
        y_lo, y_hi = 0.70, 0.95
    if y_hi - y_lo < 0.08:
        y_mid = (y_lo + y_hi) / 2.0
        y_lo, y_hi = y_mid - 0.05, y_mid + 0.05

    def x_px(rank: int) -> float:
        return pad_l + (float(rank - 1) / 24.0) * pw

    def y_px(conf: float) -> float:
        span = y_hi - y_lo
        if span <= 1e-9:
            return pad_t + ph / 2.0
        t = (conf - y_lo) / span
        return pad_t + (1.0 - t) * ph

    ttl_id = dom_id + "-ttl"
    ttl_id_esc = html.escape(ttl_id, quote=False)
    parts: list[str] = [
        f'<svg class="beast-scatter-svg" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-labelledby="{ttl_id_esc}">',
        f'<title id="{ttl_id_esc}">{html.escape(caption)}</title>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#0f172a" opacity="0.35" rx="3"/>',
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" y2="{pad_t + ph}" stroke="#64748b" stroke-width="1"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#64748b" stroke-width="1"/>',
    ]
    for rx in (1, 5, 10, 15, 20, 25):
        x = x_px(rx)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t + ph}" x2="{x:.1f}" y2="{pad_t + ph + 5}" stroke="#64748b" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{H - 22}" text-anchor="middle" font-size="9" fill="#94a3b8" font-family="system-ui,sans-serif">{rx}</text>'
        )

    for yi_f in (y_lo, (y_lo + y_hi) / 2.0, y_hi):
        y = y_px(yi_f)
        parts.append(
            f'<line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l}" y2="{y:.1f}" stroke="#475569" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="#94a3b8" font-family="system-ui,sans-serif">{yi_f:.2f}</text>'
        )

    parts.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{H - 6}" text-anchor="middle" font-size="10" fill="#cbd5e1" '
        f'font-family="system-ui,sans-serif">Beast rank (#) — 1 is best projection on this board</text>'
    )
    parts.append(
        f'<text transform="translate(12 {pad_t + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
        f'font-size="10" fill="#cbd5e1" font-family="system-ui,sans-serif">Confidence (numeric)</text>'
    )

    # Draw ordinary dots first, then picks on top, then small last-name labels for picks.
    ordinary: list[tuple[float, float, str]] = []
    picked: list[tuple[float, float, str, str, int]] = []
    for rank, conf, tip, bn in pts:
        cx, cy = x_px(rank), y_px(conf)
        tip_esc = html.escape(tip)
        ln = _beast_last_name_for_label(bn)
        if rank in pick_ranks:
            picked.append((cx, cy, tip_esc, ln, rank))
        else:
            ordinary.append((cx, cy, tip_esc))

    dot_r = 5.0

    for cx, cy, tip_esc in ordinary:
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{dot_r}" class="beast-scatter-dot" fill="#38bdf8" stroke="#0c4a6e" '
            f'stroke-width="0.6" opacity="0.9"><title>{tip_esc}</title></circle>'
        )

    for cx, cy, tip_esc, _ln, _rank in picked:
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{dot_r}" class="beast-scatter-dot beast-scatter-dot--pick" '
            f'fill="#fbbf24" stroke="#92400e" stroke-width="0.75" opacity="0.95"><title>{tip_esc}</title></circle>'
        )

    # Place pick labels with greedy non-overlap (same bubble size as non-picks).
    fs_lab = 7.0
    label_color = "#9ca3af"
    margin_x = 4.0
    margin_y = 3.0
    plot_right = pad_l + pw - margin_x
    plot_bottom = pad_t + ph - margin_y
    offset_candidates = [
        (9.0, -11.0),
        (9.0, 13.0),
        (-52.0, -11.0),
        (-52.0, 13.0),
        (9.0, -25.0),
        (-52.0, -25.0),
        (9.0, 27.0),
        (-52.0, 27.0),
        (9.0, -39.0),
        (-52.0, -39.0),
        (28.0, 4.0),
        (-72.0, 4.0),
    ]
    placed_bboxes: list[tuple[float, float, float, float]] = []
    for cx, cy, tip_esc, ln, rank in sorted(picked, key=lambda t: t[4]):
        lbl = html.escape(ln)
        nlen = len(ln)
        chosen: tuple[float, float, tuple[float, float, float, float]] | None = None
        for ox, oy in offset_candidates:
            tx = cx + ox
            ty = cy + oy
            bb = _beast_scatter_label_bbox(tx, ty, nlen, fs_lab)
            # Keep inside plot (labels sit to the side or above/below points).
            if bb[0] < pad_l + margin_x:
                shift = pad_l + margin_x - bb[0]
                tx += shift
                bb = _beast_scatter_label_bbox(tx, ty, nlen, fs_lab)
            if bb[2] > plot_right:
                shift = plot_right - bb[2]
                tx += shift
                bb = _beast_scatter_label_bbox(tx, ty, nlen, fs_lab)
            if bb[1] < pad_t + margin_y:
                shift = pad_t + margin_y - bb[1]
                ty += shift
                bb = _beast_scatter_label_bbox(tx, ty, nlen, fs_lab)
            if bb[3] > plot_bottom:
                shift = plot_bottom - bb[3]
                ty += shift
                bb = _beast_scatter_label_bbox(tx, ty, nlen, fs_lab)
            if any(_beast_scatter_bboxes_overlap(bb, pb, pad=3.0) for pb in placed_bboxes):
                continue
            chosen = (tx, ty, bb)
            break
        if chosen is None:
            ty_try = pad_t + margin_y + fs_lab + float(len(placed_bboxes)) * (fs_lab + 5.0)
            chosen_inner = None
            for _attempt in range(14):
                tx = cx + 9.0
                ty_try = min(ty_try, plot_bottom - fs_lab)
                bb = _beast_scatter_label_bbox(tx, ty_try, nlen, fs_lab)
                if not any(_beast_scatter_bboxes_overlap(bb, pb, pad=3.0) for pb in placed_bboxes):
                    chosen_inner = (tx, ty_try, bb)
                    break
                ty_try += fs_lab + 5.0
            if chosen_inner is None:
                tx = cx + 9.0
                ty_try = pad_t + margin_y + fs_lab
                bb = _beast_scatter_label_bbox(tx, ty_try, nlen, fs_lab)
                chosen_inner = (tx, ty_try, bb)
            chosen = chosen_inner
        tx, ty, bb = chosen
        placed_bboxes.append(bb)
        parts.append(
            f'<text class="beast-scatter-label" x="{tx:.1f}" y="{ty:.1f}" '
            f'font-size="{fs_lab}" fill="{label_color}" font-family="system-ui,sans-serif" '
            f'font-weight="400">{lbl}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _beast_scatter_figure(rows: list[dict], kind: str, title: str, dom_id: str) -> str:
    cap = f"{title}: Beast rank (#) vs numeric confidence"
    svg = _beast_scatter_svg(rows, kind, cap, dom_id)
    did = html.escape(dom_id, quote=True)
    return (
        f'<figure class="beast-scatter-fig" id="{did}">'
        f"<figcaption>{html.escape(title)}</figcaption>{svg}</figure>"
    )


def _beast_scatter_block_html(
    S_hit_b: list[dict],
    S_xbh_b: list[dict],
    S_hr_b: list[dict],
    tid,
) -> str:
    """Scatter charts: each plot in its own collapsible card; top 5 picks highlighted in-chart."""
    intro = (
        '<p class="intro beast-scatter-intro">Each circle is one matchup on that Beast board. '
        "<strong>Horizontal:</strong> printed rank (#) — <strong>1</strong> is the best projection on that list, "
        "<strong>25</strong> is the softest of the 25 shown. "
        "<strong>Vertical:</strong> numeric confidence multiplier (same number under Beast confidence in the table). "
        "<strong>Gold dots:</strong> top 5 on this chart by "
        "<code>confidence × (26 − rank) / 25</code> — best blend of strong Beast rank and high confidence. "
        "Small yellow labels show batter last name. Hover any dot for full matchup.</p>"
    )
    charts: list[str] = []
    for title, rows_arg, kind, slug in (
        ("Hits", S_hit_b, "hit", "hits"),
        ("XBH", S_xbh_b, "xbh", "xbh"),
        ("Home runs", S_hr_b, "hr", "hr"),
    ):
        fid = tid(f"beast-scatter-{slug}")
        charts.append(
            '<details class="card beast-scatter-chart-details">'
            f"<summary>Chart · {html.escape(title)} · Beast rank (#) vs confidence</summary>"
            '<div class="cardbody beast-scatter-chart-body">'
            + _beast_scatter_figure(rows_arg, kind, title, fid)
            + "</div></details>"
        )
    return '<div class="beast-scatter-gallery">' + intro + "".join(charts) + "</div>"


def _beast_as_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def _beast_pitch_mix_pct(x: object) -> float | None:
    """Interpret rates as fractions (0–1) or already-percent (>1-ish); return percent on 100 scale."""
    v = _beast_as_float(x)
    if v is None:
        return None
    if 0 <= v <= 1.06:
        return v * 100.0
    return v


def _beast_latest_season_profile_row(df: pd.DataFrame, key_col: str, mlb_key: object) -> pd.Series | None:
    if df is None or getattr(df, "empty", True) or key_col not in df.columns:
        return None
    try:
        kid = int(float(mlb_key))
    except (TypeError, ValueError):
        return None
    vals = pd.to_numeric(df[key_col], errors="coerce")
    sub = df.loc[vals == kid]
    if sub.shape[0] == 0:
        return None
    if "season" not in df.columns:
        return sub.iloc[-1]
    seas = pd.to_numeric(sub["season"], errors="coerce")
    if seas.notna().any():
        mx = seas.max()
        late = sub.loc[seas == mx]
        if not late.empty:
            return late.iloc[-1]
    return sub.iloc[-1]


def _beast_narr_ev_mph(hit_narrative: str) -> tuple[float | None, str | None]:
    if not hit_narrative:
        return None, None
    m = re.search(r"Exit velo vs\s+([LR])HP:\s*(\d+(?:\.\d+)?)\s*mph", hit_narrative, flags=re.I)
    if m:
        return _beast_as_float(m.group(2)), m.group(1).upper()
    m2 = re.search(r"Exit velo\s*:\s*(\d+(?:\.\d+)?)\s*mph", hit_narrative, flags=re.I)
    return (_beast_as_float(m2.group(1)) if m2 else None), None


def _beast_k_narr_rates(k_narrative: str) -> tuple[float | None, float | None]:
    if not k_narrative:
        return None, None
    pk = ww = None
    mk = re.search(r"Pitcher K rate:\s*(\d+(?:\.\d+)?)\s*%", k_narrative, flags=re.I)
    if mk:
        pk = _beast_as_float(mk.group(1))
    mw = re.search(r"Whiff rate:\s*(\d+(?:\.\d+)?)\s*%", k_narrative, flags=re.I)
    if mw:
        ww = _beast_as_float(mw.group(1))
    return pk, ww


def _beast_k_narr_first_delta_pct(k_narrative: str) -> float | None:
    """First +/-x.x% vs career token on k_narrative."""
    if not k_narrative:
        return None
    mm = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)%\s*vs\s*career", k_narrative.strip(), flags=re.I)
    return _beast_as_float(mm.group(1)) if mm else None


def _beast_platoon_split_words(pitcher_arm_abbr: str) -> tuple[str, str]:
    """(when-facing phrase for Statcast hitter splits, parquet column suffix)."""
    if pitcher_arm_abbr == "RHP":
        return ("when Statcast catalogs swings against right-handed pitching", "ev_vs_rhp")
    if pitcher_arm_abbr == "LHP":
        return ("when Statcast catalogs swings against left-handed pitching", "ev_vs_lhp")
    return ("in aggregate Statcast hitter logs", "")


def _beast_row_adj_pa(r: dict | None, kind: str) -> float | None:
    """Calibrated-ish per-PA probability used on Beast leaderboard (fraction 0–1)."""
    if not r:
        return None
    if kind == "hit":
        adj = _beast_as_float(r.get("adj_p_hit"))
        if adj is None:
            adj = _beast_as_float(r.get("p_hit_calibrated")) or _beast_as_float(r.get("p_hit"))
    elif kind == "hr":
        adj = _beast_as_float(r.get("adj_p_hr"))
        if adj is None:
            adj = _beast_as_float(r.get("p_hr_calibrated")) or _beast_as_float(r.get("p_hr"))
    else:
        adj = _beast_as_float(r.get("adj_p_xbh"))
        if adj is None:
            adj = _beast_as_float(r.get("p_xbh_calibrated")) or _beast_as_float(r.get("p_xbh"))
    return adj


def _beast_leaderboard_adj_label(kind: str) -> str:
    return {"hit": "adj P(hit)", "hr": "adj P(HR)", "xbh": "adj P(XBH)"}[kind]


def _beast_leaderboard_pick_context(
    leaderboard_rows: list[dict],
    kind: str,
    rank_display: int,
    tier: int,
    focused_row: dict,
) -> dict[str, object]:
    chunk = [x for x in (leaderboard_rows or [])[:25] if isinstance(x, dict)]
    row_fracs = [_beast_row_adj_pa(rr, kind) for rr in chunk]
    have = sorted([z for z in row_fracs if z is not None], reverse=True)

    curr = _beast_row_adj_pa(focused_row, kind)
    tu = str(focused_row.get("batter_team") or "").strip().upper()

    team_ranks: list[int] = []
    for j, rr in enumerate(chunk):
        if tu and str(rr.get("batter_team") or "").strip().upper() == tu:
            team_ranks.append(j + 1)
    team_best = min(team_ranks) if team_ranks else None

    top_f = have[0] if have else None
    tail_f = have[-1] if have else None
    med_idx = len(have) // 2 if have else 0
    med_f = have[med_idx] if have else None

    behind_top = (
        None
        if (curr is None or top_f is None or abs(top_f) < 1e-9)
        else (top_f - curr) * 100.0
    )
    vs_median_pp = (
        None if (curr is None or med_f is None) else (curr - med_f) * 100.0
    )

    return {
        "n_board": len(chunk),
        "rank_display": int(rank_display),
        "tier": int(tier),
        "curr_frac": curr,
        "top_frac": top_f,
        "med_frac": med_f,
        "tail_frac": tail_f,
        "behind_top_pp": behind_top,
        "vs_median_pp": vs_median_pp,
        "team_abbr": tu or "",
        "team_slots": len(team_ranks),
        "team_best_rank": int(team_best) if team_best is not None else None,
        "adj_label": _beast_leaderboard_adj_label(kind),
    }


def _beast_pick_note_three_sentences(
    r: dict,
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
    leaderboard_ctx: dict[str, object] | None = None,
) -> tuple[str, str, str, str]:
    """Four short scouting-style blurbs: model read, swing vs stuff, history/Ks, how to weight it."""
    kw = {"hit": "hits", "hr": "home runs", "xbh": "extra-base hits"}[kind]
    prop_plain = {"hit": "hits", "hr": "homers", "xbh": "extra bases"}[kind]
    per_trip = {"hit": "a knock", "hr": "a ball over the fence", "xbh": "an extra-base knock"}[kind]

    lbl = ((_bet_score_label_kind(r, kind)).strip() or "Neutral")
    lbl_l = lbl.lower()
    cf_disp = ((_bet_conf_label_kind(r, kind)).strip() or "?")
    cf_l = cf_disp.lower()

    bn = str(r.get("batter_name") or "This hitter").strip()
    pn = str(r.get("pitcher_name") or "pitcher").strip()
    pitch_tag = _pitcher_hand_abbr(r)
    pt = _pitcher_throws_letter_row(r)

    adj = _beast_row_adj_pa(r, kind)
    if kind == "hit":
        career_prop = _beast_as_float(r.get("career_hit"))
        lg_prop = _BEAST_CTX_LEAGUE_HIT
    elif kind == "hr":
        career_prop = _beast_as_float(r.get("career_hr"))
        lg_prop = _BEAST_CTX_LEAGUE_HR
    else:
        career_prop = _beast_as_float(r.get("career_xbh"))
        lg_prop = _BEAST_CTX_LEAGUE_XBH

    pp_df, bp_df = _beast_profile_frames()
    mix_med = _median_latest_pitcher_profiles(
        pp_df,
        ("pct_fastball", "pct_breaking", "pct_offspeed", "k_rate", "whiff_rate"),
    )
    mfb_med = float(mix_med.get("pct_fastball") or 52.0)
    mbrk_med = float(mix_med.get("pct_breaking") or 32.0)
    mos_med = float(mix_med.get("pct_offspeed") or 16.0)
    mk_med = float(mix_med.get("k_rate") or 22.0)
    mw_med = float(mix_med.get("whiff_rate") or 24.0)

    prow = _beast_latest_season_profile_row(pp_df, "pitcher", r.get("pitcher_mlbam_id"))
    brow = _beast_latest_season_profile_row(bp_df, "batter", r.get("batter_mlbam_id"))
    _, ev_col = _beast_platoon_split_words(pitch_tag)

    bid = _beast_as_float(r.get("batter_mlbam_id"))
    rec_raw: dict[str, float | int | None] = {}
    if bid is not None and _BEAST_RECENT_BIP_EV is not None:
        hit = _BEAST_RECENT_BIP_EV.get(int(bid))
        if isinstance(hit, dict):
            rec_raw = hit

    lg_evb = _BEAST_CTX_LEAGUE_EV_BIP
    lg_k_bat = _BEAST_CTX_LEAGUE_BATTER_K

    plat_blob = r.get("platoon_raw_shrink") or {}
    plat_key = {"hit": "platoon_raw_hit_mult", "xbh": "platoon_raw_xbh_mult", "hr": "platoon_raw_hr_mult"}[kind]
    plat_mult = _beast_as_float(plat_blob.get(plat_key))

    # Row 1 — what the model thinks, in dugout English
    parts1: list[str] = [f"{bn} vs {pn}: Beast tags this as {lbl} for {prop_plain}."]
    if adj is not None:
        parts1.append(
            f"The model lands near a {adj * 100:.1f}% shot at {per_trip} each time he steps in the box."
        )
    if adj is not None and career_prop is not None:
        dp = (adj - career_prop) * 100.0
        if abs(dp) >= 0.35:
            parts1.append(
                f"Stack that next to his career book ({career_prop * 100:.1f}% on {prop_plain}) and tonight reads "
                f"{'hotter' if dp > 0 else 'colder'} by about {abs(dp):.1f} percentage points."
            )
        else:
            parts1.append(
                f"That's basically in line with what his career numbers say ({career_prop * 100:.1f}% on {prop_plain})."
            )
    if adj is not None:
        if adj >= lg_prop + 0.006:
            parts1.append(
                f"Versus a generic big-league baseline (~{lg_prop * 100:.1f}% for this shape of prop), that's a green light."
            )
        elif adj + 0.006 <= lg_prop:
            parts1.append(
                f"Versus a generic big-league baseline (~{lg_prop * 100:.1f}%), the projection is a little light."
            )
        else:
            parts1.append(
                f"Versus a generic big-league baseline (~{lg_prop * 100:.1f}%), he's right in the pack."
            )
    if plat_mult is not None and abs(plat_mult - 1.0) > 0.03:
        parts1.append(
            f"We already trimmed the number for the same-side matchup (kept about {plat_mult * 100:.0f}% of a straight-up platoon guess)."
        )
    if cf_l in ("very low", "low"):
        parts1.append(
            f"Data confidence is only {cf_disp} — double-check he's in the lineup and hitting where you think before you fire."
        )
    else:
        parts1.append(f"Data confidence reads {cf_disp}.")
    s1 = " ".join(parts1)

    n_recent = int(rec_raw.get("n_recent") or 0)
    m_recent = rec_raw.get("mean_recent")
    if pt == "R":
        mh_n = int(rec_raw.get("n_vs_rhp") or 0)
        mh_mu = rec_raw.get("mean_vs_rhp")
        hand_human = "RHP"
    elif pt == "L":
        mh_n = int(rec_raw.get("n_vs_lhp") or 0)
        mh_mu = rec_raw.get("mean_vs_lhp")
        hand_human = "LHP"
    else:
        mh_n, mh_mu, hand_human = 0, None, ""

    # Row 2 — swing feel vs what this pitcher actually throws
    parts2: list[str] = []
    if n_recent >= 18 and isinstance(m_recent, (int, float)):
        dev = float(m_recent) - lg_evb
        if dev >= 1.25:
            parts2.append(
                f"When he's put the ball in play lately it's jumped off the bat (~{float(m_recent):.1f} mph exit "
                f"on a real sample of balls), a tick or two north of what you see league-wide."
            )
        elif dev <= -1.25:
            parts2.append(
                f"Recent balls in play have been a little dead (~{float(m_recent):.1f} mph exit avg) — "
                "not necessarily broken, but not smoking either."
            )
        else:
            parts2.append(
                f"Exit velo on balls in play lately (~{float(m_recent):.1f} mph) looks normal for a big-league bat."
            )
    elif n_recent > 0:
        parts2.append(
            f"We only have {n_recent} recent balls in play logged — too thin to chase a hot/cold narrative off Statcast alone."
        )
    else:
        parts2.append("No fresh Statcast window on balls in play here — lean on the season line and your eyes on BP.")

    if hand_human and mh_n >= 12 and isinstance(mh_mu, (int, float)):
        parts2.append(
            f"Against {hand_human}s like the guy on the mound tonight he's averaged ~{float(mh_mu):.1f} mph exit "
            f"on the last {mh_n} balls in play we have."
        )

    fb = _beast_pitch_mix_pct(prow.get("pct_fastball")) if prow is not None else None
    br = _beast_pitch_mix_pct(prow.get("pct_breaking")) if prow is not None else None
    ofs = _beast_pitch_mix_pct(prow.get("pct_offspeed")) if prow is not None else None
    vfb = _beast_as_float(prow.get("velo_fastball")) if prow is not None else None
    top_code = str(prow.get("top_pitch_1") or "").strip() if prow is not None else ""
    top_pct = _beast_pitch_mix_pct(prow.get("top_pitch_1_pct")) if prow is not None else None

    if prow is None:
        parts2.append(f"We don't have {pn}'s pitch mix in this pull — use the Proof column or a scouting card.")
    elif fb is not None:
        if fb >= mfb_med + 7.0:
            parts2.append(
                f"{pn} lives on the heater: about {fb:.0f}% fastballs, well above a league-average starter (~{mfb_med:.0f}%)."
            )
        elif ofs is not None and ofs >= mos_med + 7.0:
            parts2.append(
                f"{pn} pitches backwards more than most — offspeed shows up ~{ofs:.0f}% of the time vs a ~{mos_med:.0f}% league-ish mark."
            )
        elif br is not None and br >= mbrk_med + 8.0:
            parts2.append(
                f"{pn} spins it a lot: breaking balls ~{br:.0f}% vs a ~{mbrk_med:.0f}% league-ish baseline."
            )
        if top_code and top_pct is not None:
            parts2.append(f"He leans on {top_code} roughly {top_pct:.0f}% of the time.")
        if vfb is not None:
            parts2.append(f"Fastball velo sits around {vfb:.1f} mph — hittable if it's flat, nasty if he's dotting.")

    ev_fb_stat = _beast_as_float(brow.get("ev_vs_fastball")) if brow is not None else None
    ev_br_stat = _beast_as_float(brow.get("ev_vs_breaking")) if brow is not None else None
    plat_ev_stat = _beast_as_float(brow.get(ev_col)) if (brow is not None and ev_col) else None
    ev_side_word = "RHP" if (ev_col and "rhp" in ev_col.lower()) else ("LHP" if ev_col else "")

    if plat_ev_stat is not None and ev_side_word:
        parts2.append(
            f"{bn} has averaged ~{plat_ev_stat:.1f} mph exit against {ev_side_word}s in our hitter file "
            f"(league contact center is ~{lg_evb:.1f} mph)."
        )
    if fb is not None and fb >= mfb_med + 4.0 and ev_fb_stat is not None:
        parts2.append(
            f"That matters because he squares fastballs at ~{ev_fb_stat:.1f} mph exit — useful if {pn} is going to feed him heaters."
        )
    if br is not None and br >= mbrk_med + 4.0 and ev_br_stat is not None:
        parts2.append(
            f"He also handles spin at ~{ev_br_stat:.1f} mph exit, which lines up if {pn} tries to bury breakers."
        )
    if len(parts2) <= 2:
        parts2.append("If the matchup detail feels thin, scroll left to Proof for the raw Statcast hooks.")
    s2 = " ".join(parts2[:6])

    bvp = _bvp_raw_parts(r, xbh_lookup)
    career_ba = _beast_as_float(r.get("career_hit"))
    ba_bench = career_ba if career_ba is not None else _BEAST_CTX_LEAGUE_HIT

    k_line = str(r.get("k_narrative") or "")
    narr_pk, narr_wh = _beast_k_narr_rates(k_line)
    pr_k = _beast_pitch_mix_pct(prow.get("k_rate")) if prow is not None else None
    pr_wh = _beast_pitch_mix_pct(prow.get("whiff_rate")) if prow is not None else None
    pr_barrel = _beast_pitch_mix_pct(prow.get("barrel_rate_allowed")) if prow is not None else None
    pk_eff = narr_pk if narr_pk is not None else pr_k
    wh_eff = narr_wh if narr_wh is not None else pr_wh

    p_k_model = _beast_as_float(r.get("p_k"))
    pk_show = None
    if p_k_model is not None:
        pk_show = p_k_model * 100.0 if 0.0 <= p_k_model <= 1.0 else (p_k_model if p_k_model <= 100.0 else None)

    parts3: list[str] = []
    if bvp is None:
        parts3.append("They barely have a Statcast-era history — you're betting the shape of the at-bat, not old box scores.")
    else:
        pa = int(bvp["pa"])
        hh = int(bvp["h"])
        kk = int(bvp["k"])
        ba_v = float(bvp["ba"])
        if pa <= 3:
            parts3.append(
                f"They've only traded a handful of PA ({hh}-for-{pa}) — fun for trivia, not for conviction."
            )
        elif pa >= 12 and ba_v >= 0.32:
            parts3.append(
                f"He's hit this arm before: {ba_v:.3f} over {pa} PA, well north of his own ~{ba_bench * 100:.1f}% career AVG."
            )
        elif pa >= 12 and ba_v <= 0.195:
            parts3.append(
                f"The old meetings are ugly — {ba_v:.3f} over {pa} PA with {kk} punchouts. That can live in a hitter's head."
            )
        else:
            parts3.append(
                f"Head-to-head is mixed: {hh} hits in {pa} trips ({ba_v:.3f}), {kk} Ks — seasoning, not the whole story."
            )

    if pk_show is not None:
        parts3.append(
            f"Tonight's strikeout risk sits around {pk_show:.0f}% in the model (think ~{lg_k_bat * 100:.1f}% as a generic big-league hitter baseline)."
        )
    if pk_eff is not None:
        if pk_eff - mk_med >= 2.0:
            parts3.append(
                f"{pn} misses bats for a living (~{pk_eff:.0f}% K rate vs a ~{mk_med:.0f}% league-ish starter)."
            )
        elif abs(pk_eff - mk_med) < 2.0:
            parts3.append(
                f"{pn} is roughly average at finishing hitters (~{pk_eff:.0f}% K rate vs ~{mk_med:.0f}% league-ish)."
            )
        else:
            parts3.append(
                f"{pn} doesn't live on the whiff (~{pk_eff:.0f}% K rate vs ~{mk_med:.0f}% league-ish) — more balls in play."
            )
    if wh_eff is not None:
        parts3.append(f"His chase-and-miss profile sits near {wh_eff:.0f}% whiffs (league-ish starter band ~{mw_med:.0f}%).")

    if pr_barrel is not None and kind in ("hr", "xbh") and pr_barrel >= 1.0:
        parts3.append(
            f"When contact happens, {pn} has been giving up barrels around {pr_barrel:.1f}% of the time — that's real HR/XBH juice if the bat catches one."
        )

    if lbl_l == "avoid":
        parts3.append("Beast is waving you off — you'd need a fat price or a weird lineup spot to talk yourself into it.")
    elif cf_l in ("very low", "low"):
        parts3.append("Confidence on the inputs is soft — make sure he's actually in the spot you think before you bet into it.")

    if not parts3:
        parts3.append("If this row feels thin, lean on the Confidence column and the Proof block to the left.")
    s3 = " ".join(parts3[:6])

    # Bottom line — where he sits on tonight's same list, what sings, what worries, how to treat it.
    lb = leaderboard_ctx or {}
    n_board = int(lb.get("n_board") or 0)
    rank_lb = int(lb.get("rank_display") or 0)
    tier_lb = int(lb.get("tier") or 0)
    adj_lbl = str(lb.get("adj_label") or _beast_leaderboard_adj_label(kind))

    curr_fb = lb.get("curr_frac")
    curr_frac: float | None
    if isinstance(curr_fb, (int, float)):
        curr_frac = float(curr_fb)
    elif isinstance(adj, (int, float)):
        curr_frac = float(adj)
    else:
        curr_frac = None

    top_fb = lb.get("top_frac")
    top_frac = float(top_fb) if isinstance(top_fb, (int, float)) else None
    med_fb = lb.get("med_frac")
    med_frac = float(med_fb) if isinstance(med_fb, (int, float)) else None

    btp = lb.get("behind_top_pp")
    behind_top_pp = float(btp) if isinstance(btp, (int, float)) else None
    vm = lb.get("vs_median_pp")
    vs_med_pp = float(vm) if isinstance(vm, (int, float)) else None

    team_abbr_lb = str(lb.get("team_abbr") or "").strip().upper()
    team_slots = int(lb.get("team_slots") or 0)
    tb = lb.get("team_best_rank")
    team_best_rank = int(tb) if isinstance(tb, (int, float)) else None

    weak_evidence = cf_l in ("very low", "low")
    above_bar = adj is not None and adj >= lg_prop + 0.005
    below_bar = adj is not None and adj <= lg_prop - 0.005

    vibe_recent = ""
    if n_recent >= 18 and isinstance(m_recent, (int, float)):
        dev_recent = float(m_recent) - lg_evb
        if dev_recent >= 1.25:
            vibe_recent = "hot"
        elif dev_recent <= -1.25:
            vibe_recent = "cold"

    ba_bvp_end = float(bvp["ba"]) if bvp else None
    pa_bvp_end = int(bvp["pa"]) if bvp else 0
    cold_bvp = pa_bvp_end >= 12 and ba_bvp_end is not None and ba_bvp_end <= 0.195
    hot_bvp = pa_bvp_end >= 12 and ba_bvp_end is not None and ba_bvp_end >= 0.32
    high_k_warn = pk_show is not None and pk_show >= 28.0

    adj_pp = curr_frac * 100.0 if curr_frac is not None else None

    sentences: list[str] = []
    if n_board and rank_lb:
        if adj_pp is not None:
            lead = (
                f"Stacked against every other {prop_plain} row we printed tonight ({n_board} deep), "
                f"{bn} shows up {rank_lb} with {adj_lbl} parked near {adj_pp:.2f}% each trip to the plate."
            )
            if top_frac is not None:
                lead += (
                    f" The cleanest read at the top of that same list is about {top_frac * 100.0:.2f}%."
                )
            sentences.append(lead)
        else:
            sentences.append(
                f"On tonight's {prop_plain} board he's {rank_lb} of {n_board}, but the model didn't hand us a clean number to quote."
            )

        if behind_top_pp is not None and behind_top_pp > 0.05 and adj_pp is not None:
            sentences.append(
                f"If you're hunting only the loudest ticket on the sheet, that's roughly {behind_top_pp:.2f} percentage points "
                "of air between him and the guy sitting first."
            )
        if med_frac is not None and vs_med_pp is not None and abs(vs_med_pp) >= 0.02:
            sentences.append(
                f"The bat parked in the middle of this same list is near {med_frac * 100.0:.2f}% — "
                f"{bn} is sitting about {abs(vs_med_pp):.2f} points {'above' if vs_med_pp > 0 else 'below'} that middle."
            )

        if team_abbr_lb and team_slots >= 2 and team_best_rank is not None:
            if rank_lb == team_best_rank:
                sentences.append(
                    f"If you're running a {team_abbr_lb} stack off this page, he's the cleanest look we have "
                    f"among the {team_slots} teammates we listed for {prop_plain}."
                )
            elif rank_lb > team_best_rank:
                sentences.append(
                    f"{team_abbr_lb} already has a shinier row higher on this same list (#{team_best_rank}) — "
                    "I'd walk there first unless this price is begging you the other way."
                )

    if not sentences:
        sentences.append(
            "We lost the page context for how he ranks versus the other names tonight — scroll the table left before you bet into it."
        )

    yarns: list[str] = []
    if adj is not None and career_prop is not None:
        dd_pp = (adj - career_prop) * 100.0
        if dd_pp >= 1.4:
            yarns.append("the model is higher on him than his own career arc usually says")
        elif dd_pp <= -1.5:
            yarns.append("the model is actually a little cold versus what his back-of-the-card numbers suggest")

    if above_bar:
        yarns.append("he clears a league-average projection for this prop")
    if vibe_recent == "hot":
        yarns.append("the ball has been jumping when he connects")
    if plat_ev_stat is not None and ev_side_word and plat_ev_stat >= lg_evb + 2.0:
        yarns.append(f"he drives it against {ev_side_word}s the way you'd want tonight")
    if fb is not None and fb >= mfb_med + 6.0 and ev_fb_stat is not None and ev_fb_stat >= lg_evb + 1.25:
        yarns.append(f"{pn} wants to feed fastballs and {bn} has shown he can dent those")
    if pr_barrel is not None and kind in ("hr", "xbh") and pr_barrel >= 7.5:
        yarns.append(f"{pn} has been bleeding barrels (~{pr_barrel:.1f}% of contact) when hitters square him")
    if hot_bvp and ba_bvp_end is not None:
        yarns.append(
            f"past meetings vs {pn} are loud (.{ba_bvp_end:.3f} over {pa_bvp_end} PA)"
        )

    warns: list[str] = []
    if cold_bvp and ba_bvp_end is not None:
        warns.append(
            f"head-to-head vs {pn} has been a rough watch (.{ba_bvp_end:.3f} over {pa_bvp_end} PA)"
        )
    if weak_evidence:
        warns.append(
            "the stat bundle behind this row is thin, so I'd trust neighbors with cleaner proof before I talk myself into a big number"
        )
    if high_k_warn and kind in ("hit", "xbh"):
        warns.append(
            f"there's a real punchout path tonight (~{pk_show:.0f}% K in the model) if you're chasing hits instead of moonshots"
        )
    if tier_lb <= 2 and n_board:
        warns.append(
            f"stacked against the other names printed in this top-{min(n_board, 25)}, the confidence meter only lights {tier_lb} bulbs out of 5"
        )
    if rank_lb >= 17 and behind_top_pp is not None and behind_top_pp >= 1.0:
        warns.append(
            "he's printed deep enough that I'd always peek at the rows above before I fall in love with the price"
        )

    if yarns:
        sentences.append(
            "The case you make in the dugout is " + ", and ".join(yarns[:4]) + "."
        )
    if warns:
        sentences.append(
            "The case the other dugout whispers is " + "; ".join(warns[:3]) + "."
        )

    if lbl_l == "avoid":
        sentences.append(
            "How I'd play it: I'm not forcing a ticket — let friendlier matchups on this same page eat first unless the book hangs a stupid number."
        )
    elif lbl_l == "lock":
        if rank_lb <= 5:
            sentences.append(
                "How I'd play it: circle him with the handful of guys you're willing to build the night around once the lineup card is real."
            )
        else:
            sentences.append(
                "How I'd play it: the model still buys the story even if he's not printed at the very top — smaller piece, same respect after you confirm the spot."
            )
    elif lbl_l == "strong":
        sentences.append(
            "How I'd play it: he's in the real conversation tonight — pair the bet with the cleaner rows above him instead of pretending he's the only name on the sheet."
        )
    elif lbl_l == "lean":
        sentences.append(
            "How I'd play it: fine as a sprinkle after the obvious chalk clears my bankroll — not the headliner unless the payout lies to you."
        )
    else:
        if below_bar:
            sentences.append(
                "How I'd play it: I'd walk unless I've got a live read the computer doesn't — brighter colors are drawn higher on this same board."
            )
        elif above_bar:
            sentences.append(
                "How I'd play it: playable if you like the number, but I still keep one eye on the names printed above him here."
            )
        else:
            sentences.append(
                "How I'd play it: true coin-flip feel — let lineup slot and the juice do the talking, not the headline alone."
            )

    s4 = " ".join(sentences).strip()

    return (s1.strip(), s2.strip(), s3.strip(), s4.strip())


def _beast_pick_note_html(
    r: dict | None,
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
    leaderboard_ctx: dict[str, object] | None = None,
) -> str:
    """Executive brief rows (label + body) between Beast Proof and Confidence."""
    if not r:
        return '<div class="take-block muted">—</div>'
    lines = _beast_pick_note_three_sentences(
        r, kind, xbh_lookup, leaderboard_ctx=leaderboard_ctx
    )
    labs = ("The read", "Swing vs stuff", "History & K\u2019s", "How I\u2019d play it")
    rows: list[str] = []
    for i, (lab, seg) in enumerate(zip(labs, lines)):
        cls = ' class="tk-row tk-conc"' if i == len(labs) - 1 else ' class="tk-row"'
        rows.append(
            f"<div{cls}><span class=\"tk-lab\">"
            + html.escape(lab)
            + '</span><span class="tk-body">'
            + html.escape(seg)
            + "</span></div>"
        )
    return f'<div class="take-block take-col beast-take-mgr">{"".join(rows)}</div>'


def _beast_pitch_hand_label_from_row(r: dict) -> str:
    pa_audit = r.get("posterior_audit") or {}
    h = str(pa_audit.get("platoon_pitch_hand_label") or "").strip()
    if h:
        return h
    pt = str(r.get("pitcher_throws") or "R")
    return "LHP" if pt.upper().startswith("L") else "RHP"


def _beast_platoon_ytd_cell(r: dict, kind: str) -> str:
    """2026 season-to-date vs pitcher hand (before slate), kind-aware emphasis."""
    pa_audit = r.get("posterior_audit") or {}
    hand = _beast_pitch_hand_label_from_row(r)
    ypa = int(pa_audit.get("ytd_pa") or 0)

    if ypa > 0 and "ytd_h" not in pa_audit:
        tip = (
            f"{ypa} PA vs {hand} in 2026 before slate — rerun matchup inference for "
            "full YTD H / AB / XBH splits."
        )
        return (
            f'<td class="beast-plat beast-plat-ytd" title="{html.escape(tip)}">'
            f'<span class="beast-plat-hand">{html.escape(hand)}</span><br/>'
            f'<span class="beast-plat-line1">{ypa} PA</span><br/>'
            f'<span class="beast-plat-line2 beast-plat-muted">rerun model</span></td>'
        )

    yh = int(pa_audit.get("ytd_h") or 0)
    yab = int(pa_audit.get("ytd_ab") or 0)
    yhr = int(pa_audit.get("ytd_hr_raw") or 0)
    yxbh = int(pa_audit.get("ytd_xbh") or 0)

    if ypa <= 0:
        tip = f"No 2026 plate appearances vs {hand} before this slate."
        return (
            f'<td class="beast-plat beast-plat-ytd" title="{html.escape(tip)}">'
            f'<span class="beast-plat-hand">{html.escape(hand)}</span>'
            f'<span class="beast-plat-muted"><br/>—</span></td>'
        )

    ba = (yh / yab) if yab > 0 else float("nan")
    ba_s = f".{ba:.3f}" if ba == ba else "—"

    if kind == "hit":
        line1 = ba_s
        line2 = f"{yh}–{yab} H–AB · {ypa} PA"
    elif kind == "hr":
        line1 = f"{yhr}/{ypa} HR"
        line2 = f"BA {ba_s} · {yxbh} XBH"
    else:
        line1 = f"{yxbh}/{ypa} XBH"
        line2 = f"BA {ba_s} · {yhr} HR"

    tip = (
        f"2026 vs {hand} (season before slate): {yh} H, {yhr} HR, {yxbh} XBH in {ypa} PA ({yab} AB)."
    )
    return (
        f'<td class="beast-plat beast-plat-ytd" title="{html.escape(tip)}">'
        f'<span class="beast-plat-hand">{html.escape(hand)}</span><br/>'
        f'<span class="beast-plat-line1">{html.escape(line1)}</span><br/>'
        f'<span class="beast-plat-line2 beast-plat-muted">{html.escape(line2)}</span></td>'
    )


def _beast_platoon_career_cell(r: dict, kind: str) -> str:
    """Decay-weighted career vs pitcher hand (prior seasons), from inference snapshot."""
    pa_audit = r.get("posterior_audit") or {}
    hand = _beast_pitch_hand_label_from_row(r)

    if "career_vs_hand_pa_eff" not in pa_audit and "career_vs_hand_ba" not in pa_audit:
        tip = "Career vs-hand splits ship with fresh matchup JSON — rerun inference + dashboard build."
        return (
            f'<td class="beast-plat beast-plat-career" title="{html.escape(tip)}">'
            f'<span class="beast-plat-hand">{html.escape(hand)}</span>'
            f'<span class="beast-plat-muted"><br/>—</span></td>'
        )

    pa_eff = float(pa_audit.get("career_vs_hand_pa_eff") or 0)
    ba = float(pa_audit.get("career_vs_hand_ba") or 0)
    ops = float(pa_audit.get("career_vs_hand_ops") or 0)
    hr_rate = float(pa_audit.get("career_vs_hand_hr_rate") or 0)
    xbh_rate = float(pa_audit.get("career_vs_hand_xbh_rate") or 0)

    if pa_eff <= 1e-6:
        tip = f"No decay-weighted career split vs {hand} on file."
        return (
            f'<td class="beast-plat beast-plat-career" title="{html.escape(tip)}">'
            f'<span class="beast-plat-hand">{html.escape(hand)}</span>'
            f'<span class="beast-plat-muted"><br/>—</span></td>'
        )

    ba_s = f".{ba:.3f}"
    ops_s = f"{ops:.3f}"
    hr_pct = f"{100.0 * hr_rate:.1f}%"
    xbh_pct = f"{100.0 * xbh_rate:.1f}%"

    if kind == "hit":
        line1 = f"{ba_s} · {ops_s} OPS"
        line2 = f"~{pa_eff:.0f} eff PA"
    elif kind == "hr":
        line1 = f"{hr_pct} HR/PA"
        line2 = f"{ba_s} BA · ~{pa_eff:.0f} PA"
    else:
        line1 = f"{xbh_pct} XBH/PA"
        line2 = f"{ba_s} BA · ~{pa_eff:.0f} PA"

    tip = (
        f"Decay-weighted career vs {hand} (prior seasons): BA {ba_s}, OPS {ops_s}, "
        f"HR/PA {hr_rate:.4f}, XBH/PA {xbh_rate:.4f}, eff PA {pa_eff:.1f}."
    )
    return (
        f'<td class="beast-plat beast-plat-career" title="{html.escape(tip)}">'
        f'<span class="beast-plat-hand">{html.escape(hand)}</span><br/>'
        f'<span class="beast-plat-line1">{html.escape(line1)}</span><br/>'
        f'<span class="beast-plat-line2 beast-plat-muted">{html.escape(line2)}</span></td>'
    )


def _rows_beast_top25(
    rows_b: list[dict],
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
) -> str:
    """Beast top-25 with an extra column: 1–5 relative confidence within the table."""
    tier_map = _beast_confidence_tier_map(rows_b, kind)
    out: list[str] = []
    for i, r in enumerate(rows_b[:25], 1):
        if not r:
            continue
        mu = _matchup_line(r)
        pr = _proof_compact_html(r, kind, xbh_lookup)
        tier = tier_map.get(_matchup_key(r), 1)
        lb_ctx = _beast_leaderboard_pick_context(rows_b, kind, i, tier, r)
        tk = _beast_pick_note_html(r, kind, xbh_lookup, leaderboard_ctx=lb_ctx)
        cf = _confidence_cell_html(r, kind)
        rel = _beast_conf_rel_cell_html(tier)
        bv = _bvp_cell_detailed(r, xbh_lookup)
        ytd_plat = _beast_platoon_ytd_cell(r, kind)
        car_plat = _beast_platoon_career_cell(r, kind)
        out.append(
            "<tr"
            + _data_teams_attr(r, None)
            + ">"
            f'<td class="rn">{i}</td>'
            f'<td class="mu1">{mu}</td>'
            f'<td class="pr1">{pr}</td>'
            f'<td class="tk1">{tk}</td>'
            f'<td class="cf1">{cf}</td>'
            f"{rel}"
            f'<td class="bvp1">{bv}</td>'
            f"{ytd_plat}"
            f"{car_plat}"
            "</tr>"
        )
    return "\n".join(out)


def _recency_rows_aligned_to_prod(prod_rows: list[dict], recency_rows: list[dict]) -> list[dict]:
    """Drop stale recency JSON if slate_date header does not match production snapshot."""
    if not recency_rows or not prod_rows:
        return []
    psd = str(prod_rows[0].get("slate_date") or "")[:10]
    rsd = str(recency_rows[0].get("slate_date") or "")[:10]
    if not psd or rsd != psd:
        return []
    return recency_rows


def _beast_rows_aligned_to_prod(prod_rows: list[dict], beast_rows: list[dict]) -> list[dict]:
    """Drop stale beast JSON if slate_date header does not match production snapshot."""
    if not beast_rows or not prod_rows:
        return []
    psd = str(prod_rows[0].get("slate_date") or "")[:10]
    bsd = str(beast_rows[0].get("slate_date") or "")[:10]
    if not psd or bsd != psd:
        return []
    return beast_rows


def _batter_day_outcome_stats(slate_str: str) -> tuple[dict[int, dict[str, int]], str | None]:
    """Calendar-day Statcast totals per batter (same definition as fill_matchup_prediction_outcomes)."""
    pa_path = RAW / "statcast_pa_level_league.parquet"
    if not pa_path.is_file():
        return {}, "Statcast league PA file not found."
    pa = pd.read_parquet(pa_path, columns=["game_date", "batter", "events"])
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce").dt.normalize()
    pa = pa.dropna(subset=["game_date", "batter"])
    pa["batter"] = pa["batter"].astype(int)
    sdt = pd.Timestamp(slate_str).normalize()
    pa = pa.loc[pa["game_date"] == sdt]
    if pa.empty:
        return {}, f"No Statcast plate appearances in the league file for {slate_str}."
    ev = pa["events"].fillna("").str.lower()
    pa = pa.assign(
        is_hit=ev.isin(_HIT_EVENTS).astype(int),
        is_hr=(ev == "home_run").astype(int),
        is_xbh=ev.isin(_XBH_EVENTS).astype(int),
    )
    g = pa.groupby("batter", as_index=False).agg(
        PA=("events", "count"),
        H=("is_hit", "sum"),
        HR=("is_hr", "sum"),
        XBH=("is_xbh", "sum"),
    )
    out: dict[int, dict[str, int]] = {}
    for t in g.itertuples(index=False):
        out[int(t.batter)] = {"pa": int(t.PA), "h": int(t.H), "hr": int(t.HR), "xbh": int(t.XBH)}
    return out, None


def _tracking_outcomes_map(slate_str: str) -> dict[tuple[str, int, int, int], dict[str, int]]:
    """Filled rows from matchup_predictions_runs.parquet keyed by (slate, game_pk, batter, pitcher)."""
    path = TRACKING / "matchup_predictions_runs.parquet"
    if not path.is_file():
        return {}
    df = pd.read_parquet(path)
    df = df[df["slate_date"].astype(str) == str(slate_str)]
    if df.empty:
        return {}
    has = df["outcome_pa"].notna() | df["outcome_h"].notna()
    df = df.loc[has]
    if df.empty:
        return {}
    if "run_timestamp" in df.columns:
        df = df.sort_values("run_timestamp")
    df = df.drop_duplicates(subset=["slate_date", "game_pk", "batter_mlbam_id", "pitcher_mlbam_id"], keep="last")
    out: dict[tuple[str, int, int, int], dict[str, int]] = {}
    for r in df.itertuples(index=False):
        if pd.isna(r.game_pk) or pd.isna(r.batter_mlbam_id) or pd.isna(r.pitcher_mlbam_id):
            continue
        key = (str(r.slate_date), int(r.game_pk), int(r.batter_mlbam_id), int(r.pitcher_mlbam_id))
        pa_c = int(r.outcome_pa) if pd.notna(r.outcome_pa) else 0
        h_c = int(r.outcome_h) if pd.notna(r.outcome_h) else 0
        hr_c = int(r.outcome_hr) if pd.notna(r.outcome_hr) else 0
        xbh_c = int(r.outcome_xbh) if pd.notna(r.outcome_xbh) else 0
        out[key] = {"pa": pa_c, "h": h_c, "hr": hr_c, "xbh": xbh_c}
    return out


def _outcome_resolver_factory(
    slate_str: str,
    tracking_map: dict[tuple[str, int, int, int], dict[str, int]],
    bat_day: dict[int, dict[str, int]],
):
    def resolve(r: dict) -> dict[str, int]:
        gid = r.get("game_pk")
        bid = r.get("batter_mlbam_id")
        pid = r.get("pitcher_mlbam_id")
        if gid is not None and bid is not None and pid is not None:
            key = (str(slate_str), int(gid), int(bid), int(pid))
            if key in tracking_map:
                return dict(tracking_map[key])
        if bid is not None:
            return dict(bat_day.get(int(bid)) or {"pa": 0, "h": 0, "hr": 0, "xbh": 0})
        return {"pa": 0, "h": 0, "hr": 0, "xbh": 0}

    return resolve


def _kind_success_fn(kind: str):
    return {
        "hit": lambda od: od["h"] > 0,
        "xbh": lambda od: od["xbh"] > 0,
        "hr": lambda od: od["hr"] > 0,
    }[kind]


def _result_cells(od: dict, pred) -> tuple[str, str]:
    if od.get("pa", 0) == 0:
        return (
            '<td class="res unk" title="No Statcast PA this calendar day for this batter">—</td>',
            '<td class="act muted">No PA</td>',
        )
    ok = pred(od)
    mark, cls, lab = ("✓", "ok", "yes") if ok else ("✗", "no", "no")
    act = html.escape(f"H {od['h']} · XBH {od['xbh']} · HR {od['hr']} · PA {od['pa']}")
    return (
        f'<td class="res {cls}" aria-label="{lab}">{mark}</td>',
        f'<td class="act">{act}</td>',
    )


def _postgame_single_row_html(
    rank: int,
    r1: dict | None,
    r2: dict | None,
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
    resolve,
    pred,
    slate_date_cell: str | None,
) -> str:
    slate_td = f'<td class="slate">{html.escape(slate_date_cell)}</td>' if slate_date_cell else ""
    mu1 = _matchup_line(r1) if r1 else "—"
    pr1 = _proof_compact_html(r1, kind, xbh_lookup) if r1 else "—"
    tk1 = _bet_takeaway_html(r1, kind, xbh_lookup) if r1 else "—"
    cf1 = _confidence_cell_html(r1, kind) if r1 else "—"
    bv1 = _bvp_cell_detailed(r1, xbh_lookup) if r1 else "—"
    if r1:
        c1a, c1b = _result_cells(resolve(r1), pred)
    else:
        c1a, c1b = ("<td>—</td>", "<td>—</td>")
    mu2 = _matchup_line(r2) if r2 else "—"
    pr2 = _proof_compact_html(r2, kind, xbh_lookup) if r2 else "—"
    tk2 = _bet_takeaway_html(r2, kind, xbh_lookup) if r2 else "—"
    cf2 = _confidence_cell_html(r2, kind) if r2 else "—"
    bv2 = _bvp_cell_detailed(r2, xbh_lookup) if r2 else "—"
    if r2:
        c2a, c2b = _result_cells(resolve(r2), pred)
    else:
        c2a, c2b = ("<td>—</td>", "<td>—</td>")
    return (
        "<tr"
        + _data_teams_attr(r1, r2)
        + ">"
        f"{slate_td}"
        f'<td class="rn">{rank}</td>'
        f'<td class="mu1">{mu1}</td>'
        f'<td class="pr1">{pr1}</td>'
        f'<td class="tk1">{tk1}</td>'
        f'<td class="cf1">{cf1}</td>'
        f"{c1a}{c1b}"
        f'<td class="bvp1">{bv1}</td>'
        f'<td class="mu2">{mu2}</td>'
        f'<td class="pr2">{pr2}</td>'
        f'<td class="tk2">{tk2}</td>'
        f'<td class="cf2">{cf2}</td>'
        f"{c2a}{c2b}"
        f'<td class="bvp2">{bv2}</td>'
        "</tr>"
    )


def _rows_compare_postgame(
    rows_m1: list[dict],
    rows_m2: list[dict],
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
    resolve,
    *,
    max_rank: int = 25,
    slate_date_cell: str | None = None,
) -> str:
    pred = _kind_success_fn(kind)
    out = []
    for i, (r1, r2) in enumerate(zip_longest(rows_m1[:max_rank], rows_m2[:max_rank], fillvalue=None), 1):
        out.append(
            _postgame_single_row_html(i, r1, r2, kind, xbh_lookup, resolve, pred, slate_date_cell)
        )
    return "\n".join(out)


def _postgame_accuracy_summary_html(
    rows_m1: list[dict],
    rows_m2: list[dict],
    kind: str,
    resolve,
) -> str:
    label = {"hit": "Hits", "xbh": "XBH", "hr": "Home runs"}[kind]
    target = {"hit": "any hit", "xbh": "any extra-base hit", "hr": "any home run"}[kind]
    pred = _kind_success_fn(kind)

    def counts(rows: list[dict]) -> tuple[int, int, int]:
        hit = miss = nop = 0
        for r in rows[:25]:
            if not r:
                continue
            od = resolve(r)
            if od.get("pa", 0) == 0:
                nop += 1
                continue
            if pred(od):
                hit += 1
            else:
                miss += 1
        return hit, miss, nop

    h1, m1, n1 = counts(rows_m1)
    h2, m2, n2 = counts(rows_m2)
    e1, e2 = h1 + m1, h2 + m2
    r1 = (100.0 * h1 / e1) if e1 else None
    r2 = (100.0 * h2 / e2) if e2 else None

    both_hit = both_elig = 0
    for a, b in zip_longest(rows_m1[:25], rows_m2[:25]):
        if not a or not b:
            continue
        oa, ob = resolve(a), resolve(b)
        if oa.get("pa", 0) == 0 or ob.get("pa", 0) == 0:
            continue
        both_elig += 1
        if pred(oa) and pred(ob):
            both_hit += 1

    bullets: list[str] = []
    if e1 == 0 and e2 == 0:
        return (
            f'<aside class="divergence-summary postgame-sum" aria-label="{html.escape(label)} postgame">'
            "<h4 class=\"diverge-h\">What happened · "
            f"{html.escape(label)}</h4>"
            "<p class=\"empty\">No evaluable Statcast PA for these top-25 batters on this slate date. "
            "Append Statcast for this day, then run <code>python3 src/fill_matchup_prediction_outcomes.py --slate …</code> "
            "to populate tracking overrides.</p></aside>"
        )

    if e1:
        bullets.append(
            f"<strong>Model 1:</strong> <strong>{h1}</strong> / {e1} picks realized "
            f"<em>{html.escape(target)}</em> "
            f"({r1:.1f}%). "
            f"No same-day PA for <strong>{n1}</strong> of 25 rows."
        )
    if e2:
        bullets.append(
            f"<strong>Model 2:</strong> <strong>{h2}</strong> / {e2} picks realized "
            f"<em>{html.escape(target)}</em> "
            f"({r2:.1f}%). "
            f"No same-day PA for <strong>{n2}</strong> of 25 rows."
        )
    if both_elig:
        bullets.append(
            f"<strong>Same rank, both with PA data:</strong> both models’ picks hit the "
            f"<em>{html.escape(target)}</em> outcome on <strong>{both_hit}</strong> / "
            f"<strong>{both_elig}</strong> rows."
        )

    if r1 is not None and r2 is not None and e1 >= 8 and e2 >= 8:
        if r1 > r2 + 5.0:
            verdict = (
                "On this small sample, production’s top 25 converted at a meaningfully higher rate for this outcome — "
                "worth noting, though one slate is not enough to crown a model."
            )
            pick = '<p class="rec-pick">Better realized top-25 hit rate here: <strong>Model 1 (production)</strong>.</p>'
        elif r2 > r1 + 5.0:
            verdict = (
                "Experiment’s top 25 saw a higher realized success rate for this outcome on evaluable rows — "
                "interesting signal, still a single-slate snapshot."
            )
            pick = '<p class="rec-pick">Better realized top-25 hit rate here: <strong>Model 2 (experiment)</strong>.</p>'
        else:
            verdict = "Realized rates are within a few points — call it a draw for this slate and prop type."
            pick = '<p class="rec-pick">Realized performance: <strong>roughly tied</strong> on evaluable picks.</p>'
    else:
        verdict = "Too few evaluable rows on one or both sides to compare rates confidently."
        pick = '<p class="rec-pick">Realized performance: <strong>inconclusive</strong> (sparse PA data).</p>'

    ul = '<ul class="diverge-ul">' + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    return (
        f'<aside class="divergence-summary postgame-sum" aria-label="{html.escape(label)} postgame">'
        f'<h4 class="diverge-h">What happened · {html.escape(label)} vs actuals</h4>'
        f"{ul}"
        f'<p class="rec-p">{html.escape(verdict)}</p>'
        f"{pick}"
        "<p class=\"rec-note\">✓ / ✗ use calendar-day batter totals (or tracking row when filled), not strictly PA vs the listed pitcher only. "
        "Same methodology as <code>fill_matchup_prediction_outcomes.py</code>.</p>"
        "</aside>"
    )


def _rows_bvp_leaderboard(rows: list[dict], xbh_lookup: dict[tuple[int, int], int]) -> str:
    trs = []
    for i, r in enumerate(rows, 1):
        p = _bvp_raw_parts(r, xbh_lookup)
        if not p:
            continue
        trs.append(
            "<tr"
            + _data_teams_attr(r)
            + ">"
            f'<td class="rn">{i}</td>'
            f"<td>{_matchup_line(r)}</td>"
            f'<td class="num">{p["h"]}</td>'
            f'<td class="num">{p["pa"]}</td>'
            f'<td class="num">{p["ba"]:.3f}</td>'
            f'<td class="num">{p["xbh"]}</td>'
            f'<td class="num">{p["hr"]}</td>'
            f"<td>{_bvp_hit_types_cell(p)}</td>"
            f'<td class="num">{p["k"]}</td>'
            f'<td class="num">{p["bb"]}</td>'
            "</tr>"
        )
    return "\n".join(trs)


def _bvp_ranked_lists(P: list[dict], xbh_lookup: dict[tuple[int, int], int]) -> tuple[list[dict], list[dict], list[dict]]:
    eligible = [r for r in P if _bvp_raw_parts(r, xbh_lookup)]
    by_hits = sorted(eligible, key=lambda r: (_bvp_raw_parts(r, xbh_lookup)["h"], _bvp_raw_parts(r, xbh_lookup)["ba"]), reverse=True)[:25]
    by_xbh = sorted(eligible, key=lambda r: (_bvp_raw_parts(r, xbh_lookup)["xbh"], _bvp_raw_parts(r, xbh_lookup)["h"]), reverse=True)[:25]
    by_hr = sorted(eligible, key=lambda r: (_bvp_raw_parts(r, xbh_lookup)["hr"], _bvp_raw_parts(r, xbh_lookup)["h"]), reverse=True)[:25]
    return by_hits, by_xbh, by_hr


def _prediction_label_to_bp(P: list[dict]) -> dict[str, tuple[int, int]]:
    idx: dict[str, tuple[int, int]] = {}
    for r in P:
        bid, pid = r.get("batter_mlbam_id"), r.get("pitcher_mlbam_id")
        if bid is None or pid is None:
            continue
        k = f"{r['batter_name']} ({r['batter_team']}) vs {r['pitcher_name']}"
        idx[k] = (int(bid), int(pid))
    return idx


def _parse_section11_table(md_path: Path) -> list[tuple[str, str, str]]:
    if not md_path.is_file():
        return []
    text = md_path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 3:
            continue
        if parts[0] == "Batter vs pitcher":
            continue
        if parts[0].startswith("---"):
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def _barrels_html(
    md_path: Path,
    label_to_bp: dict[str, tuple[int, int]],
    barrel_outcomes: dict[tuple[int, int], str],
    *,
    table_id: str = "tbl-barrels",
) -> str:
    data = _parse_section11_table(md_path)
    if not data:
        return '<p class="empty">No <code>section_11.md</code> — run <code>python3 src/gen_sections_11_13.py --recent-days 7</code>.</p>'
    trs = []
    for a, b, c in data:
        bp = label_to_bp.get(a)
        hit_types = barrel_outcomes.get(bp, "—") if bp else "—"
        if hit_types != "—":
            hit_types = html.escape(hit_types)
        trs.append(
            "<tr"
            + _data_teams_attr_from_label(a)
            + ">"
            f"<td>{html.escape(a)}</td>"
            f'<td class="num">{html.escape(b)}</td>'
            f"<td>{hit_types}</td>"
            f"<td>{html.escape(c)}</td>"
            "</tr>"
        )
    id_esc = html.escape(str(table_id), quote=True)
    return (
        f'<table class="grid sortable" id="{id_esc}"><thead><tr>'
        "<th>Batter vs pitcher</th><th>Barrels</th><th>Barrel batted-ball outcomes (1B·2B·3B·HR)</th><th>Dates</th>"
        "</tr></thead>"
        f"<tbody>{''.join(trs)}</tbody></table>"
    )


def _inline_md(s: str) -> str:
    parts = s.split("**")
    out: list[str] = []
    for i, p in enumerate(parts):
        e = html.escape(p)
        if i % 2 == 1:
            out.append(f"<strong>{e}</strong>")
        else:
            out.append(e)
    return "".join(out)


def _table_md_to_html(
    lines: list[str],
    *,
    table_id: str | None = None,
    csv_filename: str | None = None,
) -> str:
    rows: list[list[str]] = []
    for line in lines:
        cells = [c.strip() for c in line.strip().split("|")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        compact = [re.sub(r"\s+", "", c) for c in cells]
        if all(c and all(ch in "-:" for ch in c) and "-" in c for c in compact):
            continue
        rows.append(cells)
    if not rows:
        return ""
    hrow = "<tr>" + "".join(f"<th>{_inline_md(c)}</th>" for c in rows[0]) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in r) + "</tr>" for r in rows[1:]
    )
    tid_attr = f' id="{html.escape(table_id)}"' if table_id else ""
    inner = f'<table class="grid prose-table sortable"{tid_attr}><thead>{hrow}</thead><tbody>{body}</tbody></table>'
    if table_id and csv_filename:
        fn_esc = html.escape(csv_filename, quote=True)
        tid_esc = html.escape(table_id, quote=True)
        return (
            f'<div class="table-wrap prose-tw">'
            f'<div class="csv-toolbar">'
            f'<button type="button" class="btn-csv" data-csv-target="{tid_esc}" data-csv-filename="{fn_esc}">'
            f"Export CSV</button></div>"
            f"{inner}</div>"
        )
    return inner


def markdown_to_html_prose(
    md: str,
    *,
    csv_basename: str | None = None,
    slate_safe: str = "export",
) -> str:
    """Lightweight MD → HTML for longform sections (headings, bold, tables, paragraphs)."""
    lines = md.splitlines()
    out: list[str] = []
    para: list[str] = []
    table_buf: list[str] = []
    table_idx = [0]

    def flush_para() -> None:
        nonlocal para
        if not para:
            return
        body = "<br/>\n".join(_inline_md(" ".join(p.split())) for p in para if p.strip())
        out.append(f'<p class="prose-p">{body}</p>')
        para = []

    def flush_table() -> None:
        nonlocal table_buf
        if table_buf:
            tid: str | None = None
            fn: str | None = None
            if csv_basename:
                table_idx[0] += 1
                slug = re.sub(r"[^\w\-]+", "-", csv_basename).strip("-") or "prose"
                tid = f"tbl-prose-{slug}-{table_idx[0]}"
                fn = f"prose_{slug}_{slate_safe}_t{table_idx[0]}.csv"
            raw = _table_md_to_html(table_buf, table_id=tid, csv_filename=fn)
            if raw:
                out.append(raw)
            table_buf = []

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush_para()
            table_buf.append(stripped)
            continue
        flush_table()

        if not stripped:
            flush_para()
            continue
        if stripped == "---":
            flush_para()
            out.append("<hr class=\"prose-hr\"/>")
            continue
        if stripped.startswith("## "):
            flush_para()
            out.append(f'<h2 class="prose-h2">{_inline_md(stripped[3:].strip())}</h2>')
            continue
        if re.match(r"^\*\*[^*]+\*\*$", stripped):
            flush_para()
            inner = stripped.strip("*")
            out.append(f'<h3 class="prose-h3">{_inline_md(inner)}</h3>')
            continue
        para.append(stripped)
    flush_table()
    flush_para()
    return "\n".join(out)


def _load_longform_path(base: str) -> Path | None:
    prod = REPORTS / f"{base}_prod.md"
    leg = REPORTS / f"{base}.md"
    if prod.is_file():
        return prod
    if leg.is_file():
        return leg
    return None


def _load_longform_path_experiment(base: str) -> Path | None:
    exp = REPORTS / f"{base}_exp.md"
    if exp.is_file() and exp.read_text(encoding="utf-8").strip():
        return exp
    return None


def _longform_tab_html(slate_safe: str, *, experiment: bool) -> str:
    parts = []
    if experiment:
        rows = (
            ("section_7", "Section 7 — Top 10 HR (experiment / model 2)", "m2-sec7"),
            ("section_8", "Section 8 — Top 10 hit (experiment / model 2)", "m2-sec8"),
            ("section_10", "Section 10 — Strikeout / whiff (experiment / model 2)", "m2-sec10"),
        )
    else:
        rows = (
            ("section_7", "Section 7 — Top 10 HR (production / model 1)", "m1-sec7"),
            ("section_8", "Section 8 — Top 10 hit (production / model 1)", "m1-sec8"),
            ("section_10", "Section 10 — Strikeout / whiff (production / model 1)", "m1-sec10"),
        )
    for sec, title, csv_slug in rows:
        path = _load_longform_path_experiment(sec) if experiment else _load_longform_path(sec)
        if path:
            body = markdown_to_html_prose(
                path.read_text(encoding="utf-8"),
                csv_basename=csv_slug,
                slate_safe=slate_safe,
            )
            parts.append(
                f'<details class="card longopen"><summary>{html.escape(title)}</summary>'
                f'<div class="cardbody prose">{body}</div></details>'
            )
        else:
            need = f"{sec}_exp.md" if experiment else f"{sec}_prod.md"
            parts.append(
                f'<p class="empty">Missing <code>{html.escape(need)}</code> — run dual model pipeline to generate longform.</p>'
            )
    return "\n".join(parts)


def _beast_meta_label(key: str) -> str:
    return {"hit": "score_label_hit", "xbh": "score_label_xbh", "hr": "score_label_hr"}[key]


def _beast_adj_raw_conf_keys(kind: str) -> tuple[str, str, str]:
    return {
        "hit": ("adj_p_hit", "p_hit", "conf_hit"),
        "xbh": ("adj_p_xbh", "p_xbh", "conf_xbh"),
        "hr": ("adj_p_hr", "p_hr", "conf_hr"),
    }[kind]


def _beast_section_blurb(kind: str, rows: list[dict]) -> str:
    """One line on whether this Hits / XBH / HR bucket is broadly a good wagering surface."""
    if not rows:
        return ""
    mk = _beast_meta_label(kind)
    adjk, _rawk, ck = _beast_adj_raw_conf_keys(kind)
    locks_strong = 0
    avoids = 0
    cfs: list[float] = []
    for r in rows:
        lb = str(r.get(mk) or "").strip()
        if lb in ("Lock", "Strong"):
            locks_strong += 1
        if lb == "Avoid":
            avoids += 1
        try:
            cfs.append(float(r.get(ck) or 0.0))
        except (TypeError, ValueError):
            cfs.append(0.0)
    mean_c = sum(cfs) / len(cfs) if cfs else 0.0
    low_cf = sum(1 for x in cfs if x < 0.75)
    try:
        top_adj = float(rows[0].get(adjk) or 0.0) * 100
    except (TypeError, ValueError):
        top_adj = 0.0
    nk = {"hit": "hits", "xbh": "XBH", "hr": "home runs"}[kind]

    if locks_strong >= 5:
        lead = (
            f"This {nk.upper()} sheet is meta-heavy ({locks_strong} Lock or Strong)—one of tonight's cleaner edges "
            "for stacking props."
        )
    elif avoids >= 8:
        head = nk.title() if kind != "hit" else "Hits"
        lead = (
            f"Crowded Avoid column ({avoids}/{len(rows)}) on {head}—treat the table as scouting, "
            "not conviction; lean small or skip."
        )
    elif mean_c >= 0.86:
        lead = (
            f"Confidence scores hug the high band (mean ~{mean_c:.2f})—good candidate bucket if you prize agreement."
        )
    elif mean_c < 0.80 or low_cf >= 10:
        lead = (
            f"Confidence is middling (~{mean_c:.2f} mean with {low_cf} softer rows)—priority props live elsewhere tonight."
        )
    else:
        lead = (
            "Balanced leaderboard: headline rows outclass the teens—tier down systematically past rank ~12 unless you "
            "see a standout BvP edge."
        )
    tail = f"Leader starts near {top_adj:.1f}% adj P per PA."
    text = lead + " " + tail
    return f'<p class="beast-section-blurb">{html.escape(text)}</p>'


def _beast_row_pick_headline(r: dict, kind: str) -> tuple[str, str]:
    """(css_mod, one-line verdict: good / not / conditional)."""
    mk = _beast_meta_label(kind)
    _, _, ck = _beast_adj_raw_conf_keys(kind)
    lab = str(r.get(mk) or "").strip()
    try:
        cf = float(r.get(ck) or 0.0)
    except (TypeError, ValueError):
        cf = 0.0
    if lab in ("Strong", "Lock"):
        return ("beast-good", "Good pick — Strong/Lock meta label lines up with a featured edge.")
    if lab == "Avoid":
        return ("beast-bad", "Not a featured pick — Avoid tag; conflicting signals or thin edge vs price.")
    if lab == "Lean":
        return ("beast-lean", "Lean only — meta says Lean; fine for a small add-on, not a core ticket.")
    if cf >= 0.88:
        return ("beast-good", "Good pick — high confidence score even without a flashy meta label.")
    if cf < 0.73:
        return ("beast-bad", "Caution — very low confidence; roster-depth or patchy matchup data.")
    return ("beast-lean", "Playable fringe — middling confidence; sizing should stay light.")


def _beast_bvp_one_liner(r: dict, xbh_lookup: dict[tuple[int, int], int]) -> str:
    p = _bvp_raw_parts(r, xbh_lookup)
    if not p:
        return "BvP: none in feed"
    return f"BvP: {p['h']}-{p['pa']} · {p['ba']:.3f}"


def _beast_row_mini_block(
    rank: int,
    r: dict,
    kind: str,
    xbh_lookup: dict[tuple[int, int], int],
) -> str:
    """Short stacked write-up: verdict line + stats line (+ BvP)."""
    cls, verdict = _beast_row_pick_headline(r, kind)
    adjk, rawk, ck = _beast_adj_raw_conf_keys(kind)
    try:
        adj = float(r.get(adjk) or 0.0)
        raw = float(r.get(rawk) or 0.0)
    except (TypeError, ValueError):
        adj, raw = 0.0, 0.0
    tpa = _three_pa(raw)
    tier = str(r.get("tier") or "").strip() or "—"
    try:
        cfn = float(r.get(ck) or 0.0)
    except (TypeError, ValueError):
        cfn = 0.0
    if kind == "hit":
        lk = str(r.get("conf_hit_label") or "").strip() or "—"
    elif kind == "xbh":
        lk = str(r.get("conf_xbh_label") or "").strip() or "—"
    else:
        lk = str(r.get("conf_hr_label") or "").strip() or "—"
    bvp_short = html.escape(_beast_bvp_one_liner(r, xbh_lookup))
    matchup = _matchup_line(r)
    stat = (
        f"Adj <strong>{adj * 100:.1f}%</strong> · raw {raw * 100:.1f}% · "
        f"~{tpa:.0f}% ≥1 over 3 PA · tier <strong>{html.escape(tier)}</strong> · "
        f"conf <strong>{html.escape(lk)}</strong> ({cfn:.2f}) · {bvp_short}"
    )
    return (
        f'<div class="beast-mini {cls}">'
        f'<div class="beast-mini-hd">{rank}. {matchup}</div>'
        f'<p class="beast-mini-verdict">{html.escape(verdict)}</p>'
        f'<p class="beast-mini-stat">{stat}</p>'
        f"</div>"
    )


def _beast_writeups_three_panels_html(
    rows_hit: list[dict],
    rows_xbh: list[dict],
    rows_hr: list[dict],
    xbh_lookup: dict[tuple[int, int], int],
) -> str:
    """Three collapsed panels: Hits / XBH / HR with leaderboard blurb + 25 condensed write-ups each."""
    blocks: list[str] = []

    def one_panel(kind: str, panel_label: str, rows: list[dict]) -> None:
        if not rows:
            return
        inner = _beast_section_blurb(kind, rows)
        inner += "".join(
            _beast_row_mini_block(i, r, kind, xbh_lookup)
            for i, r in enumerate(rows[:25], 1)
        )
        blocks.append(
            f'<details class="card beast-writeups-collapsed">'
            f"<summary>{html.escape(panel_label)} — short write-ups (top 25)</summary>"
            f'<div class="cardbody prose beast-longform-prose">{inner}</div>'
            "</details>"
        )

    one_panel("hit", "Beast · Hits", rows_hit)
    one_panel("xbh", "Beast · XBH", rows_xbh)
    one_panel("hr", "Beast · Home runs", rows_hr)
    return "".join(blocks)


_POSTGAME_MERGE_KEYS = ["slate_date", "game_pk", "batter_mlbam_id", "pitcher_mlbam_id"]


def _df_dedupe_matchup_rows(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Drop duplicate matchup keys, keeping the first row (caller must sort by model prob first)."""
    if df.empty:
        return df
    keys = [c for c in (subset or _POSTGAME_MERGE_KEYS) if c in df.columns]
    if len(keys) < 3:
        return df
    return df.drop_duplicates(subset=keys, keep="first")


def _list_dedupe_sorted_preds(rows: list[dict]) -> list[dict]:
    """Unique (game_pk, batter_mlbam_id, pitcher_mlbam_id) per list; input must be pre-sorted by desired rank."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        try:
            gpk = int(r.get("game_pk") or 0)
        except (TypeError, ValueError):
            gpk = 0
        try:
            bid = int(r.get("batter_mlbam_id") or 0)
        except (TypeError, ValueError):
            bid = 0
        try:
            pid = int(r.get("pitcher_mlbam_id") or 0)
        except (TypeError, ValueError):
            pid = 0
        k = (gpk, bid, pid)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _top_reasons_from_row(row: pd.Series) -> list:
    raw = row.get("top_reasons_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _pred_from_dual_row(row: pd.Series, prod: bool) -> dict:
    sfx = "_prod" if prod else "_exp"
    ms = "prod" if prod else "exp"
    tier_col = "tier_prod" if prod else "tier_exp"
    tier = row.get(tier_col) or row.get("tier_prod") or row.get("tier_exp") or "Average"
    if isinstance(tier, float) and pd.isna(tier):
        tier = "Average"
    tier = str(tier)
    return {
        "slate_date": str(row["slate_date"]),
        "game_pk": int(row["game_pk"]) if pd.notna(row.get("game_pk")) else None,
        "batter_mlbam_id": int(row["batter_mlbam_id"]),
        "pitcher_mlbam_id": int(row["pitcher_mlbam_id"]),
        "batter_name": row["batter_name"],
        "batter_team": row["batter_team"],
        "pitcher_name": row["pitcher_name"],
        "pitcher_throws": str(row.get("pitcher_throws") or "R"),
        "tier": str(tier),
        "adj_p_hit": float(row[f"adj_p_hit{sfx}"]),
        "p_hit": float(row[f"p_hit{sfx}"]),
        "adj_p_xbh": float(row[f"adj_p_xbh{sfx}"]),
        "p_xbh": float(row[f"p_xbh{sfx}"]),
        "adj_p_hr": float(row[f"adj_p_hr{sfx}"]),
        "p_hr": float(row[f"p_hr{sfx}"]),
        "conf_hit": float(row["conf_hit"]) if pd.notna(row.get("conf_hit")) else 0.0,
        "conf_xbh": float(row["conf_xbh"]) if pd.notna(row.get("conf_xbh")) else 0.0,
        "conf_hr": float(row["conf_hr"]) if pd.notna(row.get("conf_hr")) else 0.0,
        "conf_hit_label": str(row.get("conf_hit_label") or "Low"),
        "conf_xbh_label": str(row.get("conf_xbh_label") or "Low"),
        "conf_hr_label": str(row.get("conf_hr_label") or "Low"),
        "career_hit": float(row["career_hit"]) if pd.notna(row.get("career_hit")) else 0.0,
        "career_hr": float(row["career_hr"]) if pd.notna(row.get("career_hr")) else 0.0,
        "career_k": float(row["career_k"]) if pd.notna(row.get("career_k")) else 0.0,
        "career_bb": float(row["career_bb"]) if pd.notna(row.get("career_bb")) else 0.0,
        "hit_narrative": str(row.get("hit_narrative") or "")[:2500],
        "top_reasons": _top_reasons_from_row(row),
        "outcome_pa": row.get("outcome_pa"),
        "outcome_h": row.get("outcome_h"),
        "outcome_hr": row.get("outcome_hr"),
        "outcome_xbh": row.get("outcome_xbh"),
        "model_source": ms,
    }


def _pred_from_runs_row(row: pd.Series) -> dict:
    tr = row.get("tier") or "Average"
    if isinstance(tr, float) and pd.isna(tr):
        tr = "Average"
    tr = str(tr)
    return {
        "slate_date": str(row["slate_date"]),
        "game_pk": int(row["game_pk"]) if pd.notna(row.get("game_pk")) else None,
        "batter_mlbam_id": int(row["batter_mlbam_id"]),
        "pitcher_mlbam_id": int(row["pitcher_mlbam_id"]),
        "batter_name": row["batter_name"],
        "batter_team": row["batter_team"],
        "pitcher_name": row["pitcher_name"],
        "pitcher_throws": str(row.get("pitcher_throws") or "R"),
        "tier": tr,
        "adj_p_hit": float(row["adj_p_hit"]),
        "p_hit": float(row["p_hit"]),
        "adj_p_xbh": float(row["adj_p_xbh"]),
        "p_xbh": float(row["p_xbh"]),
        "adj_p_hr": float(row["adj_p_hr"]),
        "p_hr": float(row["p_hr"]),
        "conf_hit": float(row["conf_hit"]) if pd.notna(row.get("conf_hit")) else 0.0,
        "conf_xbh": float(row["conf_xbh"]) if pd.notna(row.get("conf_xbh")) else 0.0,
        "conf_hr": float(row["conf_hr"]) if pd.notna(row.get("conf_hr")) else 0.0,
        "conf_hit_label": str(row.get("conf_hit_label") or "Low"),
        "conf_xbh_label": str(row.get("conf_xbh_label") or "Low"),
        "conf_hr_label": str(row.get("conf_hr_label") or "Low"),
        "career_hit": float(row["career_hit"]) if pd.notna(row.get("career_hit")) else 0.0,
        "career_hr": float(row["career_hr"]) if pd.notna(row.get("career_hr")) else 0.0,
        "career_k": float(row["career_k"]) if pd.notna(row.get("career_k")) else 0.0,
        "career_bb": float(row["career_bb"]) if pd.notna(row.get("career_bb")) else 0.0,
        "hit_narrative": str(row.get("hit_narrative") or "")[:2500],
        "top_reasons": _top_reasons_from_row(row),
        "outcome_pa": row.get("outcome_pa"),
        "outcome_h": row.get("outcome_h"),
        "outcome_hr": row.get("outcome_hr"),
        "outcome_xbh": row.get("outcome_xbh"),
        "model_source": "prod",
    }


def _load_postgame_tracking_merged() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dual rows + outcomes/conf/career from runs; prod-only rows for slates missing dual export."""
    path_runs = TRACKING / "matchup_predictions_runs.parquet"
    path_dual = TRACKING / "matchup_dual_model_predictions.parquet"
    empty = (pd.DataFrame(), pd.DataFrame())
    if not path_runs.is_file():
        return empty
    runs = pd.read_parquet(path_runs)
    runs["slate_date"] = runs["slate_date"].astype(str)
    if "run_timestamp" in runs.columns:
        runs = runs.sort_values("run_timestamp")
    dk = list(_POSTGAME_MERGE_KEYS)
    runs = runs.drop_duplicates(subset=dk, keep="last")
    extra = [
        "outcome_pa",
        "outcome_h",
        "outcome_hr",
        "outcome_xbh",
        "hit_narrative",
        "top_reasons_json",
        "conf_hit",
        "conf_xbh",
        "conf_hr",
        "conf_hit_label",
        "conf_xbh_label",
        "conf_hr_label",
        "career_hit",
        "career_hr",
        "career_k",
        "career_bb",
        "pitcher_throws",
    ]
    oc_cols = [c for c in dk + extra if c in runs.columns]
    oc = runs[oc_cols].copy()
    if not path_dual.is_file():
        return pd.DataFrame(), runs
    dual = pd.read_parquet(path_dual)
    dual["slate_date"] = dual["slate_date"].astype(str)
    # Exports can append the same matchup multiple times; merge would duplicate rows and postgame tables.
    dual = dual.drop_duplicates(subset=dk, keep="last")
    merged = dual.merge(oc, on=dk, how="left")
    merged = merged.drop_duplicates(subset=dk, keep="last")
    dual_slates = set(merged["slate_date"].unique())
    runs_only = runs.loc[~runs["slate_date"].isin(dual_slates)].copy()
    return merged, runs_only if not runs_only.empty else pd.DataFrame()


def _sorted_pred_lists_for_slate(
    merged: pd.DataFrame,
    runs_only: pd.DataFrame,
    slate_date: str,
    kind: str,
) -> tuple[list[dict], list[dict]]:
    adj_dual = {
        "hit": ("adj_p_hit_prod", "adj_p_hit_exp"),
        "xbh": ("adj_p_xbh_prod", "adj_p_xbh_exp"),
        "hr": ("adj_p_hr_prod", "adj_p_hr_exp"),
    }[kind]
    adj_run = {"hit": "p_hit", "xbh": "p_xbh", "hr": "p_hr"}[kind]
    if not merged.empty and slate_date in set(merged["slate_date"].unique()):
        sub = merged.loc[merged["slate_date"] == slate_date]
        a1, a2 = adj_dual
        s1 = _df_dedupe_matchup_rows(sub.sort_values(a1, ascending=False))
        s2 = _df_dedupe_matchup_rows(sub.sort_values(a2, ascending=False))
        m1 = [_pred_from_dual_row(r, True) for _, r in s1.iterrows()]
        m2 = [_pred_from_dual_row(r, False) for _, r in s2.iterrows()]
        return m1, m2
    if runs_only.empty or slate_date not in set(runs_only["slate_date"].unique()):
        return [], []
    sub = runs_only.loc[runs_only["slate_date"] == slate_date]
    sort_col = adj_run if adj_run in sub.columns else {"hit": "adj_p_hit", "xbh": "adj_p_xbh", "hr": "adj_p_hr"}[kind]
    s1 = _df_dedupe_matchup_rows(sub.sort_values(sort_col, ascending=False))
    m1 = [_pred_from_runs_row(r) for _, r in s1.iterrows()]
    m2 = [None] * len(m1)
    return m1, m2


def _load_prod_exp_json_for_slate(sd: str) -> tuple[list[dict], list[dict]] | None:
    """Load flat prod/exp prediction lists for a slate from archive or current root JSON."""
    sd10 = str(sd)[:10]
    arch_p = REPORTS / "archive" / sd10 / "todays_matchup_predictions.json"
    if arch_p.is_file():
        prod = json.loads(arch_p.read_text(encoding="utf-8"))
        arch_e = REPORTS / "archive" / sd10 / "todays_matchup_predictions_exp.json"
        exp = json.loads(arch_e.read_text(encoding="utf-8")) if arch_e.is_file() else []
        return prod, exp
    root_p = REPORTS / "todays_matchup_predictions.json"
    root_e = REPORTS / "todays_matchup_predictions_exp.json"
    if not root_p.is_file():
        return None
    prod = json.loads(root_p.read_text(encoding="utf-8"))
    if not prod or str(prod[0].get("slate_date") or "")[:10] != sd10:
        return None
    exp = json.loads(root_e.read_text(encoding="utf-8")) if root_e.is_file() else []
    return prod, exp


def _sorted_pred_lists_from_json(
    prod: list[dict],
    exp: list[dict],
    kind: str,
) -> tuple[list[dict], list[dict]]:
    raw = {"hit": "p_hit", "xbh": "p_xbh", "hr": "p_hr"}[kind]
    adj = {"hit": "adj_p_hit", "xbh": "adj_p_xbh", "hr": "adj_p_hr"}[kind]
    s1 = sorted(prod, key=lambda r: float(r.get(raw) or r.get(adj) or 0.0), reverse=True)
    s2 = sorted(exp, key=lambda r: float(r.get(raw) or r.get(adj) or 0.0), reverse=True) if exp else []
    s1 = _list_dedupe_sorted_preds(s1)
    s2 = _list_dedupe_sorted_preds(s2) if s2 else []
    return s1, s2


def _build_postgame_interactive_payload(
    xbh_lookup: dict[tuple[int, int], int],
    *,
    primary_slate: str | None = None,
) -> dict | None:
    merged, runs_only = _load_postgame_tracking_merged()
    slates_m = set(merged["slate_date"].tolist()) if not merged.empty else set()
    slates_r = set(runs_only["slate_date"].tolist()) if not runs_only.empty else set()
    slates_parquet = slates_m | slates_r
    arch_slates = _archive_prediction_slate_dates()
    # Union of Parquet slates + archive snapshot folders; rows built from Parquet when present else from JSON on disk.
    all_slates = sorted(slates_parquet | arch_slates)
    if not all_slates:
        return None
    kinds = ("hit", "xbh", "hr")
    rows: dict[str, list[dict]] = {k: [] for k in kinds}
    bat_cache: dict[str, dict[int, dict[str, int]]] = {}
    track_cache: dict[str, dict[tuple[str, int, int, int], dict[str, int]]] = {}

    def _bat_map(sd: str) -> dict[int, dict[str, int]]:
        if sd not in bat_cache:
            bat_cache[sd], _ = _batter_day_outcome_stats(sd)
        return bat_cache[sd]

    def _track_map_for(sd: str) -> dict[tuple[str, int, int, int], dict[str, int]]:
        if sd not in track_cache:
            track_cache[sd] = _tracking_outcomes_map(sd)
        return track_cache[sd]

    row_slates: set[str] = set()
    for sd in all_slates:
        for kind in kinds:
            pred_fn = _kind_success_fn(kind)
            m1, m2 = _sorted_pred_lists_for_slate(merged, runs_only, sd, kind)
            if not m1:
                bundle = _load_prod_exp_json_for_slate(sd)
                if not bundle:
                    continue
                prod_j, exp_j = bundle
                m1, m2 = _sorted_pred_lists_from_json(prod_j, exp_j, kind)
                if not m1:
                    continue
            # Always resolve actuals via tracking fills (matchup-specific) then Statcast
            # calendar-day batter rollup — not outcome_* columns alone on dual-model rows.
            # Otherwise recent slates in the Parquet export show "No PA" until runs back-fill.
            resolve_f = _outcome_resolver_factory(sd, _track_map_for(sd), _bat_map(sd))
            top = min(50, len(m1))
            row_slates.add(sd)
            for i in range(1, top + 1):
                r1 = m1[i - 1]
                r2 = m2[i - 1] if i - 1 < len(m2) else None
                od1 = resolve_f(r1)
                od2 = resolve_f(r2) if r2 else {"pa": 0, "h": 0, "hr": 0, "xbh": 0}
                elig1 = od1.get("pa", 0) > 0
                elig2 = r2 is not None and od2.get("pa", 0) > 0
                ok1 = bool(pred_fn(od1)) if elig1 else None
                ok2 = bool(pred_fn(od2)) if elig2 else None
                ht = _postgame_single_row_html(
                    i, r1, r2, kind, xbh_lookup, resolve_f, pred_fn, sd
                )
                rows[kind].append(
                    {
                        "sd": sd,
                        "rk": i,
                        "m1e": elig1,
                        "m1o": ok1,
                        "m2e": elig2,
                        "m2o": ok2,
                        "html": ht,
                    }
                )
    dmin, dmax = all_slates[0], all_slates[-1]
    yd = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _clamp(d: str) -> str:
        if d < dmin:
            return dmin
        if d > dmax:
            return dmax
        return d

    # Default window: day *before* primary slate when that day has postgame rows (completed games /
    # evaluable actuals while Matchups tab shows the upcoming slate). Else primary if it has rows;
    # else calendar yesterday.
    ps = (primary_slate or "").strip()[:10]
    prev_c = ""
    if ps and re.match(r"^\d{4}-\d{2}-\d{2}$", ps):
        try:
            prev_c = _clamp((pd.Timestamp(ps) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        except (ValueError, OSError):
            prev_c = ""
    if prev_c and prev_c in row_slates:
        d_start = d_end = prev_c
    elif ps and ps in row_slates:
        d_start = d_end = _clamp(ps)
    else:
        d_start = d_end = _clamp(yd)
    return {
        "sortedSlates": all_slates,
        "slatesWithRows": sorted(row_slates),
        "dateMin": dmin,
        "dateMax": dmax,
        "daysCount": len(all_slates),
        "yesterday": yd,
        "primarySlate": ps or None,
        "defaultStart": d_start,
        "defaultEnd": d_end,
        "defaultTopN": 25,
        "rows": rows,
    }


def _postgame_kpi_row_html(kind_slug: str) -> str:
    return (
        f'<div class="postgame-kpis" id="kpi-row-{kind_slug}" aria-live="polite">'
        '<div class="kpi-card"><div class="kpi-label">Model 1 · correct</div>'
        f'<div class="kpi-value" id="kpi-{kind_slug}-m1c">—</div></div>'
        '<div class="kpi-card"><div class="kpi-label">Model 1 · evaluable picks</div>'
        f'<div class="kpi-value" id="kpi-{kind_slug}-m1t">—</div></div>'
        '<div class="kpi-card"><div class="kpi-label">Model 1 · hit rate</div>'
        f'<div class="kpi-value" id="kpi-{kind_slug}-m1p">—</div></div>'
        '<div class="kpi-card"><div class="kpi-label">Model 2 · correct</div>'
        f'<div class="kpi-value" id="kpi-{kind_slug}-m2c">—</div></div>'
        '<div class="kpi-card"><div class="kpi-label">Model 2 · evaluable picks</div>'
        f'<div class="kpi-value" id="kpi-{kind_slug}-m2t">—</div></div>'
        '<div class="kpi-card"><div class="kpi-label">Model 2 · hit rate</div>'
        f'<div class="kpi-value" id="kpi-{kind_slug}-m2p">—</div></div>'
        "</div>"
    )


def _postgame_interactive_shell(kind_slug: str, table_id: str, summary_id: str, title_suffix: str) -> str:
    return (
        f'<div class="postgame-section" data-pg-kind="{html.escape(kind_slug)}">'
        f"{_postgame_kpi_row_html(kind_slug)}"
        f'<aside class="divergence-summary postgame-sum" id="{html.escape(summary_id)}">'
        f'<h4 class="diverge-h">{html.escape(title_suffix)}</h4>'
        f'<p class="rec-p" id="{html.escape(summary_id)}-line"></p>'
        '<p class="rec-note">✓ / ✗ use calendar-day batter totals from tracking fills where present. '
        "Filters restrict which slate dates and top-N ranks are included.</p>"
        "</aside>"
        '<div class="scroll"><table class="dual compare postgame striped sortable" id="'
        + html.escape(table_id)
        + '"><thead><tr>'
        "<th>Slate</th><th>#</th>"
        "<th>Model 1 · matchup</th><th>Model 1 · proof</th><th>Model 1 · bet read</th><th>Model 1 · confidence</th>"
        "<th>Result</th><th>Actual H·XBH·HR·PA</th><th>Model 1 · BvP</th>"
        "<th>Model 2 · matchup</th><th>Model 2 · proof</th><th>Model 2 · bet read</th><th>Model 2 · confidence</th>"
        "<th>Result</th><th>Actual H·XBH·HR·PA</th><th>Model 2 · BvP</th>"
        "</tr></thead><tbody></tbody></table></div></div>"
    )


POSTGAME_JS = r"""
(function() {
  function readData() {
    var el = document.getElementById("postgame-app-data");
    if (!el || !el.textContent) return null;
    try { return JSON.parse(el.textContent); } catch (e) { return null; }
  }
  function pct(c, t) {
    if (!t) return "—";
    return (100.0 * c / t).toFixed(1) + "%";
  }
  function clampDate(s, lo, hi) {
    if (!s || String(s).length < 10) return lo;
    s = String(s).slice(0, 10);
    if (s < lo) return lo;
    if (s > hi) return hi;
    return s;
  }
  function applyPostgame() {
    var D = readData();
    if (!D || !D.rows) return;
    var lo = D.dateMin, hi = D.dateMax;
    var inp0 = document.getElementById("pg-date-start");
    var inp1 = document.getElementById("pg-date-end");
    var raw0 = (inp0 && inp0.value) ? inp0.value : (D.defaultStart || lo);
    var raw1 = (inp1 && inp1.value) ? inp1.value : (D.defaultEnd || hi);
    var d0 = clampDate(raw0, lo, hi);
    var d1 = clampDate(raw1, lo, hi);
    if (d0 > d1) { var tmp = d0; d0 = d1; d1 = tmp; }
    if (inp0 && inp0.value !== d0) inp0.value = d0;
    if (inp1 && inp1.value !== d1) inp1.value = d1;
    var topN = parseInt(document.getElementById("pg-topn").value, 10) || 25;
    var meta = document.getElementById("pg-meta-days");
    if (meta) {
      var yd = D.yesterday || "";
      var sw = D.slatesWithRows || [];
      var ps = D.primarySlate || "";
      var ds = D.defaultStart || "";
      var de = D.defaultEnd || "";
      meta.textContent = "Date range " + lo + " … " + hi + " (" + (D.daysCount || 0) + " calendar day(s)). "
        + "Rows built for " + sw.length + " day(s): " + (sw.length ? sw.join(", ") : "—")
        + " (predictions from tracking Parquet when present, otherwise from data/reports/archive JSON; outcomes use tracking fills plus Statcast calendar-day totals). "
        + "Calendar yesterday was " + yd + ". Default filter window: " + ds + "–" + de
        + (ps ? " (primary slate " + ps + ")." : ".");
    }
    ["hit","xbh","hr"].forEach(function(kind) {
      var rows = D.rows[kind] || [];
      var parts = [];
      var m1c = 0, m1t = 0, m2c = 0, m2t = 0;
      for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        if (r.sd < d0 || r.sd > d1) continue;
        if (r.rk > topN) continue;
        parts.push(r.html);
        if (r.m1e) { m1t++; if (r.m1o) m1c++; }
        if (r.m2e) { m2t++; if (r.m2o) m2c++; }
      }
      var tb = document.querySelector("#tbl-post-" + kind + " tbody");
      if (tb) tb.innerHTML = parts.join("");
      function set(id, v) { var x = document.getElementById(id); if (x) x.textContent = v; }
      set("kpi-" + kind + "-m1c", String(m1c));
      set("kpi-" + kind + "-m1t", String(m1t));
      set("kpi-" + kind + "-m1p", pct(m1c, m1t));
      set("kpi-" + kind + "-m2c", String(m2c));
      set("kpi-" + kind + "-m2t", String(m2t));
      set("kpi-" + kind + "-m2p", pct(m2c, m2t));
      var ln = document.getElementById("pg-aside-" + kind + "-line");
      if (ln) {
        ln.textContent = "Window " + d0 + "–" + d1 + ", top " + topN + " per slate: model 1 " + m1c + "/" + m1t + " (" + pct(m1c, m1t) + "); model 2 " + m2c + "/" + m2t + " (" + pct(m2c, m2t) + ").";
      }
    });
    if (typeof window.__applyTeamFilter === "function") window.__applyTeamFilter();
  }
  function wire() {
    var D = readData();
    if (!D || !D.sortedSlates || !D.sortedSlates.length) return;
    if (window.__POSTGAME_WIRED) return;
    window.__POSTGAME_WIRED = true;
    var ds = document.getElementById("pg-date-start");
    var de = document.getElementById("pg-date-end");
    if (!ds || !de) return;
    ds.min = de.min = D.dateMin;
    ds.max = de.max = D.dateMax;
    ds.value = D.defaultStart || D.dateMin;
    de.value = D.defaultEnd || D.dateMax;
    var tn = document.getElementById("pg-topn");
    if (tn) tn.value = String(D.defaultTopN || 25);
    function bind(el) {
      if (!el) return;
      el.addEventListener("input", applyPostgame);
      el.addEventListener("change", applyPostgame);
    }
    bind(ds); bind(de); bind(tn);
    applyPostgame();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
  else wire();
})();
"""


SLATE_SWITCH_JS = r"""
(function() {
  var sel = document.getElementById("matchup-slate-select");
  if (!sel) return;
  function sync() {
    var sd = sel.value;
    document.querySelectorAll(".main-slate-view").forEach(function(el) {
      el.classList.toggle("hidden", el.getAttribute("data-slate") !== sd);
    });
    document.querySelectorAll(".recency-slate-view").forEach(function(el) {
      el.classList.toggle("hidden", el.getAttribute("data-slate") !== sd);
    });
    document.querySelectorAll(".beast-slate-view").forEach(function(el) {
      el.classList.toggle("hidden", el.getAttribute("data-slate") !== sd);
    });
    document.querySelectorAll(".residual-slate-view").forEach(function(el) {
      el.classList.toggle("hidden", el.getAttribute("data-slate") !== sd);
    });
    document.querySelectorAll(".slate-weather-slate-view").forEach(function(el) {
      el.classList.toggle("hidden", el.getAttribute("data-slate") !== sd);
    });
    var vis = document.querySelector(".main-slate-view:not(.hidden)");
    var d = document.getElementById("hdr-slate-d");
    var n = document.getElementById("hdr-slate-n");
    if (vis && d && n) {
      d.textContent = vis.getAttribute("data-slate") || "";
      n.textContent = vis.getAttribute("data-n") || "0";
    }
  }
  sel.addEventListener("change", sync);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", sync);
  else sync();
})();
"""


NOTEPAD_JS = r"""
(function() {
  var KEY = "lad_matchup_dashboard_notes_v1";
  var ta = document.getElementById("dash-notepad-field");
  var aside = document.getElementById("dash-notepad");
  var btn = document.getElementById("dash-notepad-toggle");
  if (!ta || !aside || !btn) return;
  function load() {
    try {
      var v = localStorage.getItem(KEY);
      if (v !== null && v !== undefined) ta.value = v;
    } catch (e) {}
  }
  function save() {
    try { localStorage.setItem(KEY, ta.value); } catch (e) {}
  }
  function setCollapsed(c) {
    aside.classList.toggle("dash-notepad--collapsed", c);
    btn.setAttribute("aria-expanded", c ? "false" : "true");
    btn.textContent = c ? "Show" : "Hide";
  }
  load();
  ta.addEventListener("input", save);
  ta.addEventListener("blur", save);
  btn.addEventListener("click", function() {
    setCollapsed(!aside.classList.contains("dash-notepad--collapsed"));
  });
})();
"""


TEAM_FILTER_JS = r"""
(function() {
  var dd = document.getElementById("team-filter-dd");
  var btn = document.getElementById("team-filter-dd-btn");
  var panel = document.getElementById("team-filter-dd-panel");
  var summary = document.getElementById("team-filter-dd-summary");
  var clr = document.getElementById("team-filter-clear");
  if (!dd || !btn || !panel) return;

  function selectedTeams() {
    return Array.prototype.map.call(
      document.querySelectorAll(".team-filter-cb:checked"),
      function(cb) { return cb.value; }
    );
  }
  function updateSummary() {
    if (!summary) return;
    var s = selectedTeams();
    if (!s.length) summary.textContent = "All teams";
    else if (s.length <= 2) summary.textContent = s.join(", ");
    else summary.textContent = s[0] + " +" + (s.length - 1) + " more";
  }
  function setOpen(open) {
    panel.classList.toggle("hidden", !open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  }
  function rowTeams(tr) {
    return (tr.getAttribute("data-teams") || "").split(",").map(function(s) { return s.trim(); }).filter(Boolean);
  }
  /** Every batter team on the row must be in the selected set (data-teams = batter_team only from generator). */
  function rowMatches(tr) {
    var want = selectedTeams();
    var have = rowTeams(tr);
    if (!want.length) return true;
    if (!have.length) return false;
    var allow = {};
    for (var w = 0; w < want.length; w++) allow[want[w]] = true;
    for (var h = 0; h < have.length; h++) {
      if (!allow[have[h]]) return false;
    }
    return true;
  }
  function cardMatches(card) {
    var want = selectedTeams();
    if (!want.length) return true;
    var t = card.textContent || "";
    for (var i = 0; i < want.length; i++) {
      if (t.indexOf("(" + want[i] + ")") >= 0) return true;
    }
    return false;
  }
  function applyTeamFilter() {
    updateSummary();
    document.querySelectorAll("tr[data-teams]").forEach(function(tr) {
      tr.classList.toggle("team-filter-hidden", !rowMatches(tr));
    });
    document.querySelectorAll("#panel-long details.card, #panel-long-exp details.card").forEach(function(card) {
      card.classList.toggle("team-filter-hidden", !cardMatches(card));
    });
  }
  window.__applyTeamFilter = applyTeamFilter;

  btn.addEventListener("click", function(ev) {
    ev.stopPropagation();
    setOpen(panel.classList.contains("hidden"));
  });
  document.addEventListener("click", function(ev) {
    if (!dd.contains(ev.target)) setOpen(false);
  });
  panel.addEventListener("click", function(ev) { ev.stopPropagation(); });
  document.addEventListener("keydown", function(ev) {
    if (ev.key === "Escape" && !panel.classList.contains("hidden")) setOpen(false);
  });
  document.querySelectorAll(".team-filter-cb").forEach(function(cb) {
    cb.addEventListener("change", applyTeamFilter);
  });
  if (clr) clr.addEventListener("click", function() {
    document.querySelectorAll(".team-filter-cb").forEach(function(c) { c.checked = false; });
    setOpen(false);
    applyTeamFilter();
  });
  applyTeamFilter();
})(); 
"""


def _dashboard_section(
    title: str,
    inner: str,
    open_first: bool = False,
    csv_filename: str | None = None,
    table_id: str | None = None,
) -> str:
    op = " open" if open_first else ""
    tb = ""
    if csv_filename and table_id:
        fn = html.escape(f"{csv_filename}.csv", quote=True)
        tid = html.escape(table_id, quote=True)
        tb = (
            f'<div class="csv-toolbar">'
            f'<button type="button" class="btn-csv" data-csv-target="{tid}" '
            f'data-csv-filename="{fn}">Export CSV</button>'
            f"</div>"
        )
    return f'<details class="card"{op}><summary>{html.escape(title)}</summary><div class="cardbody">{tb}{inner}</div></details>'


def _gather_slate_prediction_bundles() -> list[dict]:
    """Current root JSON plus distinct archive dates (excluding archive copy of the same day as root)."""
    prod_path = REPORTS / "todays_matchup_predictions.json"
    exp_path = REPORTS / "todays_matchup_predictions_exp.json"
    rec_path = REPORTS / "todays_matchup_predictions_recency.json"
    beast_path = REPORTS / "todays_matchup_predictions_beast.json"
    if not prod_path.is_file() or not exp_path.is_file():
        raise SystemExit(f"Missing predictions: {prod_path} and/or {exp_path}")
    P0 = json.loads(prod_path.read_text(encoding="utf-8"))
    E0 = json.loads(exp_path.read_text(encoding="utf-8"))
    R0_raw = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else []
    B0_raw = json.loads(beast_path.read_text(encoding="utf-8")) if beast_path.is_file() else []
    R0 = _recency_rows_aligned_to_prod(P0, R0_raw)
    B0 = _beast_rows_aligned_to_prod(P0, B0_raw)
    primary_sd = str(P0[0].get("slate_date") or "")[:10] if P0 else ""
    by_date: dict[str, dict] = {}
    if P0:
        key = primary_sd or "unknown"
        by_date[key] = {"slate": key, "P": P0, "E": E0, "R": R0, "B": B0}
    arch = REPORTS / "archive"
    if arch.is_dir():
        for sub in sorted(arch.iterdir()):
            if not sub.is_dir():
                continue
            cand = sub.name[:10]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", cand):
                continue
            pj = sub / "todays_matchup_predictions.json"
            if not pj.is_file():
                continue
            if cand == primary_sd and primary_sd:
                continue
            ej = sub / "todays_matchup_predictions_exp.json"
            rj = sub / "todays_matchup_predictions_recency.json"
            bj = sub / "todays_matchup_predictions_beast.json"
            Pd = json.loads(pj.read_text(encoding="utf-8"))
            Ed = json.loads(ej.read_text(encoding="utf-8")) if ej.is_file() else []
            Rraw = json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else []
            Braw = json.loads(bj.read_text(encoding="utf-8")) if bj.is_file() else []
            Rd = _recency_rows_aligned_to_prod(Pd, Rraw)
            Bd = _beast_rows_aligned_to_prod(Pd, Braw)
            by_date[cand] = {"slate": cand, "P": Pd, "E": Ed, "R": Rd, "B": Bd}
    if not by_date and P0:
        by_date[primary_sd or "unknown"] = {
            "slate": primary_sd or "unknown",
            "P": P0,
            "E": E0,
            "R": R0,
            "B": B0,
        }
    return [by_date[k] for k in sorted(by_date.keys(), reverse=True)]


def _archive_prediction_slate_dates() -> set[str]:
    """Calendar dates that have a dashboard prediction snapshot under data/reports/archive/."""
    out: set[str] = set()
    arch = REPORTS / "archive"
    if not arch.is_dir():
        return out
    for sub in arch.iterdir():
        if not sub.is_dir():
            continue
        cand = sub.name[:10]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", cand) and (sub / "todays_matchup_predictions.json").is_file():
            out.add(cand)
    return out


def _recency_tab_inner_html(
    R_list: list,
    slate: str,
    xbh_lu: dict[tuple[int, int], int],
    *,
    table_id_suffix: str,
    for_embedded_main: bool = False,
) -> str:
    """Recency top-25 (hits / XBH / HR). Tab: empty-state copy if no JSON. Main embed: omit when no data; distinct table ids when present."""
    slate_safe = re.sub(r"[^\w\-]+", "_", str(slate))
    R_use = R_list or []
    if not R_use:
        if for_embedded_main:
            return ""
        return (
            '<p class="intro">No recency predictions for this slate. When '
            "<code>todays_matchup_predictions_recency.json</code> is built for the same "
            "<code>slate_date</code> as production, tables appear here after the next "
            '<code>gen_matchup_dashboard_html.py</code> run. See <code>docs/recency-model.md</code>.</p>'
        )

    id_prefix = "tbl-recency-main" if for_embedded_main else "tbl-recency"
    csv_prefix = "matchup_recency_main_" if for_embedded_main else "matchup_recency_"

    def tid(kind: str) -> str:
        base = f"{id_prefix}-{kind}"
        return f"{base}{table_id_suffix}" if table_id_suffix else base

    S_hit_r = sorted(R_use, key=lambda r: float(r.get("p_hit") or r.get("adj_p_hit") or 0), reverse=True)[:25]
    S_xbh_r = sorted(R_use, key=lambda r: float(r.get("p_xbh") or r.get("adj_p_xbh") or 0), reverse=True)[:25]
    S_hr_r = sorted(R_use, key=lambda r: float(r.get("p_hr") or r.get("adj_p_hr") or 0), reverse=True)[:25]

    tbl_recency = (
        '<div class="scroll"><table class="dual compare striped sortable" id="{tid}">'
        "<thead><tr>"
        "<th>#</th>"
        "<th>Recency · matchup</th><th>Recency · proof</th><th>Recency · bet read</th><th>Recency · confidence</th><th>Recency · BvP</th>"
        "</tr></thead><tbody>{body}</tbody></table></div>"
    )
    rec_intro = (
        '<p class="intro">Optional <strong>recency</strong> model (L3/L5 short rolls + optional weighted training): '
        "<code>todays_matchup_predictions_recency.json</code> for this slate. "
        "See <code>docs/recency-model.md</code>. Rankings use only this model’s adjusted probabilities.</p>"
    )
    return (
        rec_intro
        + _dashboard_section(
            "Hits — Recency model top 25",
            tbl_recency.format(
                tid=tid("hits"),
                body=_rows_recency_top25(S_hit_r, "hit", xbh_lu),
            ),
            open_first=True,
            csv_filename=f"{csv_prefix}hits_{slate_safe}",
            table_id=tid("hits"),
        )
        + _dashboard_section(
            "XBH — Recency model top 25",
            tbl_recency.format(
                tid=tid("xbh"),
                body=_rows_recency_top25(S_xbh_r, "xbh", xbh_lu),
            ),
            csv_filename=f"{csv_prefix}xbh_{slate_safe}",
            table_id=tid("xbh"),
        )
        + _dashboard_section(
            "Home runs — Recency model top 25",
            tbl_recency.format(
                tid=tid("hr"),
                body=_rows_recency_top25(S_hr_r, "hr", xbh_lu),
            ),
            csv_filename=f"{csv_prefix}hr_{slate_safe}",
            table_id=tid("hr"),
        )
    )


def _beast_tab_inner_html(
    B_list: list,
    slate: str,
    xbh_lu: dict[tuple[int, int], int],
    *,
    table_id_suffix: str,
) -> str:
    """Single-model rows for optional Beast bundle (caller sorts by adj_*)."""
    B_use = B_list or []
    if not B_use:
        return (
            '<p class="intro">No Beast predictions for this slate. Build '
            "<code>todays_matchup_predictions_beast.json</code> for the same "
            "<code>slate_date</code> as production, then rerun "
            '<code>gen_matchup_dashboard_html.py</code>.</p>'
        )
    tid = (lambda base: f"{base}{table_id_suffix}") if table_id_suffix else (lambda base: base)
    slate_safe = re.sub(r"[^\w\-]+", "_", str(slate))
    intro = (
        '<p class="intro">Optional <strong>Beast</strong> model top-25 rankings '
        "from <code>todays_matchup_predictions_beast.json</code> (same slate alignment as production). "
        "Rankings use this model’s adjusted probabilities only. "
        "Between <strong>Proof</strong> and <strong>Confidence</strong>, each row spells out <strong>why</strong> Beast leaned that way: calibrated priors vs MLB baselines, rolling Statcast contact versus the league midpoint, how the pitch mix meshes (or clashes) with hitter strengths, plus honest BvP and strikeout context. "
        "<strong>Rel</strong> = 1–5 vs the other rows in that table (quintiles of the evidence multiplier); "
        "<strong>5</strong> = strongest confidence in the top 25. "
        "After <strong>BvP</strong>: <strong>2026 vs LHP/RHP</strong> is season-to-date Statcast vs tonight's pitcher hand "
        "(before slate); <strong>Career vs LHP/RHP</strong> is λ-decayed career split from the same hand.</p>"
    )
    S_hit_b = sorted(B_use, key=lambda r: float(r.get("p_hit") or r.get("adj_p_hit") or 0), reverse=True)[:25]
    S_xbh_b = sorted(B_use, key=lambda r: float(r.get("p_xbh") or r.get("adj_p_xbh") or 0), reverse=True)[:25]
    S_hr_b = sorted(B_use, key=lambda r: float(r.get("p_hr") or r.get("adj_p_hr") or 0), reverse=True)[:25]

    warm_ids: set[int] = set()
    for grp in (S_hit_b, S_xbh_b, S_hr_b):
        for row in grp:
            bid = row.get("batter_mlbam_id")
            if bid is not None:
                try:
                    warm_ids.add(int(bid))
                except (TypeError, ValueError):
                    pass
    _beast_warm_recent_bip_ev(warm_ids)

    beast_rel_th = (
        '<th title="Relative confidence among these 25 rows only: quintiles of evidence multiplier '
        '(5 = strongest in this slice, 1 = weakest).">Rel</th>'
    )
    tbl_beast = (
        '<div class="scroll"><table class="dual compare striped sortable beast-top25" id="{tid}">'
        "<thead><tr>"
        "<th>#</th>"
        "<th>Beast · matchup</th><th>Beast · proof</th>"
        '<th title="Why Beast leaned this way: calibrated story, contact+arsenal fit, BvP/K risks with MLB anchors.">Beast · takeaway</th>'
        "<th>Beast · confidence</th>"
        + beast_rel_th
        + "<th>Beast · BvP</th>"
        + '<th title="2026 season-to-date vs tonight pitcher hand (before slate date).">2026 vs LHP/RHP</th>'
        + '<th title="Decay-weighted career vs tonight pitcher hand (prior seasons, λ-decayed splits).">Career vs LHP/RHP</th>'
        "</tr></thead><tbody>{body}</tbody></table></div>"
    )
    return (
        intro
        + _beast_writeups_three_panels_html(S_hit_b, S_xbh_b, S_hr_b, xbh_lu)
        + _dashboard_section(
            "Hits — Beast model top 25",
            tbl_beast.format(
                tid=tid("tbl-beast-hits"),
                body=_rows_beast_top25(S_hit_b, "hit", xbh_lu),
            ),
            open_first=True,
            csv_filename=f"matchup_beast_hits_{slate_safe}",
            table_id=tid("tbl-beast-hits"),
        )
        + _dashboard_section(
            "XBH — Beast model top 25",
            tbl_beast.format(
                tid=tid("tbl-beast-xbh"),
                body=_rows_beast_top25(S_xbh_b, "xbh", xbh_lu),
            ),
            csv_filename=f"matchup_beast_xbh_{slate_safe}",
            table_id=tid("tbl-beast-xbh"),
        )
        + _dashboard_section(
            "Home runs — Beast model top 25",
            tbl_beast.format(
                tid=tid("tbl-beast-hr"),
                body=_rows_beast_top25(S_hr_b, "hr", xbh_lu),
            ),
            csv_filename=f"matchup_beast_hr_{slate_safe}",
            table_id=tid("tbl-beast-hr"),
        )
        + _beast_scatter_block_html(S_hit_b, S_xbh_b, S_hr_b, tid)
    )


def _rows_residual_longshots(
    scored: list[dict],
    xbh_lookup: dict[tuple[int, int], int],
) -> str:
    """Production-only residual longshot rows; scored precomputed by residual_longshots module."""
    out: list[str] = []
    for i, d in enumerate(scored[:25], 1):
        r = d["row"]
        ap = float(r["adj_p_hr"])
        pr = float(d["prior"])
        lift = float(d["lift"])
        sp = float(d["slate_pct"])
        ob = float(d["obviousness"])
        p3 = 1.0 - (1.0 - ap) ** 3
        p4 = 1.0 - (1.0 - ap) ** 4
        yrt = d["ytd_rate"]
        yhr, ypa = d["ytd_hr"], d["ytd_pa"]
        if yrt is not None and ypa is not None and yhr is not None:
            ytd_sort = f"{yrt:.4f}"
            ytd_label = f"{yhr}/{ypa} ({yrt * 100:.1f}%)"
            ytd_cell = (
                f'{ytd_sort} <span class="proof-muted">{html.escape(ytd_label)}</span>'
            )
        else:
            ytd_cell = "—"
        tags = d["tags"]
        drv = ", ".join(tags)
        if d["beast_review"]:
            drv += " · Beast≠prod"
        drv_esc = html.escape(drv)
        cf_raw = r.get("conf_hr")
        try:
            cf_sort = f"{float(cf_raw):.4f}" if cf_raw is not None else "0"
        except (TypeError, ValueError):
            cf_sort = "0"
        mu = _matchup_line(r)
        prf = _proof_compact_html(r, "hr", xbh_lookup)
        tk_html = _bet_takeaway_html(r, "hr", xbh_lookup)
        cf_html = _confidence_cell_html(r, "hr")
        bv = _bvp_cell_detailed(r, xbh_lookup)
        out.append(
            "<tr"
            + _data_teams_attr(r, None)
            + ">"
            f'<td class="rn">{i}</td>'
            f'<td class="mu1">{mu}</td>'
            f'<td class="num">{ap * 100:.2f}%</td>'
            f'<td class="num">{p3 * 100:.1f}%</td>'
            f'<td class="num">{p4 * 100:.1f}%</td>'
            f'<td class="num">{pr * 100:.2f}%</td>'
            f'<td class="num">{lift:.3f}×</td>'
            f'<td class="num">{sp * 100:.1f}%</td>'
            f'<td class="num">{ytd_cell}</td>'
            f'<td class="num">{ob * 100:.1f}%</td>'
            f'<td class="num">{cf_sort}</td>'
            f'<td class="drivers">{drv_esc}</td>'
            f'<td class="pr1">{prf}</td>'
            f'<td class="tk1">{tk_html}</td>'
            f'<td class="cf1">{cf_html}</td>'
            f'<td class="bvp1">{bv}</td>'
            "</tr>"
        )
    return "\n".join(out)


_RESIDUAL_HISTORY_CACHE: dict | None = None


def _clear_residual_history_cache() -> None:
    """Backtest aggregates must recompute on every HTML build (not a long-lived process cache)."""
    global _RESIDUAL_HISTORY_CACHE
    _RESIDUAL_HISTORY_CACHE = None
    _clear_beast_profile_cache()


def _get_residual_history() -> dict:
    """Cached so multi-slate dashboards don't reload Statcast PA per panel."""
    global _RESIDUAL_HISTORY_CACHE
    if _RESIDUAL_HISTORY_CACHE is None:
        try:
            _RESIDUAL_HISTORY_CACHE = (
                evaluate_residual_history(
                    REPORTS / "archive",
                    RAW / "statcast_pa_level_league.parquet",
                )
                or {}
            )
        except Exception:
            _RESIDUAL_HISTORY_CACHE = {}
    return _RESIDUAL_HISTORY_CACHE


def _residual_history_html() -> str:
    h = _get_residual_history()
    if not h:
        return (
            '<p class="intro proof-muted">No archived slates with Statcast outcomes yet — '
            "evaluation panel will appear after a few past slate JSONs are available under "
            "<code>data/reports/archive/</code>.</p>"
        )
    slates = int(h.get("slates", 0))
    res_n = int(h.get("residual_rows", 0))
    band_n = int(h.get("band_rows", 0))
    ra = float(h.get("residual_tier_a_rate", 0)) * 100
    rb = float(h.get("residual_tier_b_rate", 0)) * 100
    ba = float(h.get("band_tier_a_rate", 0)) * 100
    bb = float(h.get("band_tier_b_rate", 0)) * 100
    da = ra - ba
    db = rb - bb

    def _arrow(d: float) -> str:
        if d >= 0.5:
            return "▲"
        if d <= -0.5:
            return "▼"
        return "≈"

    verdict = (
        "Residual list is currently <strong>not adding HR value</strong> over a random "
        f"in-band pick (Tier A {da:+.1f} pp, Tier B {db:+.1f} pp). "
        "Treat thresholds in <code>src/residual_longshots.py</code> as a hypothesis "
        "and tune with the calibration backtest before trusting the list."
        if (da < 0 and db < 0)
        else (
            "Residual list is mixed vs same-band baseline — keep monitoring; "
            "add more slates and tighten thresholds if Tier A doesn't lead."
            if (da < 0 or db < 0)
            else "Residual list is currently <strong>beating</strong> a same-band baseline on both tiers."
        )
    )
    return (
        '<div class="card-eval">'
        '<p class="intro"><strong>Residual screen — slate-day backtest</strong> '
        f"(<strong>{slates}</strong> archived slates, "
        f"<strong>{res_n}</strong> residual rows vs <strong>{band_n}</strong> "
        f"in-band baseline rows in <code>adj_p_hr ∈ [{ADJ_HR_MIN * 100:.1f}%, {ADJ_HR_MAX * 100:.1f}%]</code>). "
        "<strong>Tier A</strong> = HR vs the listed starter on slate date (strict). "
        "<strong>Tier B</strong> = any HR by that batter on slate date (lenient).</p>"
        '<div class="scroll"><table class="dual compare striped">'
        "<thead><tr>"
        "<th>Bucket</th><th>Rows</th><th>Tier A HR%</th><th>Tier B HR%</th>"
        "</tr></thead><tbody>"
        f'<tr><td>Residual list</td><td class="num">{res_n}</td>'
        f'<td class="num">{ra:.2f}%</td><td class="num">{rb:.2f}%</td></tr>'
        f'<tr><td>Same-band baseline</td><td class="num">{band_n}</td>'
        f'<td class="num">{ba:.2f}%</td><td class="num">{bb:.2f}%</td></tr>'
        f'<tr><td>Δ (residual − baseline)</td><td class="num">—</td>'
        f'<td class="num">{_arrow(da)} {da:+.2f} pp</td>'
        f'<td class="num">{_arrow(db)} {db:+.2f} pp</td></tr>'
        "</tbody></table></div>"
        f'<p class="intro proof-muted">{verdict}</p>'
        "</div>"
    )


def _residual_ytd_source_stamp_html() -> str:
    q = RAW / "qualifying_batters_2026.csv"
    try:
        if q.is_file():
            m = datetime.fromtimestamp(q.stat().st_mtime).strftime("%Y-%m-%d %H:%M local")
            return (
                '<p class="intro proof-muted">YTD join: <code>data/raw/qualifying_batters_2026.csv</code> '
                f"last modified <strong>{html.escape(m)}</strong> (refresh via daily run or "
                "<code>python3 src/fetch_qualifying_batters.py</code>).</p>"
            )
    except OSError:
        pass
    return (
        '<p class="intro proof-muted">YTD join: <code>data/raw/qualifying_batters_2026.csv</code> '
        "<strong>missing</strong> — residual obviousness uses defaults until you fetch qualifying batters.</p>"
    )


def _residual_tab_inner_html(
    P_list: list,
    beast_list: list | None,
    slate: str,
    xbh_lu: dict[tuple[int, int], int],
    *,
    table_id_suffix: str,
) -> str:
    """Residual longshots from production JSON + real YTD join; optional Beast divergence tag."""
    P_use = P_list or []
    stamp = _residual_ytd_source_stamp_html()
    if not P_use:
        return stamp + '<p class="intro">No production matchups for this slate.</p>'
    scored = compute_residual_longshots(P_use, beast_rows=beast_list or None)
    if not scored:
        return (
            stamp
            + '<p class="intro">No rows passed the <strong>residual longshots</strong> filters for this slate '
            "(band on <code>adj_p_hr</code>, lift vs max(league, career), slate percentile cap, "
            f"min conf, YTD from <code>data/raw/qualifying_batters_2026.csv</code>). "
            "After model retunes this is often <strong>empty</strong> until you relax thresholds in "
            "<code>src/residual_longshots.py</code> (e.g. lower <code>MIN_LIFT</code> or raise "
            "<code>MAX_SLATE_PCT</code>).</p>"
        )
    tid = (lambda base: f"{base}{table_id_suffix}") if table_id_suffix else (lambda base: base)
    slate_safe = re.sub(r"[^\w\-]+", "_", str(slate))
    intro = (
        '<p class="intro"><strong>Residual longshots</strong> (production model only, '
        "<strong>vs the listed starter only</strong>): matchups where "
        "<code>adj_p_hr</code> is meaningfully <strong>above</strong> a conservative prior "
        "<code>max(3.1%, career HR rate)</code>, while staying outside the very top of tonight’s "
        "HR projections (slate percentile) and blending in season <strong>YTD HR/PA</strong> from the "
        "qualifying batters file for obviousness. This tab does <strong>not</strong> change model weights; "
        "it re-ranks and filters the same <code>todays_matchup_predictions.json</code> rows. "
        "<strong>P(HR ≥1 in 3 PA)</strong> and <strong>P(HR ≥1 in 4 PA)</strong> are "
        "<code>1−(1−adj_p_hr)<sup>n</sup></code> — game-level proxies that ignore relievers / "
        "lineup turnover. Optional <strong>Beast≠prod</strong> in Drivers flags a large HR delta "
        "vs Beast when that JSON exists.</p>"
    )
    eval_html = _residual_history_html()
    tbl = (
        '<div class="scroll"><table class="dual compare striped sortable" id="{tid}">'
        "<thead><tr>"
        "<th>#</th>"
        "<th>Matchup</th>"
        "<th>Adj P(HR)</th><th>P(≥1 HR · 3 PA)</th><th>P(≥1 HR · 4 PA)</th>"
        "<th>Prior</th><th>Lift</th><th>Slate %</th>"
        "<th>YTD HR rate</th><th>Obvious %</th><th>Conf</th><th>Drivers</th>"
        "<th>Proof</th><th>Bet read</th><th>Confidence</th><th>BvP</th>"
        "</tr></thead><tbody>{body}</tbody></table></div>"
    )
    return (
        stamp
        + intro
        + eval_html
        + _dashboard_section(
            "Residual longshots — top 25 (production)",
            tbl.format(
                tid=tid("tbl-residual-hr"),
                body=_rows_residual_longshots(scored, xbh_lu),
            ),
            open_first=True,
            csv_filename=f"matchup_residual_hr_{slate_safe}",
            table_id=tid("tbl-residual-hr"),
        )
    )


def _main_matchup_inner_html(
    P: list,
    E_list: list,
    slate: str,
    xbh_lu: dict[tuple[int, int], int],
    *,
    table_id_suffix: str,
    R_list: list | None = None,
) -> str:
    """HTML for Matchups & BvP sections. When multiple slates are embedded, use a unique table_id_suffix (e.g. '-d20260420')."""
    tid = (lambda base: f"{base}{table_id_suffix}") if table_id_suffix else (lambda base: base)

    pair_set: set[tuple[int, int]] = set()
    for r in P:
        bid, pid = r.get("batter_mlbam_id"), r.get("pitcher_mlbam_id")
        if bid is not None and pid is not None:
            pair_set.add((int(bid), int(pid)))
    label_to_bp = _prediction_label_to_bp(P)
    barrel_outcomes = _barrel_outcome_strings(pair_set)
    laser_map, hard_ev_map = _laser_hard_bvp_by_pair(pair_set, str(slate))
    slate_esc = html.escape(str(slate))
    laser_hr_intro = (
        "<p class=\"intro\">"
        "<strong>Laser HR</strong> = Statcast home run with <code>launch_speed ≥ 100 mph</code>. "
        "Counts are career BvP in this feed (this batter vs tonight’s listed pitcher only). "
        f"<strong>On slate ({slate_esc})</strong> is the same pairing when <code>game_date</code> "
        "matches the slate (calendar day).</p>"
    )
    laser_hr_block = (
        laser_hr_intro
        + '<div class="scroll">'
        + _laser_hr_table_html(
            label_to_bp, laser_map, str(slate), table_id=tid("tbl-lasers-hr")
        )
        + "</div>"
    )
    hard_ev_intro = (
        "<p class=\"intro\">"
        "<strong>Hard contact (non-HR)</strong> = batted balls at <code>launch_speed ≥ 100 mph</code> "
        "where the outcome was not a home run (singles through loud outs). "
        "Same BvP scope and <strong>on slate</strong> definition as the laser HR table.</p>"
    )
    hard_ev_block = (
        hard_ev_intro
        + '<div class="scroll">'
        + _hard_ev_non_hr_table_html(
            label_to_bp, hard_ev_map, str(slate), table_id=tid("tbl-lasers-hard")
        )
        + "</div>"
    )

    S_hit_m1 = sorted(P, key=lambda r: float(r.get("p_hit") or r.get("adj_p_hit") or 0), reverse=True)[:25]
    S_hit_m2 = sorted(E_list, key=lambda r: float(r.get("p_hit") or r.get("adj_p_hit") or 0), reverse=True)[:25]
    S_xbh_m1 = sorted(P, key=lambda r: float(r.get("p_xbh") or r.get("adj_p_xbh") or 0), reverse=True)[:25]
    S_xbh_m2 = sorted(E_list, key=lambda r: float(r.get("p_xbh") or r.get("adj_p_xbh") or 0), reverse=True)[:25]
    S_hr_m1 = sorted(P, key=lambda r: float(r.get("p_hr") or r.get("adj_p_hr") or 0), reverse=True)[:25]
    S_hr_m2 = sorted(E_list, key=lambda r: float(r.get("p_hr") or r.get("adj_p_hr") or 0), reverse=True)[:25]

    slate_safe = re.sub(r"[^\w\-]+", "_", str(slate))

    bvp_hits_rank, bvp_xbh_rank, bvp_hr_rank = _bvp_ranked_lists(P, xbh_lu)

    tbl_compare = (
        '<div class="scroll"><table class="dual compare striped sortable" id="{tid}">'
        "<thead><tr>"
        "<th>#</th>"
        "<th>Model 1 · matchup</th><th>Model 1 · proof</th><th>Model 1 · bet read</th><th>Model 1 · confidence</th><th>Model 1 · BvP</th>"
        "<th>Model 2 · matchup</th><th>Model 2 · proof</th><th>Model 2 · bet read</th><th>Model 2 · confidence</th><th>Model 2 · BvP</th>"
        "</tr></thead><tbody>{body}</tbody></table></div>"
    )

    tbl_bvp = (
        '<div class="scroll"><table class="grid bvpgrid sortable" id="{tid}">'
        "<thead><tr>"
        "<th>#</th><th>Matchup</th><th>H</th><th>PA</th><th>AVG</th><th>XBH</th><th>HR</th>"
        "<th>Hit types (1B · 2B+3B · HR)</th><th>K</th><th>BB</th>"
        "</tr></thead><tbody>{body}</tbody></table></div>"
    )

    hit_block = tbl_compare.format(
        tid=tid("tbl-compare-hits"), body=_rows_compare_top25(S_hit_m1, S_hit_m2, "hit", xbh_lu)
    ) + _compare_models_summary_html(S_hit_m1, S_hit_m2, "hit")
    xbh_block = tbl_compare.format(
        tid=tid("tbl-compare-xbh"), body=_rows_compare_top25(S_xbh_m1, S_xbh_m2, "xbh", xbh_lu)
    ) + _compare_models_summary_html(S_xbh_m1, S_xbh_m2, "xbh")
    hr_block = tbl_compare.format(
        tid=tid("tbl-compare-hr"), body=_rows_compare_top25(S_hr_m1, S_hr_m2, "hr", xbh_lu)
    ) + _compare_models_summary_html(S_hr_m1, S_hr_m2, "hr")

    rec_embed = _recency_tab_inner_html(
        R_list or [], slate, xbh_lu, table_id_suffix=table_id_suffix, for_embedded_main=True
    )

    return f"""
{_dashboard_section("Hits — Top 25 (rank i = model 1 rank i vs model 2 rank i)", hit_block, open_first=True, csv_filename=f"matchup_compare_hits_{slate_safe}", table_id=tid("tbl-compare-hits"))}
{_dashboard_section("XBH — Top 25 (rank i = model 1 rank i vs model 2 rank i)", xbh_block, csv_filename=f"matchup_compare_xbh_{slate_safe}", table_id=tid("tbl-compare-xbh"))}
{_dashboard_section("Home runs — Top 25 (rank i = model 1 rank i vs model 2 rank i)", hr_block, csv_filename=f"matchup_compare_hr_{slate_safe}", table_id=tid("tbl-compare-hr"))}
{_dashboard_section("BvP leaderboard — Top 25 by career hits vs tonight's pitcher (raw)", tbl_bvp.format(tid=tid("tbl-bvp-hits"), body=_rows_bvp_leaderboard(bvp_hits_rank, xbh_lu)), csv_filename=f"bvp_leader_hits_{slate_safe}", table_id=tid("tbl-bvp-hits"))}
{_dashboard_section("BvP leaderboard — Top 25 by career XBH vs tonight's pitcher (raw)", tbl_bvp.format(tid=tid("tbl-bvp-xbh"), body=_rows_bvp_leaderboard(bvp_xbh_rank, xbh_lu)), csv_filename=f"bvp_leader_xbh_{slate_safe}", table_id=tid("tbl-bvp-xbh"))}
{_dashboard_section("BvP leaderboard — Top 25 by career HR vs tonight's pitcher (raw)", tbl_bvp.format(tid=tid("tbl-bvp-hr"), body=_rows_bvp_leaderboard(bvp_hr_rank, xbh_lu)), csv_filename=f"bvp_leader_hr_{slate_safe}", table_id=tid("tbl-bvp-hr"))}
{_dashboard_section("Barrels vs tonight's pitcher (career)", '<div class="scroll">' + _barrels_html(REPORTS / "section_11.md", label_to_bp, barrel_outcomes, table_id=tid("tbl-barrels")) + "</div>", csv_filename=f"barrels_vs_pitcher_{slate_safe}", table_id=tid("tbl-barrels"))}
{_dashboard_section("Laser HRs — ≥100 mph vs tonight's pitcher (career + slate day)", laser_hr_block, csv_filename=f"lasers_hr_bvp_{slate_safe}", table_id=tid("tbl-lasers-hr"))}
{_dashboard_section("≥100 mph EV, not a HR — vs tonight's pitcher (career + slate day)", hard_ev_block, csv_filename=f"lasers_hard_nonhr_bvp_{slate_safe}", table_id=tid("tbl-lasers-hard"))}
{rec_embed}
"""


# ---------------------------------------------------------------------------
# New tabs (Bucket Health / Conviction Picks / No HR Model)
# Self-contained: SVG helpers, status palette, three builder functions.
# ---------------------------------------------------------------------------

_BUCKET_STATUS_PALETTE = {
    "HEALTHY": ("#22c55e", "#0f1419"),
    "WARN": ("#fbbf24", "#0f1419"),
    "CRITICAL": ("#ef4444", "#fff"),
    "INSUFFICIENT": ("#6b7280", "#fff"),
}


def _status_badge_html(status: str) -> str:
    s = (status or "").strip().upper()
    bg, fg = _BUCKET_STATUS_PALETTE.get(s, ("#374151", "#fff"))
    return (
        f'<span class="status-pill" style="background:{bg};color:{fg};">'
        f'{html.escape(s or "—")}</span>'
    )


def _svg_sparkline(values: list[float], *, width: int = 80, height: int = 22,
                    baseline: float | None = None, color: str = "#3b82f6") -> str:
    """Inline SVG line chart for a small numeric series."""
    if not values or len(values) < 2:
        return '<span class="muted">—</span>'
    vmin = min(values)
    vmax = max(values)
    if baseline is not None:
        vmin = min(vmin, baseline)
        vmax = max(vmax, baseline)
    span = max(vmax - vmin, 1e-9)

    def _x(i):
        return 1 + (width - 2) * (i / max(len(values) - 1, 1))

    def _y(v):
        return height - 1 - (height - 2) * (v - vmin) / span

    pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(values))
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" class="spark">'
    ]
    if baseline is not None:
        by = _y(baseline)
        parts.append(
            f'<line x1="0" y1="{by:.1f}" x2="{width}" y2="{by:.1f}" '
            'stroke="#64748b" stroke-width="0.5" stroke-dasharray="2 2" />'
        )
    parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{pts}" />')
    last_x = _x(len(values) - 1)
    last_y = _y(values[-1])
    parts.append(f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="1.6" fill="{color}" />')
    parts.append("</svg>")
    return "".join(parts)


def _svg_streak_bars(daily: list[str], *, cells: int = 14, width: int = 90,
                      height: int = 22) -> str:
    """14 colored cells. Each cell encodes a day's outcome:
       'hit_power' (HR/XBH) -> amber, 'hit' -> green, 'no' -> grey, 'dnp' -> dark.
    """
    if not daily:
        return '<span class="muted">—</span>'
    daily = (daily + ["dnp"] * cells)[-cells:]
    cell_w = (width - (cells - 1)) / cells
    palette = {
        "hit_power": "#fbbf24",
        "hit": "#22c55e",
        "no": "#475569",
        "dnp": "#1f2937",
    }
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" class="streak">'
    ]
    for i, kind in enumerate(daily):
        x = i * (cell_w + 1)
        c = palette.get(kind, "#1f2937")
        parts.append(
            f'<rect x="{x:.1f}" y="2" width="{cell_w:.1f}" height="{height-4}" '
            f'rx="1.5" fill="{c}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_stacked_bar(parts_data: list[tuple[str, float, str]], *,
                      width: int = 160, height: int = 14) -> str:
    """Horizontal stacked bar. parts_data: list of (label, value, color)."""
    total = sum(max(v, 0.0) for _, v, _ in parts_data)
    if total <= 0:
        return '<span class="muted">—</span>'
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" class="lambda-stack-svg">'
    ]
    x = 0.0
    for label, value, color in parts_data:
        w = (max(value, 0.0) / total) * width
        parts.append(
            f'<rect x="{x:.2f}" y="0" width="{w:.2f}" height="{height}" '
            f'fill="{color}"><title>{html.escape(label)}: {value:.2f}</title></rect>'
        )
        x += w
    parts.append("</svg>")
    return "".join(parts)


def _drift_log_history(target: str, bucket: str, days: int = 30) -> list[float]:
    """Pull rolling realized rates for (target, bucket) from the drift log parquet."""
    p = ROOT / "data" / "tracking" / "calibration_drift_log.parquet"
    if not p.exists():
        return []
    try:
        df = pd.read_parquet(p)
    except Exception:
        return []
    if "target" not in df.columns or "bucket" not in df.columns:
        return []
    sub = df[(df["target"] == target) & (df["bucket"] == bucket)].copy()
    if sub.empty:
        return []
    if "computed_at" in sub.columns:
        sub = sub.sort_values("computed_at")
    sub = sub.tail(days)
    vals = sub["realized"].dropna().astype(float).tolist()
    return vals


def _parse_md_table_rows(md_path: Path, header_must_contain: str | None = None) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) from the first markdown table whose header contains the given substring."""
    if not md_path.exists():
        return [], []
    text = md_path.read_text()
    rows: list[list[str]] = []
    headers: list[str] = []
    capturing = False
    in_table = False
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            cells = [c.strip() for c in stripped.strip().strip("|").split("|")]
            if not in_table:
                if header_must_contain and header_must_contain.lower() not in stripped.lower():
                    continue
                headers = cells
                in_table = True
                capturing = True
                continue
            compact = [re.sub(r"\s+", "", c) for c in cells]
            if all(c and all(ch in "-:" for ch in c) and "-" in c for c in compact):
                continue
            if capturing:
                rows.append(cells)
        else:
            if in_table:
                break
    return headers, rows


def _build_bucket_health_panel() -> str:
    drift_md = REPORTS / "section_0_drift.md"
    if not drift_md.exists():
        return (
            '<main><p class="intro">Section 0 not generated yet. '
            'Run <code>python3 src/monitor_calibration_drift.py</code>.</p></main>'
        )
    full_md = drift_md.read_text()
    # Pull WARN / CRITICAL callout lines verbatim
    callouts = []
    for line in full_md.splitlines():
        if line.lstrip().startswith("> **CRITICAL"):
            callouts.append(f'<div class="callout-critical">{_inline_md(line.lstrip("> "))}</div>')
        elif line.lstrip().startswith("> **WARN"):
            callouts.append(f'<div class="callout-warn">{_inline_md(line.lstrip("> "))}</div>')
    headers, rows = _parse_md_table_rows(drift_md, header_must_contain="Target")
    if not rows:
        return f'<main><p class="intro">No drift table found in {drift_md.name}.</p></main>'
    # Find col indices
    def _idx(name):
        for i, h in enumerate(headers):
            if h.lower() == name.lower():
                return i
        return -1
    i_target = _idx("Target")
    i_bucket = _idx("Bucket")
    i_status = _idx("Status")
    i_realized = _idx("Realized")
    i_lift = _idx("Lift")

    head_row = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "<th>30d trend</th></tr>"
    body_lines = []
    for r in rows:
        cells = list(r) + ([""] * max(0, len(headers) - len(r)))
        td_html = []
        for i, c in enumerate(cells):
            if i == i_status:
                td_html.append(f"<td>{_status_badge_html(c)}</td>")
            else:
                td_html.append(f"<td>{_inline_md(c)}</td>")
        # 30d trend sparkline
        baseline_str = ""
        if i_target >= 0 and i_bucket >= 0:
            tgt = cells[i_target].lower()
            bkt = cells[i_bucket]
            history = _drift_log_history(tgt.upper() if tgt in ("hr", "hit", "xbh") else tgt, bkt, days=30)
            # drift log stores Target as 'HR' / 'Hit' / 'XBH' — try both casings
            if not history:
                history = _drift_log_history(cells[i_target], bkt, days=30)
            try:
                realized_now = float(cells[i_realized])
            except Exception:
                realized_now = None
            spark = _svg_sparkline(history, baseline=realized_now if realized_now is not None else None)
            td_html.append(f"<td>{spark}</td>")
        else:
            td_html.append("<td>—</td>")
        body_lines.append("<tr>" + "".join(td_html) + "</tr>")
    table_html = (
        '<div class="scroll"><table class="grid sortable bucket-health" id="tbl-bucket-health">'
        f"<thead>{head_row}</thead><tbody>{''.join(body_lines)}</tbody></table></div>"
    )
    intro = (
        '<p class="intro">Per-conf-bucket realized rate vs predicted on the trailing 14-day window. '
        '<strong>Lock</strong>/<strong>High</strong> rows are the betting deliverables — read STATUS first. '
        'Drift &gt;5% triggers an automatic per-bucket isotonic refit before the next slate. '
        'Sparkline shows last-30 entries from <code>data/tracking/calibration_drift_log.parquet</code>.</p>'
    )
    callout_html = "".join(callouts)
    return f'<main>{intro}{callout_html}{table_html}</main>'


def _park_lookup_key_mlb(home_team: str) -> str:
    """Map MLB API / slate home abbrev to row key in data/raw/park_lookup.csv."""
    u = str(home_team or "").strip().upper()
    if u == "AZ":
        return "ARI"
    if u == "ATH":
        return "OAK"
    return u


def _park_venue_name(home_team: str) -> str:
    key = _park_lookup_key_mlb(home_team)
    path = RAW / "park_lookup.csv"
    if not path.is_file():
        return str(home_team or "?")
    try:
        df = pd.read_csv(path)
        sub = df[df["home_team"].astype(str).str.upper() == key]
        if sub.empty:
            return f"{home_team} (park?)"
        return str(sub.iloc[0].get("venue_name") or home_team)
    except Exception:
        return str(home_team or "?")


# IANA tz for first-pitch local time (home ballpark). ATH uses west-coast A's home.
_TZ_FOR_PARK_HOME: dict[str, str] = {
    "ARI": "America/Phoenix",
    "ATL": "America/New_York",
    "BAL": "America/New_York",
    "BOS": "America/New_York",
    "CHC": "America/Chicago",
    "CIN": "America/New_York",
    "CLE": "America/New_York",
    "CWS": "America/Chicago",
    "COL": "America/Denver",
    "DET": "America/New_York",
    "HOU": "America/Chicago",
    "KC": "America/Chicago",
    "LAA": "America/Los_Angeles",
    "LAD": "America/Los_Angeles",
    "MIA": "America/New_York",
    "MIL": "America/Chicago",
    "MIN": "America/Chicago",
    "NYM": "America/New_York",
    "NYY": "America/New_York",
    "OAK": "America/Los_Angeles",
    "PHI": "America/New_York",
    "PIT": "America/New_York",
    "SD": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "SF": "America/Los_Angeles",
    "STL": "America/Chicago",
    "TB": "America/New_York",
    "TEX": "America/Chicago",
    "TOR": "America/Toronto",
    "WSH": "America/New_York",
    "ATH": "America/Los_Angeles",
    "AZ": "America/Phoenix",
}


def _wind_from_cardinal(deg: object) -> str:
    if deg is None:
        return "—"
    try:
        d = float(deg) % 360.0
    except (TypeError, ValueError):
        return "—"
    labels = (
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    )
    idx = int((d + 11.25) / 22.5) % 16
    return f"{d:.0f}° ({labels[idx]}) — wind from {labels[idx]}"


def _wind_vs_stadium_sentence(orientation: str, indoor: bool) -> str:
    if indoor:
        return "Under roof — model treats weather as neutral for carry (no outdoor wind)."
    o = str(orientation or "").strip().lower()
    if o == "out_to_cf":
        return "Component toward center field: helps fly-ball carry vs that park’s CF axis."
    if o == "in_from_cf":
        return "Component from center field: suppresses carry vs that park’s CF axis."
    if o == "cross":
        return "Mostly cross-field relative to CF: small carry signal vs dead-center."
    if o == "calm":
        return "Calm / light air: wind is a minor factor; temperature dominates among wx terms."
    if o == "indoor":
        return "Indoor play — no outdoor wind effect on the model multiplier."
    if not o:
        return "Orientation not classified — see raw wind from direction."
    return o.replace("_", " ").title() + " (see park CF bearing in priors)."


def _hr_weather_label(mult: object, indoor: bool) -> tuple[str, str]:
    if indoor:
        return ("Neutral", "wx-hr-neutral")
    try:
        m = float(mult)
    except (TypeError, ValueError):
        return ("Neutral", "wx-hr-neutral")
    if m > 1.02:
        return ("Up", "wx-hr-up")
    if m < 0.98:
        return ("Down", "wx-hr-down")
    return ("Neutral", "wx-hr-neutral")


def _wx_notes_cell(g: dict) -> str:
    parts: list[str] = []
    if g.get("weather_indoor"):
        parts.append("Indoor / roof-neutral multiplier")
    rp = g.get("weather_precip_prob_pct")
    if rp is not None and float(rp) >= 15:
        parts.append(f"Precip chance **{float(rp):.0f}%** (hourly model)")
    rs = g.get("roof_state_today")
    if rs is not None and str(rs).strip():
        parts.append(f"Roof/reported: {rs}")
    src = g.get("weather_source")
    if src:
        parts.append(f"Source: {src}")
    fa = g.get("weather_fetched_at")
    if fa:
        parts.append(f"Fetched: {fa}")
    if not parts:
        parts.append("—")
    return " · ".join(parts)


def _wx_expected_hr_game_total(g: dict) -> str:
    """Expected total HRs for the game from Section 15 payload (same as E[HR] in no-HR model)."""
    v = g.get("expected_hr")
    if v is None:
        v = g.get("lambda_total_adj")
    if v is None:
        v = g.get("lambda_total_raw")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.2f}"


def _wx_scene_description(g: dict) -> str:
    """Short lay summary for the Scene column (plain text; caller may run _inline_md)."""
    matchup = str(g.get("matchup") or "")
    temp = g.get("weather_temp_f")
    wmph = g.get("weather_wind_mph")
    indoor = bool(g.get("weather_indoor"))
    orient = str(g.get("weather_wind_orientation") or "")
    mult = g.get("weather_hr_mult")
    try:
        m = float(mult)
    except (TypeError, ValueError):
        m = 1.0
    bits: list[str] = []
    if indoor:
        bits.append("Game is under roof; no outdoor carry wind.")
    else:
        if temp is not None:
            bits.append(f"**{float(temp):.0f}°F** at the sampled first-pitch hour")
        if wmph is not None:
            bits.append(f"**{float(wmph):.0f} mph** sustained wind")
        if orient:
            bits.append(
                f"**{orient.replace('_', ' ')}** vs this stadium’s center-field line (projected onto CF bearing)"
            )
    if m > 1.02:
        bits.append("Combined temp×wind **lifts** HR carry vs a mild 70°F / calm baseline.")
    elif m < 0.98:
        bits.append("Combined temp×wind **dampens** HR carry vs baseline.")
    else:
        bits.append("Combined temp×wind is **near neutral** for HR carry.")
    if matchup:
        bits.append(f"({matchup})")
    return " ".join(bits)


def _zero_hr_predictions_path(slate_date: str) -> Path:
    """Prefer archived zero-HR JSON for that calendar slate so weather matches archive slates."""
    sd = str(slate_date).strip()[:10]
    arch = REPORTS / "archive" / sd / "todays_zero_hr_predictions.json"
    if arch.is_file():
        return arch
    return REPORTS / "todays_zero_hr_predictions.json"


def _build_slate_weather_overview_html(slate_date: str, *, table_id: str = "tbl-slate-weather") -> str:
    """Top-of-dashboard table: stadium, first pitch, wx, wind vs CF, HR impact, notes."""
    zpath = _zero_hr_predictions_path(slate_date)
    mpath = RAW / "todays_matchups.json"
    if not zpath.is_file():
        return (
            '<section class="slate-weather-overview" aria-label="Slate weather and first pitch">'
            "<h2>Slate weather &amp; first pitch</h2>"
            '<p class="intro muted slate-wx-intro">No <code>todays_zero_hr_predictions.json</code> yet — run '
            "<code>python3 src/gen_zero_hr_predictions.py</code> (usually via the daily pipeline) "
            "to populate stadium wx for this slate.</p>"
            "</section>"
        )
    try:
        payload = json.loads(zpath.read_text())
    except Exception as ex:
        return (
            '<section class="slate-weather-overview" aria-label="Slate weather and first pitch">'
            "<h2>Slate weather &amp; first pitch</h2>"
            f"<p class=\"intro muted slate-wx-intro\">Could not read wx JSON: {html.escape(str(ex))}</p>"
            "</section>"
        )
    games = payload.get("games") or []
    if not games:
        return (
            '<section class="slate-weather-overview" aria-label="Slate weather and first pitch">'
            "<h2>Slate weather &amp; first pitch</h2>"
            '<p class="intro muted slate-wx-intro">Zero-HR JSON has no games for this build.</p>'
            "</section>"
        )

    sd = str(slate_date).strip()[:10]
    by_pk: dict[int, dict] = {}
    if mpath.is_file():
        try:
            raw_games = json.loads(mpath.read_text())
            if isinstance(raw_games, list):
                for row in raw_games:
                    if not isinstance(row, dict):
                        continue
                    gdt = str(row.get("game_date") or "")[:10]
                    if sd and gdt and gdt != sd:
                        continue
                    try:
                        pk = int(row["game_pk"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    by_pk[pk] = row
        except Exception:
            by_pk = {}

    def sort_key(g: dict) -> str:
        pk = int(g.get("game_pk") or 0)
        row = by_pk.get(pk) or {}
        return str(row.get("game_datetime_utc") or g.get("weather_forecast_time_utc") or "")

    games_sorted = sorted(games, key=sort_key)
    pks_on_slate = set(by_pk.keys())
    if pks_on_slate:
        games_sorted = [g for g in games_sorted if int(g.get("game_pk") or 0) in pks_on_slate]

    head = (
        "<tr>"
        "<th>Matchup</th>"
        "<th>Stadium</th>"
        "<th>First pitch (local)</th>"
        "<th>Temp</th>"
        "<th>Wind</th>"
        "<th>Wind from (met)</th>"
        "<th>Vs stadium (CF)</th>"
        "<th>HR weather</th>"
        '<th title="Section 15: expected total HRs in the game (park × weather-adjusted λ stack; same as E[HR] in the No HR model).">'
        "Pred HRs<br/><span class='muted' style='font-weight:500'>(game total)</span></th>"
        "<th>Notes</th>"
        "<th>Scene</th>"
        "</tr>"
    )
    rows: list[str] = []
    for g in games_sorted:
        home = str(g.get("home_team") or "")
        venue = _park_venue_name(home)
        matchup = str(g.get("matchup") or "?")
        pk = int(g.get("game_pk") or 0)
        mrow = by_pk.get(pk) or {}
        iso = mrow.get("game_datetime_utc") or g.get("weather_forecast_time_utc")
        pk_key = _park_lookup_key_mlb(home)
        tz_name = _TZ_FOR_PARK_HOME.get(home) or _TZ_FOR_PARK_HOME.get(pk_key) or "UTC"
        time_local = "—"
        if iso:
            try:
                raw = str(iso).replace("Z", "+00:00")
                dtu = datetime.fromisoformat(raw)
                if dtu.tzinfo is None:
                    dtu = dtu.replace(tzinfo=ZoneInfo("UTC"))
                loc = dtu.astimezone(ZoneInfo(tz_name))
                h12 = loc.hour % 12 or 12
                tz_disp = loc.tzname() or tz_name
                time_local = (
                    f"{loc.strftime('%a %b')} {loc.day}, {loc.year} "
                    f"{h12}:{loc.minute:02d} {loc.strftime('%p')} {tz_disp}"
                )
            except Exception:
                time_local = str(iso)[:19].replace("T", " ") + " UTC"

        temp = g.get("weather_temp_f")
        temp_s = f"{float(temp):.0f}°F" if temp is not None else "—"
        wmph = g.get("weather_wind_mph")
        indoor = bool(g.get("weather_indoor"))
        if indoor:
            wind_s = "— (indoor)"
        elif wmph is not None:
            wind_s = f"{float(wmph):.0f} mph"
        else:
            wind_s = "—"

        wdeg = g.get("weather_wind_dir_deg")
        wind_from = "—" if indoor else _wind_from_cardinal(wdeg)

        orient = str(g.get("weather_wind_orientation") or "")
        vs_stad = _wind_vs_stadium_sentence(orient, indoor)
        vs_stad_html = html.escape(vs_stad)

        mult = g.get("weather_hr_mult")
        label, cls = _hr_weather_label(mult, indoor)

        ehr = _wx_expected_hr_game_total(g)
        ehr_cell = html.escape(ehr)
        if ehr != "—":
            try:
                ehr_sort = f"{float(ehr):.6f}"
            except ValueError:
                ehr_sort = "0"
        else:
            ehr_sort = "0"

        notes_raw = _wx_notes_cell(g)
        notes_html = _inline_md(notes_raw)

        scene = _wx_scene_description(g)
        scene_html = _inline_md(scene)

        rows.append(
            "<tr>"
            f"<td>{html.escape(matchup)}</td>"
            f"<td>{html.escape(venue)} <span class='muted'>({html.escape(home)})</span></td>"
            f"<td>{html.escape(time_local)}</td>"
            f"<td>{html.escape(temp_s)}</td>"
            f"<td>{html.escape(wind_s)}</td>"
            f"<td>{html.escape(wind_from)}</td>"
            f"<td class='wx-vs-stad'>{vs_stad_html}</td>"
            f"<td class='{cls}'><strong>{html.escape(label)}</strong></td>"
            f"<td class='num wx-ehr' data-sort-value='{html.escape(ehr_sort, quote=True)}'>{ehr_cell}</td>"
            f"<td class='wx-notes'>{notes_html}</td>"
            f"<td class='wx-desc'>{scene_html}</td>"
            "</tr>"
        )

    if not rows:
        rows.append(
            "<tr><td colspan=\"11\" class=\"muted\">No weather rows for this slate: no "
            "<code>game_pk</code> in the loaded zero-HR file matched "
            f"<code>todays_matchups.json</code> with <code>game_date</code> "
            f"<strong>{html.escape(sd)}</strong>. Regenerate matchups / Section 15 for this date.</td></tr>"
        )

    intro = (
        '<p class="intro slate-wx-intro">Open-Meteo at each park for the <strong>first-pitch hour</strong>. '
        "<strong>Wind from</strong> uses meteorological degrees (direction the wind blows <em>from</em>). "
        "<strong>Vs stadium (CF)</strong> is wind projected onto this park’s center-field bearing (same signal as Section 15 <code>weather_wind_orientation</code>). "
        "<strong>HR weather</strong>: <span class='wx-hr-up'>Up</span> = carry multiplier &gt;1.02; "
        "<span class='wx-hr-down'>Down</span> = &lt;0.98; else neutral. "
        "<strong>Pred HRs (game total)</strong> = Section 15 expected homers in that full game "
        "(<code>expected_hr</code> / λ stack, same as the No HR model’s <strong>E[HR]</strong>).</p>"
    )
    tid = html.escape(table_id, quote=True)
    table = (
        '<div class="scroll slate-wx-scroll">'
        f'<table class="grid sortable slate-wx-table" id="{tid}">'
        f"<thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    return (
        '<section class="slate-weather-overview" aria-label="Slate weather and first pitch">'
        "<h2>Slate weather &amp; first pitch</h2>"
        f"{intro}{table}"
        "</section>"
    )


def _build_no_hr_panel() -> str:
    json_path = REPORTS / "todays_zero_hr_predictions.json"
    if not json_path.exists():
        return (
            '<main><p class="intro">Section 15 not generated yet. '
            'Run <code>python3 src/gen_zero_hr_predictions.py</code>.</p></main>'
        )
    try:
        payload = json.loads(json_path.read_text())
    except Exception as ex:
        return f'<main><p class="intro">Could not parse {json_path.name}: {html.escape(str(ex))}</p></main>'
    games = payload.get("games", [])
    meta = payload.get("_meta", {})
    if not games:
        return '<main><p class="intro">No games scored for the No HR Model on this slate.</p></main>'

    games_sorted = sorted(games, key=lambda g: -(g.get("p_zero_hr") or 0.0))
    head_row = (
        "<tr><th>Rank</th><th>Matchup</th><th>Park</th><th>Enc.</th><th>PF</th>"
        "<th>Temp</th><th>Wind</th><th>Wx Mult</th>"
        "<th>λ_SP_a</th><th>λ_SP_h</th><th>λ_pen_a</th><th>λ_pen_h</th>"
        "<th>Pen A</th><th>Pen H</th>"
        "<th>λ decomposition</th><th>λ_total</th><th>E[HR]</th>"
        "<th>P(0 HR)</th><th>Away SP (BF̂)</th><th>Home SP (BF̂)</th></tr>"
    )
    rows_html = []
    for i, g in enumerate(games_sorted, 1):
        sp_a = g.get("lambda_sp_away", 0.0) or 0.0
        sp_h = g.get("lambda_sp_home", 0.0) or 0.0
        pen_a = g.get("lambda_pen_away", 0.0) or 0.0
        pen_h = g.get("lambda_pen_home", 0.0) or 0.0
        stack = _svg_stacked_bar([
            ("Away SP", sp_a, "#3b82f6"),
            ("Home SP", sp_h, "#1d4ed8"),
            ("Away pen", pen_a, "#fbbf24"),
            ("Home pen", pen_h, "#f59e0b"),
        ])
        p0 = g.get("p_zero_hr", 0.0) or 0.0
        # Color the P(0 HR) cell against the league baseline (~12%)
        baseline_p0 = 0.12
        cls = "p0-good" if p0 > baseline_p0 + 0.03 else ("p0-bad" if p0 < baseline_p0 - 0.03 else "")
        away_cell = f"{html.escape(g.get('away_pitcher','?'))} ({g.get('bf_hat_away',0):.0f})"
        home_cell = f"{html.escape(g.get('home_pitcher','?'))} ({g.get('bf_hat_home',0):.0f})"
        enc = g.get("park_enclosure", "outdoor")

        # Weather cells (Phase 1)
        temp = g.get("weather_temp_f")
        temp_str = f"{temp:.0f}&deg;F" if temp is not None else "&mdash;"
        wind_mph = g.get("weather_wind_mph")
        wind_orient = g.get("weather_wind_orientation") or ""
        if wind_mph is None:
            wind_str = "&mdash;"
        elif g.get("weather_indoor"):
            wind_str = '<span class="muted">indoor</span>'
        else:
            orient_disp = wind_orient.replace("_", " ")
            wind_str = f"{wind_mph:.0f} mph<br/><small class='muted'>{html.escape(orient_disp)}</small>"
        wx_mult = float(g.get("weather_hr_mult", 1.0) or 1.0)
        if wx_mult > 1.03:
            wx_cls = "wx-helping"  # HR-friendly weather (lowers P(0 HR))
        elif wx_mult < 0.97:
            wx_cls = "wx-suppressing"  # HR-suppressing weather (raises P(0 HR))
        else:
            wx_cls = ""

        # Per-team bullpen rate cells (Phase: Section 15 v2)
        pen_rate_a = float(g.get("pen_rate_away") or 0.0)
        pen_rate_h = float(g.get("pen_rate_home") or 0.0)
        league_pen = 0.0272  # league bullpen HR/PA from priors
        pen_a_cls = ("pen-leaky" if pen_rate_a > league_pen + 0.0015 else
                     "pen-suppressing" if pen_rate_a < league_pen - 0.0015 else "")
        pen_h_cls = ("pen-leaky" if pen_rate_h > league_pen + 0.0015 else
                     "pen-suppressing" if pen_rate_h < league_pen - 0.0015 else "")

        rows_html.append(
            f"<tr><td>{i}</td><td>{html.escape(g.get('matchup','?'))}</td>"
            f"<td>{html.escape(g.get('home_team','?'))}</td>"
            f"<td>{html.escape(str(enc)[:3])}</td>"
            f"<td>{g.get('park_pf_hr',1.0):.2f}</td>"
            f"<td>{temp_str}</td>"
            f"<td>{wind_str}</td>"
            f"<td class=\"{wx_cls}\">{wx_mult:.3f}</td>"
            f"<td>{sp_a:.2f}</td><td>{sp_h:.2f}</td><td>{pen_a:.2f}</td><td>{pen_h:.2f}</td>"
            f"<td class=\"{pen_a_cls}\">{pen_rate_a*100:.2f}%</td>"
            f"<td class=\"{pen_h_cls}\">{pen_rate_h*100:.2f}%</td>"
            f"<td>{stack}</td>"
            f"<td>{g.get('lambda_total_adj',0.0):.2f}</td>"
            f"<td>{g.get('expected_hr',0.0):.2f}</td>"
            f"<td class=\"{cls}\"><strong>{p0*100:.1f}%</strong></td>"
            f"<td>{away_cell}</td><td>{home_cell}</td></tr>"
        )
    legend = (
        '<div class="lambda-legend">'
        '<span><i style="background:#3b82f6"></i>Away SP</span>'
        '<span><i style="background:#1d4ed8"></i>Home SP</span>'
        '<span><i style="background:#fbbf24"></i>Away pen</span>'
        '<span><i style="background:#f59e0b"></i>Home pen</span>'
        '</div>'
    )
    intro = (
        '<p class="intro">Per-game P(0 HR), sorted high to low. '
        f'<strong>NB dispersion k = {meta.get("k_nb", "?")}</strong>; '
        f'league HR/PA = {meta.get("league_hr_pa", 0.031):.4f}; '
        f'mean PA/team/game = {meta.get("team_pa_per_game", 38.0):.0f}. '
        'Park PF from <code>data/priors/park_hr_factors.json</code> (two-sided, FanGraphs-style). '
        'BF̂ for each starter conditioned on Section 9 projected runs. '
        '<strong>Pen A / Pen H</strong>: per-team bullpen HR/PA blend from <code>data/priors/team_bullpen_hr.json</code> '
        '(Beta-Binomial EB shrunk; league avg 2.72%; CLE 2.44% to LAA 3.09%). '
        '<strong>Weather</strong>: live Open-Meteo forecast at stadium for first-pitch hour. Wx Mult > 1 means HR-friendly conditions (lowers P(0 HR)); &lt; 1 means HR-suppressing (raises P(0 HR)). Indoor games (fixed dome OR retractable+closed) get mult 1.0. '
        'League baseline P(0 HR) ≈ 12% — green = above baseline (good zero-HR target), red = below.</p>'
    )
    table_html = (
        '<div class="scroll"><table class="grid sortable no-hr-grid" id="tbl-no-hr">'
        f"<thead>{head_row}</thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    return f"<main>{intro}{legend}{table_html}</main>"


def _conviction_streak_for_batter(bid: int, days: int = 14) -> list[str]:
    """Per-day outcome label for the batter's last `days` calendar days from PA-level league parquet."""
    pa_path = ROOT / "data" / "raw" / "statcast_pa_level_league.parquet"
    if not pa_path.exists():
        return []
    try:
        df = pd.read_parquet(pa_path, columns=["game_date", "batter", "events"])
    except Exception:
        return []
    df = df[df["batter"].astype("Int64") == int(bid)].copy()
    if df.empty:
        return []
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["game_date"])
    if df.empty:
        return []
    max_d = df["game_date"].max()
    cutoff = max_d - pd.Timedelta(days=days - 1)
    grid = pd.date_range(cutoff, max_d, freq="D")
    df["events"] = df["events"].fillna("").str.lower()
    df["is_hit"] = df["events"].isin({"single", "double", "triple", "home_run"}).astype(int)
    df["is_power"] = df["events"].isin({"double", "triple", "home_run"}).astype(int)
    by_day = df.groupby("game_date").agg(hit=("is_hit", "max"), power=("is_power", "max"))
    out = []
    for d in grid:
        if d not in by_day.index:
            out.append("dnp")
        else:
            row = by_day.loc[d]
            if int(row["power"]) > 0:
                out.append("hit_power")
            elif int(row["hit"]) > 0:
                out.append("hit")
            else:
                out.append("no")
    return out


def _build_conviction_panel() -> str:
    md_path = REPORTS / "section_14_conviction_picks.md"
    pred_path = REPORTS / "todays_matchup_predictions.json"
    if not pred_path.exists():
        return (
            '<main><p class="intro">No predictions JSON available. '
            'Run <code>python3 src/build_matchup_dashboard.py</code> first.</p></main>'
        )
    try:
        preds = json.loads(pred_path.read_text())
    except Exception as ex:
        return f'<main><p class="intro">Could not parse predictions JSON: {html.escape(str(ex))}</p></main>'

    # Per target: Section 14 parity — M1 Lock ∩ hand conf Medium+, else legacy High; cap rows.
    has_meta = any(p.get("score_label_hr") for p in preds)

    sections_html = []
    label_sources_used = []
    for tgt_short, tgt_disp, p_field in [("hr", "HR", "p_hr"),
                                            ("hit", "Hit", "p_hit"),
                                            ("xbh", "XBH", "p_xbh")]:
        gate_ok = (
            lambda p, ts=tgt_short: (p.get(f"conf_{ts}_label") or "")
            in SEC14_MIN_HAND_LABEL
        )
        m1_lock = (
            [p for p in preds if p.get(f"score_label_{tgt_short}") == "Lock"]
            if has_meta
            else []
        )
        m1_dual = [p for p in m1_lock if gate_ok(p)]
        legacy_picks = [p for p in preds if (p.get(f"conf_{tgt_short}_label") or "") == "High"]
        if m1_dual:
            picks = m1_dual
            tgt_source = "M1 Lock + hand conf ≥ Medium"
        elif legacy_picks:
            picks = legacy_picks
            if has_meta and m1_lock:
                tgt_source = "hand-tuned High (M1 Lock failed dual gate)"
            else:
                tgt_source = "hand-tuned High (M1 yielded 0 Lock picks for this target)"
        else:
            picks = []
            tgt_source = (
                "M1 Lock gated out (no dual-gate rows)"
                if has_meta and m1_lock
                else ("M1 yielded 0 Lock" if has_meta else "legacy High empty")
            )
        label_sources_used.append(f"{tgt_disp}: {tgt_source}")
        # Sort by score (raw P × conf)
        picks.sort(key=lambda x: -(x.get(f"score_{tgt_short}") or x.get(f"adj_p_{tgt_short}") or 0.0))
        picks = picks[:SEC14_MAX_PICKS_PER_TARGET]

        head_row = (
            "<tr><th>#</th><th>Matchup</th><th>Score</th>"
            f"<th>Cal P({tgt_disp})</th><th>Raw P({tgt_disp})</th><th>3-PA</th>"
            "<th>BvP</th><th>14-day streak</th></tr>"
        )
        if not picks:
            sections_html.append(
                f'<h3 class="conv-h">{tgt_disp} Locks (n=0)</h3>'
                f'<p class="muted">No {tgt_disp} Lock picks for this slate.</p>'
            )
            continue
        body_rows = []
        for i, p in enumerate(picks, 1):
            score = p.get(f"score_{tgt_short}") or p.get(f"adj_p_{tgt_short}") or 0.0
            cal_p = p.get(f"p_{tgt_short}_calibrated") or p.get(p_field) or 0.0
            raw_p = p.get(p_field) or 0.0
            three_pa = (1 - (1 - raw_p) ** 3) * 100
            hand = _pitcher_hand_abbr(p)
            hand_esc = html.escape(hand)
            pitcher_mid = html.escape(p.get("pitcher_name", "?"))
            team_esc = html.escape(p.get("pitcher_team", "?"))
            if hand:
                pitcher_cell = f"{pitcher_mid} ({hand_esc}) ({team_esc})"
            else:
                pitcher_cell = f"{pitcher_mid} ({team_esc})"
            matchup = (
                f"{html.escape(p.get('batter_name','?'))} ({html.escape(p.get('batter_team','?'))}) "
                f"vs {pitcher_cell}"
            )
            bvp_text = p.get("bvp_text") or "—"
            bid = int(p.get("batter_mlbam_id") or 0)
            streak = _svg_streak_bars(_conviction_streak_for_batter(bid)) if bid else '<span class="muted">—</span>'
            body_rows.append(
                f"<tr><td>{i}</td><td>{matchup}</td>"
                f"<td><strong>{score*100:.2f}</strong></td>"
                f"<td>{cal_p*100:.1f}%</td><td>{raw_p*100:.1f}%</td>"
                f"<td>{three_pa:.1f}%</td>"
                f"<td>{html.escape(bvp_text)}</td>"
                f"<td>{streak}</td></tr>"
            )
        table_html = (
            f'<div class="scroll"><table class="grid sortable conv-grid" id="tbl-conv-{tgt_short}">'
            f"<thead>{head_row}</thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        )
        sections_html.append(
            f'<h3 class="conv-h">{tgt_disp} Locks (n={len(picks)})</h3>{table_html}'
        )

    streak_legend = (
        '<div class="streak-legend">'
        '<strong>14-day streak legend:</strong> '
        '<span><i style="background:#fbbf24"></i>HR/XBH</span>'
        '<span><i style="background:#22c55e"></i>Hit only</span>'
        '<span><i style="background:#475569"></i>No outcome</span>'
        '<span><i style="background:#1f2937"></i>Did not play</span>'
        '</div>'
    )
    sources_html = "<br/>".join(html.escape(s) for s in label_sources_used)
    intro = (
        '<p class="intro">High-conviction picks tonight. '
        "<strong>Score</strong> = raw P × evidence multiplier (<em>ranking number</em>, not a probability). "
        '<strong>Cal P</strong> = isotonic-calibrated probability; calibration buckets prefer quartiles '
        'of numeric <code>conf_*</code> when <code>calibration_isotonic.json</code> stores cutpoints. '
        'Display tier (<code>conf_*_label</code>) stays a heuristic band for humans and drift Section 0. '
        'Read Section 0 (Bucket Health) first to verify the conviction system is healthy on the rolling window.<br/>'
        f'<small class="muted">Label source per target: {sources_html}</small></p>'
    )
    return f'<main>{intro}{streak_legend}{"".join(sections_html)}</main>'


def build_dashboard_html(slate: str | None = None) -> str:
    _clear_residual_history_cache()
    _reset_lineup_lookup_dashboard_cache()
    bundles = _gather_slate_prediction_bundles()
    multi = len(bundles) > 1
    primary = bundles[0]
    P = primary["P"]
    E_list = primary["E"]
    slate_primary = str(primary["slate"])
    want = (slate or "").strip()[:10] if slate else ""
    default_sd = want if want and any(str(b["slate"]) == want for b in bundles) else slate_primary

    xbh_lu = _xbh_lookup()
    postgame_payload = _build_postgame_interactive_payload(xbh_lu, primary_slate=default_sd)

    S_hit_m1 = sorted(P, key=lambda r: float(r.get("p_hit") or r.get("adj_p_hit") or 0), reverse=True)[:25]
    S_hit_m2 = sorted(E_list, key=lambda r: float(r.get("p_hit") or r.get("adj_p_hit") or 0), reverse=True)[:25]
    S_xbh_m1 = sorted(P, key=lambda r: float(r.get("p_xbh") or r.get("adj_p_xbh") or 0), reverse=True)[:25]
    S_xbh_m2 = sorted(E_list, key=lambda r: float(r.get("p_xbh") or r.get("adj_p_xbh") or 0), reverse=True)[:25]
    S_hr_m1 = sorted(P, key=lambda r: float(r.get("p_hr") or r.get("adj_p_hr") or 0), reverse=True)[:25]
    S_hr_m2 = sorted(E_list, key=lambda r: float(r.get("p_hr") or r.get("adj_p_hr") or 0), reverse=True)[:25]

    slate_safe_primary = re.sub(r"[^\w\-]+", "_", str(slate_primary))

    track_map = _tracking_outcomes_map(str(slate_primary))
    bat_day, bat_warn = _batter_day_outcome_stats(str(slate_primary))
    resolve_pg = _outcome_resolver_factory(str(slate_primary), track_map, bat_day)

    tbl_postgame = (
        '<div class="scroll"><table class="dual compare postgame striped sortable" id="{tid}">'
        "<thead><tr>"
        "<th>#</th>"
        "<th>Model 1 · matchup</th><th>Model 1 · proof</th><th>Model 1 · bet read</th><th>Model 1 · confidence</th>"
        "<th>Result</th><th>Actual H·XBH·HR·PA</th><th>Model 1 · BvP</th>"
        "<th>Model 2 · matchup</th><th>Model 2 · proof</th><th>Model 2 · bet read</th><th>Model 2 · confidence</th>"
        "<th>Result</th><th>Actual H·XBH·HR·PA</th><th>Model 2 · BvP</th>"
        "</tr></thead><tbody>{body}</tbody></table></div>"
    )

    intro_pg: list[str] = [
        "Rank <em>i</em> matches the pre-game top-25 tables. <strong>✓</strong> means the batter met the row’s target outcome "
        "on the slate calendar day (see methodology note under each summary). "
        "<strong>Actual</strong> columns show same-day Statcast H, XBH, HR, and PA totals."
    ]
    if bat_warn:
        intro_pg.append(f"<strong>Statcast:</strong> {html.escape(bat_warn)}")
    if track_map:
        intro_pg.append(
            f"Using <strong>{len(track_map)}</strong> filled row(s) from <code>data/tracking/matchup_predictions_runs.parquet</code> "
            "for exact <code>game_pk</code> matchups where available; other rows use the calendar-day batter rollup."
        )
    postgame_intro = '<p class="intro postgame-intro">' + " ".join(intro_pg) + "</p>"

    if postgame_payload:
        pg_json = json.dumps(postgame_payload, ensure_ascii=False).replace("<", "\\u003c")
        topn_opts = "".join(f'<option value="{n}">{n}</option>' for n in (1, 3, 5, 10, 15, 20, 25, 50))
        ix_intro = [
            "Rows list <strong>every saved slate</strong> in your tracking exports (dual predictions + outcomes). "
            "<strong>Start date</strong> and <strong>End date</strong> default to the <strong>calendar day before the primary matchup slate</strong> "
            "when that day has postgame rows (so you see last night’s actuals while Matchups shows tonight’s slate); "
            "else the primary slate if it has rows; else <strong>calendar yesterday</strong>, "
            "each clamped to the min/max calendar range. "
            "For each day, predictions come from tracking Parquet when available; otherwise from archived (or current) matchup JSON. "
            "Actuals: <code>matchup_predictions_runs.parquet</code> fills when present for that <code>game_pk</code> matchup; "
            "otherwise same-day Statcast batter totals from <code>statcast_pa_level_league.parquet</code> (keep that file updated through the prior calendar day when rebuilding this HTML).",
            "Rank <em>i</em> is the pre-game sort for that slate. <strong>✓</strong> / <strong>✗</strong> use calendar-day batter totals "
            "(and <code>game_pk</code> fills from runs where present). <strong>Top N</strong> limits which ranks appear and what the KPI cards count.",
        ]
        if bat_warn:
            ix_intro.append(f"<strong>Statcast:</strong> {html.escape(bat_warn)}")
        ix_intro.append(
            "Data merge: <code>matchup_dual_model_predictions.parquet</code> plus outcomes from "
            "<code>matchup_predictions_runs.parquet</code>; slates that exist only in runs use model 1 rows with model 2 blank."
        )
        postgame_intro_ix = '<p class="intro postgame-intro">' + " ".join(ix_intro) + "</p>"
        postgame_toolbar = (
            '<div class="postgame-toolbar">'
            '<p class="pg-meta" id="pg-meta-days"></p>'
            '<div class="pg-date-row">'
            '<label class="pg-date-lab">Start date '
            '<input type="date" id="pg-date-start" autocomplete="off" />'
            "</label>"
            '<label class="pg-date-lab">End date '
            '<input type="date" id="pg-date-end" autocomplete="off" />'
            "</label>"
            '<label class="pg-topn-lab">Top N per slate<br/>'
            f'<select id="pg-topn">{topn_opts}</select>'
            "</label>"
            "</div></div>"
            f'<script type="application/json" id="postgame-app-data">{pg_json}</script>'
        )
        pg_hit = _postgame_interactive_shell("hit", "tbl-post-hit", "pg-aside-hit", "Hits — filtered summary")
        pg_xbh = _postgame_interactive_shell("xbh", "tbl-post-xbh", "pg-aside-xbh", "XBH — filtered summary")
        pg_hr = _postgame_interactive_shell("hr", "tbl-post-hr", "pg-aside-hr", "Home runs — filtered summary")
        postgame_inner = f"""
{postgame_intro_ix}
{postgame_toolbar}
{_dashboard_section("Hits — vs actuals (saved slates)", pg_hit, open_first=True, csv_filename=f"postgame_hits_{slate_safe_primary}", table_id="tbl-post-hit")}
{_dashboard_section("XBH — vs actuals (saved slates)", pg_xbh, csv_filename=f"postgame_xbh_{slate_safe_primary}", table_id="tbl-post-xbh")}
{_dashboard_section("Home runs — vs actuals (saved slates)", pg_hr, csv_filename=f"postgame_hr_{slate_safe_primary}", table_id="tbl-post-hr")}
"""
    else:
        pg_hit = tbl_postgame.format(
            tid="tbl-post-hit",
            body=_rows_compare_postgame(S_hit_m1, S_hit_m2, "hit", xbh_lu, resolve_pg),
        ) + _postgame_accuracy_summary_html(S_hit_m1, S_hit_m2, "hit", resolve_pg)
        pg_xbh = tbl_postgame.format(
            tid="tbl-post-xbh",
            body=_rows_compare_postgame(S_xbh_m1, S_xbh_m2, "xbh", xbh_lu, resolve_pg),
        ) + _postgame_accuracy_summary_html(S_xbh_m1, S_xbh_m2, "xbh", resolve_pg)
        pg_hr = tbl_postgame.format(
            tid="tbl-post-hr",
            body=_rows_compare_postgame(S_hr_m1, S_hr_m2, "hr", xbh_lu, resolve_pg),
        ) + _postgame_accuracy_summary_html(S_hr_m1, S_hr_m2, "hr", resolve_pg)
        postgame_inner = f"""
{postgame_intro}
{_dashboard_section("Hits — top 25 vs actuals (calendar day)", pg_hit, open_first=True, csv_filename=f"postgame_hits_{slate_safe_primary}", table_id="tbl-post-hit")}
{_dashboard_section("XBH — top 25 vs actuals (calendar day)", pg_xbh, csv_filename=f"postgame_xbh_{slate_safe_primary}", table_id="tbl-post-xbh")}
{_dashboard_section("Home runs — top 25 vs actuals (calendar day)", pg_hr, csv_filename=f"postgame_hr_{slate_safe_primary}", table_id="tbl-post-hr")}
"""

    long_inner = _longform_tab_html(slate_safe_primary, experiment=False)
    long_exp_inner = _longform_tab_html(slate_safe_primary, experiment=True)

    # New tabs: Bucket Health (S0), Conviction Picks (S14), No HR Model (S15).
    # Each builder is defensive: missing source files render an inline "not yet generated" stub.
    try:
        bucket_health_panel_html = _build_bucket_health_panel()
    except Exception as ex:
        bucket_health_panel_html = f'<main><p class="intro">Bucket Health tab error: {html.escape(str(ex))}</p></main>'
    try:
        conviction_panel_html = _build_conviction_panel()
    except Exception as ex:
        conviction_panel_html = f'<main><p class="intro">Conviction Picks tab error: {html.escape(str(ex))}</p></main>'
    try:
        no_hr_panel_html = _build_no_hr_panel()
    except Exception as ex:
        no_hr_panel_html = f'<main><p class="intro">No HR Model tab error: {html.escape(str(ex))}</p></main>'

    hdr_bundle = next(b for b in bundles if str(b["slate"]) == default_sd)
    hdr_n = len(hdr_bundle["P"])
    if multi:
        sel_opts: list[str] = []
        view_parts: list[str] = []
        for b in bundles:
            sd = str(b["slate"])
            Pb, Eb = b["P"], b["E"]
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", sd[:10])
            suff = f"-d{m.group(1)}{m.group(2)}{m.group(3)}" if m else "-d" + re.sub(r"[^\w0-9]+", "_", sd)
            inner_html = _main_matchup_inner_html(
                Pb, Eb, sd, xbh_lu, table_id_suffix=suff, R_list=b.get("R") or []
            )
            sel_opts.append(
                f'<option value="{html.escape(sd, quote=True)}"'
                f'{" selected" if sd == default_sd else ""}>'
                f"{html.escape(sd)} ({len(Pb)} matchups)</option>"
            )
            hv = "" if sd == default_sd else " hidden"
            view_parts.append(
                f'<div class="main-slate-view{hv}" data-slate="{html.escape(sd)}" data-n="{len(Pb)}">'
                f"<main>{inner_html}</main></div>"
            )
        recency_view_parts: list[str] = []
        for b in bundles:
            sd = str(b["slate"])
            Pb = b["P"]
            Rb = b.get("R") or []
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", sd[:10])
            suff = f"-d{m.group(1)}{m.group(2)}{m.group(3)}" if m else "-d" + re.sub(r"[^\w0-9]+", "_", sd)
            r_inner = _recency_tab_inner_html(Rb, sd, xbh_lu, table_id_suffix=suff)
            hv = "" if sd == default_sd else " hidden"
            recency_view_parts.append(
                f'<div class="recency-slate-view{hv}" data-slate="{html.escape(sd)}" data-n="{len(Pb)}">'
                f"<main>{r_inner}</main></div>"
            )
        beast_view_parts: list[str] = []
        for b in bundles:
            sd = str(b["slate"])
            Pb = b["P"]
            Bb = b.get("B") or []
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", sd[:10])
            suff = f"-d{m.group(1)}{m.group(2)}{m.group(3)}" if m else "-d" + re.sub(r"[^\w0-9]+", "_", sd)
            b_inner = _beast_tab_inner_html(Bb, sd, xbh_lu, table_id_suffix=suff)
            hv = "" if sd == default_sd else " hidden"
            beast_view_parts.append(
                f'<div class="beast-slate-view{hv}" data-slate="{html.escape(sd)}" data-n="{len(Pb)}">'
                f"<main>{b_inner}</main></div>"
            )
        residual_view_parts: list[str] = []
        for b in bundles:
            sd = str(b["slate"])
            Pb = b["P"]
            Bb = b.get("B") or []
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", sd[:10])
            suff = f"-d{m.group(1)}{m.group(2)}{m.group(3)}" if m else "-d" + re.sub(r"[^\w0-9]+", "_", sd)
            res_inner = _residual_tab_inner_html(Pb, Bb, sd, xbh_lu, table_id_suffix=suff)
            hv = "" if sd == default_sd else " hidden"
            residual_view_parts.append(
                f'<div class="residual-slate-view{hv}" data-slate="{html.escape(sd)}" data-n="{len(Pb)}">'
                f"<main>{res_inner}</main></div>"
            )
        slate_picker = (
            '<div class="slate-picker-row">'
            '<label for="matchup-slate-select">Matchups slate</label> '
            '<select id="matchup-slate-select" aria-label="Select slate date for matchups, BvP, recency, Beast, and residual longshots">'
            f'{"".join(sel_opts)}</select>'
            '<span class="slate-picker-note">Past days load from <code>data/reports/archive/YYYY-MM-DD/</code> '
            "(<code>todays_matchup_predictions.json</code>, <code>…_exp.json</code>, optional <code>…_recency.json</code>, optional <code>…_beast.json</code>). "
            "The slate picker updates <strong>Matchups &amp; BvP</strong>, <strong>Matchups · recency</strong>, "
            "<strong>Matchups · Beast</strong>, and <strong>Residual longshots</strong>. "
            "Sections 7–10 still use the latest <code>section_*</code> markdown on disk.</span>"
            "</div>"
        )
        main_panel_html = "".join(view_parts)
        recency_panel_html = "".join(recency_view_parts)
        beast_panel_html = "".join(beast_view_parts)
        residual_panel_html = "".join(residual_view_parts)
    else:
        slate_picker = ""
        main_panel_html = (
            f"<main>{_main_matchup_inner_html(P, E_list, slate_primary, xbh_lu, table_id_suffix='', R_list=primary.get('R') or [])}</main>"
        )
        recency_panel_html = (
            f"<main>{_recency_tab_inner_html(primary.get('R') or [], slate_primary, xbh_lu, table_id_suffix='')}</main>"
        )
        beast_panel_html = f"<main>{_beast_tab_inner_html(primary.get('B') or [], slate_primary, xbh_lu, table_id_suffix='')}</main>"
        residual_panel_html = (
            f"<main>{_residual_tab_inner_html(primary['P'], primary.get('B') or [], slate_primary, xbh_lu, table_id_suffix='')}</main>"
        )

    uniq_teams: set[str] = set()
    for b in bundles:
        for r in b["P"]:
            v = r.get("batter_team")
            if isinstance(v, str) and v.strip():
                uniq_teams.add(v.strip().upper())
        for r in b["E"]:
            v = r.get("batter_team")
            if isinstance(v, str) and v.strip():
                uniq_teams.add(v.strip().upper())
        for r in b.get("R") or []:
            v = r.get("batter_team")
            if isinstance(v, str) and v.strip():
                uniq_teams.add(v.strip().upper())
        for r in b.get("B") or []:
            v = r.get("batter_team")
            if isinstance(v, str) and v.strip():
                uniq_teams.add(v.strip().upper())
    team_cbs = "".join(
        '<label class="team-filter-cb-lab">'
        f'<input type="checkbox" class="team-filter-cb" value="{html.escape(t, quote=True)}"/> '
        f"{html.escape(t)}</label>"
        for t in sorted(uniq_teams)
    )
    if team_cbs:
        team_filter_bar_html = (
            '<div class="team-filter-bar" id="team-filter-bar" role="region" aria-label="Filter tables by team">'
            '<span class="team-filter-label" id="team-filter-label">Teams</span>'
            '<div class="team-filter-dd" id="team-filter-dd">'
            '<button type="button" class="team-filter-dd-btn" id="team-filter-dd-btn" '
            'aria-expanded="false" aria-haspopup="true" aria-controls="team-filter-dd-panel">'
            '<span class="team-filter-dd-summary" id="team-filter-dd-summary">All teams</span>'
            '<span class="team-filter-dd-caret" aria-hidden="true">▾</span>'
            "</button>"
            '<div class="team-filter-dd-panel hidden" id="team-filter-dd-panel" role="group" '
            'aria-labelledby="team-filter-label">'
            '<div class="team-filter-dd-actions">'
            '<button type="button" class="team-filter-clear" id="team-filter-clear">Clear all</button>'
            "</div>"
            f'<div class="team-filter-dd-list">{team_cbs}</div>'
            "</div>"
            "</div>"
            '<span class="team-filter-hint">Open the menu to pick teams (no selection = all). A table row stays only if '
            "<strong>every</strong> batter on that row plays for a checked team (model 1 vs model 2 top-25 rows pair two batters — "
            "check both of their teams, or only those rows where both hitters are on teams you selected will show). "
            "<strong>Matchups · recency</strong>, <strong>Matchups · Beast</strong>, and <strong>Residual longshots</strong> rows list one batter each — one team match is enough. "
            "Sections 7–10: show a card if its text includes any checked team in parentheses.</span>"
            "</div>"
        )
    else:
        team_filter_bar_html = ""

    postgame_panel_html = f"<main>{postgame_inner}</main>"

    notepad_html = (
        '<aside class="dash-notepad" id="dash-notepad" aria-label="Personal notes">'
        '<div class="dash-notepad-toolbar">'
        '<span class="dash-notepad-label">Notes</span>'
        '<button type="button" class="dash-notepad-toggle" id="dash-notepad-toggle" '
        'aria-expanded="true" aria-controls="dash-notepad-field">Hide</button>'
        "</div>"
        '<textarea id="dash-notepad-field" class="dash-notepad-field" rows="10" spellcheck="true" '
        'placeholder="Scratch space — saved in this browser only (localStorage)."></textarea>'
        "</aside>"
    )

    try:
        if multi:
            wx_parts: list[str] = []
            for b in bundles:
                sd_wx = str(b["slate"])
                m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", sd_wx[:10])
                suff = (
                    f"d{m.group(1)}{m.group(2)}{m.group(3)}"
                    if m
                    else "d" + re.sub(r"[^\w0-9]+", "_", sd_wx)
                )
                inner_wx = _build_slate_weather_overview_html(
                    sd_wx, table_id=f"tbl-slate-weather-{suff}"
                )
                hv_wx = "" if sd_wx == default_sd else " hidden"
                wx_parts.append(
                    f'<div class="slate-weather-slate-view{hv_wx}" data-slate="{html.escape(sd_wx, quote=True)}" '
                    f'role="region" aria-label="Weather for slate {html.escape(sd_wx)}">'
                    f"{inner_wx}</div>"
                )
            weather_overview_html = "".join(wx_parts)
        else:
            weather_overview_html = _build_slate_weather_overview_html(str(default_sd))
    except Exception as ex:
        weather_overview_html = (
            '<section class="slate-weather-overview" aria-label="Slate weather and first pitch">'
            "<h2>Slate weather &amp; first pitch</h2>"
            f'<p class="intro muted slate-wx-intro">Weather overview error: {html.escape(str(ex))}</p>'
            "</section>"
        )

    body = f"""{notepad_html}
<header>
  <h1>Matchup dashboard</h1>
{slate_picker}
  <p class="meta" id="hdr-slate-line">Slate <strong id="hdr-slate-d">{html.escape(default_sd)}</strong> · <span id="hdr-slate-n">{hdr_n}</span> matchups</p>
</header>
{weather_overview_html}
{team_filter_bar_html}
<nav class="tabbar" role="tablist">
  <button type="button" class="tabbtn active" role="tab" aria-selected="true" data-panel="main">Matchups &amp; BvP</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="recency">Matchups · recency</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="beast">Matchups · Beast</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="residual">Residual longshots</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="postgame">Results vs actuals</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="bucket-health">Bucket Health (S0)</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="conviction">Conviction Picks (S14)</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="no-hr">No HR Model (S15)</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="long">Sections 7–10 · model 1 (production)</button>
  <button type="button" class="tabbtn" role="tab" aria-selected="false" data-panel="long-exp">Sections 7–10 · model 2 (experiment)</button>
</nav>
<section id="panel-main" class="tabpanel" role="tabpanel">
{main_panel_html}
</section>
<section id="panel-recency" class="tabpanel hidden" role="tabpanel">
{recency_panel_html}
</section>
<section id="panel-beast" class="tabpanel hidden" role="tabpanel">
{beast_panel_html}
</section>
<section id="panel-residual" class="tabpanel hidden" role="tabpanel">
{residual_panel_html}
</section>
<section id="panel-postgame" class="tabpanel hidden" role="tabpanel">
{postgame_panel_html}
</section>
<section id="panel-bucket-health" class="tabpanel hidden" role="tabpanel">
{bucket_health_panel_html}
</section>
<section id="panel-conviction" class="tabpanel hidden" role="tabpanel">
{conviction_panel_html}
</section>
<section id="panel-no-hr" class="tabpanel hidden" role="tabpanel">
{no_hr_panel_html}
</section>
<section id="panel-long" class="tabpanel hidden" role="tabpanel">
  <main class="longmain">
    <p class="intro">Production longform from <code>section_*_prod.md</code>. Click column headers to sort. Use Export CSV on each markdown table.</p>
    {long_inner}
  </main>
</section>
<section id="panel-long-exp" class="tabpanel hidden" role="tabpanel">
  <main class="longmain">
    <p class="intro">Experiment longform from <code>section_*_exp.md</code>. Click column headers to sort. Use Export CSV on each markdown table.</p>
    {long_exp_inner}
  </main>
</section>
<footer>
  <p>Matchups: <code>todays_matchup_predictions.json</code> + <code>todays_matchup_predictions_exp.json</code> (current slate, <strong>Matchups &amp; BvP</strong> tab). Optional <code>todays_matchup_predictions_recency.json</code> powers <strong>Matchups · recency</strong>, optional <code>todays_matchup_predictions_beast.json</code> powers <strong>Matchups · Beast</strong>, and <strong>Residual longshots</strong> re-filters production rows with YTD from <code>data/raw/qualifying_batters_2026.csv</code> (no new prediction JSON). Older slates are read from <code>data/reports/archive/</code> when present.
  Statcast BvP counts (including barrels and lasers) stay in the <strong>Matchups &amp; BvP</strong> tab.
  Writeups: model 1 <code>section_7_prod.md</code> … <code>section_10_prod.md</code>;
  model 2 <code>section_7_exp.md</code> … <code>section_10_exp.md</code>.
  Postgame tab: Statcast calendar-day batter totals (and optional <code>matchup_predictions_runs.parquet</code> fills).</p>
</footer>"""

    css = r"""
:root {
  --bg: #0f1419;
  --card: #1a2332;
  --text: #e7ecf3;
  --muted: #8b9bb4;
  --accent: #3b82f6;
  --border: #2d3a4d;
  --adj: #fbbf24;
  --raw: #a7f3d0;
  --tpa: #93c5fd;
  --bvp-col: #c4b5fd;
}
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text); margin: 0; padding: 1rem 1.25rem 2rem; line-height: 1.45; }
header h1 { margin: 0 0 0.25rem; font-size: 1.5rem; }
.meta { color: var(--muted); margin: 0 0 1rem; font-size: 0.95rem; }
.slate-weather-overview {
  margin: 0 0 1.15rem;
  padding: 0.85rem 1rem;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.slate-weather-overview h2 {
  margin: 0 0 0.5rem;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--accent);
}
.slate-weather-overview > .slate-wx-intro:first-child { margin-top: 0; }
.main-slate-view > main > .intro:first-child,
.recency-slate-view > main > .intro:first-child,
.beast-slate-view > main > .intro:first-child,
.residual-slate-view > main > .intro:first-child {
  margin-top: 0;
}
.slate-wx-intro { margin-top: 0; margin-bottom: 0.65rem; font-size: 0.88rem; }
.slate-wx-scroll { max-width: 100%; margin-top: 0.35rem; }
table.slate-wx-table { min-width: 1160px; font-size: 0.8rem; }
table.slate-wx-table td.wx-ehr { white-space: nowrap; font-variant-numeric: tabular-nums; }
table.slate-wx-table th, table.slate-wx-table td {
  padding: 0.42rem 0.5rem;
  vertical-align: top;
  border-color: var(--border);
}
table.slate-wx-table td.wx-vs-stad { max-width: 14rem; white-space: normal; }
table.slate-wx-table td.wx-notes { max-width: 16rem; white-space: normal; }
table.slate-wx-table td.wx-desc { max-width: 22rem; white-space: normal; line-height: 1.35; }
.wx-hr-up { color: #fca5a5; font-weight: 700; }
.wx-hr-down { color: #86efac; font-weight: 700; }
.wx-hr-neutral { color: #fcd34d; }
.tabbar { display: flex; gap: 0.35rem; margin-bottom: 1rem; flex-wrap: wrap; }
.tabbtn {
  cursor: pointer; border: 1px solid var(--border); background: #131c28; color: var(--text);
  padding: 0.55rem 1rem; border-radius: 8px; font-size: 0.95rem; font-weight: 600;
}
.tabbtn:hover { background: #243044; }
.tabbtn.active { background: var(--accent); color: #0f1419; border-color: var(--accent); }
.tabpanel.hidden { display: none; }

/* New tabs (Section 0 / 14 / 15) */
.status-pill {
  display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em;
}
.callout-critical, .callout-warn {
  border-radius: 8px; padding: 0.55rem 0.85rem; margin: 0.35rem 0;
  font-size: 0.92rem; border-left: 4px solid;
}
.callout-critical { background: rgba(239, 68, 68, 0.10); border-left-color: #ef4444; color: #fecaca; }
.callout-warn     { background: rgba(251, 191, 36, 0.10); border-left-color: #fbbf24; color: #fde68a; }
table.bucket-health td, table.no-hr-grid td, table.conv-grid td { vertical-align: middle; }
table.bucket-health svg.spark, table.no-hr-grid svg.lambda-stack-svg,
table.conv-grid svg.streak { display: block; }
.lambda-legend, .streak-legend {
  display: flex; flex-wrap: wrap; gap: 0.85rem; align-items: center;
  font-size: 0.85rem; color: var(--muted); margin: 0.35rem 0 0.7rem;
}
.lambda-legend i, .streak-legend i {
  display: inline-block; width: 12px; height: 12px; border-radius: 2px;
  margin-right: 0.3rem; vertical-align: middle;
}
.no-hr-grid td.p0-good { color: #86efac; font-weight: 700; }
.no-hr-grid td.p0-bad { color: #fca5a5; }
.no-hr-grid td.wx-helping { color: #fca5a5; font-weight: 600; }
.no-hr-grid td.wx-suppressing { color: #86efac; font-weight: 600; }
.no-hr-grid td.pen-leaky { color: #fca5a5; font-weight: 600; }
.no-hr-grid td.pen-suppressing { color: #86efac; font-weight: 600; }
.conv-h { margin: 1.1rem 0 0.5rem; font-size: 1.05rem; color: var(--accent); }
.muted { color: var(--muted); }
.dash-notepad {
  position: fixed;
  top: 0.65rem;
  right: 0.65rem;
  z-index: 10000;
  width: min(17.5rem, calc(100vw - 1.35rem));
  max-width: 100%;
  background: #1a2332;
  border: 1px solid var(--border);
  border-radius: 10px;
  box-shadow: 0 10px 36px rgba(0, 0, 0, 0.55);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.dash-notepad--collapsed .dash-notepad-field { display: none; }
.dash-notepad-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.45rem 0.55rem;
  background: #243044;
  border-bottom: 1px solid var(--border);
}
.dash-notepad-label {
  font-size: 0.8rem;
  font-weight: 700;
  color: #e7ecf3;
  letter-spacing: 0.02em;
}
.dash-notepad-toggle {
  cursor: pointer;
  border: 1px solid var(--border);
  background: #131c28;
  color: var(--text);
  padding: 0.28rem 0.55rem;
  border-radius: 6px;
  font-size: 0.72rem;
  font-weight: 600;
}
.dash-notepad-toggle:hover { border-color: var(--accent); color: #fff; }
.dash-notepad-field {
  width: 100%;
  margin: 0;
  border: none;
  resize: vertical;
  min-height: 8.5rem;
  max-height: min(50vh, 22rem);
  background: #0f1419;
  color: var(--text);
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 0.84rem;
  line-height: 1.4;
  padding: 0.55rem 0.65rem;
  box-sizing: border-box;
}
.dash-notepad-field::placeholder { color: var(--muted); opacity: 0.9; }
.dash-notepad-field:focus { outline: 2px solid var(--accent); outline-offset: -2px; }
@media (max-width: 520px) {
  .dash-notepad {
    top: 0.45rem;
    right: 0.45rem;
    width: min(12.5rem, calc(100vw - 0.9rem));
    max-height: 42vh;
  }
  .dash-notepad-field { min-height: 5.5rem; max-height: min(32vh, 14rem); }
}
.main-slate-view.hidden { display: none; }
.recency-slate-view.hidden { display: none; }
.beast-slate-view.hidden { display: none; }
.residual-slate-view.hidden { display: none; }
.slate-weather-slate-view.hidden { display: none; }
.slate-picker-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.65rem 1rem;
  margin: 0 0 0.75rem; padding: 0.65rem 0.85rem; background: #121a26;
  border: 1px solid var(--border); border-radius: 8px; max-width: 56rem;
}
.slate-picker-row label { font-weight: 600; color: var(--muted); font-size: 0.9rem; }
.slate-picker-row select {
  padding: 0.4rem 0.65rem; border-radius: 6px; border: 1px solid var(--border);
  background: #0f1419; color: var(--text); font-size: 0.9rem; min-width: 12rem;
}
.slate-picker-note { font-size: 0.78rem; color: var(--muted); max-width: 36rem; line-height: 1.35; }
.team-filter-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.55rem 0.85rem;
  margin: 0 0 1rem;
  padding: 0.5rem 0.85rem;
  max-width: 58rem;
  background: #121a26;
  border: 1px solid var(--border);
  border-radius: 8px;
}
.team-filter-label {
  font-weight: 700;
  font-size: 0.88rem;
  color: #e7ecf3;
  flex: 0 0 auto;
}
.team-filter-dd {
  position: relative;
  flex: 1 1 14rem;
  min-width: 11rem;
  max-width: min(22rem, 100%);
}
.team-filter-dd-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  cursor: pointer;
  text-align: left;
  padding: 0.42rem 0.65rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: #0f1419;
  color: var(--text);
  font-size: 0.86rem;
  font-weight: 600;
}
.team-filter-dd-btn:hover { border-color: var(--accent); }
.team-filter-dd-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.team-filter-dd-summary { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.team-filter-dd-caret { flex: 0 0 auto; color: var(--muted); font-size: 0.7rem; }
.team-filter-dd-panel {
  position: absolute;
  left: 0;
  right: 0;
  top: calc(100% + 0.25rem);
  z-index: 80;
  max-height: min(18rem, 55vh);
  display: flex;
  flex-direction: column;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #131c28;
  box-shadow: 0 8px 24px rgba(0,0,0,0.45);
}
.team-filter-dd-panel.hidden { display: none; }
.team-filter-dd-actions {
  flex: 0 0 auto;
  padding: 0.4rem 0.5rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: flex-end;
}
.team-filter-dd-list {
  overflow: auto;
  padding: 0.35rem 0.25rem 0.5rem;
}
.team-filter-cb-lab {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.28rem 0.5rem;
  font-size: 0.84rem;
  cursor: pointer;
  border-radius: 4px;
}
.team-filter-cb-lab:hover { background: #1a2636; }
.team-filter-cb { accent-color: var(--accent); cursor: pointer; }
.team-filter-clear {
  cursor: pointer;
  border: 1px solid var(--border);
  background: #243044;
  color: var(--text);
  padding: 0.28rem 0.55rem;
  border-radius: 6px;
  font-size: 0.74rem;
  font-weight: 600;
}
.team-filter-clear:hover { border-color: var(--accent); }
.team-filter-hint {
  flex: 1 1 100%;
  font-size: 0.74rem;
  color: var(--muted);
  line-height: 1.35;
  margin: 0;
}
tr.team-filter-hidden,
details.card.team-filter-hidden { display: none !important; }
.intro { color: var(--muted); margin: 0 0 1rem; max-width: 52rem; }
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  margin-bottom: 0.75rem; overflow: hidden;
}
.card > summary {
  cursor: pointer; padding: 0.85rem 1rem; font-weight: 600; font-size: 1.02rem;
  list-style: none; user-select: none;
}
.card > summary::-webkit-details-marker { display: none; }
.card > summary::before {
  content: "▸"; color: var(--accent); margin-right: 0.45rem; display: inline-block;
  transition: transform 0.15s;
}
.card[open] > summary::before { transform: rotate(90deg); }
.card > summary:hover { background: #243044; }
.cardbody { padding: 0 1rem 1rem; }
.csv-toolbar { margin-bottom: 0.5rem; }
.btn-csv {
  cursor: pointer; border: 1px solid var(--border); background: #243044; color: var(--text);
  padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.82rem; font-weight: 600;
}
.btn-csv:hover { background: #2d3f56; border-color: var(--accent); }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table.dual {
  width: 100%; border-collapse: collapse; font-size: 0.84rem; min-width: 640px;
}
table.dual th, table.dual td {
  border: 1px solid var(--border); padding: 0.45rem 0.55rem; vertical-align: top;
}
table.dual thead th {
  background: #243044; text-align: left; font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted);
}
td.rn { width: 2rem; text-align: right; color: var(--muted); }
table.dual.compare { min-width: 1280px; font-size: 0.78rem; }
table.dual.compare.postgame { min-width: 1540px; font-size: 0.76rem; }
/* Hits / XBH / HR compare tables: light zebra body for scanability */
table.dual.compare.striped thead th {
  background: #cbd5e1 !important;
  color: #1e293b;
  border-color: #94a3b8;
  font-weight: 700;
}
table.dual.compare.striped tbody tr:nth-child(odd) > td {
  background: #ffffff !important;
  color: #0f172a;
  border-color: #cbd5e1;
}
table.dual.compare.striped tbody tr:nth-child(even) > td {
  background: #e8edf3 !important;
  color: #0f172a;
  border-color: #cbd5e1;
}
table.dual.compare.striped td.rn { color: #64748b !important; font-weight: 600; }
table.dual.compare.striped td.mu1, table.dual.compare.striped td.mu2 {
  background: inherit !important;
  color: #0f172a;
  font-weight: 600;
  max-width: 11rem;
}
table.dual.compare.striped td.pr1, table.dual.compare.striped td.pr2 {
  background: inherit !important;
  max-width: 24rem;
  vertical-align: top;
}
table.dual.compare.striped td.tk1, table.dual.compare.striped td.tk2 {
  background: inherit !important;
  max-width: 14rem;
  min-width: 7.5rem;
  vertical-align: top;
}
table.dual.compare.striped .take-block.take-col .take-s {
  margin: 0 0 0.28rem;
  font-size: 0.71rem;
  line-height: 1.38;
  color: #526077;
}
table.dual.compare.striped .take-block.take-col .take-s:last-child { margin-bottom: 0; }
/* Beast leaderboard: narrower matchup/proof, widest readable takeaway */
table.dual.compare.striped.beast-top25 {
  table-layout: fixed;
  width: 100%;
}
table.dual.compare.striped.beast-top25 thead th:nth-child(2),
table.dual.compare.striped.beast-top25 tbody td.mu1 {
  width: 11%;
  max-width: none;
  min-width: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
  hyphens: auto;
  font-weight: 600;
  font-size: 0.72rem;
  line-height: 1.3;
}
table.dual.compare.striped.beast-top25 thead th:nth-child(3),
table.dual.compare.striped.beast-top25 tbody td.pr1 {
  width: 15%;
  max-width: none;
  min-width: 0;
  vertical-align: top;
}
table.dual.compare.striped.beast-top25 thead th:nth-child(4),
table.dual.compare.striped.beast-top25 tbody td.tk1 {
  width: 34%;
  min-width: 11rem;
  max-width: none;
  vertical-align: top;
}
table.dual.compare.striped.beast-top25 .beast-take-mgr {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
table.dual.compare.striped.beast-top25 .beast-take-mgr .tk-row {
  display: grid;
  grid-template-columns: 6.75rem minmax(0, 1fr);
  gap: 0.35rem 0.55rem;
  align-items: start;
}
table.dual.compare.striped.beast-top25 .beast-take-mgr .tk-lab {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.045em;
  text-transform: uppercase;
  color: #6b7280;
  line-height: 1.35;
  padding-top: 0.1rem;
}
table.dual.compare.striped.beast-top25 .beast-take-mgr .tk-body {
  font-size: 0.84rem;
  line-height: 1.48;
  color: #1f2937;
}
table.dual.compare.striped.beast-top25 .beast-take-mgr .tk-row.tk-conc {
  padding-top: 0.38rem;
  margin-top: 0.06rem;
  border-top: 1px solid #e5e7eb;
}
table.dual.compare.striped.beast-top25 .beast-take-mgr .tk-row.tk-conc .tk-body {
  font-weight: 600;
  font-size: 0.85rem;
  line-height: 1.46;
  color: #111827;
}
table.dual.compare.striped.beast-top25 td.rn {
  width: 2rem;
}
table.dual.compare.striped.beast-top25 td.cf1 {
  width: 4.6rem;
  min-width: 4rem;
}
table.dual.compare.striped.beast-top25 td.bvp1 {
  width: auto;
  min-width: 6rem;
  max-width: 14%;
}
table.dual.compare.striped.beast-top25 td.beast-plat {
  width: 7rem;
  max-width: 9rem;
  vertical-align: top;
  font-size: 0.68rem;
  line-height: 1.28;
  font-variant-numeric: tabular-nums;
}
table.dual.compare.striped.beast-top25 td.beast-plat .beast-plat-hand {
  font-weight: 700;
  font-size: 0.62rem;
  color: var(--accent, #3b82f6);
  letter-spacing: 0.02em;
}
table.dual.compare.striped.beast-top25 td.beast-plat .beast-plat-line1 {
  font-weight: 600;
  color: var(--text, #e7ecf3);
}
table.dual.compare.striped.beast-top25 td.beast-plat .beast-plat-line2 {
  display: block;
  margin-top: 0.1rem;
}
table.dual.compare.striped.beast-top25 td.beast-plat .beast-plat-muted {
  color: var(--muted, #8b9bb4);
}
table.dual.compare.striped td.cf1, table.dual.compare.striped td.cf2 {
  background: inherit !important;
  width: 5.5rem;
  min-width: 4.8rem;
  max-width: 6rem;
  text-align: center;
  vertical-align: middle;
  font-size: 0.76rem;
}
/* Striped compare tables: text only black / grey (no accent colors in cells) */
table.dual.compare.striped td.bvp1, table.dual.compare.striped td.bvp2 {
  background: inherit !important;
  color: #374151 !important;
  font-size: 0.74rem;
  max-width: 12rem;
}
table.dual.compare.striped .proof-block {
  color: #4b5563;
  font-size: 0.74rem;
  line-height: 1.4;
}
table.dual.compare.striped .proof-s {
  margin: 0 0 0.3rem;
  color: #4b5563;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
table.dual.compare.striped .proof-s:last-child { margin-bottom: 0; }
table.dual.compare.striped .proof-block strong { color: #111827; font-weight: 700; }
table.dual.compare.striped .proof-muted { color: #6b7280; }
table.dual.compare.striped .conflev { font-weight: 700; color: #111827; line-height: 1.2; }
table.dual.compare.striped .confnum { font-size: 0.68rem; margin-top: 0.15rem; color: #6b7280 !important; }
table.dual.compare.striped td.res.ok { color: #22c55e !important; }
table.dual.compare.striped td.res.no { color: #ef4444 !important; }
table.dual.compare.striped td.res.unk { color: #94a3b8 !important; }
table.dual.compare.striped td.act { color: #4b5563 !important; }
table.dual.compare.striped.sortable thead th:hover { color: #111827 !important; }
td.res { text-align: center; font-size: 1.05rem; width: 2.4rem; font-weight: 700; }
td.res.ok { color: #22c55e; }
td.res.no { color: #ef4444; }
td.res.unk { color: #94a3b8; font-weight: 600; font-size: 0.95rem; }
td.act { font-size: 0.72rem; white-space: nowrap; color: #1e40af; min-width: 7.5rem; }
.postgame-intro strong { color: #374151; }
.postgame-sum { margin-top: 0.75rem; }
td.mu1, td.mu2 { min-width: 140px; max-width: 200px; font-weight: 500; font-size: 0.8rem; background: #141c26; }
td.pr1, td.pr2 { width: 17%; min-width: 120px; background: #131c28; }
td.tk1, td.tk2 {
  width: 11%;
  min-width: 88px;
  max-width: 180px;
  background: #131c28;
  vertical-align: top;
}
td.cf1, td.cf2 { width: 5rem; min-width: 4.5rem; text-align: center; vertical-align: middle; font-size: 0.76rem; }
td.bvp1, td.bvp2 {
  width: 14%; min-width: 100px; background: #1e1a2e; font-size: 0.78rem; color: var(--bvp-col);
  font-weight: 500; line-height: 1.35;
}
.mh { font-size: 0.68rem; text-transform: uppercase; color: var(--muted); margin-bottom: 0.3rem; letter-spacing: 0.04em; }
table.grid { width: 100%; border-collapse: collapse; font-size: 0.84rem; min-width: 520px; }
table.grid#tbl-barrels { min-width: 720px; }
table.grid#tbl-lasers-hr,
table.grid#tbl-lasers-hard { min-width: 820px; }
table.grid th, table.grid td { border: 1px solid var(--border); padding: 0.4rem 0.5rem; text-align: left; }
table.grid thead th { background: #243044; font-size: 0.72rem; text-transform: uppercase; color: var(--muted); }
table.grid .num { text-align: right; font-variant-numeric: tabular-nums; }
.empty { color: var(--muted); font-size: 0.9rem; }
footer { margin-top: 2rem; font-size: 0.78rem; color: var(--muted); }
code { background: #131c28; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.85em; }
.prose { max-width: 52rem; font-size: 0.92rem; color: #dce6f2; }
.prose-h2 { font-size: 1.15rem; margin: 1.25rem 0 0.5rem; color: #93c5fd; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }
.prose-h3 { font-size: 1.02rem; margin: 1rem 0 0.4rem; color: #fbbf24; }
.prose-p { margin: 0.45rem 0; line-height: 1.55; }
.prose-hr { border: none; border-top: 1px dashed var(--border); margin: 1.25rem 0; }
.prose-table { margin: 0; font-size: 0.82rem; }
.prose-table th { background: #243044; }
.prose .table-wrap { margin: 0.75rem 0; }
.prose .table-wrap .prose-table { margin: 0; }
.beast-writeups-collapsed { margin-bottom: 1rem; }
.beast-longform-prose {
  max-height: min(70vh, 48rem);
  overflow-y: auto;
  padding-right: 0.35rem;
}
.beast-section-blurb {
  font-size: 0.88rem; line-height: 1.45; color: #c7d4e5; margin: 0 0 0.85rem;
  padding: 0.55rem 0.65rem; background: #131c28; border-radius: 6px; border-left: 3px solid #3b82f6;
}
.beast-mini {
  margin: 0.55rem 0; padding: 0.5rem 0.55rem; border-radius: 6px; border: 1px solid var(--border);
  background: #0f1419;
}
.beast-mini-hd { font-weight: 600; font-size: 0.86rem; color: #e7ecf3; margin-bottom: 0.35rem; }
.beast-mini-verdict { margin: 0.2rem 0 0.35rem; font-size: 0.83rem; line-height: 1.4; }
.beast-mini-stat { margin: 0; font-size: 0.78rem; line-height: 1.45; color: #b8c5d6; }
.beast-conf-rel { text-align: center; width: 3rem; vertical-align: middle; }
.beast-conf-rel-num { font-weight: 700; font-size: 0.95rem; }
.beast-conf-rel.tier-5 .beast-conf-rel-num { color: #4ade80; }
.beast-conf-rel.tier-4 .beast-conf-rel-num { color: #86efac; }
.beast-conf-rel.tier-3 .beast-conf-rel-num { color: #fde68a; }
.beast-conf-rel.tier-2 .beast-conf-rel-num { color: #fdba74; }
.beast-conf-rel.tier-1 .beast-conf-rel-num { color: #fca5a5; }
.beast-good { border-left: 3px solid #22c55e; }
.beast-good .beast-mini-verdict { color: #bbf7d0; }
.beast-bad { border-left: 3px solid #f87171; }
.beast-bad .beast-mini-verdict { color: #fecaca; }
.beast-lean { border-left: 3px solid #fbbf24; }
.beast-lean .beast-mini-verdict { color: #fde68a; }
.beast-scatter-gallery { margin-top: 0.75rem; margin-bottom: 1rem; }
.beast-scatter-gallery .beast-scatter-intro {
  margin-top: 0; margin-bottom: 0.85rem; font-size: 0.84rem; line-height: 1.45; color: #b8c5d6;
}
.beast-scatter-gallery .beast-scatter-intro code {
  font-size: 0.78rem; background: #131c28; padding: 0.06rem 0.28rem; border-radius: 4px;
}
.beast-scatter-chart-details {
  margin-bottom: 0.65rem;
}
.beast-scatter-chart-details summary {
  cursor: pointer; font-weight: 600; color: #e7ecf3;
}
.beast-scatter-chart-details summary:hover { color: #93c5fd; }
.beast-scatter-chart-body { padding-top: 0.25rem; }
figure.beast-scatter-fig { margin: 0; }
figure.beast-scatter-fig figcaption { font-size: 0.82rem; font-weight: 600; color: #93c5fd; margin-bottom: 0.35rem; }
.beast-scatter-svg { width: 100%; max-width: 520px; height: auto; display: block; border-radius: 6px; }
.beast-scatter-dot { cursor: default; }
.beast-scatter-dot:hover { fill: #fbbf24; stroke: #92400e; }
.beast-scatter-dot--pick:hover { fill: #fcd34d !important; stroke: #b45309 !important; }
.beast-scatter-label {
  pointer-events: none;
  fill: #9ca3af;
  font-weight: 400;
}
.longopen { margin-bottom: 1rem; }
.divergence-summary {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin-top: 0.85rem;
  background: #121a26;
  font-size: 0.84rem;
  line-height: 1.45;
}
.diverge-h { margin: 0 0 0.45rem; font-size: 0.95rem; color: #93c5fd; font-weight: 600; }
.diverge-ul { margin: 0.25rem 0 0.65rem 1.15rem; padding: 0; color: #dce6f2; }
.diverge-ul li { margin: 0.3rem 0; }
.diverge-ul li strong { color: #fbbf24; }
.rec-p { margin: 0.45rem 0; color: #e7ecf3; }
.rec-pick { margin: 0.35rem 0; font-weight: 600; color: #a7f3d0; }
.rec-note { margin: 0.55rem 0 0; font-size: 0.76rem; color: var(--muted); line-height: 1.4; }
.postgame-toolbar {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem 1rem 1rem;
  margin-bottom: 1rem;
  background: #121a26;
}
.postgame-toolbar .pg-meta { margin: 0 0 0.65rem; font-size: 0.88rem; color: #dce6f2; }
.postgame-toolbar .pg-date-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
  align-items: flex-end;
}
.postgame-toolbar .pg-date-lab {
  font-size: 0.88rem;
  color: var(--muted);
  font-weight: 600;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-width: 11rem;
}
.postgame-toolbar input[type="date"] {
  padding: 0.4rem 0.55rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: #0f1419;
  color: var(--text);
  font-size: 0.9rem;
  font-family: inherit;
  min-width: 11.5rem;
}
.postgame-toolbar .pg-topn-lab select { margin-top: 0.35rem; padding: 0.35rem 0.5rem; border-radius: 6px; border: 1px solid var(--border); background: #0f1419; color: var(--text); }
.postgame-section { margin-bottom: 1.5rem; }
.postgame-kpis {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-bottom: 0.65rem;
}
.kpi-card {
  background: #131c28;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.45rem 0.65rem;
  min-width: 6.5rem;
}
.kpi-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
.kpi-value { font-size: 1.05rem; font-weight: 700; color: #e7ecf3; }
table.dual.compare.postgame td.slate { white-space: nowrap; font-size: 0.72rem; color: #64748b; }
table.sortable thead th {
  cursor: pointer;
  user-select: none;
}
table.sortable thead th:hover { color: #c7d2fe; }
table.sortable thead th.sort-asc::after { content: " ▲"; font-size: 0.65em; opacity: 0.85; }
table.sortable thead th.sort-desc::after { content: " ▼"; font-size: 0.65em; opacity: 0.85; }
"""

    js = r"""
function tableToCsv(table) {
  var rows = table.querySelectorAll("tr");
  var lines = [];
  for (var r = 0; r < rows.length; r++) {
    var cells = rows[r].querySelectorAll("th, td");
    var vals = [];
    for (var c = 0; c < cells.length; c++) {
      var t = (cells[c].innerText || "").replace(/\r?\n/g, " ").replace(/"/g, '""').trim();
      vals.push('"' + t + '"');
    }
    lines.push(vals.join(","));
  }
  return lines.join("\n");
}
document.body.addEventListener("click", function(ev) {
  var btn = ev.target.closest(".btn-csv");
  if (!btn) return;
  var id = btn.getAttribute("data-csv-target");
  var name = btn.getAttribute("data-csv-filename") || "export.csv";
  var tbl = id ? document.getElementById(id) : null;
  if (!tbl) return;
  var csv = tableToCsv(tbl);
  var blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = name.length >= 4 && name.slice(-4).toLowerCase() === ".csv" ? name : name + ".csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});
function proseCellText(tr, colIdx) {
  var c = tr.cells[colIdx];
  return c ? (c.innerText || "").trim() : "";
}
function detectSortMode(rows, colIdx) {
  var texts = rows.map(function(tr) { return proseCellText(tr, colIdx); }).filter(function(t) { return t.length; });
  if (!texts.length) return "str";
  function strip(s) { return s.replace(/\s/g, ""); }
  if (texts.every(function(t) { return /^\d+$/.test(strip(t)); })) return "int";
  if (texts.every(function(t) { return /^-?\d+\.?\d*%?$/.test(strip(t)); })) return "num";
  var heads = texts.map(function(t) {
    var m = t.match(/-?\d+\.?\d*/);
    return m ? parseFloat(m[0]) : NaN;
  });
  if (heads.every(function(n) { return !isNaN(n); })) return "headnum";
  return "str";
}
function sortKey(text, mode) {
  if (mode === "int") return parseInt(text.replace(/\s/g, ""), 10) || 0;
  if (mode === "num") return parseFloat(text.replace(/%/g, "").replace(/\s/g, "")) || 0;
  if (mode === "headnum") {
    var m = text.match(/-?\d+\.?\d*/);
    return m ? parseFloat(m[0]) : 0;
  }
  return (text || "").toLowerCase();
}
document.body.addEventListener("click", function(ev) {
  var th = ev.target.closest("th");
  if (!th || !th.closest("thead")) return;
  var table = th.closest("table.sortable");
  if (!table) return;
  var tbody = table.tBodies[0];
  if (!tbody) return;
  var colIdx = th.cellIndex;
  if (colIdx < 0) return;
  var rows = Array.prototype.slice.call(tbody.rows);
  if (!rows.length) return;
  var mode = detectSortMode(rows, colIdx);
  var sameCol = table.dataset.sortCol === String(colIdx);
  var nextDir = sameCol && table.dataset.sortDir === "asc" ? "desc" : "asc";
  var dir = nextDir === "asc" ? 1 : -1;
  table.dataset.sortCol = String(colIdx);
  table.dataset.sortDir = nextDir;
  var headRow = th.parentNode;
  if (headRow) {
    var ths = headRow.querySelectorAll("th");
    for (var i = 0; i < ths.length; i++) {
      ths[i].classList.remove("sort-asc", "sort-desc");
    }
  }
  th.classList.add(nextDir === "asc" ? "sort-asc" : "sort-desc");
  rows.sort(function(a, b) {
    var ta = proseCellText(a, colIdx);
    var tb = proseCellText(b, colIdx);
    var va = sortKey(ta, mode);
    var vb = sortKey(tb, mode);
    var cmp = 0;
    if (mode === "str") {
      cmp = String(va).localeCompare(String(vb));
    } else if (va === vb) {
      cmp = 0;
    } else {
      cmp = va < vb ? -1 : 1;
    }
    return dir * cmp;
  });
  rows.forEach(function(tr) { tbody.appendChild(tr); });
});
document.querySelectorAll('.tabbtn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tabbtn').forEach(function(b) {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.tabpanel').forEach(function(p) { p.classList.add('hidden'); });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
    var id = 'panel-' + btn.getAttribute('data-panel');
    var el = document.getElementById(id);
    if (el) el.classList.remove('hidden');
  });
});
"""
    if postgame_payload is not None:
        js += "\n" + POSTGAME_JS
    if multi:
        js += "\n" + SLATE_SWITCH_JS
    js += "\n" + NOTEPAD_JS
    js += "\n" + TEAM_FILTER_JS

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Matchup dashboard · {html.escape(str(default_sd))}</title>
  <style>{css}</style>
</head>
<body>
{body}
<script>{js}</script>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", help="Slate date label in title (default: from JSON)")
    p.add_argument("-o", "--output", type=Path, default=REPORTS / "matchup_dashboard.html", help="Output path")
    args = p.parse_args()
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = build_dashboard_html(slate=args.date)
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
