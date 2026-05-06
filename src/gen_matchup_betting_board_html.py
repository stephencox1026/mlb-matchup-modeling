#!/usr/bin/env python3
"""
Build docs/matchup_betting_board.html — matchup selector + Hit/XBH/HR tables
with split platoon columns (2026 vs hand / career vs hand) and bet-take blurbs.

Regenerate after slate refresh:
  PYTHONPATH=src python3 src/gen_matchup_betting_board_html.py
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import DOCS_DIR, RAW_DIR, REPORTS_DIR  # noqa: E402

PRED_JSON = REPORTS_DIR / "todays_matchup_predictions.json"
MATCHUPS_JSON = RAW_DIR / "todays_matchups.json"
STATCAST = RAW_DIR / "statcast_pa_level_league.parquet"
OUT_HTML = DOCS_DIR / "matchup_betting_board.html"

LEAGUE_HIT = 0.222
LEAGUE_HR = 0.031
LEAGUE_XBH = 0.076

HIT_EVENTS = {"single", "double", "triple", "home_run"}


def _fmt_game_time_et(iso_utc: object | None) -> str:
    """First pitch in America/New_York for dropdown labels (e.g. ``7:20 PM ET``)."""
    if iso_utc is None:
        return ""
    s = str(iso_utc).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        et = dt.astimezone(ZoneInfo("America/New_York"))
    except (ValueError, TypeError, OSError):
        return ""
    h24 = et.hour
    h12 = h24 % 12 or 12
    ampm = "PM" if h24 >= 12 else "AM"
    return f"{h12}:{et.minute:02d} {ampm} ET"


def _ab_den(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    non_ab = df["events"].isin({"walk", "intent_walk", "hit_by_pitch"}).sum()
    return max(len(df) - int(non_ab), 1)


def _load_platoon_frame() -> pd.DataFrame | None:
    if not STATCAST.exists():
        return None
    cols = ["batter", "p_throws", "game_year", "events", "launch_speed"]
    try:
        pa = pd.read_parquet(STATCAST, columns=cols)
    except Exception:
        cols = ["batter", "p_throws", "game_year", "events"]
        pa = pd.read_parquet(STATCAST, columns=cols)
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_xbh"] = pa["events"].isin({"double", "triple", "home_run"}).astype(int)
    pa["is_k"] = pa["events"].str.contains("strikeout", case=False, na=False).astype(int)
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    return pa


_PLATOON_MEMO: dict[tuple[int, bool], tuple[dict, dict]] = {}


def platoon_pack(pa: pd.DataFrame | None, batter_id: int, vs_rhp: bool) -> tuple[dict, dict]:
    """Returns (y2026_dict, career_dict) with keys pa, h, ab, avg, k_pct, bb_pct, xbh, hr, xbh_pct, hr_pct."""
    if pa is None:
        return {}, {}
    memo_k = (batter_id, vs_rhp)
    if memo_k in _PLATOON_MEMO:
        return _PLATOON_MEMO[memo_k]

    side = pa[pa["p_throws"] == ("R" if vs_rhp else "L")]
    b = side[side["batter"] == batter_id]

    def pack(year_df: pd.DataFrame) -> dict:
        if year_df.empty:
            return {}
        n = len(year_df)
        h = int(year_df["is_hit"].sum())
        ab = _ab_den(year_df)
        avg_ev = None
        if "launch_speed" in year_df.columns:
            evs = pd.to_numeric(year_df["launch_speed"], errors="coerce").dropna()
            if len(evs) >= 1:
                avg_ev = round(float(evs.mean()), 1)
        return {
            "pa": n,
            "h": h,
            "avg": round(h / ab, 3),
            "k_pct": round(100 * int(year_df["is_k"].sum()) / n, 1),
            "bb_pct": round(100 * int(year_df["is_bb"].sum()) / n, 1),
            "xbh": int(year_df["is_xbh"].sum()),
            "hr": int(year_df["is_hr"].sum()),
            "xbh_pct": round(100 * int(year_df["is_xbh"].sum()) / n, 1),
            "hr_pct": round(100 * int(year_df["is_hr"].sum()) / n, 1),
            "doubles": int((year_df["events"] == "double").sum()),
            "triples": int((year_df["events"] == "triple").sum()),
            "avg_ev": avg_ev,
        }

    y = b[b["game_year"] == 2026]
    out = (pack(y), pack(b))
    _PLATOON_MEMO[memo_k] = out
    return out


def fmt_platoon_hit(d: dict, hand_lbl: str) -> str:
    if not d:
        return f"No Statcast vs {hand_lbl} in file."
    return (
        f"{d['pa']} PA · {d['h']} H · AVG {d['avg']:.3f} · "
        f"{d['k_pct']:.1f}% K/PA · {d['bb_pct']:.1f}% BB/PA"
    )


def fmt_platoon_xbh(d: dict, hand_lbl: str) -> str:
    if not d:
        return f"No Statcast vs {hand_lbl} in file."
    return (
        f"{d['pa']} PA · {d['xbh']} XBH ({d['xbh_pct']:.1f}%) · "
        f"2B {d['doubles']} · 3B {d['triples']} · HR {d['hr']} · "
        f"{d['h']} H · AVG {d['avg']:.3f} · {d['k_pct']:.1f}% K/PA"
    )


def fmt_platoon_hr(d: dict, hand_lbl: str) -> str:
    if not d:
        return f"No Statcast vs {hand_lbl} in file."
    d3 = d["doubles"] + d["triples"]
    return (
        f"{d['pa']} PA · {d['hr']} HR ({d['hr_pct']:.1f}%) · "
        f"{d['xbh']} XBH ({d['xbh_pct']:.1f}%) · 2B+3B {d3} · "
        f"AVG {d['avg']:.3f} · {d['k_pct']:.1f}% K/PA"
    )


def fmt_bvp(m: dict) -> str:
    r = m.get("bvp_career_vs_pitcher")
    if not r or not r.get("bvp_pa"):
        return "No BvP"
    pa = int(r["bvp_pa"])
    h = int(r["bvp_hits"])
    k = int(r["bvp_k"])
    ba = float(r["bvp_ba"])
    return f"{pa} PA · {h}-for-{pa} ({ba:.3f}) · {k} K"


def _conf_pack(m: dict, target: str) -> tuple[float, str]:
    if target == "hit":
        return float(m.get("conf_hit") or 0), str(m.get("conf_hit_label") or "").strip()
    if target == "xbh":
        return float(m.get("conf_xbh") or 0), str(m.get("conf_xbh_label") or "").strip()
    return float(m.get("conf_hr") or 0), str(m.get("conf_hr_label") or "").strip()


def _adj_sort_key(m: dict, target: str) -> float:
    """Sort by raw per-PA P (post posterior); fall back to calibrated then score."""
    raw_k = "p_hit" if target == "hit" else ("p_xbh" if target == "xbh" else "p_hr")
    v = m.get(raw_k)
    if v is not None and v != "":
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    cal_k = "p_hit_calibrated" if target == "hit" else ("p_xbh_calibrated" if target == "xbh" else "p_hr_calibrated")
    try:
        return float(m.get(cal_k) or 0)
    except (TypeError, ValueError):
        pass
    k = "adj_p_hit" if target == "hit" else ("adj_p_xbh" if target == "xbh" else "adj_p_hr")
    try:
        return float(m.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def _plain_readme(
    target: str,
    hand_lbl: str,
    cal: float,
    raw: float,
    y26: dict,
    car: dict,
    *,
    conf: float,
    conf_label: str,
    bvp: dict | None,
    pitcher_name: str,
) -> str:
    """Plain-language column: short clauses (bat vs arm, EV, career/2026, BvP, trust)."""
    arm = str(hand_lbl or "RHP").strip().upper()
    if arm not in {"LHP", "RHP"}:
        arm = "RHP" if arm.startswith("R") else "LHP"
    pn = (pitcher_name or "").strip() or "this pitcher"
    parts: list[str] = []

    if target == "hit":
        parts.append(f"Bat vs {arm}: tonight hit chance about {cal:.0%} (model raw {raw:.0%}).")
    elif target == "xbh":
        parts.append(f"Bat vs {arm}: tonight XBH chance about {cal:.0%} (model raw {raw:.0%}).")
    else:
        parts.append(f"Bat vs {arm}: tonight HR chance about {cal:.0%} (model raw {raw:.0%}).")

    if car.get("pa", 0) >= 40:
        parts.append(f"Career vs {arm}: AVG {car['avg']:.3f}, K% {car['k_pct']:.1f}% in {car['pa']} PA.")
        if target == "hr":
            parts.append(f"Career vs {arm}: HR% {float(car['hr_pct']):.1f}%.")
        if target == "xbh":
            parts.append(f"Career vs {arm}: XBH% {float(car['xbh_pct']):.1f}%.")
        evc = car.get("avg_ev")
        if evc is not None:
            parts.append(f"Career avg EV vs {arm}: {evc:.1f} mph (tracked batted balls).")
    elif car.get("pa", 0) >= 1:
        parts.append(f"Career vs {arm}: only {car['pa']} PA in file — treat slash/EV as a small sample.")

    if y26.get("pa", 0) >= 8:
        yev = y26.get("avg_ev")
        ev_y = f", avg EV {yev:.1f} mph" if yev is not None else ""
        parts.append(f"2026 vs {arm}: AVG {y26['avg']:.3f} in {y26['pa']} PA{ev_y}.")
        if target == "hr" and y26.get("pa", 0) >= 20:
            parts.append(f"2026 vs {arm}: HR% {float(y26['hr_pct']):.1f}%.")
        if target == "xbh" and y26.get("pa", 0) >= 15:
            parts.append(f"2026 vs {arm}: XBH% {float(y26['xbh_pct']):.1f}%.")

    bv = bvp or {}
    bpa = int(bv.get("bvp_pa") or 0)
    if bpa >= 1:
        bhits = int(bv.get("bvp_hits") or 0)
        bks = int(bv.get("bvp_k") or 0)
        try:
            ba_txt = f"{float(bv.get('bvp_ba')):.3f}"
        except (TypeError, ValueError):
            ba_txt = "—"
        parts.append(f"Vs {pn} (career BvP): {bhits}-for-{bpa}, BA {ba_txt}, {bks} K.")

    if conf <= 0.58:
        parts.append(f"Model trust {conf:.2f} ({conf_label or 'Low'}) — noisier read.")
    elif conf >= 0.78:
        parts.append(f"Model trust {conf:.2f} ({conf_label or 'High'}) — steadier read.")

    return " ".join(parts)


def bet_hit(m: dict, y26: dict, car: dict, hand_lbl: str) -> tuple[str, str]:
    raw, cal = float(m["p_hit"]), float(m["p_hit_calibrated"])
    cr = float(m.get("career_hit") or LEAGUE_HIT)
    conf = float(m.get("conf_hit") or 0.7)
    label = str(m.get("conf_hit_label") or "")
    bvp = m.get("bvp_career_vs_pitcher") or {}
    bpa = int(bvp.get("bvp_pa") or 0)
    bba = float(bvp["bvp_ba"]) if bpa and bvp.get("bvp_ba") is not None else None

    stat: list[str] = []
    if cal >= LEAGUE_HIT + 0.018:
        stat.append(f"Cal P(hit) **{cal:.1%}** vs league-ish **{LEAGUE_HIT:.1%}** baseline.")
    elif cal <= LEAGUE_HIT - 0.018:
        stat.append(f"Cal P(hit) **{cal:.1%}** sits **below** **{LEAGUE_HIT:.1%}** league baseline.")

    if raw > cr + 0.025:
        stat.append(f"Raw **{raw:.1%}** > career hit rate **{cr:.1%}**.")
    elif raw + 0.025 < cr:
        stat.append(f"Raw **{raw:.1%}** < career **{cr:.1%}**.")

    if cal > raw * 1.04:
        stat.append(f"Isotonic **lifts** raw (**{raw:.1%}** → **{cal:.1%}**).")
    elif cal < raw * 0.93:
        stat.append(f"Isotonic **trims** raw (**{raw:.1%}** → **{cal:.1%}**).")

    if conf >= 0.78:
        stat.append(f"Hit confidence **{conf:.2f}** ({label or 'High'}).")
    elif conf <= 0.58:
        stat.append(f"Hit confidence **{conf:.2f}** ({label or 'Low'}).")

    if y26.get("pa", 0) >= 25:
        if y26["avg"] >= 0.28:
            stat.append(f"2026 vs this hand: **{y26['avg']:.3f}** AVG over **{y26['pa']}** PA.")
        elif y26["avg"] <= 0.19:
            stat.append(f"2026 vs this hand: **{y26['avg']:.3f}** AVG (**{y26['pa']}** PA).")

    if bpa >= 12 and bba is not None:
        stat.append(f"BvP **{bpa}** PA at **{bba:.3f}** vs this exact pitcher.")
    elif bpa in range(1, 12):
        stat.append(f"BvP only **{bpa}** PA — high variance.")

    if not stat:
        stat.append(f"Neutral profile: Cal **{cal:.1%}**, raw **{raw:.1%}**, career hit **{cr:.1%}**.")

    plain = _plain_readme(
        "hit",
        hand_lbl,
        cal,
        raw,
        y26,
        car,
        conf=conf,
        conf_label=label,
        bvp=bvp,
        pitcher_name=str(m.get("pitcher_name") or ""),
    )
    return " ".join(stat), plain


def bet_xbh(m: dict, y26: dict, car: dict, hand_lbl: str) -> tuple[str, str]:
    raw, cal = float(m["p_xbh"]), float(m["p_xbh_calibrated"])
    conf = float(m.get("conf_xbh") or 0.65)
    label = str(m.get("conf_xbh_label") or "")
    bvp = m.get("bvp_career_vs_pitcher") or {}

    stat: list[str] = []
    if cal >= LEAGUE_XBH + 0.012:
        stat.append(f"Cal P(XBH) **{cal:.1%}** vs ~**{LEAGUE_XBH:.1%}** league-ish XBH/PA.")
    elif cal <= LEAGUE_XBH - 0.012:
        stat.append(f"Cal P(XBH) **{cal:.1%}** **under** ~**{LEAGUE_XBH:.1%}** baseline.")

    if y26.get("pa", 0) >= 20:
        stat.append(f"2026 XBH% vs hand: **{y26.get('xbh_pct', 0):.1f}%** (**{y26['pa']}** PA).")

    if car.get("pa", 0) >= 200:
        stat.append(f"Career XBH% vs hand: **{car.get('xbh_pct', 0):.1f}%** over **{car['pa']}** PA.")

    if conf <= 0.55:
        stat.append(f"XBH confidence **{conf:.2f}** ({label or 'Low'}).")

    if not stat:
        stat.append(f"XBH Cal **{cal:.1%}**, raw **{raw:.1%}** — mid board.")

    plain = _plain_readme(
        "xbh",
        hand_lbl,
        cal,
        raw,
        y26,
        car,
        conf=conf,
        conf_label=label,
        bvp=bvp,
        pitcher_name=str(m.get("pitcher_name") or ""),
    )
    return " ".join(stat), plain


def bet_hr(m: dict, y26: dict, car: dict, hand_lbl: str) -> tuple[str, str]:
    raw, cal = float(m["p_hr"]), float(m["p_hr_calibrated"])
    cr = float(m.get("career_hr") or LEAGUE_HR)
    conf = float(m.get("conf_hr") or 0.65)
    label = str(m.get("conf_hr_label") or "")
    bvp = m.get("bvp_career_vs_pitcher") or {}
    bpa = int(bvp.get("bvp_pa") or 0)
    bhr = int(bvp.get("bvp_hr") or 0)

    stat: list[str] = []
    if cal >= LEAGUE_HR + 0.012:
        stat.append(f"Cal P(HR) **{cal:.1%}** vs ~**{LEAGUE_HR:.1%}** league HR/PA.")
    elif cal <= LEAGUE_HR - 0.008:
        stat.append(f"Cal P(HR) **{cal:.1%}** sits **below** **{LEAGUE_HR:.1%}** baseline.")

    if raw > cr + 0.012:
        stat.append(f"Raw **{raw:.1%}** > career HR% **{cr:.1%}**.")
    elif raw + 0.012 < cr:
        stat.append(f"Raw **{raw:.1%}** < career HR% **{cr:.1%}**.")

    if y26.get("pa", 0) >= 30 and y26.get("hr_pct", 0) >= 5:
        stat.append(f"2026 HR% vs hand **{y26['hr_pct']:.1f}%** (**{y26['pa']}** PA).")
    if car.get("pa", 0) >= 300 and car.get("hr_pct", 0) >= 4:
        stat.append(f"Career HR% vs hand **{car['hr_pct']:.1f}%**.")

    if bpa >= 8 and bhr > 0:
        stat.append(f"BvP includes **{bhr}** HR in **{bpa}** PA vs this pitcher.")

    if conf <= 0.55:
        stat.append(f"HR confidence **{conf:.2f}** ({label or 'Low'}).")

    if not stat:
        stat.append(f"HR Cal **{cal:.1%}**, raw **{raw:.1%}**, career HR% **{cr:.1%}**.")

    plain = _plain_readme(
        "hr",
        hand_lbl,
        cal,
        raw,
        y26,
        car,
        conf=conf,
        conf_label=label,
        bvp=bvp,
        pitcher_name=str(m.get("pitcher_name") or ""),
    )
    return " ".join(stat), plain


def row_dict(
    m: dict,
    pa_df: pd.DataFrame | None,
    vs_rhp: bool,
    hand_lbl: str,
    target: str,
    *,
    rank: int,
) -> dict:
    bid = int(m["batter_mlbam_id"])
    y26, car = platoon_pack(pa_df, bid, vs_rhp)
    if target == "hit":
        y_s, c_s = fmt_platoon_hit(y26, hand_lbl), fmt_platoon_hit(car, hand_lbl)
        st, pl = bet_hit(m, y26, car, hand_lbl)
    elif target == "xbh":
        y_s, c_s = fmt_platoon_xbh(y26, hand_lbl), fmt_platoon_xbh(car, hand_lbl)
        st, pl = bet_xbh(m, y26, car, hand_lbl)
    else:
        y_s, c_s = fmt_platoon_hr(y26, hand_lbl), fmt_platoon_hr(car, hand_lbl)
        st, pl = bet_hr(m, y26, car, hand_lbl)

    raw_k = "p_hit" if target == "hit" else ("p_xbh" if target == "xbh" else "p_hr")
    cal_k = raw_k + "_calibrated"
    cf, lab = _conf_pack(m, target)
    pname = str(m.get("batter_name") or "")
    conf_disp = f"{lab} {cf:.3f}".strip() if lab else f"{cf:.3f}"
    return {
        "rank": int(rank),
        "player": pname,
        "player_sort": pname.strip().lower(),
        "raw": round(float(m[raw_k]), 5),
        "cal": round(float(m[cal_k]), 5),
        "conf": round(cf, 5),
        "conf_disp": conf_disp,
        "bvp": fmt_bvp(m),
        "platoon_2026": y_s,
        "platoon_career": c_s,
        "stat_take": st,
        "plain_take": pl,
    }


def build_game_payload(game: dict, preds_by_game: dict[int, list], pa_df: pd.DataFrame | None) -> dict:
    gpk = int(game["game_pk"])
    rows = preds_by_game.get(gpk, [])
    away, home = game["away_team"], game["home_team"]
    apid = int(game.get("away_pitcher_id") or 0)
    hpid = int(game.get("home_pitcher_id") or 0)
    apn = game.get("away_pitcher_name", "TBD")
    hpn = game.get("home_pitcher_name", "TBD")

    def throws(pid: int, name: str) -> bool:
        for r in rows:
            if int(r.get("pitcher_mlbam_id") or 0) == pid:
                return str(r.get("pitcher_throws", "R")).upper().startswith("R")
        return True

    home_rhp = throws(hpid, hpn)
    away_rhp = throws(apid, apn)
    hand_home = "RHP" if home_rhp else "LHP"
    hand_away = "RHP" if away_rhp else "LHP"

    def collect(pid: int):
        return [m for m in rows if int(m.get("pitcher_mlbam_id") or 0) == pid]

    away_vs_home = collect(hpid)
    home_vs_away = collect(apid)

    def sort_rows(lst: list, target: str):
        return sorted(lst, key=lambda x: -_adj_sort_key(x, target))

    def build_table(title: str, lst: list, target: str, vs_rhp: bool, hand_lbl: str):
        out_rows = []
        for idx, m in enumerate(sort_rows(lst, target), start=1):
            out_rows.append(row_dict(m, pa_df, vs_rhp, hand_lbl, target, rank=idx))
        col_y = f"2026 vs {hand_lbl[0]}HP"
        col_c = f"Career vs {hand_lbl[0]}HP"
        return {
            "title": title,
            "target": target,
            "col_2026": col_y,
            "col_career": col_c,
            "rows": out_rows,
        }

    tables = []
    if away_vs_home:
        tables.append(build_table(f"{away} @ {home} — {away} vs {hpn} ({hand_home}) — Hits",
                                  away_vs_home, "hit", home_rhp, hand_home))
        tables.append(build_table(f"{away} @ {home} — {away} vs {hpn} ({hand_home}) — XBH",
                                  away_vs_home, "xbh", home_rhp, hand_home))
        tables.append(build_table(f"{away} @ {home} — {away} vs {hpn} ({hand_home}) — HR",
                                  away_vs_home, "hr", home_rhp, hand_home))
    if home_vs_away:
        tables.append(build_table(f"{away} @ {home} — {home} vs {apn} ({hand_away}) — Hits",
                                  home_vs_away, "hit", away_rhp, hand_away))
        tables.append(build_table(f"{away} @ {home} — {home} vs {apn} ({hand_away}) — XBH",
                                  home_vs_away, "xbh", away_rhp, hand_away))
        tables.append(build_table(f"{away} @ {home} — {home} vs {apn} ({hand_away}) — HR",
                                  home_vs_away, "hr", away_rhp, hand_away))

    gd = str(game.get("game_date") or "").strip()[:10] or None
    gdt_utc = game.get("game_datetime_utc")
    time_et = _fmt_game_time_et(gdt_utc)
    return {
        "game_pk": gpk,
        "game_date": gd,
        "game_datetime_utc": str(gdt_utc).strip() if gdt_utc else None,
        "game_time_display": time_et,
        "label": f"{away} @ {home}",
        "subtitle": f"{apn} ({away}) vs {hpn} ({home}) · game_pk {gpk}"
        + (f" · {time_et}" if time_et else ""),
        "tables": tables,
    }


def _default_slate_date(games: list[dict], default_pk: int) -> str:
    for p in games:
        if int(p.get("game_pk") or 0) == int(default_pk):
            d = p.get("game_date")
            return str(d).strip()[:10] if d else ""
    if games:
        d0 = games[0].get("game_date")
        return str(d0).strip()[:10] if d0 else ""
    return ""


def render_html(games: list[dict], default_pk: int) -> str:
    default_sd = _default_slate_date(games, default_pk)
    payload = json.dumps(
        {"games": games, "defaultGamePk": default_pk, "defaultSlateDate": default_sd},
        ensure_ascii=False,
    )
    payload_esc = html.escape(payload, quote=True)
    # embed as text — we'll use JSON.parse on a script text node without escaping issues
    # Safer: base64 or put JSON in script type="application/json" without extra escaping
    payload_b64 = __import__("base64").b64encode(payload.encode("utf-8")).decode("ascii")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Matchup betting board</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3b82f6;
      --border: #2d3a4d;
      --good: #22c55e;
      --bad: #f87171;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.45; font-size: 14px;
    }}
    header {{
      padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
      background: var(--panel); position: sticky; top: 0; z-index: 20;
    }}
    h1 {{ margin: 0 0 0.35rem 0; font-size: 1.15rem; font-weight: 600; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: flex-end; }}
    .control-block {{ display: flex; flex-direction: column; gap: 0.2rem; }}
    label {{ color: var(--muted); font-size: 0.8rem; }}
    select {{
      background: var(--bg); color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; padding: 0.45rem 0.65rem; min-width: 280px; font-size: 0.9rem;
    }}
    #dateFilter {{ min-width: 160px; }}
    main {{ padding: 1rem 1.25rem 2.5rem; max-width: 100%; }}
    .game-meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 1.25rem; }}
    details.board {{
      margin-bottom: 1rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    details.board summary.board-summary {{
      cursor: pointer;
      font-size: 1rem;
      margin: 0;
      padding: 0.65rem 0.85rem;
      color: var(--accent);
      font-weight: 600;
      list-style: none;
      user-select: none;
    }}
    details.board summary.board-summary::-webkit-details-marker {{ display: none; }}
    details.board summary.board-summary::marker {{ content: ""; }}
    details.board summary.board-summary:hover {{ filter: brightness(1.08); }}
    details.board .wrap {{
      border: none;
      border-radius: 0;
      border-top: 1px solid var(--border);
    }}
    .wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }}
    table {{
      width: 100%; border-collapse: collapse; min-width: 1100px;
      background: var(--panel);
    }}
    th, td {{
      border-bottom: 1px solid var(--border); padding: 0.5rem 0.6rem;
      text-align: left; vertical-align: top;
    }}
    th {{
      background: #243044; font-weight: 600; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.04em; color: var(--muted); white-space: nowrap;
    }}
    table.board-sortable thead th {{
      cursor: pointer;
      user-select: none;
    }}
    table.board-sortable thead th:hover {{ color: #c7d2fe; }}
    table.board-sortable thead th.sort-asc::after {{
      content: " ▲"; font-size: 0.65em; opacity: 0.85;
    }}
    table.board-sortable thead th.sort-desc::after {{
      content: " ▼"; font-size: 0.65em; opacity: 0.85;
    }}
    tr:hover td {{ background: rgba(59, 130, 246, 0.06); }}
    td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
    td.small {{ font-size: 0.78rem; color: #c5d0e0; max-width: 320px; }}
    td.take {{ font-size: 0.78rem; max-width: 340px; }}
    td.take strong {{ color: #fbbf24; }}
    .empty {{ color: var(--muted); padding: 1rem; }}
    footer {{ padding: 1rem; color: var(--muted); font-size: 0.75rem; border-top: 1px solid var(--border); }}
  </style>
</head>
<body>
  <header>
    <h1>Matchup betting board</h1>
    <div class="controls">
      <div class="control-block">
        <label for="dateFilter">Slate date</label>
        <select id="dateFilter" aria-label="Filter games by slate date"></select>
      </div>
      <div class="control-block">
        <label for="gameSelect">Game</label>
        <select id="gameSelect" aria-label="Select game"></select>
      </div>
    </div>
  </header>
  <main id="main"></main>
  <footer>
    Regenerate: <code>PYTHONPATH=src python3 src/gen_matchup_betting_board_html.py</code>
    · Platoon columns use Statcast <code>p_throws</code> vs batter (2026 / career).
    · “Stat take” / “Plain” are heuristic blurbs — not picks advice.
  </footer>
  <script>
    const _raw = atob("{payload_b64}");
    const DATA = JSON.parse(_raw);
    const allGames = (DATA.games || []).slice();
    const defaultPk = DATA.defaultGamePk;
    const defaultSlateDate = String(DATA.defaultSlateDate || "").slice(0, 10);

    const dateSel = document.getElementById("dateFilter");
    const sel = document.getElementById("gameSelect");
    const main = document.getElementById("main");

    function gameDate(g) {{
      return String(g && g.game_date != null ? g.game_date : "").slice(0, 10);
    }}

    function uniqueSlateDates() {{
      const s = new Set();
      allGames.forEach(function (g) {{
        const d = gameDate(g);
        if (d) s.add(d);
      }});
      return Array.from(s).sort();
    }}

    let visibleGames = allGames.slice();

    function applyDateFilter() {{
      const v = (dateSel && dateSel.value) ? dateSel.value : "";
      if (!v) {{
        visibleGames = allGames.slice();
      }} else {{
        visibleGames = allGames.filter(function (g) {{ return gameDate(g) === v; }});
      }}
      rebuildGameSelect();
    }}

    function rebuildGameSelect() {{
      const prev = parseInt(String(sel.value || "0"), 10) || 0;
      sel.innerHTML = "";
      const list = visibleGames.slice().sort(function (a, b) {{
        const ta = String(a.game_datetime_utc || "");
        const tb = String(b.game_datetime_utc || "");
        if (ta && tb && ta !== tb) {{
          return ta.localeCompare(tb);
        }}
        return (a.label || "").localeCompare(b.label || "");
      }});
      for (let i = 0; i < list.length; i++) {{
        const g = list[i];
        const o = document.createElement("option");
        o.value = String(g.game_pk);
        const ttime = g.game_time_display ? (" · " + g.game_time_display) : "";
        o.textContent = (g.label || "?") + ttime + " (" + g.game_pk + ")";
        sel.appendChild(o);
      }}
      let pick = 0;
      if (list.some(function (g) {{ return g.game_pk === prev; }})) {{
        pick = prev;
      }} else if (list.some(function (g) {{ return g.game_pk === defaultPk; }})) {{
        pick = defaultPk;
      }} else if (list.length) {{
        pick = list[0].game_pk;
      }}
      sel.value = String(pick);
      if (pick) {{
        renderGame(pick);
      }} else {{
        main.innerHTML = '<p class="empty">No games for this date.</p>';
      }}
    }}

    (function initDateFilter() {{
      const dates = uniqueSlateDates();
      dateSel.innerHTML = "";
      if (!dates.length) {{
        const o = document.createElement("option");
        o.value = "";
        o.textContent = "All games (no game_date in matchups)";
        dateSel.appendChild(o);
      }} else if (dates.length > 1) {{
        const o0 = document.createElement("option");
        o0.value = "";
        o0.textContent = "All dates";
        dateSel.appendChild(o0);
        for (let i = 0; i < dates.length; i++) {{
          const ox = document.createElement("option");
          ox.value = dates[i];
          ox.textContent = dates[i];
          dateSel.appendChild(ox);
        }}
      }} else {{
        const ox = document.createElement("option");
        ox.value = dates[0];
        ox.textContent = dates[0];
        dateSel.appendChild(ox);
      }}
      if (dates.length === 1) {{
        dateSel.value = dates[0];
      }} else if (defaultSlateDate && dates.indexOf(defaultSlateDate) >= 0) {{
        dateSel.value = defaultSlateDate;
      }} else {{
        dateSel.value = "";
      }}
      dateSel.addEventListener("change", applyDateFilter);
    }})();

    function pct(x) {{
      return (100 * x).toFixed(2) + "%";
    }}

    function esc(s) {{
      const d = document.createElement("div");
      d.textContent = s;
      return d.innerHTML;
    }}

    /** Allow light markdown **bold** in stat_take strings */
    function fmtTake(htmlish) {{
      return esc(htmlish).replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
    }}

    function attrEsc(s) {{
      return String(s || "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;");
    }}

    function cellSortVal(tr, colIdx) {{
      const c = tr.cells[colIdx];
      if (!c) return "";
      const ds = c.getAttribute("data-sort-value");
      if (ds !== null && ds !== "") return ds;
      return (c.innerText || "").trim();
    }}

    function detectSortMode(rows, colIdx) {{
      const texts = rows.map((tr) => cellSortVal(tr, colIdx)).filter((t) => t.length);
      if (!texts.length) return "str";
      function strip(s) {{ return s.replace(/\\s/g, ""); }}
      if (texts.every((t) => /^\\d+$/.test(strip(t)))) return "int";
      if (texts.every((t) => /^-?\\d+\\.?\\d*%?$/.test(strip(t)))) return "num";
      const heads = texts.map((t) => {{
        const m = t.match(/-?\\d+\\.?\\d*/);
        return m ? parseFloat(m[0]) : NaN;
      }});
      if (heads.every((n) => !isNaN(n))) return "headnum";
      return "str";
    }}

    function sortKey(text, mode) {{
      if (mode === "int") return parseInt(text.replace(/\\s/g, ""), 10) || 0;
      if (mode === "num") return parseFloat(text.replace(/%/g, "").replace(/\\s/g, "")) || 0;
      if (mode === "headnum") {{
        const m = text.match(/-?\\d+\\.?\\d*/);
        return m ? parseFloat(m[0]) : 0;
      }}
      return (text || "").toLowerCase();
    }}

    function wireBoardSortOnce(root) {{
      if (root.dataset.boardSortBound) return;
      root.dataset.boardSortBound = "1";
      root.addEventListener("click", function (ev) {{
        const th = ev.target.closest("thead th");
        if (!th) return;
        const table = th.closest("table.board-sortable");
        if (!table || !root.contains(table)) return;
        const tbody = table.tBodies[0];
        if (!tbody) return;
        const colIdx = th.cellIndex;
        if (colIdx < 0) return;
        const rows = Array.prototype.slice.call(tbody.rows);
        if (!rows.length) return;
        if (rows.length === 1 && rows[0].cells.length <= 1) return;
        const mode = detectSortMode(rows, colIdx);
        const sameCol = table.dataset.sortCol === String(colIdx);
        const nextDir = sameCol && table.dataset.sortDir === "asc" ? "desc" : "asc";
        const dir = nextDir === "asc" ? 1 : -1;
        table.dataset.sortCol = String(colIdx);
        table.dataset.sortDir = nextDir;
        const headRow = th.parentNode;
        if (headRow) {{
          headRow.querySelectorAll("th").forEach((h) => h.classList.remove("sort-asc", "sort-desc"));
        }}
        th.classList.add(nextDir === "asc" ? "sort-asc" : "sort-desc");
        rows.sort((a, b) => {{
          const ta = cellSortVal(a, colIdx);
          const tb = cellSortVal(b, colIdx);
          const va = sortKey(ta, mode);
          const vb = sortKey(tb, mode);
          let cmp = 0;
          if (mode === "str") {{
            cmp = String(va).localeCompare(String(vb));
          }} else if (va === vb) {{
            cmp = 0;
          }} else {{
            cmp = va < vb ? -1 : 1;
          }}
          return dir * cmp;
        }});
        rows.forEach((tr) => tbody.appendChild(tr));
      }});
    }}

    function renderGame(gpk) {{
      const g = visibleGames.find((x) => x.game_pk === gpk);
      if (!g) {{
        main.innerHTML = '<p class="empty">No data for this game.</p>';
        return;
      }}
      let html = `<div class="game-meta">${{esc(g.subtitle)}}</div>`;
      for (const tbl of g.tables) {{
        html += `<details class="board"><summary class="board-summary">${{esc(tbl.title)}}</summary><div class="wrap"><table class="board-sortable"><thead><tr>`;
        html += '<th title="Model rank (1 = best on this table). Stays fixed when you sort other columns.">#</th>'
        html += '<th title="Click to sort">Player</th><th title="Click to sort">Conf</th>';
        html += '<th title="Click to sort">Raw</th><th title="Click to sort">Cal</th><th title="Click to sort">BvP</th>';
        html += `<th title="Click to sort">${{esc(tbl.col_2026 || "2026 vs hand")}}</th>`;
        html += `<th title="Click to sort">${{esc(tbl.col_career || "Career vs hand")}}</th>`;
        html += '<th title="Click to sort">Stat take</th><th title="Click to sort">Plain language</th></tr></thead><tbody>';
        if (!tbl.rows.length) {{
          html += '<tr><td colspan="10" class="empty">No rows</td></tr>';
        }} else {{
          tbl.rows.forEach((r, i) => {{
            const ps = r.player_sort != null ? r.player_sort : String(r.player || "").toLowerCase();
            const rk = r.rank != null ? Number(r.rank) : i + 1;
            html += "<tr>";
            html += `<td class="num" data-sort-value="${{rk}}">${{rk}}</td>`;
            html += `<td data-sort-value="${{attrEsc(ps)}}">${{esc(r.player)}}</td>`;
            html += `<td class="num" data-sort-value="${{r.conf}}">${{esc(r.conf_disp || String(r.conf))}}</td>`;
            html += `<td class="num" data-sort-value="${{r.raw}}">${{pct(r.raw)}}</td>`;
            html += `<td class="num" data-sort-value="${{r.cal}}">${{pct(r.cal)}}</td>`;
            html += `<td class="small" data-sort-value="${{attrEsc(r.bvp)}}">${{esc(r.bvp)}}</td>`;
            html += `<td class="small" data-sort-value="${{attrEsc(r.platoon_2026)}}">${{esc(r.platoon_2026)}}</td>`;
            html += `<td class="small" data-sort-value="${{attrEsc(r.platoon_career)}}">${{esc(r.platoon_career)}}</td>`;
            html += `<td class="take" data-sort-value="${{attrEsc(r.stat_take)}}">${{fmtTake(r.stat_take)}}</td>`;
            html += `<td class="take" data-sort-value="${{attrEsc(r.plain_take)}}">${{esc(r.plain_take)}}</td>`;
            html += "</tr>";
          }});
        }}
        html += "</tbody></table></div></details>";
      }}
      main.innerHTML = html;
      wireBoardSortOnce(main);
    }}

    applyDateFilter();

    sel.addEventListener("change", () => {{
      renderGame(parseInt(sel.value, 10));
    }});
  </script>
</body>
</html>
"""


def _default_game_pk(games_json: list, preds_by_game: dict[int, list]) -> int:
    """Prefer earliest Cubs (CHC) game by first pitch; else earliest game on slate."""
    scored: list[tuple[str, int]] = []
    chc_scored: list[tuple[str, int]] = []
    for g in games_json:
        gpk = int(g.get("game_pk") or 0)
        if gpk not in preds_by_game:
            continue
        t = str(g.get("game_datetime_utc") or "")
        scored.append((t, gpk))
        if g.get("home_team") == "CHC" or g.get("away_team") == "CHC":
            chc_scored.append((t, gpk))
    pick = sorted(chc_scored, key=lambda x: x[0]) if chc_scored else sorted(scored, key=lambda x: x[0])
    return pick[0][1] if pick else 0


def main() -> int:
    global _PLATOON_MEMO
    _PLATOON_MEMO = {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, default=PRED_JSON)
    ap.add_argument("--matchups", type=Path, default=MATCHUPS_JSON)
    ap.add_argument("--out", type=Path, default=OUT_HTML)
    ap.add_argument(
        "--default-pk",
        type=int,
        default=None,
        help="Initial game_pk in dropdown; omit for earliest CHC game else earliest slate",
    )
    args = ap.parse_args()

    if not args.pred.exists():
        print(f"Missing {args.pred}", file=sys.stderr)
        return 1
    preds = json.loads(args.pred.read_text(encoding="utf-8"))
    preds_by_game: dict[int, list] = {}
    batter_ids: set[int] = set()
    for m in preds:
        gpk = int(m.get("game_pk") or 0)
        if gpk <= 0:
            continue
        preds_by_game.setdefault(gpk, []).append(m)
        try:
            batter_ids.add(int(m["batter_mlbam_id"]))
        except (KeyError, TypeError, ValueError):
            pass

    games_json = []
    if args.matchups.exists():
        games_json = json.loads(args.matchups.read_text(encoding="utf-8"))

    pa_df = _load_platoon_frame()
    if pa_df is not None and batter_ids:
        pa_df = pa_df[pa_df["batter"].isin(batter_ids)]
    payloads = []
    default_pk = args.default_pk if args.default_pk is not None else _default_game_pk(
        games_json, preds_by_game
    )
    for game in games_json:
        gpk = int(game.get("game_pk") or 0)
        if gpk not in preds_by_game:
            continue
        payloads.append(build_game_payload(game, preds_by_game, pa_df))

    if not payloads:
        print("No games with predictions; abort.", file=sys.stderr)
        return 1

    # ensure default exists
    if not any(p["game_pk"] == default_pk for p in payloads):
        default_pk = payloads[0]["game_pk"]

    html_out = render_html(payloads, default_pk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.out} ({len(payloads)} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
