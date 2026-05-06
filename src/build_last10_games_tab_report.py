#!/usr/bin/env python3
"""
Build a tabbed HTML report: last 10 games per batter (game-by-game),
standard + advanced stats from Statcast PA parquet, with rows highlighted
when the (approximate) opposing starter handedness matches tonight's starter.

Tonight's matchup comes from data/reports/todays_matchup_predictions.json.

Opposing starter rule: pitcher on this batter's *first PA of the game*
(min at_bat_number). For regulars who bat from inning 1, that is almost
always the true opposing SP; pinch-only appearances can mislabel.

Data: reads `statcast_pa_level_league.parquet` (bulk baseline) **filtered to
these batters**, then optionally merges **live** pitch rows from pybaseball
`statcast_batter` from the day after the parquet max date through today
(`--no-fetch` skips network).
"""
from __future__ import annotations

import argparse
import json
import html
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_bulk_pa import pitches_to_pa  # noqa: E402

RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "data" / "reports"
STATCAST_PA = RAW / "statcast_pa_level_league.parquet"
PREDICTIONS = REPORTS / "todays_matchup_predictions.json"
OUT_HTML = REPORTS / "last10_games_selected_batters.html"
OUT_HTML_LEGACY = REPORTS / "last10_games_six_batters.html"

# MLBAM ids from qualifying_batters_2026.csv / Statcast
BATTERS: list[tuple[str, int]] = [
    ("Matt Olson", 621566),
    ("Andrew Benintendi", 643217),
    ("Jorge Polanco", 593871),
    ("Julio Rodríguez", 677594),
    ("Aaron Judge", 592450),
    ("Yordan Alvarez", 670541),
    ("Eugenio Suárez", 553993),
    ("Spencer Torkelson", 679529),
]


def strip_accents(s: str) -> str:
    nk = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nk if not unicodedata.combining(c))


def norm_key(name: str) -> str:
    return strip_accents(name).lower().strip()


def parquet_max_game_date(path: Path) -> date:
    s = pd.read_parquet(path, columns=["game_date"], engine="pyarrow")
    t = pd.to_datetime(s["game_date"], errors="coerce").max()
    if pd.isna(t):
        return date.today()
    return pd.Timestamp(t).normalize().date()


def batter_id_list() -> list[int]:
    return [bid for _, bid in BATTERS]


def load_base_pa_for_batters(path: Path, batter_ids: list[int]) -> pd.DataFrame:
    return pd.read_parquet(path, filters=[("batter", "in", batter_ids)], engine="pyarrow")


def fetch_incremental_pas(batter_ids: list[int], start: date, end: date) -> pd.DataFrame:
    from pybaseball import cache, statcast_batter

    cache.enable()
    if start > end:
        return pd.DataFrame()
    sd, ed = start.isoformat(), end.isoformat()
    frames: list[pd.DataFrame] = []
    for bid in batter_ids:
        try:
            raw = statcast_batter(sd, ed, bid)
        except Exception as ex:
            print(f"  statcast_batter({bid}) {sd}..{ed}: {ex}")
            time.sleep(0.45)
            continue
        if raw is not None and not raw.empty:
            pa = pitches_to_pa(raw)
            if not pa.empty:
                frames.append(pa)
                print(f"  batter {bid}: +{len(pa)} PAs ({sd} to {ed})")
        time.sleep(0.45)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def align_to_league_columns(pa: pd.DataFrame, league_cols: list[str]) -> pd.DataFrame:
    out = pa.copy()
    for c in league_cols:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in league_cols]
    if extras:
        out = out.drop(columns=extras)
    return out[league_cols]


def build_working_frame(no_fetch: bool) -> tuple[pd.DataFrame, str, str]:
    """PA-level rows for BATTERS only; (df, html_note_suffix, data_through_date)."""
    import pyarrow.parquet as pq

    batter_ids = batter_id_list()
    league_cols = list(pq.ParquetFile(STATCAST_PA).schema_arrow.names)

    base = load_base_pa_for_batters(STATCAST_PA, batter_ids)
    if base.empty:
        return base, " <strong>No rows</strong> in parquet for these batter IDs.", ""

    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")
    max_ts = base["game_date"].max()
    max_base_d = max_ts.normalize().date() if pd.notna(max_ts) else parquet_max_game_date(STATCAST_PA)

    today = date.today()
    if no_fetch or today <= max_base_d:
        data_through = str(max_base_d)
        return base, "", data_through

    inc_start = max_base_d + timedelta(days=1)
    fresh_raw = fetch_incremental_pas(batter_ids, inc_start, today)
    if fresh_raw.empty:
        data_through = str(max_base_d)
        suffix = (
            f" Live Statcast pull for <strong>{inc_start}</strong>–<strong>{today}</strong> returned no new PAs "
            "(offline, API error, or no games in range)."
        )
        return base, suffix, data_through

    fresh = align_to_league_columns(fresh_raw, league_cols)
    fresh["game_date"] = pd.to_datetime(fresh["game_date"], errors="coerce")
    merged = pd.concat([base, fresh], ignore_index=True)
    merged = merged.drop_duplicates(subset=["game_pk", "at_bat_number", "batter"], keep="last")
    merged = merged.sort_values(["batter", "game_date", "at_bat_number"], kind="mergesort")
    max_m = merged["game_date"].max().date()
    data_through = str(max_m)
    suffix = (
        f" Includes <strong>live Statcast</strong> (<code>statcast_batter</code>) for these batters "
        f"from <strong>{inc_start}</strong> through <strong>{today}</strong>, merged on top of the parquet "
        f"(baseline last date <strong>{max_base_d}</strong>)."
    )
    return merged, suffix, data_through


def load_tonight_pitchers(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data:
        bn = row.get("batter_name")
        if not bn:
            continue
        k = norm_key(bn)
        if k not in out:
            out[k] = {
                "pitcher_name": row.get("pitcher_name", ""),
                "pitcher_throws": (row.get("pitcher_throws") or "R").upper()[:1],
                "pitcher_team": row.get("pitcher_team", ""),
                "batter_team": row.get("batter_team", ""),
            }
    return out


def mlb_people_names(ids: set[int]) -> dict[int, tuple[str, str]]:
    """Return mlbam -> (fullName, pitchHand code L/R/S for pitchers)."""
    mapping: dict[int, tuple[str, str]] = {}
    ids_list = sorted(ids)
    for i in range(0, len(ids_list), 45):
        chunk = ids_list[i : i + 45]
        url = "https://statsapi.mlb.com/api/v1/people"
        r = requests.get(url, params={"personIds": ",".join(map(str, chunk))}, timeout=30)
        r.raise_for_status()
        for p in r.json().get("people", []):
            pid = int(p["id"])
            name = p.get("fullName", str(pid))
            ph = (p.get("pitchHand") or {}).get("code", "R")
            mapping[pid] = (name, ph)
    return mapping


def is_ab_event(e: str) -> bool:
    if not isinstance(e, str):
        return False
    if e in ("walk", "intent_walk", "hit_by_pitch", "sac_bunt", "catcher_interf"):
        return False
    if e.startswith("sac_fly"):
        return False
    return True


def aggregate_game(g: pd.DataFrame) -> dict[str, Any]:
    e = g["events"].astype(str)
    PA = len(g)
    BB = e.isin(["walk", "intent_walk"]).sum()
    HBP = (e == "hit_by_pitch").sum()
    SF = e.str.startswith("sac_fly", na=False).sum()
    SH = e.str.startswith("sac_bunt", na=False).sum()
    AB_mask = e.map(is_ab_event)
    AB = int(AB_mask.sum())

    doubles = (e == "double").sum()
    triples = (e == "triple").sum()
    hr = (e == "home_run").sum()
    singles = (e == "single").sum()
    H = int(singles + doubles + triples + hr)
    SO = e.str.contains("strikeout", case=False, na=False).sum()

    avg = H / AB if AB else 0.0
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    slg = tb / AB if AB else 0.0
    obp_denom = AB + BB + HBP + SF + SH
    obp = (H + BB + HBP) / obp_denom if obp_denom else 0.0
    ops = obp + slg

    iso = slg - avg
    babip_denom = AB - SO - hr + SF
    babip = (H - hr) / babip_denom if babip_denom > 0 else float("nan")

    wmask = g["woba_denom"].fillna(0) == 1
    woba_num = g.loc[wmask, "woba_value"].fillna(0).sum()
    woba_den = int(wmask.sum())
    woba = woba_num / woba_den if woba_den else float("nan")

    xw = g["estimated_woba_using_speedangle"]
    xwoba = float(xw.mean()) if xw.notna().any() else float("nan")

    barrels = int(g["barrel"].fillna(0).sum())
    brl_pct = barrels / PA if PA else float("nan")

    ev = g["launch_speed"]
    ev_mask = ev.notna()
    ev_mean = float(ev[ev_mask].mean()) if ev_mask.any() else float("nan")
    ev_max = float(ev[ev_mask].max()) if ev_mask.any() else float("nan")
    # Hard-hit rate: 95+ mph among PA ending in a batted-ball event (exclude pure walks/K/HBP).
    non_bip = (
        e.isin(["walk", "intent_walk", "hit_by_pitch", "game_advisory", "truncated_pa", "ejection"])
        | e.str.contains("strikeout", case=False, na=False)
    )
    bip_mask = ~non_bip
    bip_n = int(bip_mask.sum())
    hh = int(((ev >= 95) & bip_mask).sum())
    hh_pct = hh / bip_n if bip_n else float("nan")

    k_pct = SO / PA if PA else float("nan")
    bb_pct = BB / PA if PA else float("nan")

    first = g.sort_values("at_bat_number").iloc[0]
    opp_id = int(first["pitcher"])
    opp_hand = str(first.get("p_throws", "R") or "R").upper()[:1]

    return {
        "PA": PA,
        "AB": AB,
        "H": H,
        "1B": int(singles),
        "2B": int(doubles),
        "3B": int(triples),
        "HR": int(hr),
        "BB": int(BB),
        "SO": int(SO),
        "HBP": int(HBP),
        "SF": int(SF),
        "AVG": avg,
        "OBP": obp,
        "SLG": slg,
        "OPS": ops,
        "ISO": iso,
        "BABIP": babip,
        "wOBA": woba,
        "xwOBA": xwoba,
        "Barrel%": brl_pct,
        "HardHit%": hh_pct,
        "Avg EV": ev_mean,
        "Max EV": ev_max,
        "K%": k_pct,
        "BB%": bb_pct,
        "opp_pitcher_id": opp_id,
        "opp_hand": opp_hand,
    }


def pct(x: float) -> str:
    if x != x:  # NaN
        return "—"
    return f"{100 * x:.1f}%"


def fmt3(x: float) -> str:
    if x != x:
        return "—"
    return f"{x:.3f}"


def fmt1(x: float) -> str:
    if x != x:
        return "—"
    return f"{x:.1f}"


def build_html(
    tonight: dict[str, dict[str, Any]],
    pitcher_names: dict[int, tuple[str, str]],
    tables: dict[str, list[dict[str, Any]]],
    data_through: str,
    data_extra_html: str = "",
) -> str:
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0f1419; color: #e7ecf3; }
    h1 { font-size: 1.25rem; margin: 1rem 1rem 0.5rem; }
    .note { font-size: 0.85rem; color: #9aa7b8; margin: 0 1rem 1rem; max-width: 960px; line-height: 1.45; }
    .tabset { padding: 0 1rem; }
    .tab-buttons { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 0.5rem; border-bottom: 1px solid #2a3444; padding-bottom: 4px; }
    .tab-buttons button { cursor: pointer; padding: 0.5rem 0.85rem; border-radius: 6px 6px 0 0; background: #1a2332; color: #b8c5d6; font-size: 0.9rem; border: none; }
    .tab-buttons button.active { background: #2d3b52; color: #fff; font-weight: 600; }
    .panel { display: none; padding: 0.5rem 0 1rem; overflow-x: auto; }
    .panel.active { display: block; }
    table { border-collapse: collapse; font-size: 0.78rem; min-width: 1100px; }
    th, td { border: 1px solid #2a3444; padding: 6px 8px; text-align: right; }
    th { background: #1a2332; color: #c5d0e0; white-space: nowrap; }
    td.text-left, th.text-left { text-align: left; }
    tr.platoon-match { background: #2a3d28; }
    tr.platoon-match td:first-child { box-shadow: inset 3px 0 0 #6bc46b; }
    .matchpill { display: inline-block; background: #356936; color: #d8ffd8; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .hdr { margin-bottom: 0.75rem; font-size: 0.95rem; color: #c5d0e0; }
    """

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Last 10 games — selected batters</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Last 10 games (Statcast PA file) — per-game splits</h1>",
        f"<p class='note'><strong>Data window:</strong> Games through <strong>{html.escape(data_through or '—')}</strong> for these batters.{data_extra_html} "
        "<strong>“Tonight”</strong> pitcher handedness is taken from "
        "<code>todays_matchup_predictions.json</code>. <strong>Opposing starter</strong> is approximated as the pitcher on this "
        "player’s <em>first plate appearance of that game</em> (same <code>p_throws</code> as Statcast uses for the matchup). "
        "Rows with the green bar match that hand (LHP vs LHP or RHP vs RHP) vs tonight’s starter.</p>",
        "<div class='tabset'>",
        "<div class='tab-buttons' role='tablist'>",
    ]
    for i, (display_name, _) in enumerate(BATTERS):
        active = " active" if i == 0 else ""
        parts.append(
            f"<button type='button' role='tab' class='tab-btn{active}' data-panel='panel{i}' "
            f"aria-selected='{'true' if i == 0 else 'false'}'>{html.escape(display_name)}</button>"
        )
    parts.append("</div>")

    for i, (display_name, _) in enumerate(BATTERS):
        nk = norm_key(display_name)
        t = tonight.get(nk, {})
        throws = t.get("pitcher_throws", "?")
        pname = t.get("pitcher_name", "Unknown")
        pteam = t.get("pitcher_team", "")
        bteam = t.get("batter_team", "")
        tonight_label = f"{html.escape(pname)} ({throws}HP){' — ' + html.escape(pteam) if pteam else ''}"
        hdr = (
            f"<div class='hdr'><strong>Tonight ({html.escape(bteam) if bteam else 'team'})</strong> vs "
            f"<strong>{tonight_label}</strong> — highlight rows where past game’s first pitcher faced is also <strong>{throws}HP</strong>.</div>"
        )

        rows_html = []
        rows = tables.get(display_name, [])
        for r in rows:
            match = r.get("platoon_match")
            trc = " platoon-match" if match else ""
            pill = " <span class='matchpill'>Same hand as tonight</span>" if match else ""
            pid = r["opp_pitcher_id"]
            pnm, _ = pitcher_names.get(pid, (str(pid), "?"))
            hand = r["opp_hand"]
            pname_cell = html.escape(pnm) + f" ({hand}HP)"
            rows_html.append(
                f"<tr class='{trc.strip()}'>"
                f"<td class='text-left'>{html.escape(r['game_date'])}{pill}</td>"
                f"<td class='text-left'>{pname_cell}</td>"
                f"<td>{r['PA']}</td><td>{r['AB']}</td><td>{r['H']}</td><td>{r['2B']}</td><td>{r['3B']}</td><td>{r['HR']}</td>"
                f"<td>{r['BB']}</td><td>{r['SO']}</td><td>{r['HBP']}</td><td>{r['SF']}</td>"
                f"<td>{fmt3(r['AVG'])}</td><td>{fmt3(r['OBP'])}</td><td>{fmt3(r['SLG'])}</td><td>{fmt3(r['OPS'])}</td>"
                f"<td>{fmt3(r['ISO'])}</td><td>{fmt3(r['BABIP'])}</td>"
                f"<td>{fmt3(r['wOBA'])}</td><td>{fmt3(r['xwOBA'])}</td>"
                f"<td>{pct(r['Barrel%'])}</td><td>{pct(r['HardHit%'])}</td>"
                f"<td>{fmt1(r['Avg EV'])}</td>"
                f"<td>{fmt1(r['Max EV'])}</td>"
                f"<td>{pct(r['K%'])}</td><td>{pct(r['BB%'])}</td>"
                f"</tr>"
            )

        thead = (
            "<thead><tr>"
            "<th class='text-left'>Game</th><th class='text-left'>First pitcher faced (SP proxy)</th>"
            "<th>PA</th><th>AB</th><th>H</th><th>2B</th><th>3B</th><th>HR</th><th>BB</th><th>SO</th><th>HBP</th><th>SF</th>"
            "<th>AVG</th><th>OBP</th><th>SLG</th><th>OPS</th><th>ISO</th><th>BABIP</th>"
            "<th>wOBA</th><th>xwOBA</th><th>Barrel%</th><th>HardHit%</th><th>Avg EV</th><th>Max EV</th><th>K%</th><th>BB%</th>"
            "</tr></thead>"
        )

        empty_row = '<tr><td colspan="25">No PA data for this player in the parquet.</td></tr>'
        tbody = "".join(rows_html) if rows_html else empty_row
        pactive = " active" if i == 0 else ""
        parts.append(
            f"<div class='panel{pactive}' id='panel{i}' role='tabpanel'>{hdr}<table>{thead}<tbody>{tbody}</tbody></table></div>"
        )

    parts.append(
        "</div><script>(function(){var b=document.querySelectorAll('.tab-btn'),p=document.querySelectorAll('.tabset .panel');"
        "function show(i){for(var j=0;j<p.length;j++){p[j].classList.toggle('active',j===i);"
        "b[j].classList.toggle('active',j===i);b[j].setAttribute('aria-selected',j===i?'true':'false');}}"
        "b.forEach(function(btn,idx){btn.addEventListener('click',function(){show(idx);});});})();</script></body></html>"
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Last-10-games tabbed HTML for selected batters.")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not call pybaseball Statcast (use parquet slice only).",
    )
    args = parser.parse_args()

    tonight = load_tonight_pitchers(PREDICTIONS)
    df, data_extra_html, data_through = build_working_frame(args.no_fetch)

    all_pitcher_ids: set[int] = set()
    tables: dict[str, list[dict[str, Any]]] = {}

    for display_name, bid in BATTERS:
        b = df[df["batter"] == bid].copy()
        if b.empty:
            tables[display_name] = []
            continue

        games = (
            b.groupby("game_pk", as_index=False)
            .agg(game_date=("game_date", "max"))
            .sort_values("game_date", ascending=False)
        )
        last_games = games.head(10)["game_pk"].tolist()

        nk = norm_key(display_name)
        tinfo = tonight.get(nk, {})
        raw_throw = (tinfo.get("pitcher_throws") or "R").upper()[:1]
        t_throw = raw_throw if raw_throw in ("L", "R") else "R"

        rows_out: list[dict[str, Any]] = []
        for gp in last_games:
            g = b[b["game_pk"] == gp]
            gd = str(g["game_date"].iloc[0])[:10]
            agg = aggregate_game(g)
            opp_hand = agg["opp_hand"]
            platoon_match = opp_hand == t_throw
            all_pitcher_ids.add(int(agg["opp_pitcher_id"]))
            row = {
                "game_date": gd,
                "platoon_match": platoon_match,
                **agg,
            }
            rows_out.append(row)
        tables[display_name] = rows_out

    pitcher_names = mlb_people_names(all_pitcher_ids)
    html_doc = build_html(tonight, pitcher_names, tables, data_through, data_extra_html)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    OUT_HTML_LEGACY.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_HTML_LEGACY} (copy)")


if __name__ == "__main__":
    main()
