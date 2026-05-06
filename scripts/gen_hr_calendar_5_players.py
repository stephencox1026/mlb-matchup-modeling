#!/usr/bin/env python3
"""2026 HR-by-date calendar HTML: five sluggers, Super 6 trios, and a custom trio tab (all qualifying batters)."""
from __future__ import annotations

import calendar
import html as html_mod
import itertools
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PA_PATH = ROOT / "data/raw/statcast_pa_level_league.parquet"
QUAL_BATTERS_CSV = ROOT / "data/raw/qualifying_batters_2026.csv"
MATCHUPS_JSON = ROOT / "data/raw/todays_matchups.json"
BEAST_MATCHUP_JSON = ROOT / "data/reports/todays_matchup_predictions_beast.json"
VAL_BEAST_PRED_PARQUET = ROOT / "data/reports/val_predictions_multi_beast.parquet"
OUT_PATH = ROOT / "data/reports/hr_calendar_5_sluggers_2026.html"

# Inclusive window for calendar cells + trio 0/1/2/3 day-count stats (same range).
STATCAST_WINDOW_START = date(2026, 3, 26)
# Third-wheel suggestions cap (inclusive) for ranking next to Custom trio.
SUGGEST_RANKING_END = date(2026, 5, 1)
# How many trio rows to show in Trio Research (and related tables).
GLOBAL_TRIO_TABLE_TOP_N = 20
# Cap batters in the slate pool before C(n,3) over the season window (schedule can be huge).
TRIO_RESEARCH_POOL_CAP = 150
# Rank, Trio, Total HR, five (count, %) pairs, one days column = 3 + 10 + 1.
TRIO_RESEARCH_NCOL = 14
# Sentinel `<option value>` for "None" in custom trio selects (clears pick).
CUSTOM_TRIO_SEL_NONE = "__none__"

BATTERS: dict[int, str] = {
    670541: "Yordan Alvarez",
    592450: "Aaron Judge",
    808959: "Munetaka Murakami",
    700250: "Ben Rice",
    656941: "Kyle Schwarber",
}

PLAYER_COLORS: dict[int, str] = {
    670541: "#f97316",
    592450: "#3b82f6",
    808959: "#22c55e",
    700250: "#a855f7",
    656941: "#ef4444",
}

# Second pool: De La Cruz, Olson, Trout, Wood, Buxton, Caminero (MLBAM IDs from qualifying file).
BATTERS_SUPER6: dict[int, str] = {
    682829: "Elly De La Cruz",
    621566: "Matt Olson",
    545361: "Mike Trout",
    695578: "James Wood",
    621439: "Byron Buxton",
    691406: "Junior Caminero",
}

PLAYER_COLORS_SUPER6: dict[int, str] = {
    682829: "#38bdf8",
    621566: "#fbbf24",
    545361: "#fb7185",
    695578: "#4ade80",
    621439: "#c084fc",
    691406: "#f97316",
}


def slate_configs() -> list[dict]:
    """Each slate: tab data-* keys, DOM ids, batter/color maps."""
    return [
        {
            "key": "f5",
            "data_tab_trios": "trios-f5",
            "data_tab_games": "trio-games-f5",
            "label_trios": "Trios (3 of 5)",
            "label_games": "Trio games — 5",
            "panel_trios": "panel-trios-f5",
            "panel_games": "panel-trio-games-f5",
            "batters": BATTERS,
            "colors": PLAYER_COLORS,
        },
        {
            "key": "s6",
            "data_tab_trios": "trios-s6",
            "data_tab_games": "trio-games-s6",
            "label_trios": "Trios — Super 6",
            "label_games": "Trio games — Super 6",
            "panel_trios": "panel-trios-s6",
            "panel_games": "panel-trio-games-s6",
            "batters": BATTERS_SUPER6,
            "colors": PLAYER_COLORS_SUPER6,
        },
    ]


def load_hr_by_date(batters: dict[int, str]) -> tuple[dict[date, dict[int, int]], dict[int, int]]:
    """Per calendar day: batter_id -> HR count (multi-HR preserved for tooltip)."""
    df = pd.read_parquet(
        PA_PATH,
        columns=["batter", "game_date", "game_year", "is_hr"],
    )
    ids = list(batters.keys())
    df = df[(df["game_year"] == 2026) & (df["batter"].isin(ids)) & (df["is_hr"] == 1)].copy()
    df["d"] = pd.to_datetime(df["game_date"]).dt.date
    by_date: dict[date, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        d = row["d"]
        b = int(row["batter"])
        by_date[d][b] += 1
    by_date = {k: dict(v) for k, v in by_date.items()}
    totals = {bid: 0 for bid in ids}
    for day_counts in by_date.values():
        for b, c in day_counts.items():
            totals[b] += c
    return by_date, totals


def load_ab_by_date(batters: dict[int, str]) -> dict[date, dict[int, int]]:
    """Per calendar day: batter_id -> official AB count (is_ab==1 PAs only)."""
    df = pd.read_parquet(
        PA_PATH,
        columns=["batter", "game_date", "game_year", "is_ab"],
    )
    ids = list(batters.keys())
    df = df[(df["game_year"] == 2026) & (df["batter"].isin(ids)) & (df["is_ab"] == 1)].copy()
    df["d"] = pd.to_datetime(df["game_date"]).dt.date
    by_date: dict[date, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        d = row["d"]
        b = int(row["batter"])
        by_date[d][b] += 1
    return {k: dict(v) for k, v in by_date.items()}


def trio_count_positive(
    combo: tuple[int, int, int],
    d: date,
    per_day: dict[date, dict[int, int]],
) -> int:
    c = per_day.get(d, {})
    return sum(1 for b in combo if c.get(b, 0) > 0)


def trio_players_positive(
    combo: tuple[int, int, int],
    d: date,
    per_day: dict[date, dict[int, int]],
    batters: dict[int, str],
) -> list[int]:
    c = per_day.get(d, {})
    return sorted((b for b in combo if c.get(b, 0) > 0), key=lambda x: batters[x])


def iter_calendar_days(first: date, last: date):
    d = first
    while d <= last:
        yield d
        d += timedelta(days=1)


def load_pa_slice_2026(batters: dict[int, str]) -> pd.DataFrame:
    """2026 PAs for given batters: game_pk, batter, is_hr (one row per PA)."""
    df = pd.read_parquet(
        PA_PATH,
        columns=["game_pk", "batter", "game_year", "is_hr"],
    )
    ids = list(batters.keys())
    return df[(df["game_year"] == 2026) & (df["batter"].isin(ids))].copy()


def beast_rank_all_qual(all_ids: list[int], beast: dict[int, float]) -> dict[int, int]:
    """Rank 1 = highest Beast HR score among all qualifying batters."""
    ordered = sorted(all_ids, key=lambda b: (-float(beast.get(b, 0.0) or 0.0), int(b)))
    return {b: i + 1 for i, b in enumerate(ordered)}


def hr_odds_rank_in_top_trios_table(
    day_rows: list[dict],
    beast: dict[int, float],
) -> dict[int, int]:
    """
    Unique batters appearing in the day's trio table (top N rows), sorted by Beast P(HR)
    descending; rank 1 = highest HR odds in that set.
    """
    ids: set[int] = set()
    for r in day_rows:
        for bid in r.get("ids") or []:
            ids.add(int(bid))
    if not ids:
        return {}
    ordered = sorted(
        ids,
        key=lambda b: (-float(beast.get(b, 0.0) or 0.0), int(b)),
    )
    return {b: i + 1 for i, b in enumerate(ordered)}


def format_trio_cell_parenthetical(season_hr: int, pool_rank: int) -> str:
    """YTD HR count and stack rank within the top-table player group (1 = most likely HR)."""
    return f"{int(season_hr)} HR | {int(pool_rank)}"


def load_scheduled_qualifying_batters_by_date(
    matchups_path: Path,
    all_qual_ids: set[int],
) -> dict[str, list[int]]:
    """
    ``game_date`` from ``todays_matchups.json`` -> sorted MLBAM ids that appear in the
    qualifying batters file (schedule-only pool for Trio Research).
    """
    if not matchups_path.exists():
        return {}
    try:
        games = json.loads(matchups_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    by_date: dict[str, set[int]] = defaultdict(set)
    for g in games:
        gd = g.get("game_date")
        if not gd:
            continue
        gd_s = str(gd)
        for side in ("away_batters", "home_batters"):
            for b in g.get(side) or []:
                mid = b.get("mlbam_id")
                if mid is None:
                    continue
                bid = int(mid)
                if bid in all_qual_ids:
                    by_date[gd_s].add(bid)
    return {k: sorted(v) for k, v in by_date.items()}


def max_game_date_in_matchups(matchups_path: Path) -> date | None:
    """Latest ``game_date`` in matchups JSON (so upcoming-slates extend the date picker)."""
    if not matchups_path.exists():
        return None
    try:
        games = json.loads(matchups_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    best: date | None = None
    for g in games:
        gd = g.get("game_date")
        if not gd:
            continue
        try:
            y, m, d0 = map(int, str(gd).split("-"))
            dd = date(y, m, d0)
        except ValueError:
            continue
        if best is None or dd > best:
            best = dd
    return best


def compute_top10_trios_same_calendar_day(
    pool_ids: list[int],
    pa_by_date: dict[date, dict[int, int]],
    hr_by_date: dict[date, dict[int, int]],
    window_first: date,
    window_last: date,
    *,
    top_n: int = GLOBAL_TRIO_TABLE_TOP_N,
) -> list[dict]:
    """
    Triples from ``pool_ids``: over each calendar day in [window_first, window_last],
    if all three had ≥1 PA that day, count the day. Rank by share of those days where
    at least one of the three had ≥1 HR (season-style prior logic).
    """
    id_set = frozenset(int(x) for x in pool_ids)
    if len(id_set) < 3:
        return []
    trios_pa: dict[tuple[int, int, int], int] = defaultdict(int)
    trios_hr: dict[tuple[int, int, int], int] = defaultdict(int)

    for d in iter_calendar_days(window_first, window_last):
        pa_day = pa_by_date.get(d, {})
        hr_day = hr_by_date.get(d, {})
        eligible = sorted(b for b in id_set if pa_day.get(b, 0) >= 1)
        if len(eligible) < 3:
            continue
        for combo in itertools.combinations(eligible, 3):
            t = tuple(int(x) for x in combo)
            trios_pa[t] += 1
            if any(hr_day.get(b, 0) >= 1 for b in t):
                trios_hr[t] += 1

    all_rows: list[dict] = []
    for t, n_days in trios_pa.items():
        nh = int(trios_hr.get(t, 0))
        pct = round(100.0 * nh / n_days, 4) if n_days else 0.0
        all_rows.append({"ids": t, "n_days": n_days, "n_hr": nh, "pct": pct})

    def sort_key(r: dict) -> tuple:
        return (-r["pct"], -r["n_days"], -r["n_hr"], r["ids"])

    for min_d in (5, 3, 1):
        cand = [r for r in all_rows if r["n_days"] >= min_d]
        if len(cand) >= top_n:
            cand.sort(key=sort_key)
            return cand[:top_n]
    cand = sorted(all_rows, key=sort_key)
    return cand[:top_n]


def enrich_trio_research_rows_season_histogram(
    ranked: list[dict],
    hr_by_date: dict[date, dict[int, int]],
    pa_by_date: dict[date, dict[int, int]],
    season_totals: dict[int, int],
    stats_first: date,
    stats_last: date,
) -> list[dict]:
    """Attach full-season (stats window) histogram + season trio HR totals."""
    out: list[dict] = []
    for row in ranked:
        t = tuple(sorted(row["ids"]))
        st = global_trio_pa_window_table_stats(
            t,
            hr_by_date,
            pa_by_date,
            season_totals,
            stats_first,
            stats_last,
        )
        out.append(
            {
                "ids": list(t),
                "total_hr": int(st["total_hr"]),
                "d0": int(st["d0"]),
                "d1": int(st["d1"]),
                "d2": int(st["d2"]),
                "d3": int(st["d3"]),
                "days_any": int(st["days_any"]),
                "days_all_three_pa": int(st["days_all_three_pa"]),
            }
        )
    return out


def trio_research_dates_to_precompute(
    picker_first: date,
    picker_last: date,
    scheduled_by_iso: dict[str, list[int]],
    today: date,
    window_last_statcast: date,
) -> list[date]:
    """
    Only these calendar dates get full trio rows embedded (keeps HTML gen fast).
    Includes: every ``game_date`` present in matchups JSON, plus today and yesterday
    (statcast window end) when they fall in the picker range.
    """
    out: set[date] = set()
    for s in scheduled_by_iso:
        try:
            y, m, d0 = map(int, str(s).split("-"))
            dd = date(y, m, d0)
        except ValueError:
            continue
        if picker_first <= dd <= picker_last:
            out.add(dd)
    for xd in (today, window_last_statcast):
        if picker_first <= xd <= picker_last:
            out.add(xd)
    if not out:
        mid = min(picker_last, max(picker_first, window_last_statcast))
        out.add(mid)
    return sorted(out)


def trio_research_pool_and_source(
    d: date,
    scheduled_by_iso: dict[str, list[int]],
    pa_by_date: dict[date, dict[int, int]],
    all_ids: list[int],
    qual_pa: dict[int, int],
) -> tuple[list[int], str, int]:
    """
    Batter pool for trios on slate date ``d``: prefer scheduled lineup pool from JSON;
    if unavailable or <3 qualifying scheduled batters, fall back to batters with PA that day.

    Returns (pool capped for ranking, source tag, eligible count before cap).
    """
    iso = d.isoformat()
    sched = scheduled_by_iso.get(iso)
    if sched is not None and len(sched) >= 3:
        raw = list(sched)
        src = "schedule"
    else:
        raw = [b for b in all_ids if int(pa_by_date.get(d, {}).get(b, 0) or 0) >= 1]
        src = "played" if sched is None else "played_fallback"

    eligible_before = len(raw)
    if len(raw) > TRIO_RESEARCH_POOL_CAP:
        raw = sorted(
            raw,
            key=lambda b: (-int(qual_pa.get(b, 0) or 0), int(b)),
        )[:TRIO_RESEARCH_POOL_CAP]
    return raw, src, eligible_before


def build_trio_research_trios_by_day(
    picker_first: date,
    picker_last: date,
    stats_first: date,
    stats_last: date,
    scheduled_by_iso: dict[str, list[int]],
    pa_by_date: dict[date, dict[int, int]],
    hr_by_date: dict[date, dict[int, int]],
    season_totals: dict[int, int],
    all_ids: list[int],
    qual_pa: dict[int, int],
    today: date,
    window_last_statcast: date,
) -> tuple[
    dict[str, list[dict]],
    dict[str, int],
    dict[str, str],
    list[str],
]:
    trios_by_day: dict[str, list[dict]] = {}
    eligible_count: dict[str, int] = {}
    pool_source: dict[str, str] = {}
    for d in trio_research_dates_to_precompute(
        picker_first,
        picker_last,
        scheduled_by_iso,
        today,
        window_last_statcast,
    ):
        iso = d.isoformat()
        pool, src, elig_n = trio_research_pool_and_source(
            d,
            scheduled_by_iso,
            pa_by_date,
            all_ids,
            qual_pa,
        )
        eligible_count[iso] = elig_n
        pool_source[iso] = src
        ranked = compute_top10_trios_same_calendar_day(
            pool,
            pa_by_date,
            hr_by_date,
            stats_first,
            stats_last,
            top_n=GLOBAL_TRIO_TABLE_TOP_N,
        )
        trios_by_day[iso] = enrich_trio_research_rows_season_histogram(
            ranked,
            hr_by_date,
            pa_by_date,
            season_totals,
            stats_first,
            stats_last,
        )
    precomputed = sorted(trios_by_day.keys())
    return trios_by_day, eligible_count, pool_source, precomputed


def global_trio_pa_window_table_stats(
    combo: tuple[int, int, int],
    hr_by_date: dict[date, dict[int, int]],
    pa_by_date: dict[date, dict[int, int]],
    season_totals: dict[int, int],
    first: date,
    last: date,
) -> dict:
    """
    Among calendar days in [first, last] where all three had ≥1 PA: histogram of how
    many of the three had ≥1 HR; days with combined HR total == 2 or >= 3.
    """
    trio = tuple(sorted(combo))
    hist = [0, 0, 0, 0]
    days_all_three_pa = 0
    days_combo_hr_eq2 = 0
    days_combo_hr_ge3 = 0
    for d in iter_calendar_days(first, last):
        pa_day = pa_by_date.get(d, {})
        if not all(pa_day.get(b, 0) >= 1 for b in trio):
            continue
        days_all_three_pa += 1
        n_hr_players = trio_count_positive(trio, d, hr_by_date)
        hist[n_hr_players] += 1
        hr_day = hr_by_date.get(d, {})
        combined = sum(int(hr_day.get(b, 0) or 0) for b in trio)
        if combined == 2:
            days_combo_hr_eq2 += 1
        if combined >= 3:
            days_combo_hr_ge3 += 1
    total_hr = sum(int(season_totals.get(b, 0) or 0) for b in trio)
    days_any = hist[1] + hist[2] + hist[3]
    return {
        "ids": trio,
        "total_hr": total_hr,
        "d0": hist[0],
        "d1": hist[1],
        "d2": hist[2],
        "d3": hist[3],
        "days_any": days_any,
        "days_combo_hr_eq2": days_combo_hr_eq2,
        "days_combo_hr_ge3": days_combo_hr_ge3,
        "days_all_three_pa": days_all_three_pa,
    }


def build_trio_research_json_payload(
    trios_by_day: dict[str, list[dict]],
    eligible_count_by_day: dict[str, int],
    pool_source_by_day: dict[str, str],
    precomputed_dates: list[str],
    all_qual: dict[int, tuple[str, str]],
    all_ids: list[int],
    season_totals: dict[int, int],
    beast_scores: dict[int, float],
    picker_first: date,
    picker_last: date,
    stats_first: date,
    stats_last: date,
    default_day: date,
) -> dict:
    """Client hydrates the table from ``trios_by_day[iso_date]``."""
    return {
        "statcast_start": STATCAST_WINDOW_START.isoformat(),
        "default_day": default_day.isoformat(),
        "min_day": picker_first.isoformat(),
        "max_day": picker_last.isoformat(),
        "stats_first": stats_first.isoformat(),
        "stats_last": stats_last.isoformat(),
        "precomputed_dates": precomputed_dates,
        "ncol": TRIO_RESEARCH_NCOL,
        "pool_cap": TRIO_RESEARCH_POOL_CAP,
        "table_top_n": GLOBAL_TRIO_TABLE_TOP_N,
        "trios_by_day": {k: v for k, v in trios_by_day.items()},
        "eligible_count_by_day": eligible_count_by_day,
        "pool_source_by_day": pool_source_by_day,
        "player_names": {str(k): all_qual[k][0] for k in all_qual},
        "beast_p_hr": {
            str(k): round(float(beast_scores.get(k, 0.0) or 0.0), 8) for k in all_ids
        },
        "season_hr": {str(k): int(season_totals.get(k, 0) or 0) for k in all_ids},
    }


def trio_research_table_rows_html(
    day_rows: list[dict],
    all_qual: dict[int, tuple[str, str]],
    beast_scores: dict[int, float],
    season_totals: dict[int, int],
) -> str:
    """Inner HTML for `<tbody>` only. Columns: rank, trio, season Σ HR, histogram + %, days."""
    if not day_rows:
        return ""
    pool_rank = hr_odds_rank_in_top_trios_table(day_rows, beast_scores)
    parts: list[str] = []
    for i, r in enumerate(day_rows, start=1):
        name_parts: list[str] = []
        for bid in r["ids"]:
            bid = int(bid)
            nm = html_mod.escape(all_qual[bid][0])
            sh = int(season_totals.get(bid, 0) or 0)
            rk = int(pool_rank.get(bid, 0))
            inner = format_trio_cell_parenthetical(sh, rk)
            name_parts.append(f"{nm} ({inner})")
        trio_cell = " · ".join(name_parts)
        den = int(r["days_all_three_pa"] or 0)
        d0 = int(r["d0"])
        d1 = int(r["d1"])
        d2 = int(r["d2"])
        d3 = int(r["d3"])
        d_any = int(r["days_any"])
        parts.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{trio_cell}</td>"
            f"<td>{int(r['total_hr'])}</td>"
            f"<td>{d0}</td>{trio_pct_of_days_cell(d0, den)}"
            f"<td>{d_any}</td>{trio_pct_of_days_cell(d_any, den)}"
            f"<td>{d1}</td>{trio_pct_of_days_cell(d1, den)}"
            f"<td>{d2}</td>{trio_pct_of_days_cell(d2, den)}"
            f"<td>{d3}</td>{trio_pct_of_days_cell(d3, den)}"
            f"<td>{den}</td>"
            "</tr>"
        )
    return "".join(parts)


def trio_research_thead_html() -> str:
    pct = '<th title="Share of days (all three had ≥1 PA)">% </th>'
    return (
        "<thead><tr>"
        "<th>Rank</th><th>Trio</th><th>Total HR</th>"
        '<th title="Days: none of the three homered">Days: 0 of 3</th>'
        + pct
        + '<th title="Days: at least one of the three homered">1+ HR</th>'
        + pct
        + "<th>1 of 3</th>"
        + pct
        + "<th>2 of 3</th>"
        + pct
        + "<th>3 of 3</th>"
        + pct
        + "<th>Days (all 3 PA)</th>"
        "</tr></thead>"
    )


def trio_research_panel_html(
    trios_by_day: dict[str, list[dict]],
    eligible_count_by_day: dict[str, int],
    pool_source_by_day: dict[str, str],
    precomputed_dates: list[str],
    all_qual: dict[int, tuple[str, str]],
    season_totals: dict[int, int],
    beast_scores: dict[int, float],
    picker_first: date,
    picker_last: date,
    stats_first: date,
    stats_last: date,
    default_day: date,
) -> str:
    """Full tab panel: slate-day picker + season histogram table + JSON + script."""
    ss = STATCAST_WINDOW_START.isoformat()
    wf_s = picker_first.isoformat()
    pl_s = picker_last.isoformat()
    def_day_s = default_day.isoformat()
    day_rows = trios_by_day.get(def_day_s, [])
    elig_n = int(eligible_count_by_day.get(def_day_s, 0))
    pool_src = pool_source_by_day.get(def_day_s, "")
    payload = build_trio_research_json_payload(
        trios_by_day,
        eligible_count_by_day,
        pool_source_by_day,
        precomputed_dates,
        all_qual,
        list(all_qual.keys()),
        season_totals,
        beast_scores,
        picker_first,
        picker_last,
        stats_first,
        stats_last,
        default_day,
    )
    json_txt = json.dumps(payload, separators=(",", ":"))
    tbody_inner = trio_research_table_rows_html(day_rows, all_qual, beast_scores, season_totals)
    nc = str(TRIO_RESEARCH_NCOL)
    if not tbody_inner and elig_n >= 3:
        tbody_inner = (
            f'<tr><td colspan="{nc}" class="trio-research-empty">No trio rows (unexpected).</td></tr>'
        )
    elif not tbody_inner:
        tbody_inner = (
            f'<tr><td colspan="{nc}" class="trio-research-empty">'
            f"Fewer than 3 batters in the slate pool for this date ({elig_n})."
            "</td></tr>"
        )
    h2 = (
        "<h2 class=\"global-top10-h2\">Trio Research — slate vs season-to-date stats</h2>"
    )
    precomp_hint = ", ".join(precomputed_dates) if precomputed_dates else "—"
    note = (
        '<p class="sub trio-research-note">Choose a <strong>precomputed</strong> slate date (see list below). '
        "The pool is <strong>scheduled</strong> batters from <code>todays_matchups.json</code> for that "
        "<code>game_date</code> (intersected with the qualifying file) when there are ≥3; otherwise that day&rsquo;s "
        f"PA from Statcast. Histogram columns are <strong>season-to-date</strong> through "
        f"<strong>{html_mod.escape(stats_last.isoformat())}</strong> "
        f"(<strong>{html_mod.escape(stats_first.isoformat())}</strong>–{html_mod.escape(stats_last.isoformat())}). "
        f"Trios are ranked by the original HR-day% rule over that window. Large pools cap at "
        f"<strong>{TRIO_RESEARCH_POOL_CAP}</strong> by YTD PA. "
        f"Each name is <code>(YTD HR | rank)</code> among <strong>unique batters in this table</strong> for the slate: "
        f"<strong>1</strong> = most likely to homer (Beast P(HR) order); the <strong>largest</strong> rank = least likely, "
        f"and equals how many <strong>unique</strong> batters appear in the top {GLOBAL_TRIO_TABLE_TOP_N} trio rows. "
        f"<em>Baked dates:</em> {html_mod.escape(precomp_hint)}</p>"
    )
    sched_note = (
        "scheduled (qualifying)"
        if pool_src == "schedule"
        else ("scheduled→fallback to PA" if pool_src == "played_fallback" else "PA that day")
    )
    elig_line = (
        f'<p class="trio-research-eligible-line" id="trio-research-eligible">'
        f"<strong>{elig_n}</strong> batters in pool · <span class=\"trio-research-pool-src\">{html_mod.escape(sched_note)}</span> · "
        f'slate <span class="trio-research-day-label">{html_mod.escape(def_day_s)}</span></p>'
    )
    filters = (
        f'<div class="trio-research-filters">'
        f'<label class="trio-research-date-label">Slate date '
        f'<input type="date" id="trio-research-day" min="{html_mod.escape(ss)}" '
        f'max="{html_mod.escape(pl_s)}" value="{html_mod.escape(def_day_s)}"></label>'
        f'<button type="button" class="btn btn-primary" id="trio-research-apply">Apply</button>'
        f"</div>"
    )
    json_block = (
        f'<script type="application/json" id="trio-research-json">{json_txt}</script>'
    )
    table_block = (
        '<div class="global-top10-table-wrap">'
        '<table class="combo-table global-top10-table" id="trio-research-table">'
        + trio_research_thead_html()
        + f'<tbody id="trio-research-tbody">{tbody_inner}</tbody>'
        + "</table></div>"
    )
    return (
        '<div id="panel-trio-research" class="tab-panel" hidden data-tab-panel="trio-research">'
        '<section class="global-top10-trios trio-research-section" aria-label="Trio Research">'
        + h2
        + note
        + filters
        + elig_line
        + table_block
        + "</section>"
        + json_block
        + trio_research_script()
        + "</div>"
    )


def trio_research_script() -> str:
    return """
<script>
(function () {
  function loadPayload() {
    var el = document.getElementById("trio-research-json");
    if (!el || !el.textContent) return null;
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      return null;
    }
  }

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function pctCell(n, den) {
    if (!den) return '<td class="trio-pct-col">—</td>';
    return (
      '<td class="trio-pct-col">' + ((100.0 * n) / den).toFixed(1) + "%</td>"
    );
  }

  function poolSrcLabel(src) {
    if (src === "schedule") return "scheduled (qualifying)";
    if (src === "played_fallback") return "scheduled→fallback to PA";
    return "PA that day";
  }

  /** Union of all batter ids in the trio table for this slate; rank 1 = highest Beast P(HR). */
  function hrOddsRankInTopTriosTable(rows, beastHr) {
    var seen = {};
    var ids = [];
    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      if (!row.ids) continue;
      for (var j = 0; j < row.ids.length; j++) {
        var bid = row.ids[j];
        if (seen[bid]) continue;
        seen[bid] = true;
        ids.push(bid);
      }
    }
    ids.sort(function (a, b) {
      var pa = Number(beastHr[String(a)] || 0);
      var pb = Number(beastHr[String(b)] || 0);
      if (pb !== pa) return pb - pa;
      return a - b;
    });
    var out = {};
    for (var i = 0; i < ids.length; i++) out[ids[i]] = i + 1;
    return out;
  }

  function trioCellParen(sh, poolRank) {
    return sh + " HR | " + poolRank;
  }

  function renderTrioResearch() {
    var payload = loadPayload();
    if (!payload || !payload.trios_by_day) return;
    var dayEl = document.getElementById("trio-research-day");
    var tbody = document.getElementById("trio-research-tbody");
    var eligP = document.getElementById("trio-research-eligible");
    if (!dayEl || !tbody) return;
    var ncol = parseInt(payload.ncol || "14", 10);
    var dayStr = dayEl.value || payload.default_day;
    if (
      payload.min_day &&
      payload.max_day &&
      (dayStr < payload.min_day || dayStr > payload.max_day)
    ) {
      tbody.innerHTML =
        '<tr><td colspan="' +
        ncol +
        '" class="trio-research-empty">Choose a date between ' +
        esc(payload.min_day) +
        " and " +
        esc(payload.max_day) +
        ".</td></tr>";
      if (eligP) eligP.textContent = "";
      if (typeof window.decorateSortableHeaders === "function") {
        window.decorateSortableHeaders();
      }
      return;
    }
    if (
      payload.trios_by_day &&
      payload.trios_by_day[dayStr] === undefined
    ) {
      var slateList = (payload.precomputed_dates || []).join(", ");
      tbody.innerHTML =
        '<tr><td colspan="' +
        ncol +
        '" class="trio-research-empty">No trio block baked for this date. Run the generator after ' +
        "<code>fetch_todays_games.py</code> for this slate, or pick a precomputed date: " +
        esc(slateList || "(none)") +
        "</td></tr>";
      if (eligP) eligP.textContent = "";
      if (typeof window.decorateSortableHeaders === "function") {
        window.decorateSortableHeaders();
      }
      return;
    }
    var rows = payload.trios_by_day[dayStr] || [];
    var eligN = parseInt(
      (payload.eligible_count_by_day && payload.eligible_count_by_day[dayStr]) || "0",
      10
    );
    var psrc =
      (payload.pool_source_by_day && payload.pool_source_by_day[dayStr]) || "";
    if (eligP) {
      eligP.innerHTML =
        "<strong>" +
        eligN +
        '</strong> batters in pool · <span class="trio-research-pool-src">' +
        esc(poolSrcLabel(psrc)) +
        '</span> · slate <span class="trio-research-day-label">' +
        esc(dayStr) +
        "</span>";
    }
    var beastHr = payload.beast_p_hr || {};
    var seasonHr = payload.season_hr || {};
    if (eligN < 3) {
      tbody.innerHTML =
        '<tr><td colspan="' +
        ncol +
        '" class="trio-research-empty">Fewer than 3 batters in the slate pool for this date (' +
        eligN +
        ").</td></tr>";
      if (typeof window.decorateSortableHeaders === "function") {
        window.decorateSortableHeaders();
      }
      return;
    }
    if (!rows.length) {
      tbody.innerHTML =
        '<tr><td colspan="' +
        ncol +
        '" class="trio-research-empty">No trio rows for this slate.</td></tr>';
      if (typeof window.decorateSortableHeaders === "function") {
        window.decorateSortableHeaders();
      }
      return;
    }
    var pnames = payload.player_names || {};
    var poolRank = hrOddsRankInTopTriosTable(rows, beastHr);
    var html = "";
    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      var trioParts = [];
      for (var j = 0; j < row.ids.length; j++) {
        var bid = row.ids[j];
        var sh = parseInt(seasonHr[String(bid)] || "0", 10);
        var rk = parseInt(poolRank[bid] || "0", 10);
        var nm = esc(pnames[String(bid)] || ("#" + bid));
        trioParts.push(nm + " (" + trioCellParen(sh, rk) + ")");
      }
      var trioCell = trioParts.join(" · ");
      var den = parseInt(row.days_all_three_pa || "0", 10);
      var d0 = parseInt(row.d0 || "0", 10);
      var d1 = parseInt(row.d1 || "0", 10);
      var d2 = parseInt(row.d2 || "0", 10);
      var d3 = parseInt(row.d3 || "0", 10);
      var dAny = parseInt(row.days_any || "0", 10);
      html +=
        "<tr>" +
        "<td>" +
        (r + 1) +
        "</td>" +
        "<td>" +
        trioCell +
        "</td>" +
        "<td>" +
        row.total_hr +
        "</td>" +
        "<td>" +
        d0 +
        "</td>" +
        pctCell(d0, den) +
        "<td>" +
        dAny +
        "</td>" +
        pctCell(dAny, den) +
        "<td>" +
        d1 +
        "</td>" +
        pctCell(d1, den) +
        "<td>" +
        d2 +
        "</td>" +
        pctCell(d2, den) +
        "<td>" +
        d3 +
        "</td>" +
        pctCell(d3, den) +
        "<td>" +
        den +
        "</td>" +
        "</tr>";
    }
    tbody.innerHTML = html;
    if (typeof window.decorateSortableHeaders === "function") {
      window.decorateSortableHeaders();
    }
  }

  function hook() {
    var btn = document.getElementById("trio-research-apply");
    var dayEl = document.getElementById("trio-research-day");
    if (btn) btn.addEventListener("click", renderTrioResearch);
    if (dayEl) dayEl.addEventListener("change", renderTrioResearch);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", hook);
  } else {
    hook();
  }
})();
</script>
"""


def trio_game_pa_hr_metrics(
    combo: tuple[int, int, int],
    dfp: pd.DataFrame,
) -> tuple[int, int, float]:
    """
    Among games (game_pk) where all three batters recorded ≥1 PA in 2026:
    return (n_games, n_games_with_any_trio_hr, pct).
    """
    trio = tuple(sorted(combo))
    bset = set(trio)
    sub = dfp[dfp["batter"].isin(bset)]
    if sub.empty:
        return 0, 0, 0.0
    bats_per_game = sub.groupby("game_pk", sort=False)["batter"].apply(
        lambda s: frozenset(int(x) for x in s.dropna().unique()),
    )
    games_all3 = [int(pk) for pk, fs in bats_per_game.items() if bset <= set(fs)]
    n = len(games_all3)
    if n == 0:
        return 0, 0, 0.0
    sub_g = sub[sub["game_pk"].isin(games_all3)]
    n_hr = int(sub_g.loc[sub_g["is_hr"] == 1, "game_pk"].nunique())
    pct = round(100.0 * n_hr / n, 1)
    return n, n_hr, pct


def trio_window_all_three_ab_then_hr(
    combo: tuple[int, int, int],
    ab_by_date: dict[date, dict[int, int]],
    hr_by_date: dict[date, dict[int, int]],
    first: date,
    last: date,
) -> tuple[int, int, float]:
    """
    In [first, last]: days where all three had ≥1 AB; among those, days where any had ≥1 HR.
    """
    trio = tuple(sorted(combo))
    d_ab = 0
    d_hr = 0
    for d in iter_calendar_days(first, last):
        if trio_count_positive(trio, d, ab_by_date) < 3:
            continue
        d_ab += 1
        if trio_count_positive(trio, d, hr_by_date) >= 1:
            d_hr += 1
    pct = round(100.0 * d_hr / d_ab, 1) if d_ab else 0.0
    return d_ab, d_hr, pct


def trio_games_panel_html(
    slate: dict,
    ranked_combos: list[dict],
    pa_df: pd.DataFrame,
    ab_by_date: dict[date, dict[int, int]],
    hr_by_date: dict[date, dict[int, int]],
    window_first: date,
    window_last: date,
) -> str:
    """Per-slate tab: same-calendar-day stats (always populated) + same-game_pk table."""
    h_day = (
        "<thead><tr><th>Rank</th><th>Trio</th>"
        "<th>Days (all 3 ≥1 AB)</th><th>Days (any HR)</th><th>HR %</th></tr></thead>"
    )
    tb_day: list[str] = ["<tbody>"]
    for i, r in enumerate(ranked_combos, start=1):
        nd, nh, pd = trio_window_all_three_ab_then_hr(
            r["ids"], ab_by_date, hr_by_date, window_first, window_last
        )
        pct_d = f"{pd}%" if nd else "—"
        tb_day.append(
            "<tr>"
            f"<td>{i}</td><td>{html_mod.escape(r['label'])}</td>"
            f"<td>{nd}</td><td>{nh}</td><td>{pct_d}</td>"
            "</tr>"
        )
    tb_day.append("</tbody>")
    table_a = (
        '<h2 class="trio-games-h2">A. Same calendar day (all three had ≥1 AB)</h2>'
        f'<table class="combo-table trio-games-table">{h_day}{"".join(tb_day)}</table>'
    )

    h_pk = (
        "<thead><tr><th>Rank</th><th>Trio</th>"
        "<th>Games (all 3 w/ PA)</th><th>Games (any HR)</th><th>HR %</th></tr></thead>"
    )
    tb_pk: list[str] = ["<tbody>"]
    for i, r in enumerate(ranked_combos, start=1):
        n_g, n_h, pp = trio_game_pa_hr_metrics(r["ids"], pa_df)
        pct_p = f"{pp}%" if n_g else "—"
        tb_pk.append(
            "<tr>"
            f"<td>{i}</td><td>{html_mod.escape(r['label'])}</td>"
            f"<td>{n_g}</td><td>{n_h}</td><td>{pct_p}</td>"
            "</tr>"
        )
    tb_pk.append("</tbody>")
    table_b = (
        '<h2 class="trio-games-h2">B. Same MLB game (<code>game_pk</code>, 2026)</h2>'
        f'<table class="combo-table trio-games-table">{h_pk}{"".join(tb_pk)}</table>'
    )

    tab = slate["data_tab_games"]
    pid = slate["panel_games"]
    return f'<div id="{pid}" class="tab-panel" hidden data-tab-panel="{tab}">{table_a}{table_b}</div>'


def window_day_count(first: date, last: date) -> int:
    """Inclusive calendar days in [first, last]."""
    if last < first:
        return 0
    return (last - first).days + 1


def fmt_pct_of(n: int, denom: int, *, decimals: int = 1, suffix: str = "") -> str:
    """HTML fragment: space + (xx.x%) of denom; empty if denom is 0."""
    if denom <= 0:
        return ""
    p = 100.0 * n / denom
    suf = html_mod.escape(suffix) if suffix else ""
    return f' <span class="pct">({p:.{decimals}f}%{suf})</span>'


def trio_pct_of_days_cell(n: int, denom: int) -> str:
    """Sortable % column; denominator = days all three had PA."""
    if denom <= 0:
        return '<td class="trio-pct-col">—</td>'
    return f'<td class="trio-pct-col">{100.0 * n / denom:.1f}%</td>'


def combo_row_stats(
    combo: tuple[int, int, int],
    hr_by_date: dict[date, dict[int, int]],
    season_totals: dict[int, int],
    first: date,
    last: date,
    batters: dict[int, str],
) -> dict:
    """Histogram over first..last: how many of the 3 had ≥1 HR that day."""
    trio = tuple(sorted(combo))
    hist = [0, 0, 0, 0]
    for d in iter_calendar_days(first, last):
        n = trio_count_positive(trio, d, hr_by_date)
        hist[n] += 1
    total_hr = sum(season_totals[b] for b in trio)
    return {
        "ids": trio,
        "label": " · ".join(batters[b] for b in trio),
        "short": " / ".join(batters[b].split()[-1] for b in trio),
        "total_hr": total_hr,
        "d0": hist[0],
        "d1": hist[1],
        "d2": hist[2],
        "d3": hist[3],
        "days_any": hist[1] + hist[2] + hist[3],
        "days_all_three": hist[3],
    }


def rank_combos(
    hr_by_date: dict[date, dict[int, int]],
    season_totals: dict[int, int],
    first: date,
    last: date,
    batters: dict[int, str],
) -> list[dict]:
    ids = sorted(batters.keys())
    rows: list[dict] = []
    for combo in itertools.combinations(ids, 3):
        rows.append(combo_row_stats(combo, hr_by_date, season_totals, first, last, batters))
    rows.sort(
        key=lambda r: (-r["total_hr"], -r["days_all_three"], -r["days_any"], r["label"]),
    )
    return rows


def trio_calendar_tooltip(
    d: date,
    combo: tuple[int, int, int],
    ab_by_date: dict[date, dict[int, int]],
    hr_by_date: dict[date, dict[int, int]],
    batters: dict[int, str],
) -> str:
    """Corner = AB participation; fill/dots = HR — spell out both in the hover."""
    ac = ab_by_date.get(d, {})
    hc = hr_by_date.get(d, {})
    n_ab = trio_count_positive(combo, d, ab_by_date)
    n_hr = trio_count_positive(combo, d, hr_by_date)
    parts = [
        d.isoformat(),
        f"Corner: {n_ab} of 3 with ≥1 AB",
        f"Fill/dots: {n_hr} of 3 with HR",
    ]
    for bid in sorted(combo, key=lambda b: batters[b]):
        abn = ac.get(bid, 0)
        hrn = hc.get(bid, 0)
        parts.append(f"{batters[bid]}: {abn} AB, {hrn} HR")
    return html_mod.escape(" | ".join(parts))


def month_blocks_trio(
    combo: tuple[int, int, int],
    ab_by_date: dict[date, dict[int, int]],
    hr_by_date: dict[date, dict[int, int]],
    y: int,
    m: int,
    window_first: date,
    window_last: date,
    batters: dict[int, str],
    colors: dict[int, str],
) -> str:
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdatescalendar(y, m)
    dow = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    parts: list[str] = [
        f'<div class="month"><h2>{calendar.month_name[m]} {y}</h2>',
        '<div class="dow">',
        "".join(f"<div>{d}</div>" for d in dow),
        "</div><div class=\"grid\">",
    ]
    for week in weeks:
        for cell_d in week:
            in_month = cell_d.month == m
            extra = " out" if not in_month else ""
            if not in_month:
                parts.append(
                    f'<div class="cell trio-cell empty-day{extra}">'
                    f'<span class="num"></span>'
                    '<div class="dots"></div></div>'
                )
                continue
            in_window = window_first <= cell_d <= window_last
            if not in_window:
                tip = html_mod.escape(f"{cell_d.isoformat()} (outside window)")
                parts.append(
                    f'<div class="cell trio-cell empty-day cal-out-window" title="{tip}">'
                    f'<span class="num">{cell_d.day}</span><div class="dots"></div></div>'
                )
                continue
            ds = cell_d.isoformat()
            n_ab = trio_count_positive(combo, cell_d, ab_by_date)
            n_hr = trio_count_positive(combo, cell_d, hr_by_date)
            players_hr = trio_players_positive(combo, cell_d, hr_by_date, batters)
            if n_hr == 0:
                body_cls = "empty-day"
            else:
                body_cls = f"has-hr trio-n{n_hr}"
            title = trio_calendar_tooltip(cell_d, combo, ab_by_date, hr_by_date, batters)
            dots = "".join(
                f'<span class="dot" style="background:{colors[b]}"></span>' for b in players_hr
            )
            dots_block = f'<div class="dots">{dots}</div>' if dots else '<div class="dots"></div>'
            parts.append(
                f'<div class="cell trio-cell {body_cls}" data-date="{ds}" title="{title}">'
                f'<span class="trio-corner-count" aria-hidden="true">{n_ab}</span>'
                f'<span class="num">{cell_d.day}</span>{dots_block}</div>'
            )
    parts.append("</div></div>")
    return "".join(parts)


def trios_panel_html(
    slate: dict,
    ranked: list[dict],
    ab_by_date: dict[date, dict[int, int]],
    hr_by_date: dict[date, dict[int, int]],
    months: list[tuple[int, int]],
    window_first: date,
    window_last: date,
    pool_combined_season_hr: int,
) -> str:
    batters = slate["batters"]
    colors = slate["colors"]
    sk = slate["key"]
    tab_key = slate["data_tab_trios"]
    panel_id = slate["panel_trios"]
    total_days = window_day_count(window_first, window_last)
    thead = (
        "<thead><tr>"
        "<th>Rank</th><th>Trio</th><th>Total HR</th>"
        "<th>Days: 0 of 3</th><th>1 of 3</th><th>2 of 3</th><th>3 of 3</th>"
        "<th>Any ≥1 HR</th><th>All 3 HR</th>"
        "</tr></thead>"
    )
    tbody = ["<tbody>"]
    for i, r in enumerate(ranked, start=1):
        hr_pct = fmt_pct_of(r["total_hr"], pool_combined_season_hr, suffix=" of combined")
        tbody.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html_mod.escape(r['label'])}</td>"
            f"<td>{r['total_hr']}{hr_pct}</td>"
            f"<td>{r['d0']}{fmt_pct_of(r['d0'], total_days)}</td>"
            f"<td>{r['d1']}{fmt_pct_of(r['d1'], total_days)}</td>"
            f"<td>{r['d2']}{fmt_pct_of(r['d2'], total_days)}</td>"
            f"<td>{r['d3']}{fmt_pct_of(r['d3'], total_days)}</td>"
            f"<td>{r['days_any']}{fmt_pct_of(r['days_any'], total_days)}</td>"
            f"<td>{r['days_all_three']}{fmt_pct_of(r['days_all_three'], total_days)}</td>"
            "</tr>"
        )
    tbody.append("</tbody>")
    table = f'<table class="combo-table">{thead}{"".join(tbody)}</table>'

    legend = (
        '<div class="legend trio-legend">'
        '<div class="legend-item"><span class="swatch trio-n1"></span>1 HR</div>'
        '<div class="legend-item"><span class="swatch trio-n2"></span>2 HR</div>'
        '<div class="legend-item"><span class="swatch trio-n3"></span>3 HR</div>'
        "</div>"
    )

    blocks: list[str] = ['<div class="trio-calendars-wrap">']
    for i, r in enumerate(ranked, start=1):
        combo = r["ids"]
        blocks.append(f'<section class="trio-block" id="trio-rank-{sk}-{i}">')
        blocks.append(f"<h3>#{i} — {html_mod.escape(r['short'])}</h3>")
        blocks.append('<div class="months trio-months">')
        for yy, mm in months:
            blocks.append(
                month_blocks_trio(
                    combo,
                    ab_by_date,
                    hr_by_date,
                    yy,
                    mm,
                    window_first,
                    window_last,
                    batters,
                    colors,
                )
            )
        blocks.append("</div></section>")
    blocks.append("</div>")

    return (
        f'<div id="{panel_id}" class="tab-panel" hidden data-tab-panel="{tab_key}">'
        f"{table}{legend}{''.join(blocks)}</div>"
    )


def hr_by_date_json(by_date: dict[date, dict[int, int]]) -> str:
    out: dict[str, dict[str, int]] = {}
    for d, counts in by_date.items():
        out[d.isoformat()] = {str(b): int(c) for b, c in counts.items()}
    return json.dumps(out, separators=(",", ":"))


def batters_json() -> str:
    return json.dumps({str(k): v for k, v in BATTERS.items()}, separators=(",", ":"))


def colors_json() -> str:
    return json.dumps({str(k): v for k, v in PLAYER_COLORS.items()}, separators=(",", ":"))


def load_all_qualifying_batters() -> dict[int, tuple[str, str]]:
    """mlbam_id -> (player name, team abbr)."""
    path = QUAL_BATTERS_CSV
    if not path.exists():
        return {k: (v, "") for k, v in BATTERS.items()}
    qb = pd.read_csv(path)
    out: dict[int, tuple[str, str]] = {}
    for _, row in qb.iterrows():
        bid = int(row["mlbam_id"])
        name = str(row["name"]).strip()
        team = str(row.get("team", "") or "").strip()
        out[bid] = (name, team)
    return out


def load_qualifying_pa_map() -> dict[int, int]:
    """mlbam_id -> 2026 PA from qualifying CSV (for ranking pool caps)."""
    if not QUAL_BATTERS_CSV.exists():
        return {}
    qb = pd.read_csv(QUAL_BATTERS_CSV)
    return {int(r.mlbam_id): int(r.get("PA", 0) or 0) for _, r in qb.iterrows()}


def load_hr_by_date_ids(batter_ids: list[int]) -> tuple[dict[date, dict[int, int]], dict[int, int]]:
    """Same as load_hr_by_date but for an explicit id list (e.g. all qualifying)."""
    ids = list(batter_ids)
    if not ids:
        return {}, {}
    df = pd.read_parquet(
        PA_PATH,
        columns=["batter", "game_date", "game_year", "is_hr"],
        filters=[("batter", "in", ids)],
    )
    df = df[(df["game_year"] == 2026) & (df["is_hr"] == 1)].copy()
    df["d"] = pd.to_datetime(df["game_date"]).dt.date
    by_date: dict[date, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        d = row["d"]
        b = int(row["batter"])
        by_date[d][b] += 1
    by_date = {k: dict(v) for k, v in by_date.items()}
    totals = {bid: 0 for bid in ids}
    for day_counts in by_date.values():
        for b, c in day_counts.items():
            totals[b] = totals.get(b, 0) + c
    return by_date, totals


def load_ab_by_date_ids(batter_ids: list[int]) -> dict[date, dict[int, int]]:
    ids = list(batter_ids)
    if not ids:
        return {}
    df = pd.read_parquet(
        PA_PATH,
        columns=["batter", "game_date", "game_year", "is_ab"],
        filters=[("batter", "in", ids)],
    )
    df = df[(df["game_year"] == 2026) & (df["is_ab"] == 1)].copy()
    df["d"] = pd.to_datetime(df["game_date"]).dt.date
    by_date: dict[date, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        d = row["d"]
        b = int(row["batter"])
        by_date[d][b] += 1
    return {k: dict(v) for k, v in by_date.items()}


def load_pa_by_date_ids(batter_ids: list[int]) -> dict[date, dict[int, int]]:
    """Per calendar day: batter -> PA count (one row per PA in league parquet)."""
    ids = list(batter_ids)
    if not ids:
        return {}
    df = pd.read_parquet(
        PA_PATH,
        columns=["batter", "game_date", "game_year"],
        filters=[("batter", "in", ids)],
    )
    df = df[df["game_year"] == 2026].copy()
    df["d"] = pd.to_datetime(df["game_date"]).dt.date
    by_date: dict[date, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        d = row["d"]
        b = int(row["batter"])
        by_date[d][b] += 1
    return {k: dict(v) for k, v in by_date.items()}


def load_games_lookup_2026() -> dict[str, dict]:
    """game_pk (str) -> { 'b': sorted batter ids, 'hr': 0|1 } for trio same-game stats in the browser."""
    df = pd.read_parquet(PA_PATH, columns=["game_pk", "game_year", "batter", "is_hr"])
    df = df[df["game_year"] == 2026]
    out: dict[str, dict] = {}
    for pk, grp in df.groupby("game_pk"):
        out[str(int(pk))] = {
            "b": sorted(int(x) for x in grp["batter"].unique()),
            "hr": int(grp["is_hr"].max()),
        }
    return out


def _fmt_split_line(sub: pd.DataFrame) -> str:
    if sub is None or sub.empty:
        return "—"
    pa = len(sub)
    ab = int(sub["is_ab"].sum())
    h = int(sub["is_hit"].sum())
    hr = int(sub["is_hr"].sum())
    k = int(sub["is_strikeout"].sum())
    if pa == 0:
        return "—"
    ba = (h / ab) if ab else 0.0
    kp = 100.0 * k / pa if pa else 0.0
    return f"{pa} PA · {ba:.3f} ({h}/{ab}) · {hr} HR · {kp:.1f}% K"


def _bvp_line(sub: pd.DataFrame) -> str:
    if sub is None or sub.empty:
        return "No BvP"
    pa = len(sub)
    ab = int(sub["is_ab"].sum())
    h = int(sub["is_hit"].sum())
    hr = int(sub["is_hr"].sum())
    k = int(sub["is_strikeout"].sum())
    ba = (h / ab) if ab else 0.0
    return f"{h}-for-{ab} ({ba:.3f}), {hr} HR, {k} K ({pa} PA)"


def _pitcher_throws_map(pitcher_ids: set[int]) -> dict[int, str]:
    if not pitcher_ids:
        return {}
    ids = sorted(pitcher_ids)
    df = pd.read_parquet(PA_PATH, columns=["pitcher", "p_throws"], filters=[("pitcher", "in", ids)])
    if df.empty:
        return {}
    out: dict[int, str] = {}
    for pid, grp in df.groupby("pitcher"):
        m = grp["p_throws"].mode()
        ch = str(m.iloc[0]) if len(m) else "R"
        out[int(pid)] = ch
    return out


def build_matchup_json_for_batters(
    batter_ids: list[int],
    sc_df: pd.DataFrame,
    slate_index: dict[int, dict],
    throws_map: dict[int, str],
    meta: dict[int, tuple[str, str]],
) -> dict[str, dict]:
    """Per batter id (str keys): tonight's row + platoon strings for embedding in HTML."""
    out: dict[str, dict] = {}
    for bid in batter_ids:
        sub_all = sc_df[sc_df["batter"] == bid]
        car_l = _fmt_split_line(sub_all[sub_all["p_throws"] == "L"])
        car_r = _fmt_split_line(sub_all[sub_all["p_throws"] == "R"])
        ytd = sub_all[sub_all["game_year"] == 2026]
        ytd_l = _fmt_split_line(ytd[ytd["p_throws"] == "L"])
        ytd_r = _fmt_split_line(ytd[ytd["p_throws"] == "R"])
        nm0, tm0 = meta.get(bid, ("", ""))
        slot = slate_index.get(bid)
        if not slot:
            out[str(bid)] = {
                "name": nm0,
                "team": tm0,
                "matchup": "Not on today's slate",
                "opp": "—",
                "bvp": "—",
                "car_L": car_l,
                "car_R": car_r,
                "ytd_L": ytd_l,
                "ytd_R": ytd_r,
            }
            continue
        oid = slot["opp_pitcher_id"]
        oname = slot["opp_pitcher_name"] or "TBD"
        oid_int = int(oid) if oid is not None else None
        th = throws_map.get(oid_int, "R") if oid_int is not None else "R"
        hpl = "LHP" if th == "L" else "RHP"
        opp_disp = f"{oname} ({hpl})" if oid_int is not None else "TBD"
        if oid_int is not None:
            bvp_sub = sub_all[sub_all["pitcher"] == oid_int]
            bvp_s = _bvp_line(bvp_sub)
        else:
            bvp_s = "—"
        out[str(bid)] = {
            "name": slot.get("name") or nm0,
            "team": slot.get("team") or tm0,
            "matchup": slot["matchup"],
            "opp": opp_disp,
            "bvp": bvp_s,
            "car_L": car_l,
            "car_R": car_r,
            "ytd_L": ytd_l,
            "ytd_R": ytd_r,
        }
    return out


def load_beast_hr_score_by_batter(batter_ids: list[int]) -> dict[int, float]:
    """
    Per batter: max(mean beast val-split P(HR) from val_predictions_multi_beast,
    max tonight's beast matchup p_hr / adj_p_hr when present).
    Used only to rank the 10 suggestion rows (1 = highest Beast signal among those 10).
    """
    ids = sorted({int(x) for x in batter_ids})
    out: dict[int, float] = {bid: 0.0 for bid in ids}
    if VAL_BEAST_PRED_PARQUET.exists():
        try:
            df = pd.read_parquet(
                VAL_BEAST_PRED_PARQUET,
                columns=["batter", "p_is_hr"],
                filters=[("batter", "in", ids)],
            )
            if not df.empty:
                for bid, val in df.groupby("batter")["p_is_hr"].mean().items():
                    bid = int(bid)
                    if bid in out:
                        out[bid] = max(out[bid], float(val))
        except (OSError, ValueError, TypeError, pd.errors.EmptyDataError):
            pass
    if BEAST_MATCHUP_JSON.exists():
        try:
            rows = json.loads(BEAST_MATCHUP_JSON.read_text(encoding="utf-8"))
            for row in rows:
                bid = row.get("batter_mlbam_id")
                if bid is None:
                    continue
                bid = int(bid)
                if bid not in out:
                    continue
                ph = row.get("p_hr")
                if ph is None:
                    ph = row.get("adj_p_hr")
                if ph is None:
                    ph = row.get("score_hr")
                try:
                    phf = float(ph) if ph is not None else 0.0
                except (TypeError, ValueError):
                    phf = 0.0
                out[bid] = max(out[bid], phf)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return out


def load_slate_batter_index(matchups_path: Path) -> tuple[dict[int, dict], set[int]]:
    """batter_id -> { matchup, team, opp_pitcher_id, opp_pitcher_name, name } + pitcher ids on slate."""
    if not matchups_path.exists():
        return {}, set()
    games = json.loads(matchups_path.read_text(encoding="utf-8"))
    idx: dict[int, dict] = {}
    pit_ids: set[int] = set()
    for g in games:
        away_team = g.get("away_team", "")
        home_team = g.get("home_team", "")
        matchup = f"{away_team} @ {home_team}"
        hid = g.get("home_pitcher_id")
        aid = g.get("away_pitcher_id")
        if hid is not None:
            pit_ids.add(int(hid))
        if aid is not None:
            pit_ids.add(int(aid))
        hname = g.get("home_pitcher_name") or ""
        aname = g.get("away_pitcher_name") or ""

        for b in g.get("away_batters") or []:
            bid = int(b["mlbam_id"])
            idx[bid] = {
                "matchup": matchup,
                "team": away_team,
                "name": str(b.get("name", "")),
                "opp_pitcher_id": hid,
                "opp_pitcher_name": hname,
            }
        for b in g.get("home_batters") or []:
            bid = int(b["mlbam_id"])
            idx[bid] = {
                "matchup": matchup,
                "team": home_team,
                "name": str(b.get("name", "")),
                "opp_pitcher_id": aid,
                "opp_pitcher_name": aname,
            }
    return idx, pit_ids


def build_month_grids(months: list[tuple[int, int]]) -> dict[str, list[list[str]]]:
    """Key 'YYYY-MM' -> weeks of ISO date strings (Sunday-first weeks, Python calendar)."""
    cal = calendar.Calendar(firstweekday=6)
    out: dict[str, list[list[str]]] = {}
    for y, m in months:
        weeks = [[d.isoformat() for d in w] for w in cal.monthdatescalendar(y, m)]
        out[f"{y}-{m:02d}"] = weeks
    return out


def month_blocks(y: int, m: int, window_first: date, window_last: date) -> str:
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdatescalendar(y, m)
    dow = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    parts: list[str] = [
        f'<div class="month"><h2>{calendar.month_name[m]} {y}</h2>',
        '<div class="dow">',
        "".join(f"<div>{d}</div>" for d in dow),
        "</div><div class=\"grid\">",
    ]
    for week in weeks:
        for cell_d in week:
            in_month = cell_d.month == m
            extra = " out" if not in_month else ""
            if not in_month:
                parts.append(
                    f'<div class="cell empty-day{extra}">'
                    f'<span class="num"></span><div class="dots"></div></div>'
                )
                continue
            in_window = window_first <= cell_d <= window_last
            if not in_window:
                tip = html_mod.escape(f"{cell_d.isoformat()} (outside HR window)")
                parts.append(
                    f'<div class="cell empty-day cal-out-window" title="{tip}">'
                    f'<span class="num">{cell_d.day}</span><div class="dots"></div></div>'
                )
                continue
            ds = cell_d.isoformat()
            parts.append(
                f'<div class="cell empty-day" data-date="{ds}">'
                f'<span class="num">{cell_d.day}</span>'
                '<div class="dots"></div>'
                "</div>"
            )
    parts.append("</div></div>")
    return "".join(parts)


def filter_panel_html() -> str:
    rows = []
    for bid, name in BATTERS.items():
        nm_key = html_mod.escape(name.lower(), quote=True)
        rows.append(
            f'<label class="filter-row" data-name="{nm_key}">'
            f'<input type="checkbox" class="player-cb" data-batter-id="{bid}" checked>'
            f'<span class="filter-swatch" style="background:{PLAYER_COLORS[bid]}"></span>'
            f"<span>{html_mod.escape(name)}</span>"
            f"</label>"
        )
    checks = "\n".join(rows)
    return f"""
<details class="filter-dropdown" id="player-filter">
  <summary>Filter players <span class="chev">▾</span></summary>
  <div class="filter-panel">
    <input type="search" id="player-filter-search" class="filter-name-search" placeholder="Type to filter by name…" autocomplete="off" aria-label="Filter player list by name">
    {checks}
    <div class="filter-actions">
      <button type="button" class="btn" id="filter-all">Select all</button>
      <button type="button" class="btn" id="filter-none">Clear all</button>
    </div>
  </div>
</details>
"""


def main_tabs_html(slates: list[dict], n_qual: int) -> str:
    n_main = len(BATTERS)
    lines = [
        '<div class="main-tabs" role="tablist">',
        f'  <button type="button" class="main-tab active" data-tab="all" role="tab" aria-selected="true">All {n_main}</button>',
    ]
    for s in slates:
        lines.append(
            f'  <button type="button" class="main-tab" data-tab="{s["data_tab_trios"]}" '
            f'role="tab" aria-selected="false">{html_mod.escape(s["label_trios"])}</button>'
        )
        lines.append(
            f'  <button type="button" class="main-tab" data-tab="{s["data_tab_games"]}" '
            f'role="tab" aria-selected="false">{html_mod.escape(s["label_games"])}</button>'
        )
    lines.append(
        f'  <button type="button" class="main-tab" data-tab="custom-trio" '
        f'role="tab" aria-selected="false">Custom trio ({n_qual} players)</button>'
    )
    lines.append(
        '  <button type="button" class="main-tab" data-tab="trio-research" '
        'role="tab" aria-selected="false">Trio Research</button>'
    )
    lines.append("</div>")
    return "\n".join(lines)


def tab_switch_script() -> str:
    return """
<script>
(function () {
  const tabs = document.querySelectorAll('.main-tab');
  const panels = document.querySelectorAll('.tab-panel[data-tab-panel]');
  if (!tabs.length || !panels.length) return;
  tabs.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var tab = btn.getAttribute('data-tab');
      tabs.forEach(function (b) {
        var on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      panels.forEach(function (p) {
        p.hidden = p.getAttribute('data-tab-panel') !== tab;
      });
    });
  });
})();
</script>
"""


def sortable_tables_script() -> str:
    """Click any ``thead`` cell on ``table.combo-table`` / ``table.summary`` to sort (numeric-aware)."""
    return """
<script>
(function () {
  function decorateSortableHeaders() {
    document.querySelectorAll("table.combo-table").forEach(function (table) {
      var tr = table.querySelector("thead tr");
      if (!tr) return;
      Array.prototype.forEach.call(tr.cells, function (th) {
        if (th.classList.contains("sortable-col")) return;
        th.classList.add("sortable-col");
        th.setAttribute("title", "Click to sort column");
        th.setAttribute("tabindex", "0");
        th.setAttribute("role", "columnheader");
      });
    });
  }

  window.decorateSortableHeaders = decorateSortableHeaders;

  function sortTable(table, colIndex, ascending) {
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (ra, rb) {
      var ca = ra.cells[colIndex];
      var cb = rb.cells[colIndex];
      var a = ca ? ca.textContent : "";
      var b = cb ? cb.textContent : "";
      var cmp = a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
      return ascending ? cmp : -cmp;
    });
    rows.forEach(function (row) {
      tbody.appendChild(row);
    });
  }

  function applySortFromHeader(th) {
    var table = th.closest("table");
    if (!table) return;
    var tr = th.parentNode;
    var col = Array.prototype.indexOf.call(tr.cells, th);
    if (col < 0) return;
    var same = table.dataset.sortCol === String(col);
    var ascending = same ? table.dataset.sortDir !== "asc" : true;
    sortTable(table, col, ascending);
    table.dataset.sortCol = String(col);
    table.dataset.sortDir = ascending ? "asc" : "desc";
    Array.prototype.forEach.call(tr.cells, function (h) {
      h.removeAttribute("aria-sort");
      h.classList.remove("sort-asc", "sort-desc");
    });
    th.setAttribute("aria-sort", ascending ? "ascending" : "descending");
    th.classList.add(ascending ? "sort-asc" : "sort-desc");
  }

  document.addEventListener("click", function (ev) {
    var th = ev.target.closest("table.combo-table thead th.sortable-col");
    if (!th) return;
    ev.preventDefault();
    applySortFromHeader(th);
  });

  document.addEventListener("keydown", function (ev) {
    if (ev.key !== "Enter" && ev.key !== " ") return;
    var th = ev.target.closest("table.combo-table thead th.sortable-col");
    if (!th) return;
    ev.preventDefault();
    applySortFromHeader(th);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", decorateSortableHeaders);
  } else {
    decorateSortableHeaders();
  }
})();
</script>
"""


def client_script(hr_json: str, batters_j: str, colors_j: str) -> str:
    return f"""
<script>
(function () {{
  const HR_BY_DATE = {hr_json};
  const BATTERS = {batters_j};
  const COLORS = {colors_j};

  const LEVELS = ['n1', 'n2', 'n3', 'n4', 'n5', 'n5p'];

  function cellLevelClass(n) {{
    if (n <= 0) return 'empty-day';
    if (n === 1) return 'has-hr n1';
    if (n === 2) return 'has-hr n2';
    if (n === 3) return 'has-hr n3';
    if (n === 4) return 'has-hr n4';
    if (n === 5) return 'has-hr n5';
    return 'has-hr n5p';
  }}

  function selectedIds() {{
    return Array.from(document.querySelectorAll('.player-cb:checked')).map((cb) => cb.dataset.batterId);
  }}

  function buildTooltip(ds, pairs) {{
    const total = pairs.reduce((s, p) => s + p[1], 0);
    const npl = pairs.length;
    const parts = [
      ds,
      npl + ' player' + (npl === 1 ? '' : 's') + ' with HR',
      total + ' HR total',
    ];
    pairs
      .slice()
      .sort((a, b) => b[1] - a[1] || BATTERS[a[0]].localeCompare(BATTERS[b[0]]))
      .forEach(([id, cnt]) => {{
        parts.push(BATTERS[id] + ': ' + cnt + (cnt === 1 ? ' HR' : ' HRs'));
      }});
    return parts.join(' | ');
  }}

  function applyCell(cell) {{
    const ds = cell.dataset.date;
    if (!ds) return;
    const counts = HR_BY_DATE[ds] || {{}};
    const sel = new Set(selectedIds());
    const pairs = [];
    for (const id of Object.keys(counts)) {{
      if (sel.has(id) && counts[id] > 0) pairs.push([id, counts[id]]);
    }}
    pairs.sort((a, b) => BATTERS[a[0]].localeCompare(BATTERS[b[0]]));
    const n = pairs.length;

    cell.classList.remove('has-hr', 'empty-day', ...LEVELS);
    const dots = cell.querySelector('.dots');
    dots.innerHTML = '';

    if (sel.size === 0) {{
      cell.classList.add('empty-day');
      cell.title = ds + ' | (no players selected)';
      return;
    }}

    if (n === 0) {{
      cell.classList.add('empty-day');
      cell.title = ds;
      return;
    }}

    const lvl = cellLevelClass(n);
    lvl.split(' ').forEach((c) => cell.classList.add(c));
    cell.title = buildTooltip(ds, pairs);
    pairs.forEach(([id]) => {{
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.background = COLORS[id] || '#94a3b8';
      dots.appendChild(dot);
    }});
  }}

  function refreshCalendar() {{
    document.querySelectorAll('#panel-all .cell[data-date]').forEach(applyCell);
  }}

  document.querySelectorAll('.player-cb').forEach((cb) => {{
    cb.addEventListener('change', refreshCalendar);
  }});
  const filterSearch = document.getElementById('player-filter-search');
  if (filterSearch) {{
    filterSearch.addEventListener('input', function () {{
      const q = filterSearch.value.trim().toLowerCase();
      document.querySelectorAll('#player-filter .filter-row').forEach(function (row) {{
        const raw = (row.getAttribute('data-name') || '').toLowerCase();
        row.style.display = !q || raw.includes(q) ? '' : 'none';
      }});
    }});
  }}
  document.getElementById('filter-all').addEventListener('click', () => {{
    document.querySelectorAll('.player-cb').forEach((cb) => {{ cb.checked = true; }});
    if (filterSearch) filterSearch.value = '';
    document.querySelectorAll('#player-filter .filter-row').forEach((row) => {{ row.style.display = ''; }});
    refreshCalendar();
  }});
  document.getElementById('filter-none').addEventListener('click', () => {{
    document.querySelectorAll('.player-cb').forEach((cb) => {{ cb.checked = false; }});
    if (filterSearch) filterSearch.value = '';
    document.querySelectorAll('#player-filter .filter-row').forEach((row) => {{ row.style.display = ''; }});
    refreshCalendar();
  }});

  refreshCalendar();
}})();
</script>
"""


def build_custom_select_options(all_qual: dict[int, tuple[str, str]], selected_id: int) -> str:
    lines: list[str] = [f'<option value="{CUSTOM_TRIO_SEL_NONE}">None</option>']
    for bid in sorted(all_qual.keys(), key=lambda i: (all_qual[i][0].lower(), i)):
        nm, tm = all_qual[bid]
        lab = f"{nm} ({tm})" if tm else nm
        sel = " selected" if bid == selected_id else ""
        lines.append(f'<option value="{bid}"{sel}>{html_mod.escape(lab)}</option>')
    return "\n".join(lines)


def custom_trio_panel_html(
    default_trio: tuple[int, int, int],
    all_qual: dict[int, tuple[str, str]],
) -> str:
    s0 = build_custom_select_options(all_qual, default_trio[0])
    s1 = build_custom_select_options(all_qual, default_trio[1])
    s2 = build_custom_select_options(all_qual, default_trio[2])
    return f"""
<div id="panel-custom-trio" class="tab-panel" hidden data-tab-panel="custom-trio">
  <div class="custom-pick-toolbar">
    <div class="custom-pick-row">
      <label class="custom-pick-field">Player 1
        <input type="search" class="custom-sel-search" data-sel-target="custom-sel-0" placeholder="Type to filter…" autocomplete="off" aria-label="Filter player 1 list">
        <select id="custom-sel-0" class="custom-sel">{s0}</select>
      </label>
      <label class="custom-pick-field">Player 2
        <input type="search" class="custom-sel-search" data-sel-target="custom-sel-1" placeholder="Type to filter…" autocomplete="off" aria-label="Filter player 2 list">
        <select id="custom-sel-1" class="custom-sel">{s1}</select>
      </label>
      <label class="custom-pick-field">Player 3
        <input type="search" class="custom-sel-search" data-sel-target="custom-sel-2" placeholder="Type to filter…" autocomplete="off" aria-label="Filter player 3 list">
        <select id="custom-sel-2" class="custom-sel">{s2}</select>
      </label>
      <label class="custom-pick-field">All 3
        <select id="custom-sel-all3" class="custom-sel custom-sel-all3" aria-label="Clear all three player slots">
          <option value="" selected>—</option>
          <option value="none">None — deselect all</option>
        </select>
      </label>
      <button type="button" class="btn btn-primary" id="custom-trio-apply">Update trio view</button>
    </div>
    <aside class="custom-trio-suggest" aria-labelledby="custom-trio-suggest-title">
      <div id="custom-trio-suggest-title" class="custom-trio-suggest-h"><span id="custom-trio-suggest-lead">Best 10 as 3rd with</span> <span id="custom-trio-suggest-pair">—</span></div>
      <ol id="custom-trio-suggest-list" class="custom-trio-suggest-list"></ol>
    </aside>
  </div>
  <p id="custom-trio-err" class="custom-err" hidden></p>
  <h2 class="trio-games-h2">Trio overlap (same window as other tabs)</h2>
  <table class="combo-table" id="custom-trio-stats">
    <thead><tr>
      <th>Trio</th><th>Total HR</th>
      <th>Days: 0 of 3</th><th>1 of 3</th><th>2 of 3</th><th>3 of 3</th><th>Any ≥1 HR</th><th>All 3 HR</th>
    </tr></thead><tbody id="custom-trio-stats-body"><tr><td colspan="8">Loading…</td></tr></tbody></table>
  <h2 class="trio-games-h2">Tonight / slate — matchup &amp; platoon</h2>
  <div class="scroll-x"><table class="combo-table matchup-wide-table" id="custom-matchup-table">
    <thead><tr>
      <th>Player</th><th>Team</th><th>Matchup</th><th>Opp SP</th><th>BvP vs SP</th>
      <th>Career vs LHP</th><th>Career vs RHP</th><th>2026 vs LHP</th><th>2026 vs RHP</th>
    </tr></thead><tbody id="custom-matchup-tbody"></tbody></table></div>
  <h2 class="trio-games-h2">A. Same calendar day (all three had ≥1 AB)</h2>
  <table class="combo-table trio-games-table"><thead><tr><th>Trio</th><th>Days (all 3 ≥1 AB)</th><th>Days (any HR)</th><th>HR %</th></tr></thead>
  <tbody id="custom-trio-day-body"></tbody></table>
  <h2 class="trio-games-h2">B. Same MLB game (<code>game_pk</code>, 2026)</h2>
  <table class="combo-table trio-games-table"><thead><tr><th>Trio</th><th>Games (all 3 w/ PA)</th><th>Games (any HR)</th><th>HR %</th></tr></thead>
  <tbody id="custom-trio-pk-body"></tbody></table>
  <div class="legend trio-legend" id="custom-trio-legend"></div>
  <div class="months trio-months" id="custom-months"></div>
</div>
"""


def custom_trio_script(
    hr_all: str,
    ab_all: str,
    games: str,
    totals: str,
    meta: str,
    matchup: str,
    month_grids: str,
    month_keys: str,
    wf_iso: str,
    wl_iso: str,
    total_days: int,
    all_ids_json: str,
    suggest_wf_iso: str,
    suggest_wl_iso: str,
    suggest_total_days: int,
    beast_hr_json: str,
) -> str:
    return f"""
<script>
(function () {{
  const HR_ALL = {hr_all};
  const AB_ALL = {ab_all};
  const GAMES = {games};
  const TOTALS = {totals};
  const META = {meta};
  const MATCHUP = {matchup};
  const MONTH_GRIDS = {month_grids};
  const MONTH_KEYS = {month_keys};
  const WF = "{wf_iso}";
  const WL = "{wl_iso}";
  const TOTAL_DAYS = {total_days};
  const ALL_IDS = {all_ids_json};
  const SUGGEST_WF = "{suggest_wf_iso}";
  const SUGGEST_WL = "{suggest_wl_iso}";
  const SUGGEST_TOTAL_DAYS = {suggest_total_days};
  const BEAST_HR = {beast_hr_json};
  const MONTH_NAMES = ["","January","February","March","April","May","June","July","August","September","October","November","December"];

  function colorForId(id) {{
    const x = Number(id);
    const h = (x * 7919 + (x % 997)) % 360;
    return "hsl(" + h + ", 72%, 55%)";
  }}

  function parseISODate(s) {{
    const p = s.split("-");
    return new Date(Number(p[0]), Number(p[1]) - 1, Number(p[2]));
  }}

  function nextIso(ds) {{
    const x = parseISODate(ds);
    x.setDate(x.getDate() + 1);
    return x.getFullYear() + "-" + String(x.getMonth() + 1).padStart(2, "0") + "-" + String(x.getDate()).padStart(2, "0");
  }}

  function trioCountPositive(combo, ds, perDay) {{
    const c = perDay[ds] || {{}};
    let n = 0;
    for (let i = 0; i < combo.length; i++) {{
      const b = String(combo[i]);
      if ((c[b] || 0) > 0) n++;
    }}
    return n;
  }}

  function trioPlayersHr(combo, ds, hrDay) {{
    const c = hrDay[ds] || {{}};
    const out = [];
    for (let i = 0; i < combo.length; i++) {{
      const bid = combo[i];
      if ((c[String(bid)] || 0) > 0) out.push(bid);
    }}
    out.sort(function (a, b) {{
      const na = (META[String(a)] && META[String(a)].name) || String(a);
      const nb = (META[String(b)] && META[String(b)].name) || String(b);
      return na.localeCompare(nb);
    }});
    return out;
  }}

  function comboRowStats(comboRaw) {{
    const trio = comboRaw.map(Number).sort(function (a, b) {{ return a - b; }});
    const hist = [0, 0, 0, 0];
    for (let d = WF; d <= WL; d = nextIso(d)) {{
      const n = trioCountPositive(trio, d, HR_ALL);
      hist[n]++;
    }}
    let totalHr = 0;
    for (let i = 0; i < trio.length; i++) totalHr += TOTALS[String(trio[i])] || 0;
    const names = trio.map(function (b) {{ return (META[String(b)] && META[String(b)].name) || String(b); }});
    const short = names.map(function (nm) {{ const p = nm.split(" "); return p[p.length - 1]; }});
    return {{
      trio: trio,
      hist: hist,
      totalHr: totalHr,
      label: names.join(" · "),
      short: short.join(" / "),
      daysAny: hist[1] + hist[2] + hist[3],
      daysAll: hist[3],
    }};
  }}

  function trioWindowAbHr(trio) {{
    let dAb = 0;
    let dHr = 0;
    for (let d = WF; d <= WL; d = nextIso(d)) {{
      if (trioCountPositive(trio, d, AB_ALL) < 3) continue;
      dAb++;
      if (trioCountPositive(trio, d, HR_ALL) >= 1) dHr++;
    }}
    const pct = dAb ? Math.round((1000 * dHr) / dAb) / 10 : 0;
    return [dAb, dHr, pct];
  }}

  function trioGamePk(trio) {{
    const need = trio.map(String);
    let n = 0;
    let nHr = 0;
    for (const pk in GAMES) {{
      const row = GAMES[pk];
      const bats = row.b.map(String);
      let ok = true;
      for (let i = 0; i < need.length; i++) {{
        if (bats.indexOf(need[i]) < 0) {{ ok = false; break; }}
      }}
      if (!ok) continue;
      n++;
      if (row.hr) nHr++;
    }}
    const pct = n ? Math.round((1000 * nHr) / n) / 10 : 0;
    return [n, nHr, pct];
  }}

  function fmtPct(num, den, suffix) {{
    if (!den || den <= 0) return "";
    const p = (100 * num) / den;
    const suf = suffix ? String(suffix) : "";
    return ' <span class="pct">(' + p.toFixed(1) + '%' + suf + ")</span>";
  }}

  function esc(s) {{
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }}

  function buildCellTooltip(ds, trio) {{
    const ac = AB_ALL[ds] || {{}};
    const hc = HR_ALL[ds] || {{}};
    const nAb = trioCountPositive(trio, ds, AB_ALL);
    const nHr = trioCountPositive(trio, ds, HR_ALL);
    const parts = [ds, "Corner: " + nAb + " of 3 with ≥1 AB", "Fill/dots: " + nHr + " of 3 with HR"];
    const order = trio.slice().sort(function (a, b) {{
      const na = (META[String(a)] || {{}}).name || String(a);
      const nb = (META[String(b)] || {{}}).name || String(b);
      return na.localeCompare(nb);
    }});
    for (let i = 0; i < order.length; i++) {{
      const bid = order[i];
      const nm = (META[String(bid)] || {{}}).name || String(bid);
      parts.push(nm + ": " + (ac[String(bid)] || 0) + " AB, " + (hc[String(bid)] || 0) + " HR");
    }}
    return parts.join(" | ");
  }}

  function renderLegend() {{
    const el = document.getElementById("custom-trio-legend");
    if (!el) return;
    el.innerHTML =
      '<span style="color:#94a3b8;font-size:0.8rem;">Trio calendar — top-left = how many of the three had <strong>≥1 AB</strong> that day. '
      + "Fill + dots = <strong>HR</strong> among the three (1 / 2 / 3); hover lists AB + HR per player:</span>"
      + '<div class="legend-item"><span class="swatch trio-n1"></span>1 of 3 w/ HR</div>'
      + '<div class="legend-item"><span class="swatch trio-n2"></span>2 of 3 w/ HR</div>'
      + '<div class="legend-item"><span class="swatch trio-n3"></span>3 of 3 w/ HR</div>';
  }}

  function renderMonths(trio) {{
    const root = document.getElementById("custom-months");
    if (!root) return;
    root.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "trio-calendars-wrap";
    const section = document.createElement("section");
    section.className = "trio-block";
    const st = comboRowStats(trio);
    const h3 = document.createElement("h3");
    h3.innerHTML =
      esc(st.short)
      + ' <span class="trio-meta">('
      + esc(st.label)
      + ") · "
      + st.totalHr
      + " HR combined · HR day split 0/1/2/3: "
      + st.hist.join("/")
      + "</span>";
    section.appendChild(h3);
    const monthsInner = document.createElement("div");
    monthsInner.className = "months trio-months";
    const dow = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

    for (let mi = 0; mi < MONTH_KEYS.length; mi++) {{
      const key = MONTH_KEYS[mi];
      const yp = key.split("-");
      const yy = Number(yp[0]);
      const mm = Number(yp[1]);
      const weeks = MONTH_GRIDS[key];
      const monthDiv = document.createElement("div");
      monthDiv.className = "month";
      const h2 = document.createElement("h2");
      h2.textContent = MONTH_NAMES[mm] + " " + yy;
      monthDiv.appendChild(h2);
      const dowRow = document.createElement("div");
      dowRow.className = "dow";
      for (let di = 0; di < dow.length; di++) {{
        const c = document.createElement("div");
        c.textContent = dow[di];
        dowRow.appendChild(c);
      }}
      monthDiv.appendChild(dowRow);
      const grid = document.createElement("div");
      grid.className = "grid";
      for (let wi = 0; wi < weeks.length; wi++) {{
        const week = weeks[wi];
        for (let ci = 0; ci < week.length; ci++) {{
          const cellD = week[ci];
          const inMonth = Number(cellD.slice(5, 7)) === mm && Number(cellD.slice(0, 4)) === yy;
          const cell = document.createElement("div");
          if (!inMonth) {{
            cell.className = "cell trio-cell empty-day out";
            cell.innerHTML = '<span class="num"></span><div class="dots"></div>';
            grid.appendChild(cell);
            continue;
          }}
          if (cellD < WF || cellD > WL) {{
            cell.className = "cell trio-cell empty-day cal-out-window";
            cell.title = cellD + " (outside window)";
            cell.innerHTML = '<span class="num">' + Number(cellD.slice(8, 10)) + '</span><div class="dots"></div>';
            grid.appendChild(cell);
            continue;
          }}
          const dayNum = Number(cellD.slice(8, 10));
          const nAb = trioCountPositive(trio, cellD, AB_ALL);
          const nHr = trioCountPositive(trio, cellD, HR_ALL);
          const playersHr = trioPlayersHr(trio, cellD, HR_ALL);
          let bodyCls = "empty-day";
          if (nHr > 0) bodyCls = "has-hr trio-n" + nHr;
          cell.className = "cell trio-cell " + bodyCls;
          cell.dataset.date = cellD;
          cell.title = buildCellTooltip(cellD, trio);
          let dotsHtml = '<div class="dots">';
          for (let pi = 0; pi < playersHr.length; pi++) {{
            dotsHtml += '<span class="dot" style="background:' + colorForId(playersHr[pi]) + '"></span>';
          }}
          dotsHtml += "</div>";
          cell.innerHTML =
            '<span class="trio-corner-count" aria-hidden="true">' + nAb + "</span>"
            + '<span class="num">' + dayNum + "</span>"
            + dotsHtml;
          grid.appendChild(cell);
        }}
      }}
      monthDiv.appendChild(grid);
      monthsInner.appendChild(monthDiv);
    }}
    section.appendChild(monthsInner);
    wrap.appendChild(section);
    root.appendChild(wrap);
  }}

  function renderMatchupRows(trio) {{
    const tb = document.getElementById("custom-matchup-tbody");
    if (!tb) return;
    tb.innerHTML = "";
    for (let i = 0; i < trio.length; i++) {{
      const bid = trio[i];
      const row = MATCHUP[String(bid)] || {{}};
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + esc(row.name) + "</td>"
        + "<td>" + esc(row.team) + "</td>"
        + "<td>" + esc(row.matchup) + "</td>"
        + "<td>" + esc(row.opp) + "</td>"
        + "<td>" + esc(row.bvp) + "</td>"
        + "<td>" + esc(row.car_L) + "</td>"
        + "<td>" + esc(row.car_R) + "</td>"
        + "<td>" + esc(row.ytd_L) + "</td>"
        + "<td>" + esc(row.ytd_R) + "</td>";
      tb.appendChild(tr);
    }}
  }}

  function renderAll(comboRaw) {{
    const err = document.getElementById("custom-trio-err");
    const trio = comboRaw.map(Number).sort(function (a, b) {{ return a - b; }});
    const st = comboRowStats(comboRaw);
    const pool = st.totalHr > 0 ? st.totalHr : 1;
    const statsBody = document.getElementById("custom-trio-stats-body");
    if (statsBody) {{
      statsBody.innerHTML =
        "<tr><td>"
        + esc(st.label)
        + "</td><td>"
        + st.totalHr
        + fmtPct(st.totalHr, pool, " of trio")
        + "</td><td>"
        + st.hist[0]
        + fmtPct(st.hist[0], TOTAL_DAYS)
        + "</td><td>"
        + st.hist[1]
        + fmtPct(st.hist[1], TOTAL_DAYS)
        + "</td><td>"
        + st.hist[2]
        + fmtPct(st.hist[2], TOTAL_DAYS)
        + "</td><td>"
        + st.hist[3]
        + fmtPct(st.hist[3], TOTAL_DAYS)
        + "</td><td>"
        + st.daysAny
        + fmtPct(st.daysAny, TOTAL_DAYS)
        + "</td><td>"
        + st.daysAll
        + fmtPct(st.daysAll, TOTAL_DAYS)
        + "</td></tr>";
    }}
    const dayRow = trioWindowAbHr(trio);
    const pkRow = trioGamePk(trio);
    const dayBody = document.getElementById("custom-trio-day-body");
    if (dayBody) {{
      dayBody.innerHTML =
        "<tr><td>"
        + esc(st.label)
        + "</td><td>"
        + dayRow[0]
        + "</td><td>"
        + dayRow[1]
        + "</td><td>"
        + (dayRow[0] ? dayRow[2] + "%" : "—")
        + "</td></tr>";
    }}
    const pkBody = document.getElementById("custom-trio-pk-body");
    if (pkBody) {{
      pkBody.innerHTML =
        "<tr><td>"
        + esc(st.label)
        + "</td><td>"
        + pkRow[0]
        + "</td><td>"
        + pkRow[1]
        + "</td><td>"
        + (pkRow[0] ? pkRow[2] + "%" : "—")
        + "</td></tr>";
    }}
    renderLegend();
    renderMonths(comboRaw);
    renderMatchupRows(trio);
    if (err) err.hidden = true;
  }}

  function parseSlotVal(v) {{
    if (v === "{CUSTOM_TRIO_SEL_NONE}" || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }}

  function readCombo() {{
    const a = document.getElementById("custom-sel-0");
    const b = document.getElementById("custom-sel-1");
    const c = document.getElementById("custom-sel-2");
    if (!a || !b || !c) return null;
    return [parseSlotVal(a.value), parseSlotVal(b.value), parseSlotVal(c.value)];
  }}

  function daysAnyHrTrioInRange(trio, wf, wl) {{
    if (!wf || !wl || wf > wl) return 0;
    let n = 0;
    for (let d = wf; d <= wl; d = nextIso(d)) {{
      if (trioCountPositive(trio, d, HR_ALL) >= 1) n++;
    }}
    return n;
  }}

  function beastScore(bid) {{
    return Number(BEAST_HR[String(bid)] || 0);
  }}

  /** Ordinal rank 1..N among rows: higher Beast HR score first; tie daysHit desc then id. */
  function rankThirdRowsByBeast(topRows) {{
    const scored = topRows.map(function (r, idx) {{
      return {{ idx: idx, p: r.p, daysHit: r.daysHit, s: beastScore(r.p) }};
    }});
    scored.sort(function (a, b) {{
      if (b.s !== a.s) return b.s - a.s;
      if (b.daysHit !== a.daysHit) return b.daysHit - a.daysHit;
      return a.p - b.p;
    }});
    const rankByIdx = {{}};
    scored.forEach(function (o, i) {{
      rankByIdx[o.idx] = i + 1;
    }});
    return rankByIdx;
  }}

  /** Ordinal 1..N: mean Beast(anchor, a, b); tie daysHit then (a,b). */
  function rankTrioMeanBeast(topRows, anchor) {{
    const scored = topRows.map(function (r, idx) {{
      const s = (beastScore(anchor) + beastScore(r.a) + beastScore(r.b)) / 3;
      return {{ idx: idx, a: r.a, b: r.b, daysHit: r.daysHit, s: s }};
    }});
    scored.sort(function (x, y) {{
      if (y.s !== x.s) return y.s - x.s;
      if (y.daysHit !== x.daysHit) return y.daysHit - x.daysHit;
      if (x.a !== y.a) return x.a - y.a;
      return x.b - y.b;
    }});
    const rankByIdx = {{}};
    scored.forEach(function (o, i) {{
      rankByIdx[o.idx] = i + 1;
    }});
    return rankByIdx;
  }}

  /** First-seen order of distinct non-null ids from three slots. */
  function distinctPickedIds(s0, s1, s2) {{
    const out = [];
    const seen = {{}};
    const slots = [s0, s1, s2];
    for (let t = 0; t < slots.length; t++) {{
      const id = slots[t];
      if (id == null) continue;
      if (seen[id]) continue;
      seen[id] = true;
      out.push(id);
    }}
    return out;
  }}

  function renderSuggestions() {{
    const a0 = document.getElementById("custom-sel-0");
    const a1 = document.getElementById("custom-sel-1");
    const a2 = document.getElementById("custom-sel-2");
    const pairEl = document.getElementById("custom-trio-suggest-pair");
    const leadEl = document.getElementById("custom-trio-suggest-lead");
    const list = document.getElementById("custom-trio-suggest-list");
    if (!a0 || !a1 || !a2 || !list) return;
    const s0 = parseSlotVal(a0.value);
    const s1 = parseSlotVal(a1.value);
    const s2 = parseSlotVal(a2.value);
    const distinct = distinctPickedIds(s0, s1, s2);
    const denom = SUGGEST_TOTAL_DAYS > 0 ? SUGGEST_TOTAL_DAYS : 1;

    if (distinct.length === 0) {{
      if (leadEl) leadEl.textContent = "Suggestions";
      if (pairEl) pairEl.textContent = "—";
      list.innerHTML =
        "<li class=\\"suggest-muted\\">Pick at least one player (not None) to see ranked combos. Use <strong>All 3 → None</strong> to clear every slot.</li>";
      return;
    }}

    if (distinct.length === 1) {{
      const anchor = distinct[0];
      const nmA = (META[String(anchor)] || {{}}).name || String(anchor);
      if (leadEl) leadEl.textContent = "Top 10 pair combos with";
      if (pairEl) pairEl.textContent = nmA + " (by % days w/ ≥1 HR)";
      const scores = [];
      for (let i = 0; i < ALL_IDS.length; i++) {{
        const pa = ALL_IDS[i];
        if (pa === anchor) continue;
        for (let j = i + 1; j < ALL_IDS.length; j++) {{
          const pb = ALL_IDS[j];
          if (pb === anchor) continue;
          const trio = [anchor, pa, pb].slice().sort(function (x, y) {{ return x - y; }});
          const daysHit = daysAnyHrTrioInRange(trio, SUGGEST_WF, SUGGEST_WL);
          scores.push({{ a: pa, b: pb, daysHit: daysHit }});
        }}
      }}
      scores.sort(function (x, y) {{
        if (y.daysHit !== x.daysHit) return y.daysHit - x.daysHit;
        const na = (META[String(x.a)] || {{}}).name || String(x.a);
        const nb = (META[String(x.b)] || {{}}).name || String(x.b);
        const nc = (META[String(y.a)] || {{}}).name || String(y.a);
        const nd = (META[String(y.b)] || {{}}).name || String(y.b);
        const sa = na + "|" + nb;
        const sb = nc + "|" + nd;
        return sa.localeCompare(sb);
      }});
      const top = scores.slice(0, 10);
      const trioBeastRk = rankTrioMeanBeast(top, anchor);
      list.innerHTML = "";
      for (let k = 0; k < top.length; k++) {{
        const row = top[k];
        const nma = (META[String(row.a)] || {{}}).name || String(row.a);
        const nmb = (META[String(row.b)] || {{}}).name || String(row.b);
        const pct = SUGGEST_TOTAL_DAYS > 0 ? ((100 * row.daysHit) / denom).toFixed(1) : "0.0";
        const br = trioBeastRk[k] || k + 1;
        const li = document.createElement("li");
        li.innerHTML =
          "<strong>"
          + esc(nma)
          + " + "
          + esc(nmb)
          + " ("
          + br
          + ")</strong> · "
          + row.daysHit
          + " day"
          + (row.daysHit === 1 ? "" : "s")
          + ' w/ ≥1 HR <span class="suggest-pct">('
          + pct
          + "% of season window)</span>";
        list.appendChild(li);
      }}
      if (top.length === 0) {{
        list.innerHTML = "<li class=\\"suggest-muted\\">No candidate pairs.</li>";
      }}
      return;
    }}

    let f0;
    let f1;
    if (s0 != null && s1 != null) {{
      f0 = s0;
      f1 = s1;
    }} else {{
      const nn = [s0, s1, s2].filter(function (x) {{ return x != null; }});
      const uo = [];
      const seen2 = {{}};
      for (let z = 0; z < nn.length; z++) {{
        const id = nn[z];
        if (seen2[id]) continue;
        seen2[id] = true;
        uo.push(id);
      }}
      if (uo.length < 2) {{
        if (leadEl) leadEl.textContent = "Suggestions";
        if (pairEl) pairEl.textContent = "—";
        list.innerHTML = "<li class=\\"suggest-muted\\">Could not resolve a fixed pair.</li>";
        return;
      }}
      f0 = uo[0];
      f1 = uo[1];
    }}
    if (!Number.isFinite(f0) || !Number.isFinite(f1)) {{
      if (leadEl) leadEl.textContent = "Best 10 as 3rd with";
      if (pairEl) pairEl.textContent = "—";
      list.innerHTML = "<li class=\\"suggest-muted\\">Invalid player selection.</li>";
      return;
    }}
    const nm0 = (META[String(f0)] || {{}}).name || String(f0);
    const nm1 = (META[String(f1)] || {{}}).name || String(f1);
    if (leadEl) leadEl.textContent = "Best 10 as 3rd with";
    if (pairEl) pairEl.textContent = nm0 + " + " + nm1;
    const scores2 = [];
    for (let i = 0; i < ALL_IDS.length; i++) {{
      const p = ALL_IDS[i];
      if (p === f0 || p === f1) continue;
      const trio = [p, f0, f1].slice().sort(function (x, y) {{ return x - y; }});
      const daysHit = daysAnyHrTrioInRange(trio, SUGGEST_WF, SUGGEST_WL);
      scores2.push({{ p: p, daysHit: daysHit }});
    }}
    scores2.sort(function (x, y) {{
      if (y.daysHit !== x.daysHit) return y.daysHit - x.daysHit;
      const na = (META[String(x.p)] || {{}}).name || "";
      const nb = (META[String(y.p)] || {{}}).name || "";
      return na.localeCompare(nb);
    }});
    const top2 = scores2.slice(0, 10);
    const beastRk2 = rankThirdRowsByBeast(top2);
    list.innerHTML = "";
    for (let j = 0; j < top2.length; j++) {{
      const row = top2[j];
      const nm = (META[String(row.p)] || {{}}).name || String(row.p);
      const pct = SUGGEST_TOTAL_DAYS > 0 ? ((100 * row.daysHit) / denom).toFixed(1) : "0.0";
      const br = beastRk2[j] || j + 1;
      const li = document.createElement("li");
      li.innerHTML =
        "<strong>"
        + esc(nm)
        + " ("
        + br
        + ")</strong> · "
        + row.daysHit
        + " day"
        + (row.daysHit === 1 ? "" : "s")
        + ' w/ ≥1 HR <span class="suggest-pct">('
        + pct
        + "% of season window)</span>";
      list.appendChild(li);
    }}
    if (top2.length === 0) {{
      list.innerHTML = "<li class=\\"suggest-muted\\">No candidates.</li>";
    }}
  }}

  function applyCustom() {{
    const err = document.getElementById("custom-trio-err");
    const combo = readCombo();
    if (!combo) return;
    renderSuggestions();
    const valid = combo.filter(function (x) {{ return x != null; }});
    if (valid.length === 0) {{
      if (err) {{
        err.hidden = false;
        err.textContent = "All slots are None — pick three players to update the trio view.";
      }}
      return;
    }}
    if (valid.length !== 3) {{
      if (err) {{
        err.hidden = false;
        err.textContent = "Select three players (replace any None slot).";
      }}
      return;
    }}
    const uniq = new Set(valid.map(String));
    if (uniq.size !== 3) {{
      if (err) {{
        err.hidden = false;
        err.textContent = "Choose three different players.";
      }}
      return;
    }}
    if (err) err.hidden = true;
    renderAll(valid);
  }}

  function wireSelectSearch(inp, sel) {{
    if (!inp || !sel) return;
    function clearFilter() {{
      inp.value = "";
      Array.from(sel.options).forEach(function (opt) {{ opt.hidden = false; }});
    }}
    function applyFilter() {{
      const q = inp.value.trim().toLowerCase();
      const cur = String(sel.value);
      Array.from(sel.options).forEach(function (opt) {{
        const t = (opt.textContent || "").toLowerCase();
        const keep =
          !q ||
          t.includes(q) ||
          String(opt.value) === cur ||
          opt.value === "{CUSTOM_TRIO_SEL_NONE}";
        opt.hidden = !keep;
      }});
    }}
    inp.addEventListener("input", applyFilter);
    sel.addEventListener("change", clearFilter);
    inp.addEventListener("keydown", function (ev) {{
      if (ev.key !== "Enter") return;
      const q = inp.value.trim().toLowerCase();
      if (!q) return;
      ev.preventDefault();
      let pick = null;
      Array.from(sel.options).forEach(function (opt) {{
        if (pick) return;
        const t = (opt.textContent || "").toLowerCase();
        if (t.includes(q)) pick = opt;
      }});
      if (pick) {{
        sel.value = pick.value;
        clearFilter();
      }}
    }});
  }}
  ["custom-sel-0", "custom-sel-1", "custom-sel-2"].forEach(function (id) {{
    const sel = document.getElementById(id);
    const inp = document.querySelector('.custom-sel-search[data-sel-target="' + id + '"]');
    wireSelectSearch(inp, sel);
    if (sel) sel.addEventListener("change", renderSuggestions);
  }});

  const all3 = document.getElementById("custom-sel-all3");
  if (all3) {{
    all3.addEventListener("change", function () {{
      if (all3.value !== "none") return;
      ["custom-sel-0", "custom-sel-1", "custom-sel-2"].forEach(function (id) {{
        const s = document.getElementById(id);
        if (s) s.value = "{CUSTOM_TRIO_SEL_NONE}";
      }});
      all3.value = "";
      renderSuggestions();
    }});
  }}

  const btn = document.getElementById("custom-trio-apply");
  if (btn) btn.addEventListener("click", applyCustom);
  applyCustom();
}})();
</script>
"""


def main() -> None:
    slates = slate_configs()
    by_date, season_totals = load_hr_by_date(BATTERS)
    today = date.today()
    yesterday = today - timedelta(days=1)
    window_first = STATCAST_WINDOW_START
    window_last = yesterday
    if window_last < window_first:
        window_last = window_first

    y, m = window_first.year, window_first.month
    months: list[tuple[int, int]] = []
    while (y, m) <= (window_last.year, window_last.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    hr_j = hr_by_date_json(by_date)
    bj = batters_json()
    cj = colors_json()

    all_qual = load_all_qualifying_batters()
    all_ids = sorted(all_qual.keys())
    sorted_name_ids = sorted(all_qual.keys(), key=lambda i: (all_qual[i][0].lower(), i))
    if len(sorted_name_ids) >= 3:
        default_trio = (sorted_name_ids[0], sorted_name_ids[1], sorted_name_ids[2])
    else:
        pool_merged: list[int] = []
        seen: set[int] = set()
        for bid in sorted_name_ids + list(BATTERS.keys()):
            if bid not in seen:
                seen.add(bid)
                pool_merged.append(bid)
        while len(pool_merged) < 3:
            pool_merged.append(pool_merged[-1])
        default_trio = (pool_merged[0], pool_merged[1], pool_merged[2])

    hr_all, totals_all_qual = load_hr_by_date_ids(all_ids)
    ab_all = load_ab_by_date_ids(all_ids)
    pa_all = load_pa_by_date_ids(all_ids)
    hr_all_j = hr_by_date_json(hr_all)
    ab_all_j = hr_by_date_json(ab_all)
    games_j = json.dumps(load_games_lookup_2026(), separators=(",", ":"))
    totals_all_j = json.dumps({str(k): int(v) for k, v in totals_all_qual.items()}, separators=(",", ":"))
    meta_j = json.dumps(
        {str(k): {"name": all_qual[k][0], "team": all_qual[k][1]} for k in all_qual},
        separators=(",", ":"),
    )
    sc_cols = ["batter", "pitcher", "game_year", "p_throws", "is_ab", "is_hit", "is_hr", "is_strikeout"]
    sc_df = pd.read_parquet(PA_PATH, columns=sc_cols, filters=[("batter", "in", all_ids)])
    slate_idx, pit_ids = load_slate_batter_index(MATCHUPS_JSON)
    throws_map = _pitcher_throws_map(pit_ids)
    matchup_obj = build_matchup_json_for_batters(all_ids, sc_df, slate_idx, throws_map, all_qual)
    matchup_j = json.dumps(matchup_obj, separators=(",", ":"))
    month_grids_j = json.dumps(build_month_grids(months), separators=(",", ":"))
    month_keys_j = json.dumps([f"{yy}-{mm:02d}" for yy, mm in months], separators=(",", ":"))
    total_days_int = window_day_count(window_first, window_last)
    suggest_end_eff = min(window_last, SUGGEST_RANKING_END)
    if suggest_end_eff < STATCAST_WINDOW_START:
        suggest_end_eff = STATCAST_WINDOW_START
    suggest_total_days_int = window_day_count(STATCAST_WINDOW_START, suggest_end_eff)
    all_ids_j = json.dumps(all_ids, separators=(",", ":"))
    beast_scores = load_beast_hr_score_by_batter(all_ids)
    qual_pa = load_qualifying_pa_map()
    scheduled_by_iso = load_scheduled_qualifying_batters_by_date(MATCHUPS_JSON, set(all_ids))
    slate_end = max_game_date_in_matchups(MATCHUPS_JSON)
    picker_last = max(window_last, today)
    if slate_end is not None:
        picker_last = max(picker_last, slate_end)
    trio_research_by_day, trio_eligible_by_day, trio_pool_source, trio_precomputed = (
        build_trio_research_trios_by_day(
            window_first,
            picker_last,
            window_first,
            window_last,
            scheduled_by_iso,
            pa_all,
            hr_all,
            totals_all_qual,
            all_ids,
            qual_pa,
            today,
            window_last,
        )
    )
    if slate_end is not None and window_first <= slate_end <= picker_last:
        trio_default_day = slate_end
    else:
        trio_default_day = today if today <= picker_last else picker_last
    beast_j = json.dumps({str(k): round(v, 8) for k, v in beast_scores.items()}, separators=(",", ":"))
    custom_script_out = custom_trio_script(
        hr_all_j,
        ab_all_j,
        games_j,
        totals_all_j,
        meta_j,
        matchup_j,
        month_grids_j,
        month_keys_j,
        window_first.isoformat(),
        window_last.isoformat(),
        total_days_int,
        all_ids_j,
        STATCAST_WINDOW_START.isoformat(),
        suggest_end_eff.isoformat(),
        suggest_total_days_int,
        beast_j,
    )

    css = """
    :root { --bg:#0f172a; --panel:#1e293b; --muted:#94a3b8; --text:#f1f5f9; }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
    h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 8px; }
    .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; max-width: 52rem; line-height: 1.45; }
    .best-callout { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.35); border-radius: 8px; padding: 12px 14px; margin: 0 0 20px; max-width: 52rem; font-size: 0.92rem; }
    .main-tabs { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .main-tab { padding: 10px 18px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: var(--text); font-size: 0.9rem; cursor: pointer; }
    .main-tab:hover { background: #334155; }
    .main-tab.active { background: #3b82f6; border-color: #60a5fa; font-weight: 600; }
    .tab-panel[hidden] { display: none !important; }
    .tab-panel:not([hidden]) { display: block !important; }
    .trio-games-h2 { font-size: 1.05rem; margin: 22px 0 10px; color: #e2e8f0; font-weight: 600; }
    .trio-games-h2:first-of-type { margin-top: 4px; }
    .combo-table { border-collapse: collapse; font-size: 0.82rem; width: 100%; max-width: 100%; margin-bottom: 20px; table-layout: auto; }
    .combo-table th, .combo-table td { border: 1px solid #334155; padding: 8px 10px; text-align: right; word-break: break-word; hyphens: auto; }
    .combo-table th:first-child, .combo-table td:first-child,
    .combo-table th:nth-child(2), .combo-table td:nth-child(2) { text-align: left; }
    .combo-table th { background: #1e293b; color: #94a3b8; font-weight: 500; white-space: normal; vertical-align: bottom; line-height: 1.25; }
    .combo-table thead th.sortable-col { cursor: pointer; user-select: none; }
    .combo-table thead th.sortable-col:hover { color: #e2e8f0; background: #334155; }
    .combo-table thead th.sortable-col.sort-asc::after { content: " \\25B2"; font-size: 0.65em; opacity: 0.85; }
    .combo-table thead th.sortable-col.sort-desc::after { content: " \\25BC"; font-size: 0.65em; opacity: 0.85; }
    .combo-caption { caption-side: bottom; text-align: left; font-size: 0.76rem; color: #94a3b8; padding: 10px 4px 0; max-width: 58rem; line-height: 1.45; }
    .combo-table .pct { color: #94a3b8; font-size: 0.78em; font-weight: 400; }
    .combo-table td.trio-pct-col { color: #94a3b8; font-size: 0.92em; }
    .trio-calendars-wrap { display: flex; flex-direction: column; gap: 36px; margin-top: 8px; }
    .trio-block h3 { font-size: 1rem; margin: 0 0 12px; color: #e2e8f0; font-weight: 600; }
    .trio-meta { font-weight: 400; color: var(--muted); font-size: 0.82rem; }
    .trio-legend { margin-bottom: 8px; }
    .global-top10-trios { background: var(--panel); border: 1px solid #334155; border-radius: 10px; padding: 14px 16px 16px; margin-bottom: 20px; max-width: 100%; box-sizing: border-box; }
    .global-top10-h2 { font-size: 1.05rem; margin: 0 0 12px; color: #e2e8f0; font-weight: 600; }
    .global-top10-table-wrap { margin-top: 4px; width: 100%; overflow-x: visible; }
    .global-top10-table { width: 100%; max-width: 100%; font-size: 0.76rem; }
    .trio-research-filters { display: flex; flex-wrap: wrap; align-items: center; gap: 14px 20px; margin-bottom: 10px; }
    .trio-research-note { margin-top: 0; margin-bottom: 12px; max-width: 58rem; }
    .trio-research-eligible-line { font-size: 0.88rem; color: #cbd5e1; margin: 0 0 14px; }
    .trio-research-day-label { font-weight: 600; color: #e2e8f0; }
    td.trio-research-empty { text-align: left; color: var(--muted); font-style: italic; }
    .trio-research-date-label { display: inline-flex; align-items: center; gap: 8px; font-size: 0.88rem; color: #cbd5e1; }
    .trio-research-date-label input[type="date"] { padding: 6px 10px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: var(--text); font-size: 0.85rem; }
    .filter-dropdown { margin-bottom: 20px; max-width: 520px; }
    .filter-dropdown summary { cursor: pointer; list-style: none; padding: 10px 14px; background: var(--panel); border-radius: 8px; border: 1px solid #334155; user-select: none; font-size: 0.95rem; font-weight: 500; display: flex; align-items: center; justify-content: space-between; }
    .filter-dropdown summary::-webkit-details-marker { display: none; }
    .filter-dropdown[open] summary .chev { transform: rotate(180deg); display: inline-block; }
    .filter-panel { margin-top: 10px; padding: 14px 16px; background: var(--panel); border-radius: 8px; border: 1px solid #334155; }
    .filter-hint { margin: 0 0 12px; font-size: 0.78rem; color: var(--muted); line-height: 1.4; }
    .filter-row { display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 0.9rem; padding: 6px 0; border-bottom: 1px solid rgba(51,65,85,0.6); }
    .filter-row:last-of-type { border-bottom: none; }
    .filter-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; border: 1px solid rgba(255,255,255,0.2); }
    .filter-actions { display: flex; gap: 10px; margin-top: 14px; padding-top: 12px; border-top: 1px solid #334155; }
    .filter-name-search { width: 100%; max-width: 28rem; padding: 8px 10px; margin-bottom: 12px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: var(--text); font-size: 0.88rem; }
    .filter-name-search::placeholder { color: #64748b; }
    .btn { padding: 6px 12px; border-radius: 6px; border: 1px solid #475569; background: #334155; color: var(--text); font-size: 0.8rem; cursor: pointer; }
    .btn:hover { background: #475569; }
    .legend { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 20px; align-items: center; }
    .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; }
    .swatch { width: 18px; height: 18px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.15); }
    .months {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px 24px;
      align-items: start;
      max-width: 1100px;
    }
    @media (max-width: 820px) {
      .months { grid-template-columns: 1fr; }
    }
    .month { background: var(--panel); border-radius: 12px; padding: 14px 16px 16px; min-width: 0; }
    .month h2 { margin: 0 0 12px; font-size: 1.05rem; color: #e2e8f0; }
    .dow { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; margin-bottom: 6px; }
    .dow div { text-align: center; font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
    .grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }
    .cell { aspect-ratio: 1; min-height: 40px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 0.72rem; position: relative; border: 1px solid transparent; }
    .cell.trio-cell { position: relative; }
    .trio-corner-count {
      position: absolute;
      top: 3px;
      left: 4px;
      z-index: 1;
      font-size: 0.68rem;
      font-weight: 800;
      color: #f8fafc;
      line-height: 1;
      letter-spacing: -0.03em;
      text-shadow: 0 1px 2px rgba(0,0,0,0.85);
      pointer-events: none;
    }
    .cell.trio-cell.empty-day:not(.has-hr) .trio-corner-count { color: rgba(248,250,252,0.5); }
    .cell.trio-cell.trio-n3 .trio-corner-count { color: #0f172a; text-shadow: 0 0 1px rgba(255,255,255,0.5); }
    .cell.out { opacity: 0.22; pointer-events: none; }
    .cell.cal-out-window { opacity: 0.26; pointer-events: none; background: rgba(255,255,255,0.02); }
    .window-line { font-size: 0.82rem; color: #94a3b8; margin: -6px 0 18px; max-width: 52rem; line-height: 1.45; }
    .cell.empty-day { background: rgba(255,255,255,0.04); }
    .cell.has-hr { cursor: default; border-color: rgba(255,255,255,0.12); }
    .cell .num { font-weight: 600; font-size: 0.82rem; }
    .cell .dots { display: flex; flex-wrap: wrap; gap: 2px; justify-content: center; max-width: 100%; padding: 0 2px; margin-top: 3px; min-height: 7px; }
    .dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
    .n1 { background: linear-gradient(135deg, #22d3ee 0%, #0891b2 100%); }
    .n2 { background: linear-gradient(135deg, #4ade80 0%, #16a34a 100%); }
    .n3 { background: linear-gradient(135deg, #facc15 0%, #ca8a04 100%); color: #1c1917; }
    .n4 { background: linear-gradient(135deg, #fb923c 0%, #ea580c 100%); }
    .n5 { background: linear-gradient(135deg, #f472b6 0%, #db2777 100%); }
    .n5p { background: linear-gradient(135deg, #c084fc 0%, #7c3aed 100%); }
    .trio-n1 { background: linear-gradient(135deg, #22d3ee 0%, #0891b2 100%); }
    .trio-n2 { background: linear-gradient(135deg, #4ade80 0%, #16a34a 100%); }
    .trio-n3 { background: linear-gradient(135deg, #f472b6 0%, #db2777 100%); }
    table.summary { border-collapse: collapse; font-size: 0.85rem; margin-top: 28px; width: 100%; max-width: 42rem; table-layout: auto; }
    table.summary th, table.summary td { border: 1px solid #334155; padding: 8px 12px; text-align: left; word-break: break-word; }
    table.summary th { background: #1e293b; color: #94a3b8; font-weight: 500; white-space: normal; vertical-align: bottom; }
    .table-note { font-size: 0.78rem; color: var(--muted); margin-top: 8px; max-width: 42rem; }
    .custom-pick-toolbar { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 22px 36px; margin-bottom: 8px; }
    .custom-pick-toolbar .custom-pick-row { display: flex; flex-wrap: wrap; gap: 14px 18px; align-items: flex-end; margin-bottom: 0; flex: 2 1 420px; }
    .custom-trio-suggest { flex: 1 1 260px; max-width: 480px; background: var(--panel); border: 1px solid #334155; border-radius: 10px; padding: 12px 14px 14px; }
    .custom-trio-suggest-h { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; line-height: 1.35; margin: 0 0 10px; }
    .custom-trio-suggest-list { margin: 0; padding-left: 1.15rem; font-size: 0.82rem; color: #cbd5e1; line-height: 1.45; }
    .custom-trio-suggest-list li { margin-bottom: 6px; }
    .suggest-pct { color: #94a3b8; font-weight: 500; }
    .suggest-muted { color: #64748b; font-style: italic; }
    .custom-pick-row label.custom-pick-field { display: flex; flex-direction: column; gap: 6px; font-size: 0.82rem; color: #cbd5e1; }
    .custom-sel-search { width: 100%; min-width: 220px; max-width: 360px; padding: 6px 10px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: var(--text); font-size: 0.82rem; }
    .custom-sel-search::placeholder { color: #64748b; }
    .custom-sel-hint { margin-top: 0; margin-bottom: 18px; max-width: 52rem; }
    .custom-sel { min-width: 240px; max-width: 360px; padding: 8px 10px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: var(--text); font-size: 0.85rem; }
    .custom-sel-all3 { min-width: 11rem; max-width: 16rem; }
    .btn-primary { background: #2563eb; border-color: #3b82f6; font-weight: 600; }
    .btn-primary:hover { background: #1d4ed8; }
    .custom-err { color: #fca5a5; font-size: 0.88rem; margin: 8px 0; }
    .scroll-x { overflow-x: visible; max-width: 100%; margin-bottom: 16px; }
    .matchup-wide-table { width: 100%; max-width: 100%; font-size: 0.76rem; table-layout: auto; }
    .matchup-wide-table td:nth-child(n+5) { white-space: normal; min-width: 0; max-width: 22rem; vertical-align: top; word-break: break-word; }
    """

    parts: list[str] = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>2026 HR calendar — 5 sluggers</title><style>",
        css,
        "</style></head><body>",
        "<h1>2026 home run calendar</h1>",
        main_tabs_html(slates, len(all_qual)),
        '<div id="panel-all" class="tab-panel" data-tab-panel="all">',
        filter_panel_html(),
        '<div class="legend">',
    ]
    for lab, cls in [("1", "n1"), ("2", "n2"), ("3", "n3"), ("4", "n4"), ("5", "n5")]:
        parts.append(f'<div class="legend-item"><span class="swatch {cls}"></span>{lab}</div>')
    parts.append("</div><div class=\"legend\">")
    for bid, name in BATTERS.items():
        parts.append(
            f'<div class="legend-item"><span class="swatch" style="background:{PLAYER_COLORS[bid]}"></span>{name}</div>'
        )
    parts.append("</div><div class=\"months\">")
    for yy, mm in months:
        parts.append(month_blocks(yy, mm, window_first, window_last))
    parts.append("</div>")

    parts.append(
        "<table class=\"combo-table summary\"><thead><tr><th>Player</th><th>2026 HR (this file)</th></tr></thead><tbody>"
    )
    for bid in sorted(BATTERS.keys(), key=lambda b: (-season_totals[b], BATTERS[b])):
        parts.append(f"<tr><td>{BATTERS[bid]}</td><td>{season_totals[bid]}</td></tr>")
    parts.append("</tbody></table>")
    parts.append(
        '<p class="table-note">Season HR totals in the table are from the full file (not affected by the filter).</p>'
    )
    parts.append("</div>")  # end panel-all

    for slate in slates:
        batters = slate["batters"]
        hr_slate, totals_slate = load_hr_by_date(batters)
        ab_slate = load_ab_by_date(batters)
        ranked_slate = rank_combos(hr_slate, totals_slate, window_first, window_last, batters)
        pa_slate = load_pa_slice_2026(batters)
        pool_combined = sum(totals_slate.values())
        parts.append(
            trios_panel_html(
                slate,
                ranked_slate,
                ab_slate,
                hr_slate,
                months,
                window_first,
                window_last,
                pool_combined,
            )
        )
        parts.append(
            trio_games_panel_html(
                slate,
                ranked_slate,
                pa_slate,
                ab_slate,
                hr_slate,
                window_first,
                window_last,
            )
        )

    parts.append(custom_trio_panel_html(default_trio, all_qual))
    parts.append(
        trio_research_panel_html(
            trio_research_by_day,
            trio_eligible_by_day,
            trio_pool_source,
            trio_precomputed,
            all_qual,
            totals_all_qual,
            beast_scores,
            window_first,
            picker_last,
            window_first,
            window_last,
            trio_default_day,
        )
    )
    parts.append(tab_switch_script())
    parts.append(sortable_tables_script())
    parts.append(client_script(hr_j, bj, cj))
    parts.append(custom_script_out)
    parts.append("</body></html>")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
