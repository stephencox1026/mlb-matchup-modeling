#!/usr/bin/env python3
"""
Sections 11–13: barrel-focused tables for today's slate.

- Section 11: career barrels vs tonight's opposing pitcher (dates + counts).
- Section 12: barrels in the last 14 days (any pitcher) for slate batters.
- Section 13: season-to-date barrel totals for slate batters + most recent barrel date.

Barrel definition: Statcast `launch_speed_angle == 6` (barrel bucket), OR `barrel==1`
when that column is populated (matches MLB / pybaseball conventions used elsewhere).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PA_PATH = "data/raw/statcast_pa_level_league.parquet"
PREDS_PATH = "data/reports/todays_matchup_predictions.json"
MATCHUPS_PATH = "data/raw/todays_matchups.json"
REPORT_COMBINED = Path("data/reports/sections_11_13.md")


def is_barrel_series(df: pd.DataFrame) -> pd.Series:
    lsa = df["launch_speed_angle"] if "launch_speed_angle" in df.columns else None
    br = df["barrel"] if "barrel" in df.columns else None
    out = pd.Series(False, index=df.index, dtype=bool)
    if lsa is not None:
        out = out | (lsa == 6)
    if br is not None:
        out = out | (br.fillna(0).astype(int) > 0)
    return out


def load_pa() -> pd.DataFrame:
    pa = pd.read_parquet(
        PA_PATH,
        columns=["game_date", "game_year", "batter", "pitcher", "launch_speed_angle", "barrel"],
    )
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")
    pa = pa.dropna(subset=["game_date", "batter", "pitcher"])
    pa["batter"] = pa["batter"].astype(int)
    pa["pitcher"] = pa["pitcher"].astype(int)
    pa["is_barrel"] = is_barrel_series(pa)
    return pa


def slate_date_from_matchups() -> pd.Timestamp:
    with open(MATCHUPS_PATH) as f:
        m = json.load(f)
    if not m:
        return pd.Timestamp.today().normalize()
    dates = [pd.to_datetime(x["game_date"], errors="coerce") for x in m if x.get("game_date")]
    dates = [d for d in dates if pd.notna(d)]
    return max(dates) if dates else pd.Timestamp.today().normalize()


def date_counts_str(dates: list[pd.Timestamp]) -> str:
    """Compress sorted dates into 'YYYY-MM-DD×n; ...'"""
    if not dates:
        return "—"
    by: dict[str, int] = defaultdict(int)
    for d in sorted(dates):
        k = pd.Timestamp(d).strftime("%Y-%m-%d")
        by[k] += 1
    parts = [f"{k}×{v}" if v > 1 else k for k, v in sorted(by.items())]
    return "; ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recent-days", type=int, default=14, help="Window for Section 12 (default 14)")
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write data/reports/sections_11_13.md (stdout only)",
    )
    args = p.parse_args()

    slate = slate_date_from_matchups().normalize()
    window_start = slate - pd.Timedelta(days=args.recent_days - 1)
    season_year = int(slate.year)

    with open(PREDS_PATH) as f:
        preds = json.load(f)
    df_pred = pd.DataFrame(preds)
    if df_pred.empty:
        print("No predictions file — nothing to emit.", file=sys.stderr)
        return

    id_to_name = (
        df_pred.drop_duplicates("batter_mlbam_id")
        .set_index("batter_mlbam_id")["batter_name"]
        .to_dict()
    )

    pairs = df_pred[
        ["batter_mlbam_id", "pitcher_mlbam_id", "batter_name", "pitcher_name", "batter_team"]
    ].drop_duplicates()
    pairs = pairs[(pairs["batter_mlbam_id"].notna()) & (pairs["pitcher_mlbam_id"].notna())]
    pairs = pairs[(pairs["pitcher_mlbam_id"].astype(int) > 0)]

    pa = load_pa()
    br = pa[pa["is_barrel"]].copy()

    # --- Section 11 ---
    rows11 = []
    for _, r in pairs.iterrows():
        bid, pid = int(r["batter_mlbam_id"]), int(r["pitcher_mlbam_id"])
        sub = br[(br["batter"] == bid) & (br["pitcher"] == pid)]
        n = len(sub)
        if n == 0:
            continue
        rows11.append(
            {
                "label": f"{r['batter_name']} ({r['batter_team']}) vs {r['pitcher_name']}",
                "n": n,
                "when": date_counts_str(sub["game_date"].tolist()),
            }
        )
    rows11.sort(key=lambda x: (-x["n"], x["label"]))

    # --- Section 12: slate batters only, any pitcher ---
    batter_ids = df_pred["batter_mlbam_id"].dropna().astype(int).unique()
    sub12 = br[
        (br["batter"].isin(batter_ids))
        & (br["game_date"] >= window_start)
        & (br["game_date"] <= slate)
    ]
    rows12_map: dict[int, list[pd.Timestamp]] = defaultdict(list)
    for bid, g in sub12.groupby("batter"):
        rows12_map[int(bid)] = g["game_date"].tolist()
    rows12 = []
    for bid, dates in rows12_map.items():
        rows12.append(
            {
                "player": id_to_name.get(bid, f"ID {bid}"),
                "n": len(dates),
                "when": date_counts_str(dates),
            }
        )
    rows12.sort(key=lambda x: (-x["n"], x["player"]))

    # --- Section 13: season barrels for slate batters ---
    if "game_year" not in br.columns:
        br["game_year"] = br["game_date"].dt.year
    sub13 = br[(br["batter"].isin(batter_ids)) & (br["game_year"] == season_year)]
    rows13 = []
    for bid, g in sub13.groupby("batter"):
        bid = int(bid)
        last_dt = g["game_date"].max()
        rows13.append(
            {
                "player": id_to_name.get(bid, f"ID {bid}"),
                "season_barrels": len(g),
                "most_recent": last_dt.strftime("%Y-%m-%d") if pd.notna(last_dt) else "—",
            }
        )
    rows13.sort(key=lambda x: (-x["season_barrels"], x["player"]))

    def section_block(num: int, title: str, header: str, sep: str, lines: list[str]) -> list[str]:
        block = [f"## Section {num}: {title}", "", header, sep, *lines, ""]
        return block

    blocks = [
        section_block(
            11,
            "Barrels — vs tonight's opposing pitcher (career in dataset)",
            "| Batter vs pitcher | Barrels vs pitcher | Dates (YYYY-MM-DD, ×n if >1 that day) |",
            "|---|---:|---|",
            [f"| {r['label']} | {r['n']} | {r['when']} |" for r in rows11]
            or ["| *(no slate batter has a recorded barrel vs their listed pitcher in this dataset)* | — | — |"],
        ),
        section_block(
            12,
            f"Recent barrels — past {args.recent_days} days (slate batters, any pitcher)",
            f"| Player | Barrels ({args.recent_days}d through {slate.date()}) | Dates |",
            "|---|---:|---|",
            [f"| {r['player']} | {r['n']} | {r['when']} |" for r in rows12]
            or ["| *(no barrels in window for tonight's projected batters)* | — | — |"],
        ),
        section_block(
            13,
            f"Season barrels ({season_year}) — slate batters, most recent noted",
            "| Player | Season barrels | Most recent barrel |",
            "|---|---:|---|",
            [f"| {r['player']} | {r['season_barrels']} | {r['most_recent']} |" for r in rows13]
            or [f"| *(no {season_year} barrels in dataset for slate batters)* | — | — |"],
        ),
    ]

    preamble = [
        f"# Sections 11–13 — snapshot {slate.date()}",
        "",
        "Barrel definition: Statcast `launch_speed_angle == 6` (barrel bucket), or `barrel==1` when present.",
        "",
    ]
    full_text = "\n".join(preamble + [line for b in blocks for line in b]).rstrip() + "\n"
    print(full_text, end="")

    if not args.no_write:
        REPORT_COMBINED.parent.mkdir(parents=True, exist_ok=True)
        REPORT_COMBINED.write_text(full_text, encoding="utf-8")
        rp = REPORT_COMBINED.parent
        (rp / "section_11.md").write_text("\n".join(blocks[0]).rstrip() + "\n", encoding="utf-8")
        (rp / "section_12.md").write_text("\n".join(blocks[1]).rstrip() + "\n", encoding="utf-8")
        (rp / "section_13.md").write_text("\n".join(blocks[2]).rstrip() + "\n", encoding="utf-8")

        html_py = os.path.join(ROOT, "src", "gen_matchup_dashboard_html.py")
        if os.path.isfile(html_py):
            sd = slate.date().isoformat() if hasattr(slate, "date") else str(slate)
            subprocess.run(
                [sys.executable, html_py, "--date", sd],
                cwd=ROOT,
                check=False,
            )


if __name__ == "__main__":
    main()
