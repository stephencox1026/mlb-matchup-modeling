#!/usr/bin/env python3
"""
Build / refresh lineup slot datasets:

  Layer 1: data/raw/lineup_slots_by_game.parquet
    One row per starter batting-order slot (game_pk, batter_id, bat_slot 1–9,
    team_id, opp_starter_id, opp_starter_throws L/R, game_date).

  Layer 2: data/raw/batter_lineup_spot_last10_vs_hand.parquet
    Per batter × opposing starter hand: last up to 10 games (game_date desc),
    median_slot, mode_slot, n_games, slots_json, asof_date.

Sources: MLB Stats API schedule + boxscore + people (pitchHand).

Daily usage (from run_dual_model_daily):
  python3 src/refresh_batter_lineup_context.py --asof YYYY-MM-DD

First-time / backfill:
  python3 src/refresh_batter_lineup_context.py --asof YYYY-MM-DD --bootstrap-days 45
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR  # noqa: E402

BASE = "https://statsapi.mlb.com/api/v1"
LAYER1_PATH = RAW_DIR / "lineup_slots_by_game.parquet"
LAYER2_PATH = RAW_DIR / "batter_lineup_spot_last10_vs_hand.parquet"
SLEEP_S = 0.12
WINDOW = 10


def _final_games_for_date(d: date) -> list[tuple[int, date]]:
    """Return (game_pk, calendar_date) for Final regular-season games on ``d``."""
    url = f"{BASE}/schedule"
    r = requests.get(
        url,
        params={"sportId": 1, "date": d.isoformat(), "gameType": "R"},
        timeout=60,
    )
    r.raise_for_status()
    out: list[tuple[int, date]] = []
    for block in r.json().get("dates", []) or []:
        for g in block.get("games", []) or []:
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            try:
                pk = int(g["gamePk"])
            except (KeyError, TypeError, ValueError):
                continue
            gd = g.get("officialDate") or (str(g.get("gameDate", ""))[:10])
            try:
                gday = date.fromisoformat(str(gd)[:10])
            except ValueError:
                gday = d
            out.append((pk, gday))
    time.sleep(SLEEP_S)
    return out


def _boxscore(pk: int) -> dict:
    r = requests.get(f"{BASE}/game/{pk}/boxscore", timeout=60)
    r.raise_for_status()
    time.sleep(SLEEP_S)
    return r.json()


def _pitch_hand(pid: int, cache: dict[int, str]) -> str:
    if pid in cache:
        return cache[pid]
    r = requests.get(f"{BASE}/people/{pid}", params={"hydrate": ""}, timeout=30)
    r.raise_for_status()
    time.sleep(SLEEP_S)
    people = r.json().get("people") or []
    code = "R"
    if people:
        ph = people[0].get("pitchHand") or {}
        code = str(ph.get("code") or "R").upper()[:1]
        if code not in ("L", "R"):
            code = "R"
    cache[pid] = code
    return code


def _team_starter_id(team_box: dict) -> int | None:
    players = team_box.get("players") or {}
    for pid in team_box.get("pitchers") or []:
        key = f"ID{pid}"
        pl = players.get(key)
        if not pl:
            continue
        pit = (pl.get("stats") or {}).get("pitching") or {}
        try:
            gs = int(pit.get("gamesStarted") or 0)
        except (TypeError, ValueError):
            gs = 0
        if gs >= 1:
            return int(pid)
    return None


def _starting_batters(team_box: dict, team_id: int, game_pk: int, game_day: date, side: str) -> list[dict]:
    """Rows dict bat_slot, batter_id for batting orders with substitution suffix 00 only."""
    rows: list[dict] = []
    players = team_box.get("players") or {}
    team_obj = (team_box.get("team") or {})
    tid = int(team_obj.get("id") or team_id)
    for _, pinfo in players.items():
        bo = pinfo.get("battingOrder")
        if bo is None:
            continue
        try:
            boi = int(bo)
        except (TypeError, ValueError):
            continue
        if boi % 100 != 0:
            continue
        slot = boi // 100
        if slot < 1 or slot > 9:
            continue
        pers = pinfo.get("person") or {}
        try:
            bid = int(pers.get("id"))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "game_pk": game_pk,
                "game_date": pd.Timestamp(game_day),
                "team_id": tid,
                "batter_id": bid,
                "bat_slot": slot,
                "is_home": 1 if side == "home" else 0,
                "source": "statsapi_boxscore",
            }
        )
    # Dedupe slot if API duplicated (keep lowest batter_id)
    by_slot: dict[int, dict] = {}
    for r in rows:
        s = int(r["bat_slot"])
        if s not in by_slot or r["batter_id"] < by_slot[s]["batter_id"]:
            by_slot[s] = r
    return [by_slot[s] for s in sorted(by_slot)]


def parse_game_lineups(box: dict, game_pk: int, game_day: date, hand_cache: dict[int, str]) -> list[dict]:
    teams = box.get("teams") or {}
    home_tb = teams.get("home") or {}
    away_tb = teams.get("away") or {}

    home_sp = _team_starter_id(home_tb)
    away_sp = _team_starter_id(away_tb)
    if home_sp is None or away_sp is None:
        return []

    away_throws = _pitch_hand(away_sp, hand_cache)
    home_throws = _pitch_hand(home_sp, hand_cache)

    home_tid = int((home_tb.get("team") or {}).get("id") or 0)
    away_tid = int((away_tb.get("team") or {}).get("id") or 0)

    out: list[dict] = []
    for side, tb, opp_sp, opp_throws, tid in (
        ("home", home_tb, away_sp, away_throws, home_tid),
        ("away", away_tb, home_sp, home_throws, away_tid),
    ):
        bat = _starting_batters(tb, tid, game_pk, game_day, side)
        for row in bat:
            row["opp_starter_id"] = int(opp_sp)
            row["opp_starter_throws"] = str(opp_throws).upper()[:1]
            row["fetched_at"] = pd.Timestamp.utcnow()
            out.append(row)
    return out


def load_existing_game_pks(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    try:
        df = pd.read_parquet(path, columns=["game_pk"])
        return set(df["game_pk"].astype(int).unique())
    except Exception:
        return set()


def append_layer1(new_rows: list[dict], path: Path) -> int:
    if not new_rows:
        return 0
    new_df = pd.DataFrame(new_rows)
    if path.is_file():
        old = pd.read_parquet(path)
        df = pd.concat([old, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["game_pk", "batter_id"], keep="last")
    else:
        df = new_df.drop_duplicates(subset=["game_pk", "batter_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(new_df)


def ingest_dates(
    dates: list[date],
    existing_pks: set[int],
    hand_cache: dict[int, str],
) -> tuple[list[dict], set[int]]:
    """Fetch boxscores for finals on listed dates; skip game_pk already ingested."""
    all_rows: list[dict] = []
    seen_new: set[int] = set()
    for d in sorted(set(dates)):
        meta = _final_games_for_date(d)
        for pk, gday in meta:
            if pk in existing_pks or pk in seen_new:
                continue
            try:
                box = _boxscore(pk)
            except requests.HTTPError:
                continue
            rows = parse_game_lineups(box, pk, gday, hand_cache)
            if rows:
                all_rows.extend(rows)
                seen_new.add(pk)
                existing_pks.add(pk)
    return all_rows, existing_pks


def rebuild_layer2(layer1: Path, asof: date, out: Path) -> int:
    if not layer1.is_file():
        return 0
    df = pd.read_parquet(layer1)
    if df.empty:
        return 0
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    cutoff = pd.Timestamp(asof)
    df = df[df["game_date"] < cutoff]

    records: list[dict] = []
    for (bid, oh), g in df.groupby(["batter_id", "opp_starter_throws"], sort=False):
        g2 = g.sort_values("game_date", ascending=False).head(WINDOW)
        slots = [int(x) for x in g2["bat_slot"].tolist()]
        n = len(slots)
        if n == 0:
            continue
        med = float(np.median(slots))
        med_int = int(round(med))
        med_int = max(1, min(9, med_int))
        ctr = Counter(slots)
        top = ctr.most_common()
        best_n = top[0][1]
        mode_candidates = sorted([s for s, c in top if c == best_n])
        mode_slot = mode_candidates[0]
        split = "vs_lhp" if str(oh).upper().startswith("L") else "vs_rhp"
        records.append(
            {
                "batter_id": int(bid),
                "opp_hand": str(oh).upper()[:1],
                "split": split,
                "asof_date": pd.Timestamp(asof).normalize(),
                "n_games": n,
                "median_slot": med_int,
                "mode_slot": int(mode_slot),
                "slots_json": json.dumps(slots),
            }
        )

    out_df = pd.DataFrame.from_records(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    return len(out_df)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asof", required=True, help="Slate/knowledge date YYYY-MM-DD (rollup excludes games on or after this)")
    ap.add_argument(
        "--bootstrap-days",
        type=int,
        default=0,
        help="If >0, ingest finals for each calendar day in [asof-bootstrap, asof-1] (first-time fill)",
    )
    ap.add_argument(
        "--layer1-only",
        action="store_true",
        help="Only append Layer 1; skip Layer 2 rebuild",
    )
    args = ap.parse_args()
    asof = date.fromisoformat(args.asof)

    hand_cache: dict[int, str] = {}
    existing = load_existing_game_pks(LAYER1_PATH)

    ingest_days: list[date] = []
    if args.bootstrap_days > 0:
        for i in range(args.bootstrap_days, 0, -1):
            ingest_days.append(asof - timedelta(days=i))
    else:
        ingest_days.append(asof - timedelta(days=1))

    print(f"+ lineup ingest dates: {[str(x) for x in ingest_days]} (existing game_pk count={len(existing)})")
    rows, _ = ingest_dates(ingest_days, existing, hand_cache)
    n_app = append_layer1(rows, LAYER1_PATH)
    print(f"  Layer 1: appended {n_app} rows → {LAYER1_PATH}")

    if args.layer1_only:
        return 0

    n2 = rebuild_layer2(LAYER1_PATH, asof, LAYER2_PATH)
    print(f"  Layer 2: {n2} batter×split rows → {LAYER2_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
