#!/usr/bin/env python3
"""
Post-process matchup predictions into per-starter expected HR allowed.

Additive only: reads existing JSON/CSV priors; does not change narrative_engine
or model outputs.

Lineup modes:
  pa_top9 (default) — opposing roster from todays_matchups.json; keep top 9 by
    season PA (qualifying_batters CSV); order by PA desc; slot s gets prior weight[s-1].
  json9 — first 9 batters listed in matchups JSON order.
  official_json — ordered 9 batter IDs from --lineup-json (see schema in docstring).

Official lineup JSON schema (optional file, e.g. data/raw/todays_official_lineups.json):
{
  "823878": {
    "away": [676475, 671056, ... ],
    "home": [682663, 596103, ... ]
  },
  ...
}
Keys are game_pk strings or ints (both accepted). Each list must be length 9.
The \"away\" array is batting order for the away team (vs home starter); \"home\" for home team.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RAW_DIR, REPORTS_DIR

DEFAULT_PRIORS = DATA_DIR / "priors" / "starter_hr_exposure.json"
DEFAULT_PA_CSV = RAW_DIR / "qualifying_batters_2026.csv"
DEFAULT_PRED = REPORTS_DIR / "todays_matchup_predictions.json"
DEFAULT_MATCHUPS = RAW_DIR / "todays_matchups.json"


def load_priors(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pa_map(path: Path) -> dict[int, int]:
    df = pd.read_csv(path)
    col = "mlbam_id" if "mlbam_id" in df.columns else "batter"
    return {int(r[col]): int(r["PA"]) for _, r in df.iterrows()}


def top9_by_pa(roster_ids: list[int], pa_map: dict[int, int]) -> list[int]:
    scored = [(bid, pa_map.get(bid, 0)) for bid in roster_ids]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [b for b, _ in scored[:9]]


def lineup_batter_ids(
    mode: str,
    game: dict[str, Any],
    pitcher_side: str,
    pa_map: dict[int, int],
    official: dict[str, Any] | None,
) -> tuple[list[int], str]:
    """
    Returns (9 batter mlbam ids in slot 1..9 order, lineup_source label).
    """
    batting_side = "away" if pitcher_side == "home" else "home"
    roster = game.get(f"{batting_side}_batters") or []
    roster_ids = [int(b["mlbam_id"]) for b in roster]

    if mode == "json9":
        ids = roster_ids[:9]
        if len(ids) < 9:
            raise ValueError(
                f"game_pk={game.get('game_pk')} {batting_side}: json9 needs 9 batters, got {len(ids)}"
            )
        return ids, "json9_first9"

    if mode == "pa_top9":
        if len(roster_ids) < 9:
            raise ValueError(
                f"game_pk={game.get('game_pk')} {batting_side}: roster has only {len(roster_ids)} batters"
            )
        ids = top9_by_pa(roster_ids, pa_map)
        return ids, "pa_top9_desc"

    if mode == "official_json":
        if not official:
            raise ValueError("official_json mode requires --lineup-json")
        gk = str(game.get("game_pk", ""))
        block = official.get(gk) or official.get(str(int(gk)) if gk.isdigit() else gk)
        if not block:
            raise KeyError(f"No official lineup for game_pk={gk} in lineup JSON")
        ids = block.get(batting_side)
        if not ids or len(ids) != 9:
            raise ValueError(
                f"game_pk={gk} side={batting_side}: need official list of 9 ids, got {ids!r}"
            )
        return [int(x) for x in ids], "official_json"

    raise ValueError(f"Unknown lineup mode: {mode}")


def load_official_lineups(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pitcher_rows_for_slate(preds: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(preds)
    if df.empty:
        return df
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Expected HR allowed per starter (additive post-process).")
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--matchups", type=Path, default=DEFAULT_MATCHUPS)
    ap.add_argument("--pa-csv", type=Path, default=DEFAULT_PA_CSV, help="PA totals for pa_top9 selection")
    ap.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    ap.add_argument(
        "--lineup-mode",
        choices=("pa_top9", "json9", "official_json"),
        default="pa_top9",
        help="How to pick/order the 9 batters (default: top 9 by PA, ordered by PA desc for slot weights)",
    )
    ap.add_argument(
        "--lineup-json",
        type=Path,
        default=None,
        help="With official_json: path to game_pk -> away/home -> [9 batter ids]",
    )
    ap.add_argument("--output", type=Path, default=None, help="JSON output path (default: reports/pitcher_expected_hr_<date>.json)")
    ap.add_argument("--md", type=Path, default=None, help="Optional markdown table path")
    args = ap.parse_args()

    if args.lineup_mode == "official_json" and (args.lineup_json is None or not args.lineup_json.exists()):
        print("ERROR: official_json requires --lineup-json pointing to an existing file", file=sys.stderr)
        return 1

    priors = load_priors(args.priors)
    weights: list[float] = priors["mean_pa_vs_starter_by_slot"]
    mean_bf: float = float(priors["mean_bf"])
    if len(weights) != 9:
        raise ValueError("priors mean_pa_vs_starter_by_slot must have length 9")
    if abs(sum(weights) - mean_bf) > 0.02:
        raise ValueError("Slot weights should sum to mean_bf in priors file")

    pa_map = load_pa_map(args.pa_csv) if args.pa_csv.exists() else {}
    if args.lineup_mode == "pa_top9" and not pa_map:
        print("ERROR: pa_top9 requires PA CSV", args.pa_csv, file=sys.stderr)
        return 1

    official = load_official_lineups(args.lineup_json)

    with open(args.predictions, encoding="utf-8") as f:
        preds = json.load(f)
    if not preds:
        print("ERROR: empty predictions", file=sys.stderr)
        return 1
    slate_date = str(preds[0].get("slate_date", "unknown"))

    with open(args.matchups, encoding="utf-8") as f:
        games = json.load(f)

    pred_df = pitcher_rows_for_slate(preds)
    lookup = pred_df.set_index(["batter_mlbam_id", "pitcher_mlbam_id"])

    rows_out: list[dict[str, Any]] = []

    for game in games:
        game_pk = game.get("game_pk")
        for pitcher_side in ("away", "home"):
            pid = game.get(f"{pitcher_side}_pitcher_id")
            pname = game.get(f"{pitcher_side}_pitcher_name")
            pteam = game.get(f"{pitcher_side}_team")
            if pid is None or int(pid) <= 0:
                continue
            pid = int(pid)

            batting_side = "away" if pitcher_side == "home" else "home"
            opponent = game.get(f"{batting_side}_team")

            try:
                batter_ids, lineup_src = lineup_batter_ids(
                    args.lineup_mode, game, pitcher_side, pa_map, official
                )
            except (ValueError, KeyError) as e:
                print(f"WARN: skip {pname} game_pk={game_pk}: {e}", file=sys.stderr)
                continue

            slot_p: list[float] = []
            missing = 0
            for bid in batter_ids:
                try:
                    row = lookup.loc[(int(bid), pid)]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    slot_p.append(float(row.get("adj_p_hr") or 0.0))
                except KeyError:
                    slot_p.append(0.0)
                    missing += 1

            exp_hr_weighted = sum(w * p for w, p in zip(weights, slot_p))
            mean_top9 = sum(slot_p) / 9.0 if slot_p else 0.0
            exp_hr_uniform = mean_top9 * mean_bf

            sub = pred_df[pred_df["pitcher_mlbam_id"] == pid]
            pred_hr_file_sum_all = float(sub["adj_p_hr"].sum()) if len(sub) else 0.0
            n_listed = int(len(sub))

            rows_out.append(
                {
                    "slate_date": slate_date,
                    "game_pk": game_pk,
                    "pitcher_mlbam_id": pid,
                    "pitcher_name": pname,
                    "pitcher_team": pteam,
                    "opponent": opponent,
                    "lineup_mode": args.lineup_mode,
                    "lineup_source": lineup_src,
                    "expected_hr_slot_weighted": round(exp_hr_weighted, 4),
                    "expected_hr_uniform_mean_bf": round(exp_hr_uniform, 4),
                    "pred_hr_file_sum_all_listed": round(pred_hr_file_sum_all, 4),
                    "n_predictions_listed": n_listed,
                    "mean_bf_prior": mean_bf,
                    "missing_matchup_cells": missing,
                    "batter_ids_ordered": batter_ids,
                }
            )

    rows_out.sort(key=lambda r: r["expected_hr_slot_weighted"], reverse=True)

    out_path = args.output
    if out_path is None:
        out_path = REPORTS_DIR / f"pitcher_expected_hr_{slate_date}.json"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "slate_date": slate_date,
        "predictions_path": str(args.predictions.resolve()),
        "matchups_path": str(args.matchups.resolve()),
        "priors_path": str(args.priors.resolve()),
        "lineup_mode": args.lineup_mode,
        "lineup_json": str(args.lineup_json) if args.lineup_json else None,
        "pitchers": rows_out,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows_out)} pitchers -> {out_path}")

    if args.md:
        lines = [
            f"# Pitcher expected HR ({slate_date})",
            "",
            f"Lineup mode: **{args.lineup_mode}**. (Each JSON row also has `lineup_source`.)",
            "",
            "| Rank | Pitcher | Team | Opp | Exp HR (slot wts) | Uniform @ mean BF | File sum (all listed) | n listed |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for i, r in enumerate(rows_out, 1):
            lines.append(
                f"| {i} | {r['pitcher_name']} | {r['pitcher_team']} | {r['opponent']} | "
                f"{r['expected_hr_slot_weighted']:.3f} | {r['expected_hr_uniform_mean_bf']:.3f} | "
                f"{r['pred_hr_file_sum_all_listed']:.3f} | {r['n_predictions_listed']} |"
            )
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote markdown -> {args.md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
