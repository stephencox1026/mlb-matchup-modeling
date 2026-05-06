#!/usr/bin/env python3
"""
Section 15 — No HR Model: per-game P(0 HR) prediction.

Slim reactivation of the deferred No HR Model plan, leveraging the now-shipped
park-aware base model (L2). For each game:

  lambda_game = (sum over both lineups: slot-weighted batter p_hr_calibrated vs starter)
                + bullpen contribution
  lambda_game *= park_pf_hr(home_team)
  P(0 HR) = (k / (k + lambda_game))^k     # Negative-Binomial-calibrated zero

NB dispersion k loaded from data/priors/nb_dispersion.json (fit on 2022-2024
bulk; k≈12.65). Park HR factor from data/priors/park_hr_factors.json (L2).

Bullpen contribution is approximated as PA_pen × league_HR_PA × park_pf_hr.
v2 hooks documented inline (team-specific bullpen priors, roof state, weather).

Output:
  data/reports/section_15_zero_hr.md
  data/reports/todays_zero_hr_predictions.json
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR, REPORTS_DIR, DATA_DIR  # noqa: E402
from weather_fetch import (  # noqa: E402
    forecast_at_first_pitch,
    hr_carry_multiplier,
    per_batter_hr_multiplier,
)

DEFAULT_MATCHUPS = RAW_DIR / "todays_matchups.json"
DEFAULT_PRED = REPORTS_DIR / "todays_matchup_predictions.json"
DEFAULT_STARTER_RUNS = REPORTS_DIR / "todays_starter_run_expectancies.json"
SLOT_WEIGHTS_JSON = DATA_DIR / "priors" / "starter_hr_exposure.json"
PARK_PF_JSON = DATA_DIR / "priors" / "park_hr_factors.json"
NB_DISPERSION_JSON = DATA_DIR / "priors" / "nb_dispersion.json"
TEAM_BULLPEN_PRIORS_JSON = DATA_DIR / "priors" / "team_bullpen_hr.json"
QUAL_BATTERS_CSV = RAW_DIR / "qualifying_batters_2026.csv"

OUT_MD = REPORTS_DIR / "section_15_zero_hr.md"
OUT_JSON = REPORTS_DIR / "todays_zero_hr_predictions.json"

LEAGUE_HR_PA = 0.031  # league HR/PA baseline (matches narrative_engine LEAGUE_AVG)
TEAM_PA_PER_GAME = 38.0  # league mean total PAs per team per game


def _load_priors() -> dict:
    slot_weights = json.loads(SLOT_WEIGHTS_JSON.read_text())
    park = json.loads(PARK_PF_JSON.read_text())
    nb = json.loads(NB_DISPERSION_JSON.read_text()) if NB_DISPERSION_JSON.exists() else {"k_mle": 10.0}
    bullpen_blob = (json.loads(TEAM_BULLPEN_PRIORS_JSON.read_text())
                    if TEAM_BULLPEN_PRIORS_JSON.exists() else {})
    bullpen_blends = {team: float(d.get("blend", LEAGUE_HR_PA))
                      for team, d in bullpen_blob.get("teams", {}).items()}
    league_pen_hr_pa = float(bullpen_blob.get("_meta", {})
                              .get("league_hr_pa_pen", LEAGUE_HR_PA))
    return {
        "weights": slot_weights["mean_pa_vs_starter_by_slot"],
        "mean_bf": float(slot_weights["mean_bf"]),
        "park": park.get("parks", {}),
        "league_hr_pa": float(park.get("_meta", {}).get("league_hr_pa", LEAGUE_HR_PA)),
        "k_nb": float(nb.get("k_mle", 10.0)),
        # Per-team bullpen blends (Beta-Binomial EB shrinkage to league mean)
        "bullpen_blends": bullpen_blends,
        "league_hr_pa_pen": league_pen_hr_pa,
    }


def _load_pa_map() -> dict[int, int]:
    if not QUAL_BATTERS_CSV.exists():
        return {}
    df = pd.read_csv(QUAL_BATTERS_CSV)
    return {int(r.mlbam_id): int(r.PA) for _, r in df.iterrows()}


def _top9_by_pa(roster_ids: list[int], pa_map: dict[int, int]) -> list[int]:
    scored = [(bid, pa_map.get(bid, 0)) for bid in roster_ids]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [b for b, _ in scored[:9]]


def _bf_hat(starter_id: int, starter_runs: dict | None, mean_bf: float) -> float:
    """Estimate BF for tonight's starter from Section 9 IP projection. Falls
    back to mean_bf when no projection."""
    if not starter_runs:
        return mean_bf
    rec = starter_runs.get(int(starter_id))
    if rec is None:
        return mean_bf
    # Rough mapping IP -> BF (4.3 BF/IP league avg). Keep within sane bounds.
    pred_runs = rec.get("predicted_runs")
    if pred_runs is None:
        return mean_bf
    # IP isn't directly in the file but interval bounds give us a range to
    # estimate. As a rough proxy, use predicted_runs to scale: starter giving
    # up many runs gets pulled earlier. Crude but defensible.
    # Use predicted_runs around 2.5-3.0 ≈ league avg ≈ ~6 IP ≈ ~25 BF.
    # If pred_runs >= 4.5, project shorter outing.
    if pred_runs >= 4.5:
        return max(15.0, mean_bf * 0.65)
    if pred_runs >= 3.5:
        return mean_bf * 0.85
    if pred_runs <= 1.5:
        return min(30.0, mean_bf * 1.15)
    return mean_bf


@functools.lru_cache(maxsize=1)
def _park_lookup_full() -> dict[str, dict]:
    """home_team -> park_lookup row (with lat/lon/enclosure/wind_orientation_cf_deg)."""
    p = RAW_DIR / "park_lookup.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {str(r.home_team).upper(): r._asdict() if hasattr(r, "_asdict") else dict(r._mapping)
            for r in df.itertuples(index=False)}


def _park_lookup_row(home_team: str) -> dict | None:
    return _park_lookup_full().get(str(home_team or "").upper())


def compute_one_game(game: dict, preds_by_pair: dict, priors: dict,
                     pa_map: dict, starter_runs: dict | None) -> dict:
    home = game["home_team"]
    weights = priors["weights"]
    mean_bf = priors["mean_bf"]
    league_hr_pa = priors["league_hr_pa"]
    k_nb = priors["k_nb"]

    park_cell = priors["park"].get(home, {})
    pf_hr = float(park_cell.get("pf_two_sided_shrunk", 1.0))

    # Phase 1: live weather forecast at first pitch, applied as a multiplier.
    park_row = _park_lookup_row(home)
    weather = forecast_at_first_pitch(home, game.get("game_datetime_utc"))
    roof_state = game.get("roof_state_today")  # populated by future refresh script
    weather_info = hr_carry_multiplier(weather, park_row, roof_state=roof_state)
    weather_mult = float(weather_info["hr_mult"])
    weather_breakdown = weather_info["breakdown"]

    bullpen_blends = priors.get("bullpen_blends", {})
    league_pen_hr_pa = priors.get("league_hr_pa_pen", league_hr_pa)

    lambdas = {}
    for sp_side, bat_side in (("away", "home"), ("home", "away")):
        sp_id = int(game.get(f"{sp_side}_pitcher_id") or 0)
        sp_name = game.get(f"{sp_side}_pitcher_name", "TBD")
        sp_team = game.get(f"{sp_side}_team")
        roster = game.get(f"{bat_side}_batters") or []
        roster_ids = [int(b["mlbam_id"]) for b in roster]
        if len(roster_ids) < 9 or sp_id <= 0:
            lambdas[sp_side] = {"sp": 0.0, "pen": 0.0, "sp_name": sp_name,
                                 "pen_rate": league_pen_hr_pa, "pen_source": "league_avg",
                                 "skip_reason": "TBD/short roster"}
            continue
        top9 = _top9_by_pa(roster_ids, pa_map)
        # Slot-weighted SP HR rate, with per-batter weather adjustment (T2.3)
        bf_pred = _bf_hat(sp_id, starter_runs, mean_bf)
        scale = bf_pred / mean_bf if mean_bf else 1.0
        sp_lambda = 0.0
        sum_per_batter_mult = 0.0
        n_used = 0
        for s, bid in enumerate(top9):
            row = preds_by_pair.get((int(bid), int(sp_id)))
            if row is None:
                continue
            p_hr = float(row.get("p_hr_calibrated") or row.get("p_hr") or 0.0)
            # Per-batter weather mult: lefty pull-power benefits more from wind-out-RF
            # than oppo-power righty at the same Wrigley game.
            pb_info = per_batter_hr_multiplier(weather, park_row, int(bid),
                                                 roof_state=roof_state)
            pb_mult = float(pb_info.get("hr_mult", 1.0))
            sum_per_batter_mult += pb_mult
            n_used += 1
            sp_lambda += weights[s] * scale * p_hr * pb_mult
        # Per-team bullpen HR/PA blend (Beta-Binomial EB shrinkage to league mean,
        # built from 2022-2025 bulk by src/build_team_bullpen_hr_priors.py).
        # Falls back to league mean for unknown teams.
        if sp_team and sp_team in bullpen_blends:
            pen_rate = bullpen_blends[sp_team]
            pen_source = "team_blend"
        else:
            pen_rate = league_pen_hr_pa
            pen_source = "league_avg_fallback"
        pa_pen = max(0.0, TEAM_PA_PER_GAME - bf_pred)
        pen_lambda = pa_pen * pen_rate
        lambdas[sp_side] = {
            "sp": float(sp_lambda),
            "pen": float(pen_lambda),
            "sp_name": sp_name,
            "bf_hat": float(bf_pred),
            "pen_rate": float(pen_rate),
            "pen_source": pen_source,
            "pen_team": sp_team,
        }

    # T2.3: per-batter weather mult is already applied INSIDE each side's sp_lambda
    # (per-slot, weighted by pull rate). We still apply a game-level weather mult to
    # the BULLPEN contribution since we don't have per-PA bullpen batter info here.
    lambda_total_raw = sum(d.get("sp", 0.0) + d.get("pen", 0.0) * weather_mult
                            for d in lambdas.values())
    lambda_total_pre_weather = (
        sum(d.get("sp", 0.0) + d.get("pen", 0.0) for d in lambdas.values()) * pf_hr
    )
    # Park PF still applies on top of (already-weather-adjusted) sums
    lambda_total_adj = lambda_total_raw * pf_hr

    # NB zero formula
    p_zero = (k_nb / (k_nb + lambda_total_adj)) ** k_nb if lambda_total_adj > 0 else 1.0

    return {
        "game_pk": game.get("game_pk"),
        "matchup": f"{game.get('away_team')} @ {game.get('home_team')}",
        "home_team": home,
        "park_pf_hr": round(pf_hr, 4),
        "park_enclosure": park_cell.get("enclosure", "outdoor")
                          if isinstance(park_cell, dict) else "outdoor",
        "lambda_sp_away": round(lambdas.get("away", {}).get("sp", 0.0), 4),
        "lambda_sp_home": round(lambdas.get("home", {}).get("sp", 0.0), 4),
        "lambda_pen_away": round(lambdas.get("away", {}).get("pen", 0.0), 4),
        "lambda_pen_home": round(lambdas.get("home", {}).get("pen", 0.0), 4),
        "pen_rate_away": round(lambdas.get("away", {}).get("pen_rate", league_pen_hr_pa), 5),
        "pen_rate_home": round(lambdas.get("home", {}).get("pen_rate", league_pen_hr_pa), 5),
        "pen_source_away": lambdas.get("away", {}).get("pen_source", "league_avg"),
        "pen_source_home": lambdas.get("home", {}).get("pen_source", "league_avg"),
        "bf_hat_away": round(lambdas.get("away", {}).get("bf_hat", priors["mean_bf"]), 2),
        "bf_hat_home": round(lambdas.get("home", {}).get("bf_hat", priors["mean_bf"]), 2),
        "lambda_total_raw": round(lambda_total_raw, 4),
        "lambda_total_pre_weather": round(lambda_total_pre_weather, 4),
        "lambda_total_adj": round(lambda_total_adj, 4),
        "p_zero_hr": round(p_zero, 4),
        "expected_hr": round(lambda_total_adj, 4),
        "p_under_1_5_hr": round(p_zero + (lambda_total_adj * p_zero / k_nb)
                                 * ((lambda_total_adj / (k_nb + lambda_total_adj)) ** 0.0)
                                 if lambda_total_adj > 0 else 1.0, 4),
        "k_nb": k_nb,
        "away_pitcher": lambdas.get("away", {}).get("sp_name", "TBD"),
        "home_pitcher": lambdas.get("home", {}).get("sp_name", "TBD"),
        # Phase 1: weather fields
        "weather_temp_f": weather.get("temp_f"),
        "weather_wind_mph": weather.get("wind_mph"),
        "weather_wind_dir_deg": weather.get("wind_dir_deg"),
        "weather_humidity_pct": weather.get("humidity_pct"),
        "weather_precip_prob_pct": weather.get("precip_prob_pct"),
        "weather_wind_orientation": weather_breakdown.get("wind_orientation"),
        "weather_wind_out_mph": weather_breakdown.get("wind_out_mph"),
        "weather_hr_mult": round(weather_mult, 4),
        "weather_temp_mult": weather_breakdown.get("temp_mult"),
        "weather_wind_mult": weather_breakdown.get("wind_mult"),
        "weather_indoor": weather_breakdown.get("indoor"),
        "weather_source": weather.get("source"),
        "weather_fetched_at": weather.get("fetched_at"),
        "weather_forecast_time_utc": weather.get("time_utc"),
        "roof_state_today": game.get("roof_state_today"),
    }


def render_md(rows: list[dict], priors: dict) -> str:
    rows_sorted = sorted(rows, key=lambda r: -(r.get("p_zero_hr") or 0.0))
    lines = [
        "## Section 15: No HR Model — P(0 HR) per game",
        "",
        f"**NB dispersion k = {priors['k_nb']:.2f}** (fit on 2022-2024 bulk; "
        "Poisson would systematically under-predict zero-HR rate by ~3 pp).",
        f"**League HR/PA baseline = {priors['league_hr_pa']:.4f}**.  "
        f"**Mean PA/team/game = {TEAM_PA_PER_GAME:.0f}**.",
        "",
        "Park PF: two-sided HR factor from `data/priors/park_hr_factors.json` (L2). "
        "BF̂ for each starter conditioned on Section 9 projected runs (heavier projection → shorter outing → more PAs to bullpen). "
        "**Bullpen rate**: per-team HR/PA blend from `data/priors/team_bullpen_hr.json` (Beta-Binomial EB shrunk; CLE 2.44% to LAA 3.09%). "
        "**Weather**: live Open-Meteo forecast at stadium for first-pitch hour; multiplier captures temp + wind-projected-onto-CF (capped ±20%); indoor games get mult 1.0.",
        "",
        "| Rank | Matchup | Park | PF | Temp | Wind | Wx Mult | Pen A | Pen H | λ_total | E[HR] | **P(0 HR)** | Away SP (BF̂) | Home SP (BF̂) |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(rows_sorted, 1):
        away_cell = f"{r['away_pitcher']} ({r['bf_hat_away']:.0f})"
        home_cell = f"{r['home_pitcher']} ({r['bf_hat_home']:.0f})"
        temp = r.get("weather_temp_f")
        temp_str = f"{temp:.0f}°F" if temp is not None else "—"
        wind_mph = r.get("weather_wind_mph")
        wind_orient = r.get("weather_wind_orientation") or "—"
        if wind_mph is None:
            wind_str = "—"
        elif r.get("weather_indoor"):
            wind_str = "indoor"
        else:
            wind_str = f"{wind_mph:.0f} mph {wind_orient}"
        wx_mult = r.get("weather_hr_mult", 1.0)
        pen_a = r.get("pen_rate_away") or 0.0
        pen_h = r.get("pen_rate_home") or 0.0
        lines.append(
            f"| {i} | {r['matchup']} | {r['home_team']} | "
            f"{r['park_pf_hr']:.2f} | {temp_str} | {wind_str} | "
            f"{wx_mult:.3f} | {pen_a*100:.2f}% | {pen_h*100:.2f}% | "
            f"{r['lambda_total_adj']:.2f} | {r['expected_hr']:.2f} | "
            f"**{r['p_zero_hr']*100:.1f}%** | {away_cell} | {home_cell} |"
        )
    lines.append("")
    lines.append("_Sorted by highest P(0 HR). Higher = better target for an Under HR / "
                  "no-HR-game prop. Wx Mult > 1 = HR-friendly conditions; < 1 = HR-suppressing._")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 15: No HR Model.")
    parser.add_argument("--matchups", type=Path, default=DEFAULT_MATCHUPS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--starter-runs", type=Path, default=DEFAULT_STARTER_RUNS)
    parser.add_argument("--output-md", type=Path, default=OUT_MD)
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    if not args.matchups.exists() or not args.predictions.exists():
        print(f"Missing inputs: matchups={args.matchups.exists()} preds={args.predictions.exists()}",
              file=sys.stderr)
        sys.exit(1)

    games = json.loads(args.matchups.read_text())
    preds = json.loads(args.predictions.read_text())
    preds_by_pair = {(int(p["batter_mlbam_id"]), int(p["pitcher_mlbam_id"])): p for p in preds}

    starter_runs = None
    if args.starter_runs.exists():
        sr = json.loads(args.starter_runs.read_text())
        # build lookup by pitcher id
        starter_runs = {}
        for r in sr:
            for side in ("away", "home"):
                pid = r.get(f"{side}_pitcher_id")
                if pid is not None:
                    starter_runs[int(pid)] = {"predicted_runs": r.get(f"{side}_predicted_runs")}

    priors = _load_priors()
    pa_map = _load_pa_map()

    out_rows = [compute_one_game(g, preds_by_pair, priors, pa_map, starter_runs) for g in games]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "_meta": {
            "k_nb": priors["k_nb"],
            "league_hr_pa": priors["league_hr_pa"],
            "team_pa_per_game": TEAM_PA_PER_GAME,
            "n_games": len(out_rows),
        },
        "games": out_rows,
    }, indent=2))
    md = render_md(out_rows, priors)
    args.output_md.write_text(md)
    print(md)
    print(f"\nWrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
