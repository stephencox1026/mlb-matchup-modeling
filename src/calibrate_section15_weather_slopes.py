#!/usr/bin/env python3
"""
Priority 3 — Calibrate Section 15 weather carry vs realized game HR totals.

Joins archived `todays_zero_hr_predictions.json` games to Statcast actual HR
counts per game_pk on that calendar date, evaluates:

  - RMSE / Poisson deviance: λ_pre_weather vs λ_adj vs actual HR
  - Grid-searched scalar **WEATHER_MULT_SHRINK** in [0, 1] that shrinks
    multipliers toward 1.0: effective_mult = 1 + s * (mult_raw - 1)

Optional `--write` persists `data/priors/weather_slope_override.json` consumed by
`weather_fetch.py` (reload via process restart).

Usage:
  PYTHONPATH=src python3 src/calibrate_section15_weather_slopes.py
  PYTHONPATH=src python3 src/calibrate_section15_weather_slopes.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_DIR  # noqa: E402

PA_PATH = ROOT / "data" / "raw" / "statcast_pa_level_league.parquet"
ARCHIVE_DIR = ROOT / "data" / "reports" / "archive"
OUT_JSON = DATA_DIR / "priors" / "weather_slope_override.json"


def _collect_archive_games(archive_dir: Path) -> list[dict]:
    rows = []
    if not archive_dir.is_dir():
        return rows
    for day_dir in sorted(archive_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        jp = day_dir / "todays_zero_hr_predictions.json"
        if not jp.exists():
            continue
        slate = day_dir.name
        try:
            blob = json.loads(jp.read_text())
        except Exception:
            continue
        gd = pd.to_datetime(slate).normalize()
        for g in blob.get("games") or []:
            gpk = g.get("game_pk")
            if gpk is None:
                continue
            rows.append({
                "slate_date": gd,
                "game_pk": int(gpk),
                "lambda_pre": float(g.get("lambda_total_pre_weather") or g.get("lambda_total_raw") or 0),
                "lambda_adj": float(g.get("lambda_total_adj") or 0),
                "weather_hr_mult": float(g["weather_hr_mult"]) if g.get("weather_hr_mult") is not None else np.nan,
                "temp_f": g.get("weather_temp_f"),
                "wx_src": g.get("weather_source"),
            })
    return rows


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance, stable for mu > 0."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2.0 * np.mean(term))


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 15 weather λ calibration vs outcomes.")
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--write", action="store_true",
                        help=f"Write shrink scalar to {OUT_JSON}")
    args = parser.parse_args()

    if not PA_PATH.exists():
        print(f"Missing Statcast PA file: {PA_PATH}")
        sys.exit(1)

    games = _collect_archive_games(args.archive_dir)
    if len(games) == 0:
        print(f"No archived Section 15 JSON under {args.archive_dir}.")
        sys.exit(0)

    pa = pd.read_parquet(PA_PATH, columns=["game_pk", "game_date", "events"])
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce").dt.normalize()

    resolved = []
    for row in games:
        gpk = row["game_pk"]
        d = row["slate_date"]
        sub = pa[(pa["game_pk"] == gpk) & (pa["game_date"] == d)]
        if sub.empty:
            continue
        ev = sub["events"].fillna("").str.lower()
        hr_tot = float((ev == "home_run").sum())
        row["actual_hr"] = hr_tot
        resolved.append(row)

    if len(resolved) < 3:
        print(f"Too few games matched Statcast after join ({len(resolved)}).")
        sys.exit(0)

    df = pd.DataFrame(resolved)
    y = df["actual_hr"].values
    pre = df["lambda_pre"].values
    adj = df["lambda_adj"].values
    wm = df["weather_hr_mult"].values

    rmse_pre = float(np.sqrt(np.mean((y - pre) ** 2)))
    rmse_adj = float(np.sqrt(np.mean((y - adj) ** 2)))
    dev_pre = _poisson_deviance(y, pre)
    dev_adj = _poisson_deviance(y, adj)

    best_s, best_rmse = 1.0, float("inf")
    for s in np.linspace(0.0, 1.0, 41):
        lam = np.maximum(1e-6, pre * (1.0 + s * (wm - 1.0)))
        rm = float(np.sqrt(np.mean((y - lam) ** 2)))
        if rm < best_rmse:
            best_rmse, best_s = rm, float(s)

    print("## Priority 3 — Section 15 weather vs realized game HR")
    print()
    print(f"- Archive games joined to Statcast: **{len(df)}**")
    print(f"- RMSE(actual HR vs λ_pre_weather): **{rmse_pre:.3f}**")
    print(f"- RMSE(actual HR vs λ_adj current): **{rmse_adj:.3f}**")
    print(f"- Mean Poisson deviance (λ_pre): **{dev_pre:.4f}**")
    print(f"- Mean Poisson deviance (λ_adj): **{dev_adj:.4f}**")
    print(f"- Best **WEATHER_MULT_SHRINK** on grid [0,1]: **{best_s:.3f}** → RMSE **{best_rmse:.3f}**")
    print()

    if rmse_adj < rmse_pre:
        print("_Weather adjustment improves RMSE vs pre-weather λ at game level._")
    else:
        print("_Pre-weather λ is closer on RMSE — review slopes or shrink toward 1._")

    out_blob = {
        "_meta": {
            "source": "calibrate_section15_weather_slopes.py",
            "n_games": int(len(df)),
            "rmse_pre": rmse_pre,
            "rmse_adj": rmse_adj,
            "best_weather_mult_shrink": best_s,
            "best_rmse_under_shrink": best_rmse,
        },
        "WEATHER_MULT_SHRINK": best_s,
    }

    if args.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(out_blob, indent=2))
        print(f"Wrote **{OUT_JSON}** — restart consumers or clear caches to pick up.")
    else:
        print(f"_Dry-run — pass `--write` to save `{OUT_JSON}`._")


if __name__ == "__main__":
    main()
