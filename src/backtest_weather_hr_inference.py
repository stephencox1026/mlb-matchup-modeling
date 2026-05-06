#!/usr/bin/env python3
"""
Priority 2 — Backtest: same-game weather multiplier at inference vs Beast HR log-loss.

Joins filled tracking rows (vs-SP outcomes) to archived Section 15 JSON
(`todays_zero_hr_predictions.json`) by (slate_date, game_pk) and compares:

  - Baseline: raw `p_hr` (and optionally `adj_p_hr`)
  - Weather-adjusted: clip(p * weather_hr_mult, 0, 1)

Uses aggregate Bernoulli Brier over PA counts (no PA expansion):

  sum_i [ HR_i * (1-p)^2 + (PA_i - HR_i) * p^2 ]

Lower is better. Prints delta vs baseline and archive coverage.

Does NOT mutate models — gate only. Pass → consider Beast+weather features after review.

Usage:
  PYTHONPATH=src python3 src/backtest_weather_hr_inference.py
  PYTHONPATH=src python3 src/backtest_weather_hr_inference.py --archive-dir data/reports/archive
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

from matchup_tracking import TRACKING_MAIN  # noqa: E402


def _load_archive_weather_map(archive_root: Path) -> dict[tuple[str, int], float]:
    """(slate_date YYYY-MM-DD, game_pk) -> weather_hr_mult."""
    out: dict[tuple[str, int], float] = {}
    if not archive_root.is_dir():
        return out
    for day_dir in sorted(archive_root.iterdir()):
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
        for g in blob.get("games") or []:
            gpk = g.get("game_pk")
            wm = g.get("weather_hr_mult")
            if gpk is None or wm is None:
                continue
            out[(slate, int(gpk))] = float(wm)
    return out


def _brier_binomial(hr: np.ndarray, pa: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Returns (weighted_sum_brier, total_pa). Invalid p rows skipped."""
    hr = np.asarray(hr, dtype=float)
    pa = np.asarray(pa, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(pa) & (pa > 0) & np.isfinite(p)
    hr = hr[mask]
    pa = pa[mask]
    p = np.clip(p[mask], 0.0, 1.0)
    contrib = hr * (1.0 - p) ** 2 + (pa - hr) * p**2
    return float(contrib.sum()), float(pa.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather-at-inference HR backtest (Brier gate).")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "data" / "reports" / "archive")
    parser.add_argument("--use-adj", action="store_true",
                        help="Compare adj_p_hr as baseline instead of raw p_hr.")
    args = parser.parse_args()

    wx_map = _load_archive_weather_map(args.archive_dir)
    if not TRACKING_MAIN.exists():
        print(f"No tracking file: {TRACKING_MAIN}")
        sys.exit(1)

    df = pd.read_parquet(TRACKING_MAIN)
    df = df[df["outcome_filled_at"].notna()].copy()
    df = df[df["outcome_pa_vs_sp"].notna() & (df["outcome_pa_vs_sp"] > 0)].copy()
    if df.empty:
        print("No filled vs-SP rows in tracking.")
        sys.exit(0)

    df["slate_str"] = pd.to_datetime(df["slate_date"]).dt.strftime("%Y-%m-%d")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce")

    mult = []
    for _, row in df.iterrows():
        key = (row["slate_str"], int(row["game_pk"])) if pd.notna(row["game_pk"]) else None
        mult.append(wx_map[key] if key and key in wx_map else np.nan)
    df["_wx_mult"] = mult

    base_col = "adj_p_hr" if args.use_adj else "p_hr"
    if base_col not in df.columns:
        print(f"Missing column {base_col}")
        sys.exit(1)

    p_base = df[base_col].astype(float).values
    hr = df["outcome_hr_vs_sp"].astype(float).values
    pa = df["outcome_pa_vs_sp"].astype(float).values

    # Weather-adjusted p (game-level multiplier applied uniformly within game)
    wx = df["_wx_mult"].values
    p_wx = np.where(np.isfinite(wx), np.clip(p_base * wx, 0.0, 1.0), np.nan)

    covered = np.isfinite(wx)
    n_cov = int(covered.sum())
    n_tot = len(df)

    b0_sum, pa0 = _brier_binomial(hr, pa, p_base)
    bwx_sum, pawx = _brier_binomial(hr[covered], pa[covered], p_wx[covered])

    mean_base = b0_sum / pa0 if pa0 > 0 else float("nan")
    mean_wx = bwx_sum / pawx if pawx > 0 else float("nan")

    print("## Priority 2 — Weather-at-inference HR backtest (vs-SP Brier)")
    print()
    print(f"- Tracking rows (filled vs-SP): **{n_tot}**")
    print(f"- Archive coverage (weather_hr_mult): **{n_cov}** ({100 * n_cov / max(n_tot, 1):.1f}%)")
    print(f"- Archived slate dirs scanned: `{args.archive_dir}`")
    print(f"- Baseline column: **`{base_col}`**")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Mean Brier (baseline) | {mean_base:.6f} |")
    wx_cell = f"{mean_wx:.6f}" if np.isfinite(mean_wx) else "—"
    print(f"| Mean Brier (p × weather_hr_mult), covered rows only | {wx_cell} |")
    delta = mean_wx - mean_base if np.isfinite(mean_wx) and np.isfinite(mean_base) else float("nan")
    d_cell = f"{delta:.6f}" if np.isfinite(delta) else "—"
    print(f"| Delta (negative = weather helps) | {d_cell} |")
    print()

    if n_cov < 50:
        print(
            "_Coverage is thin — expand `data/reports/archive/*/todays_zero_hr_predictions.json` "
            "history before trusting the gate._"
        )
        print()

    if n_cov == 0:
        print(
            "**Gate read:** no overlapping (slate_date, game_pk) between tracking and archive — "
            "cannot evaluate inference-time weather yet."
        )
    elif mean_wx < mean_base:
        print(
            "**Gate read:** weather multiplier improves mean Brier on covered rows → "
            "worth a supervised feature experiment after hr-p3 calibration review."
        )
    else:
        print(
            "**Gate read:** weather scaling does not beat baseline Brier here → "
            "do not add live weather to Beast training yet (or shrink multipliers in hr-p3 first)."
        )


if __name__ == "__main__":
    main()
