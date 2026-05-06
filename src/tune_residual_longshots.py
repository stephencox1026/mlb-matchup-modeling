#!/usr/bin/env python3
"""
Grid-search residual longshot thresholds against evaluate_residual_history.

Usage (from project root, venv on):
  PYTHONPATH=src python3 src/tune_residual_longshots.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from residual_longshots import (
    ResidualConfig,
    evaluate_slate_bundles,
    load_slate_backtest_bundles,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive",
        type=Path,
        default=Path("data/reports/archive"),
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/raw/statcast_pa_level_league.parquet"),
    )
    ap.add_argument("--min-rows", type=int, default=50, help="min residual rows in backtest")
    ap.add_argument("--max-slates", type=int, default=90)
    args = ap.parse_args()

    base = ResidualConfig.defaults()
    # Primary grid: behavior knobs that don't collapse the in-band definition
    max_slate_pcts = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]
    min_lifts = [1.01, 1.03, 1.05, 1.07, 1.10]
    min_confs = [0.35, 0.42, 0.50, 0.58, 0.65]

    # Optional weight mixes (keeps sum ~1, focuses obviousness re-weighting)
    weight_profiles = [
        (0.45, 0.30, 0.25),  # current
        (0.50, 0.25, 0.25),  # more slate
        (0.40, 0.20, 0.40),  # more career
        (0.35, 0.15, 0.50),  # very career-forward
    ]

    bundles = load_slate_backtest_bundles(
        args.archive, args.parquet, max_slates=args.max_slates
    )
    if not bundles:
        print("No slate bundles (missing archive or Statcast).", file=sys.stderr)
        return 1
    print(f"Loaded {len(bundles)} slates; running grid...")

    results: list[dict] = []
    for msp in max_slate_pcts:
        for ml in min_lifts:
            for mcf in min_confs:
                for ws, wy, wc in weight_profiles:
                    cfg = replace(
                        base,
                        max_slate_pct=msp,
                        min_lift=ml,
                        min_conf_hr=mcf,
                        w_slate=ws,
                        w_ytd=wy,
                        w_career=wc,
                    )
                    ev = evaluate_slate_bundles(
                        bundles,
                        config=cfg,
                    )
                    if not ev:
                        continue
                    n = int(ev.get("residual_rows", 0))
                    if n < args.min_rows:
                        continue
                    ra = float(ev.get("residual_tier_a_rate", 0))
                    rb = float(ev.get("residual_tier_b_rate", 0))
                    ba = float(ev.get("band_tier_a_rate", 0))
                    bb = float(ev.get("band_tier_b_rate", 0))
                    d_a = ra - ba
                    d_b = rb - bb
                    results.append(
                        {
                            "d_tier_a": d_a,
                            "d_tier_b": d_b,
                            "res_tier_a": ra,
                            "res_tier_b": rb,
                            "band_tier_a": ba,
                            "band_tier_b": bb,
                            "n_residual": n,
                            "n_band": int(ev.get("band_rows", 0)),
                            "slates": int(ev.get("slates", 0)),
                            "config": {
                                "max_slate_pct": msp,
                                "min_lift": ml,
                                "min_conf_hr": mcf,
                                "w_slate": ws,
                                "w_ytd": wy,
                                "w_career": wc,
                            },
                        }
                    )

    if not results:
        print("No results (empty archive, missing parquet, or all configs < min_rows).", file=sys.stderr)
        return 1

    # Sort: Tier A lift first, then Tier B, then more rows (stability)
    results.sort(
        key=lambda d: (d["d_tier_a"], d["d_tier_b"], d["n_residual"]),
        reverse=True,
    )

    best = results[0]
    print("=" * 72)
    print("Residual longshot grid search (in-band baseline vs same adj_hr range)")
    print(f"  archive: {args.archive}")
    print(f"  min residual rows: {args.min_rows}")
    print("=" * 72)
    print(
        f"Best by Tier-A delta: d_a={best['d_tier_a']*100:+.2f} pp, "
        f"d_b={best['d_tier_b']*100:+.2f} pp | "
        f"res A={best['res_tier_a']*100:.2f}% vs band A={best['band_tier_a']*100:.2f}% | "
        f"n={best['n_residual']} (slates={best['slates']})"
    )
    print("Config (apply to ResidualConfig / module constants):")
    print(json.dumps(best["config"], indent=2))
    print()
    print("Top 12 by Tier-A delta:")
    for i, d in enumerate(results[:12], 1):
        c = d["config"]
        print(
            f"  {i:2}. dA={d['d_tier_a']*100:+5.2f} pp  dB={d['d_tier_b']*100:+5.2f} pp  "
            f"msp={c['max_slate_pct']:.2f} lift>={c['min_lift']:.2f} conf>={c['min_conf_hr']:.2f}  "
            f"w=({c['w_slate']},{c['w_ytd']},{c['w_career']})  n={d['n_residual']}"
        )

    any_pos = [d for d in results if d["d_tier_a"] > 0.001]
    if not any_pos:
        print()
        print("No config achieved Tier A > baseline on this sample (d_tier_a > 0).")
        print("Top 5 by least bad Tier A:")
        least_bad = sorted(results, key=lambda d: d["d_tier_a"], reverse=True)[:5]
        for d in least_bad:
            c = d["config"]
            print(
                f"  dA={d['d_tier_a']*100:+5.2f} pp  msp={c['max_slate_pct']:.2f}  "
                f"lift>={c['min_lift']:.2f}  conf>={c['min_conf_hr']:.2f}  "
                f"w=({c['w_slate']},{c['w_ytd']},{c['w_career']})  n={d['n_residual']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
