#!/usr/bin/env python3
"""
H4: Audit which conf sub-factors actually add lift in the High bucket.

Strip each sub-factor (set to 1.0), recompute conf, recompute the bucket
assignment, measure the new High-bucket realized rate. Compare to baseline.
A factor that adds zero lift when stripped is decorative.

Usage:
  python3 src/audit_conf_factors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import REPORTS_DIR  # noqa: E402
from matchup_tracking import TRACKING_MAIN  # noqa: E402

OUT_MD = REPORTS_DIR / "conf_factors_audit.md"

CF_KEYS = ["pitcher_data", "bvp_hr", "bvp_hit", "staleness",
           "convergence_hr", "convergence_hit"]


def _label(conf: float) -> str:
    if conf >= 1.05:
        return "High"
    if conf >= 0.88:
        return "Medium"
    if conf >= 0.75:
        return "Low"
    return "Very Low"


def _conf_for_target(factors: dict, target: str) -> float:
    """Mirror narrative_engine.compute_confidence composite formula."""
    if target == "hr":
        return (factors.get("pitcher_data", 1.0) * factors.get("bvp_hr", 1.0)
                * factors.get("staleness", 1.0) * factors.get("convergence_hr", 1.0))
    if target == "hit":
        return (factors.get("pitcher_data", 1.0) * factors.get("bvp_hit", 1.0)
                * factors.get("staleness", 1.0) * factors.get("convergence_hit", 1.0))
    if target == "xbh":
        c_hr = (factors.get("pitcher_data", 1.0) * factors.get("bvp_hr", 1.0)
                * factors.get("staleness", 1.0) * factors.get("convergence_hr", 1.0))
        c_hit = (factors.get("pitcher_data", 1.0) * factors.get("bvp_hit", 1.0)
                 * factors.get("staleness", 1.0) * factors.get("convergence_hit", 1.0))
        return (c_hr + c_hit) / 2.0
    return 1.0


def _wilson(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    res = scistats.binomtest(int(hits), int(n)).proportion_ci(0.95, method="wilson")
    return float(res.low), float(res.high)


def main() -> None:
    df = pd.read_parquet(TRACKING_MAIN)
    filled = df[df["outcome_pa_vs_sp"].notna() & (df["outcome_pa_vs_sp"] > 0)].copy()
    filled["cf"] = filled["conf_factors_json"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s else {}
    )

    lines = ["# H4: Confidence Factor Driver Audit", "",
             f"**Window:** {filled['slate_date'].min()} → {filled['slate_date'].max()}  "
             f"({len(filled)} vs-SP outcomes)", ""]

    for target, outcome_col in [("hr", "outcome_hr_flag_vs_sp"),
                                  ("hit", "outcome_hit_flag_vs_sp"),
                                  ("xbh", "outcome_xbh_flag_vs_sp")]:
        sub = filled[filled[outcome_col].notna()].copy()
        if sub.empty:
            continue
        baseline = float(sub[outcome_col].mean())

        # Baseline High realized
        sub["_conf_base"] = sub["cf"].apply(lambda d, t=target: _conf_for_target(d, t))
        sub["_label_base"] = sub["_conf_base"].apply(_label)
        high_base = sub[sub["_label_base"] == "High"]
        n_high_base = len(high_base)
        rate_high_base = high_base[outcome_col].mean() if n_high_base else float("nan")
        lift_base = rate_high_base / baseline if baseline > 0 else float("nan")

        lines.append(f"## {target.upper()} target")
        lines.append("")
        lines.append(f"Baseline ({len(sub)} picks, all): realized {baseline:.4f}")
        ci_lo, ci_hi = _wilson(int(high_base[outcome_col].sum()) if n_high_base else 0, n_high_base)
        ci_str = (f"[{ci_lo:.3f}, {ci_hi:.3f}]" if n_high_base else "n/a")
        lines.append(f"Baseline High bucket: n={n_high_base}, realized={rate_high_base:.3f}, "
                     f"lift={lift_base:.2f}x, 95% CI {ci_str}")
        lines.append("")

        lines.append("**Marginal contribution: strip each factor, recompute bucket assignments**")
        lines.append("")
        lines.append("| Factor stripped | New High n | New High realized | New lift | Δ vs baseline |")
        lines.append("|---|---:|---:|---:|---:|")

        for factor_to_strip in CF_KEYS:
            def _strip(d, f=factor_to_strip):
                d2 = dict(d)
                d2[f] = 1.0
                return d2
            sub["_cf_strip"] = sub["cf"].apply(_strip)
            sub["_conf_strip"] = sub["_cf_strip"].apply(lambda d, t=target: _conf_for_target(d, t))
            sub["_label_strip"] = sub["_conf_strip"].apply(_label)
            high_strip = sub[sub["_label_strip"] == "High"]
            n = len(high_strip)
            r = high_strip[outcome_col].mean() if n else float("nan")
            lift = r / baseline if baseline > 0 else float("nan")
            delta = (lift_base - lift) if not np.isnan(lift_base) and not np.isnan(lift) else float("nan")
            sign = "+" if delta > 0 else ""
            lines.append(f"| {factor_to_strip} | {n} | {r:.3f} | {lift:.2f}x | {sign}{delta:.2f}x |")

        lines.append("")
        lines.append("Interpretation: a factor whose Δ > 0 is **adding lift** to the High bucket "
                     "(stripping it makes High realize worse). Δ ≤ 0 means the factor is "
                     "**not earning its keep** for this target.")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
