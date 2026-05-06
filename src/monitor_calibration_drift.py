#!/usr/bin/env python3
"""
M3: Per-conf-bucket drift monitor + auto-recalibration trigger.

Computes rolling 14-day per-(target, bucket) realized rate vs predicted, flags
drift and auto-triggers a refit of the C1 isotonic calibration when needed.
Emits Section 0 markdown for the daily chat output.

Status grades:
  HEALTHY  — bucket realized rate within 5% of recent baseline
  WARN     — 5%-10% drift; refit triggered
  CRITICAL — High bucket lift dropped below 1.0 OR drift > 10%

Usage:
  python3 src/monitor_calibration_drift.py
  python3 src/monitor_calibration_drift.py --window-days 14
  python3 src/monitor_calibration_drift.py --no-refit       # report only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import REPORTS_DIR  # noqa: E402
from matchup_tracking import TRACKING_MAIN  # noqa: E402

DRIFT_LOG = ROOT / "data" / "tracking" / "calibration_drift_log.parquet"
SECTION0_MD = REPORTS_DIR / "section_0_drift.md"

BUCKETS = ["Very Low", "Low", "Medium", "High"]

# (target, raw_col, conf_label_col, vs_sp_outcome, vs_sp_pa)
TARGETS = [
    ("HR", "p_hr", "conf_hr_label", "outcome_hr_vs_sp", "outcome_pa_vs_sp"),
    ("Hit", "p_hit", "conf_hit_label", "outcome_h_vs_sp", "outcome_pa_vs_sp"),
    ("XBH", "p_xbh", "conf_xbh_label", "outcome_xbh_vs_sp", "outcome_pa_vs_sp"),
]

WARN_DRIFT = 0.05
CRITICAL_DRIFT = 0.10
HIGH_BUCKET_LIFT_FLOOR = 1.0


def _load_filled() -> pd.DataFrame:
    df = pd.read_parquet(TRACKING_MAIN)
    return df[df["outcome_filled_at"].notna()].copy()


def _window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty or days <= 0:
        return df
    max_d = pd.to_datetime(df["slate_date"]).max()
    cutoff = max_d - pd.Timedelta(days=days)
    return df[pd.to_datetime(df["slate_date"]) >= cutoff].copy()


def _bucket_per_pa(g: pd.DataFrame, raw_col: str, outcome_col: str, pa_col: str) -> dict:
    sub = g[g[pa_col].notna() & (g[pa_col] > 0)]
    n = len(sub)
    if n == 0:
        return {"n": 0, "n_pa": 0, "predicted_rate": float("nan"), "realized_rate": float("nan")}
    n_pa = float(sub[pa_col].sum())
    realized = float(sub[outcome_col].sum() / n_pa)
    predicted = float(np.average(sub[raw_col], weights=sub[pa_col]))
    return {"n": int(n), "n_pa": n_pa, "predicted_rate": predicted, "realized_rate": realized}


def _grade(predicted: float, realized: float, bucket: str, baseline: float) -> tuple[str, str]:
    if np.isnan(predicted) or np.isnan(realized):
        return "INSUFFICIENT", "n=0"
    drift = abs(predicted - realized) / max(predicted, 1e-6)
    lift = realized / baseline if baseline > 0 else float("nan")

    if bucket == "High" and lift < HIGH_BUCKET_LIFT_FLOOR:
        return "CRITICAL", f"High bucket lift {lift:.2f}x < {HIGH_BUCKET_LIFT_FLOOR:.1f}x floor"
    if drift > CRITICAL_DRIFT:
        return "CRITICAL", f"drift {drift*100:.1f}% > {CRITICAL_DRIFT*100:.0f}%"
    if drift > WARN_DRIFT:
        return "WARN", f"drift {drift*100:.1f}% > {WARN_DRIFT*100:.0f}%; refit triggered"
    return "HEALTHY", f"drift {drift*100:.1f}%"


def compute_drift(window_days: int = 14) -> dict:
    df = _load_filled()
    if df.empty:
        return {"_meta": {"n_rows": 0}, "rows": []}

    df_w = _window(df, window_days)
    rows = []
    for label, raw_col, conf_label_col, outcome_col, pa_col in TARGETS:
        # Baseline = realized rate on all picks in the window (not bucket-filtered)
        all_pa = df_w[df_w[pa_col].notna() & (df_w[pa_col] > 0)]
        baseline = float(all_pa[outcome_col].sum() / all_pa[pa_col].sum()) if len(all_pa) else float("nan")
        for bucket in BUCKETS:
            g = df_w[df_w[conf_label_col] == bucket]
            stats = _bucket_per_pa(g, raw_col, outcome_col, pa_col)
            grade, note = _grade(stats["predicted_rate"], stats["realized_rate"], bucket, baseline)
            lift = (stats["realized_rate"] / baseline
                    if baseline and baseline > 0 and not np.isnan(stats["realized_rate"])
                    else float("nan"))
            rows.append({
                "target": label,
                "bucket": bucket,
                "n": stats["n"],
                "n_pa": int(stats["n_pa"]),
                "predicted": stats["predicted_rate"],
                "realized": stats["realized_rate"],
                "baseline": baseline,
                "lift": lift,
                "grade": grade,
                "note": note,
            })

    return {
        "_meta": {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "window_days": window_days,
            "n_rows": int(len(df_w)),
        },
        "rows": rows,
    }


def render_section0(report: dict) -> str:
    meta = report["_meta"]
    rows = report["rows"]
    order_t = {"HR": 0, "Hit": 1, "XBH": 2}
    rows = sorted(
        rows,
        key=lambda r: (
            order_t.get(r["target"], 9),
            BUCKETS.index(r["bucket"]) if r["bucket"] in BUCKETS else 99,
        ),
    )
    lines = ["## Section 0: Bucket Health (drift monitor)", ""]
    if meta.get("n_rows", 0) == 0:
        lines.append("_No filled outcomes yet — drift monitor inactive._")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"**Window:** trailing {meta['window_days']} days "
                 f"({meta['n_rows']} filled outcomes)")
    lines.append("")
    hr_rows = [r for r in rows if r["target"] == "HR"]
    if hr_rows:
        brief = " · ".join(f"{r['bucket']} → **{r['grade']}**" for r in hr_rows)
        lines.append(f"**HR (Beast) buckets:** {brief}")
        lines.append("")
    lines.append("| Target | Bucket | n | n PA | Predicted | Realized | Lift | Status | Note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        pred_s = "—" if np.isnan(r["predicted"]) else f"{r['predicted']:.4f}"
        real_s = "—" if np.isnan(r["realized"]) else f"{r['realized']:.4f}"
        lift_s = "—" if np.isnan(r["lift"]) else f"{r['lift']:.2f}x"
        lines.append(
            f"| {r['target']} | {r['bucket']} | {r['n']} | {r['n_pa']} | "
            f"{pred_s} | {real_s} | {lift_s} | {r['grade']} | {r['note']} |"
        )
    lines.append("")

    # Surface any CRITICAL flags at the top
    crits = [r for r in rows if r["grade"] == "CRITICAL"]
    warns = [r for r in rows if r["grade"] == "WARN"]
    if crits:
        lines.insert(2, "")
        lines.insert(2, "> **CRITICAL drift** in: " + ", ".join(
            f"{r['target']}/{r['bucket']} ({r['note']})" for r in crits
        ))
    if warns:
        idx = 2 + (1 if crits else 0)
        lines.insert(idx, "> **WARN drift triggered refit** in: " + ", ".join(
            f"{r['target']}/{r['bucket']}" for r in warns
        ))
        lines.insert(idx, "")

    return "\n".join(lines)


def append_log(report: dict) -> None:
    DRIFT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(report["rows"])
    new["computed_at"] = report["_meta"]["computed_at"]
    new["window_days"] = report["_meta"]["window_days"]
    if DRIFT_LOG.exists():
        existing = pd.read_parquet(DRIFT_LOG)
        out = pd.concat([existing, new], ignore_index=True)
    else:
        out = new
    out.to_parquet(DRIFT_LOG, index=False)


def trigger_refit(report: dict) -> bool:
    """Auto-refit the C1 calibration if any bucket drift > WARN threshold."""
    needs_refit = any(r["grade"] in ("WARN", "CRITICAL") and r["n"] >= 5 for r in report["rows"])
    if not needs_refit:
        return False
    print("Drift exceeds WARN threshold — refitting per-bucket calibration ...")
    res = subprocess.run(
        [sys.executable, str(ROOT / "src" / "calibrate_predictions.py")],
        capture_output=True, text=True
    )
    print(res.stdout)
    if res.returncode != 0:
        print("WARN: recal subprocess failed:", res.stderr, file=sys.stderr)
        return False
    # Reset the in-process cache so subsequent inferences use the new map.
    try:
        from calibrate_predictions import reset_cache
        reset_cache()
    except Exception:
        pass
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-bucket calibration drift monitor.")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--no-refit", action="store_true",
                        help="Report only; do not auto-trigger refit on drift.")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip appending to drift log.")
    args = parser.parse_args()

    report = compute_drift(window_days=args.window_days)
    md = render_section0(report)
    SECTION0_MD.parent.mkdir(parents=True, exist_ok=True)
    SECTION0_MD.write_text(md)
    print(md)

    if not args.no_log:
        append_log(report)

    if not args.no_refit:
        trigger_refit(report)

    try:
        from gen_hr_calibration_snapshot import write_snapshot

        write_snapshot(window_days=args.window_days)
    except Exception as ex:
        print(f"WARN: HR calibration snapshot skipped: {ex}", file=sys.stderr)


if __name__ == "__main__":
    main()
