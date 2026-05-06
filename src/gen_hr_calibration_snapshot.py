#!/usr/bin/env python3
"""
Priority 1 — Beast HR calibration snapshot (markdown).

Writes data/reports/hr_calibration_snapshot.md with:
  - Section 0 HR bucket rows (14-day drift window)
  - C1 isotonic / mult_constant summary for target `hr` only from calibration_isotonic.json

Run standalone or after monitor_calibration_drift (same window).

Usage:
  python3 src/gen_hr_calibration_snapshot.py
  python3 src/gen_hr_calibration_snapshot.py --window-days 21
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import REPORTS_DIR  # noqa: E402
from monitor_calibration_drift import compute_drift  # noqa: E402

CALIBRATION_JSON = ROOT / "data" / "priors" / "calibration_isotonic.json"
OUT_MD = REPORTS_DIR / "hr_calibration_snapshot.md"
BUCKETS = ["Very Low", "Low", "Medium", "High"]


def write_snapshot(window_days: int = 14) -> str:
    report = compute_drift(window_days=window_days)
    meta = report.get("_meta", {})
    rows = [r for r in report.get("rows", []) if r.get("target") == "HR"]

    lines = [
        "## Beast HR — calibration snapshot",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC_",
        "",
        f"**Drift window:** {window_days} days | **filled tracking rows:** {meta.get('n_rows', 0)}",
        "",
        "### Section 0 — HR buckets only",
        "",
        "| Bucket | n | n PA | Predicted | Realized | Lift | Status | Note |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    if not rows:
        lines.append("| — | — | — | — | — | — | — | _No HR drift rows (empty tracking or insufficient outcomes)._ |")
    else:
        for r in rows:
            pred_s = "—" if np.isnan(r["predicted"]) else f"{r['predicted']:.4f}"
            real_s = "—" if np.isnan(r["realized"]) else f"{r['realized']:.4f}"
            lift_s = "—" if np.isnan(r["lift"]) else f"{r['lift']:.2f}x"
            lines.append(
                f"| {r['bucket']} | {r['n']} | {r['n_pa']} | {pred_s} | {real_s} | {lift_s} "
                f"| {r['grade']} | {r['note']} |"
            )

    lines.extend(["", "### C1 calibration — HR target only", ""])

    if not CALIBRATION_JSON.exists():
        lines.append("_No `calibration_isotonic.json` — run `python3 src/calibrate_predictions.py`._")
    else:
        cal = json.loads(CALIBRATION_JSON.read_text())
        cm = cal.get("_meta", {})
        lines.append(
            f"_Trained {cm.get('trained_at', '—')} | rows={cm.get('n_rows_total', '—')} | "
            f"slates {cm.get('slate_date_range', '—')}_"
        )
        lines.extend(["", "| Bucket | Method | n_rows | n_pa | mean_pred | mean_real | detail |", "|---|---|---:|---:|---:|---:|---|"])
        buckets = cal.get("targets", {}).get("hr", {}).get("buckets", {})
        for b in BUCKETS:
            cell = buckets.get(b) or {}
            method = cell.get("method", "—")
            detail = ""
            if method == "mult_constant":
                detail = f"x{cell.get('constant', 1.0):.4f}"
            elif method == "isotonic":
                detail = f"{len(cell.get('knots_x', []))} knots"
            elif method == "identity":
                detail = cell.get("note") or "identity"
            mp = cell.get("mean_pred")
            mr = cell.get("mean_realized")
            mp_s = "—" if mp is None else f"{float(mp):.4f}"
            mr_s = "—" if mr is None else f"{float(mr):.4f}"
            lines.append(
                f"| {b} | {method} | {cell.get('n_rows', '—')} | {cell.get('n_pa', '—')} "
                f"| {mp_s} | {mr_s} | {detail} |"
            )

    lines.append("")
    text = "\n".join(lines)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Beast HR calibration snapshot markdown.")
    parser.add_argument("--window-days", type=int, default=14)
    args = parser.parse_args()
    md = write_snapshot(window_days=args.window_days)
    print(md)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
