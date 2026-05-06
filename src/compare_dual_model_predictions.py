#!/usr/bin/env python3
"""Summarize prod vs exp matchup probabilities from dual-model tracking Parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import REPORTS_DIR  # noqa: E402
from matchup_tracking import DUAL_MODEL_PARQUET  # noqa: E402

OUT_MD = REPORTS_DIR / "dual_model_diff_summary.md"


def write_dual_summary(
    parquet: Path | None = None,
    run_timestamp: str | None = None,
    out_md: Path | None = None,
) -> Path:
    """Load dual-model Parquet, optionally filter to one run, write markdown summary."""
    path = parquet or DUAL_MODEL_PARQUET
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if run_timestamp:
        df = df[df["run_timestamp"] == run_timestamp].copy()
    elif "run_timestamp" in df.columns and len(df):
        latest = df["run_timestamp"].max()
        df = df[df["run_timestamp"] == latest].copy()
    md = summarize(df)
    out = out_md or OUT_MD
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return out


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 10:
        return None
    return float(s["a"].corr(s["b"]))


def summarize(df: pd.DataFrame) -> str:
    lines = ["# Dual model: production vs experiment", "", f"Rows: **{len(df):,}**", ""]
    pairs = [
        ("p_hit_prod", "p_hit_exp", "hit"),
        ("p_hr_prod", "p_hr_exp", "hr"),
        ("p_xbh_prod", "p_xbh_exp", "xbh"),
        ("p_k_prod", "p_k_exp", "k"),
        ("p_bb_prod", "p_bb_exp", "bb"),
    ]
    lines.append("| target | Pearson r | mean |Δp| | pct abs Δ < 0.02 |")
    lines.append("|---|---:|---:|---:|")
    for pc, pe, name in pairs:
        if pc not in df.columns or pe not in df.columns:
            continue
        sub = df[[pc, pe]].dropna()
        if sub.empty:
            continue
        r = _corr(sub[pc], sub[pe])
        d = (sub[pe] - sub[pc]).astype(float)
        mae = float(d.abs().mean())
        pct = float((d.abs() < 0.02).mean()) if len(d) else 0.0
        r_str = f"{r:.4f}" if r is not None else "—"
        lines.append(f"| {name} | {r_str} | {mae:.4f} | {pct:.1%} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DUAL_MODEL_PARQUET,
        help="Dual-model tracking Parquet",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help="Filter to this run_timestamp (default: latest in file)",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"Missing {args.parquet}")
        sys.exit(1)

    out = write_dual_summary(parquet=args.parquet, run_timestamp=args.run_timestamp)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
