#!/usr/bin/env python3
"""
Runtime data-quality gate.

Two phases:
  pre-run  (default):  validate freshness/completeness of source files BEFORE predictions.
                       Fails fast on stale or missing data so the pipeline doesn't run
                       on dummy / default-laden inputs.
  post-run (--post):   audit the prediction JSON AFTER the pipeline writes it.
                       Counts rows by data_quality tag, flags rows that are pure-default,
                       and writes data/reports/data_quality_report.md.

Usage:
  python3 src/data_quality_check.py                # pre-run (default)
  python3 src/data_quality_check.py --post         # post-run audit on todays_matchup_predictions.json
  python3 src/data_quality_check.py --post --slate-date 2026-05-01

Break-glass: every consumer that wires this in supports --skip-dq to bypass.

Rolling-feature freshness vs PA data
------------------------------------
``features_val_league.parquet`` is gated below on ``game_date`` with ``max_age_days=2``.
After ``append_statcast_day_to_league_pa.py`` updates ``statcast_pa_level_league.parquet``,
this val file can still pass while being older than the newest PA rows (e.g. sitting at the
2-day boundary). That does not invalidate the run; it only means legacy rolling snapshots
have not been rebuilt.

Primary **Beast** matchup inference reads ``features_val_league_beast.parquet``
(``config.BEAST_VAL_FEATURES``), which is **not** included in ``PRE_CHECKS`` unless you add it.

To align rolling features with the latest Statcast PA file, rebuild train/val feature
parquets from your usual pipeline (e.g. ``src/features.py`` / model-specific builders).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MASTER = ROOT / "data" / "master"
REPORTS = ROOT / "data" / "reports"


# Each entry: (path, kind, freshness rule, rule kwargs)
# kind in {"parquet_max_date", "parquet_has_year", "csv_mtime", "json_mtime", "exists"}
PRE_CHECKS = [
    {
        "path": RAW / "statcast_pa_level_league.parquet",
        "kind": "parquet_max_date",
        "max_age_days": 2,
        "date_col": "game_date",
        "label": "Statcast PA-level (master)",
    },
    {
        "path": RAW / "pitcher_profiles_by_season.parquet",
        "kind": "parquet_has_year",
        "year_col": "season",
        "year": "current",
        "label": "Pitcher profiles (current-year rows)",
    },
    # Legacy prod / tooling val slice — see module docstring "Rolling-feature freshness vs PA data".
    {
        "path": MASTER / "features_val_league.parquet",
        "kind": "parquet_max_date",
        "max_age_days": 2,
        "date_col": "game_date",
        "label": "Val features (rolling stats source)",
    },
    {
        "path": RAW / "qualifying_pitchers_2026.csv",
        "kind": "csv_mtime",
        "max_age_days": 3,
        "label": "Qualifying pitchers 2026",
    },
    {
        "path": RAW / "qualifying_batters_2026.csv",
        "kind": "csv_mtime",
        "max_age_days": 3,
        "label": "Qualifying batters 2026",
    },
    {
        "path": RAW / "batter_pitch_profiles.parquet",
        "kind": "parquet_has_year",
        "year_col": "season",
        "year": "current",
        "label": "Batter pitch profiles (current-year rows)",
    },
    {
        "path": RAW / "todays_matchups.json",
        "kind": "json_mtime",
        "max_age_days": 1,
        "label": "Today's matchups JSON",
    },
]


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def _check_parquet_max_date(path: Path, max_age_days: int, date_col: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "MISSING file"
    try:
        df = pd.read_parquet(path, columns=[date_col])
    except Exception as ex:
        return False, f"read failed: {ex}"
    if df.empty:
        return False, "empty"
    max_d = pd.Timestamp(df[date_col].max())
    today = pd.Timestamp(date.today())
    age_days = (today - max_d).days
    ok = age_days <= max_age_days
    return ok, f"max_date={max_d.date()} (age={age_days}d, limit={max_age_days}d)"


def _check_parquet_has_year(path: Path, year_col: str, year) -> tuple[bool, str]:
    if not path.exists():
        return False, "MISSING file"
    try:
        df = pd.read_parquet(path, columns=[year_col])
    except Exception as ex:
        return False, f"read failed: {ex}"
    if df.empty:
        return False, "empty"
    target = date.today().year if year == "current" else int(year)
    yrs = set(int(y) for y in df[year_col].dropna().unique())
    ok = target in yrs
    return ok, f"years_present={sorted(yrs)[-3:]} (need {target})"


def _check_csv_mtime(path: Path, max_age_days: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "MISSING file"
    age = time.time() - os.path.getmtime(path)
    ok = age <= max_age_days * 86400
    return ok, f"mtime={time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))} (age={_fmt_age(age)}, limit={max_age_days}d)"


def _check_json_mtime(path: Path, max_age_days: int) -> tuple[bool, str]:
    return _check_csv_mtime(path, max_age_days)


def run_pre_checks(verbose: bool = True) -> tuple[bool, list[dict]]:
    """Returns (all_passed, results). Results: [{label, ok, detail, path, kind}]."""
    results = []
    for c in PRE_CHECKS:
        kind = c["kind"]
        if kind == "parquet_max_date":
            ok, detail = _check_parquet_max_date(c["path"], c["max_age_days"], c["date_col"])
        elif kind == "parquet_has_year":
            ok, detail = _check_parquet_has_year(c["path"], c["year_col"], c["year"])
        elif kind == "csv_mtime":
            ok, detail = _check_csv_mtime(c["path"], c["max_age_days"])
        elif kind == "json_mtime":
            ok, detail = _check_json_mtime(c["path"], c["max_age_days"])
        else:
            ok, detail = False, f"unknown kind {kind}"
        results.append({
            "label": c["label"],
            "ok": ok,
            "detail": detail,
            "path": str(c["path"].relative_to(ROOT)),
            "kind": kind,
        })

    if verbose:
        print("=" * 78)
        print("DATA QUALITY CHECK (pre-run)")
        print("=" * 78)
        for r in results:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['label']:<48} {r['detail']}")
            print(f"         {r['path']}")
        n_fail = sum(1 for r in results if not r["ok"])
        print("-" * 78)
        print(f"  {'PASS' if n_fail == 0 else 'FAIL'}: {len(results) - n_fail}/{len(results)} checks passed")
        print("=" * 78)
    return all(r["ok"] for r in results), results


def run_post_checks(predictions_json: Path, slate_date: str | None = None, verbose: bool = True) -> tuple[bool, dict]:
    """Validate the prediction JSON written by predict_matchups.

    Returns (hard_pass, summary). Hard fail if any row is a pure-default zombie:
      data_quality == 'x_is_none_league_avg' AND ytd_pa == 0 AND vs_hand_eff_pa == 0 AND no BvP PA.
    """
    if not predictions_json.exists():
        if verbose:
            print(f"FAIL: predictions JSON missing: {predictions_json}")
        return False, {"error": "missing_predictions_json"}

    rows = json.load(open(predictions_json))
    n = len(rows)

    by_quality: dict[str, int] = {}
    bpt_imputed = 0
    g_roll_imputed = 0
    pitcher_missing = 0
    pitcher_tbd = 0
    pure_defaults: list[dict] = []
    not_engaged = 0
    engaged = 0

    for r in rows:
        aud = r.get("posterior_audit") or {}
        dq = r.get("data_quality") or aud.get("data_quality") or "unknown"
        by_quality[dq] = by_quality.get(dq, 0) + 1
        if aud.get("bpt_imputed"):
            bpt_imputed += 1
        if aud.get("g_roll_imputed_count"):
            g_roll_imputed += 1
        if aud.get("pitcher_profile_missing"):
            pitcher_missing += 1
        if aud.get("pitcher_is_tbd"):
            pitcher_tbd += 1
        if aud.get("engagement_gate") is False:
            not_engaged += 1
        elif aud.get("engagement_gate") is True:
            engaged += 1

        if dq == "x_is_none_league_avg":
            ypa = float(aud.get("ytd_pa", 0) or 0)
            eff_pa = float(aud.get("vs_hand_eff_pa", 0) or 0)
            bvp_pa = int(aud.get("bvp_pa", 0) or 0)
            if ypa == 0 and eff_pa == 0 and bvp_pa == 0:
                pure_defaults.append({
                    "batter_name": r.get("batter_name"),
                    "batter_team": r.get("batter_team"),
                    "pitcher_name": r.get("pitcher_name"),
                    "p_hit": r.get("p_hit"),
                    "p_hr": r.get("p_hr"),
                    "p_xbh": r.get("p_xbh"),
                })

    full_gbdt = by_quality.get("full_gbdt", 0)
    pct_full = full_gbdt / max(n, 1)
    summary = {
        "n_total": n,
        "by_quality": by_quality,
        "pct_full_gbdt": round(pct_full, 4),
        "bpt_imputed": bpt_imputed,
        "g_roll_imputed": g_roll_imputed,
        "pitcher_profile_missing_rows": pitcher_missing,
        "pitcher_tbd_rows": pitcher_tbd,
        "engagement_gate_engaged": engaged,
        "engagement_gate_not_engaged": not_engaged,
        "pure_default_rows": pure_defaults,
        "warning_low_full_gbdt": pct_full < 0.85,
        "hard_fail": len(pure_defaults) > 0,
    }

    # Always write a markdown report.
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = _post_report_md(summary, predictions_json, slate_date)
    out = REPORTS / "data_quality_report.md"
    out.write_text(md, encoding="utf-8")
    if slate_date:
        archive_dir = REPORTS / "archive" / slate_date
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "data_quality_report.md").write_text(md, encoding="utf-8")

    if verbose:
        print("=" * 78)
        print("DATA QUALITY CHECK (post-run)")
        print("=" * 78)
        print(f"  Predictions JSON: {predictions_json}")
        print(f"  Total rows: {n}")
        print(f"  By data_quality:")
        for k, v in sorted(by_quality.items(), key=lambda x: -x[1]):
            print(f"    {k:<28} {v:>5}  ({v/max(n,1):.1%})")
        print(f"  bpt_imputed: {bpt_imputed}, g_roll_imputed: {g_roll_imputed}")
        print(f"  pitcher_profile_missing: {pitcher_missing}, pitcher_tbd: {pitcher_tbd}")
        print(f"  engagement_gate engaged: {engaged}, not_engaged: {not_engaged}")
        if summary["warning_low_full_gbdt"]:
            print(f"  WARN: only {pct_full:.1%} of rows have full GBDT coverage (target >= 85%).")
        if pure_defaults:
            print(f"  HARD FAIL: {len(pure_defaults)} row(s) are pure defaults (no career, no YTD, no BvP):")
            for pd_row in pure_defaults[:10]:
                print(f"    - {pd_row['batter_name']} ({pd_row['batter_team']}) vs {pd_row['pitcher_name']}")
        print(f"  Report → {out}")
        print("=" * 78)

    return (not summary["hard_fail"]), summary


def _post_report_md(summary: dict, predictions_json: Path, slate_date: str | None) -> str:
    lines = ["# Data Quality Report"]
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Predictions JSON: `{predictions_json.relative_to(ROOT)}`")
    if slate_date:
        lines.append(f"- Slate date: {slate_date}")
    lines.append(f"- Total rows: **{summary['n_total']}**")
    lines.append("")
    lines.append("## Coverage by data_quality")
    lines.append("")
    lines.append("| data_quality | rows | share |")
    lines.append("|---|---:|---:|")
    n = max(summary["n_total"], 1)
    for k, v in sorted(summary["by_quality"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} | {v/n:.1%} |")
    lines.append("")
    lines.append("## Imputation counters")
    lines.append("")
    lines.append("| signal | rows | share |")
    lines.append("|---|---:|---:|")
    lines.append(f"| bpt_imputed (pitch-type EV missing) | {summary['bpt_imputed']} | {summary['bpt_imputed']/n:.1%} |")
    lines.append(f"| g_roll_imputed (rolling NaN → league) | {summary['g_roll_imputed']} | {summary['g_roll_imputed']/n:.1%} |")
    lines.append(f"| pitcher_profile_missing | {summary['pitcher_profile_missing_rows']} | {summary['pitcher_profile_missing_rows']/n:.1%} |")
    lines.append(f"| pitcher_is_tbd | {summary['pitcher_tbd_rows']} | {summary['pitcher_tbd_rows']/n:.1%} |")
    lines.append(f"| engagement_gate engaged | {summary['engagement_gate_engaged']} | {summary['engagement_gate_engaged']/n:.1%} |")
    lines.append(f"| engagement_gate NOT engaged | {summary['engagement_gate_not_engaged']} | {summary['engagement_gate_not_engaged']/n:.1%} |")
    lines.append("")
    lines.append(f"- Full-GBDT coverage: **{summary['pct_full_gbdt']:.1%}** (warn threshold 85%).")
    if summary["warning_low_full_gbdt"]:
        lines.append("- WARNING: full-GBDT coverage below 85% — predictions are leaning heavily on the X-is-None league-avg path.")
    lines.append("")
    lines.append("## Hard-fail check (pure-default rows)")
    lines.append("")
    if summary["pure_default_rows"]:
        lines.append(f"- HARD FAIL: {len(summary['pure_default_rows'])} row(s) have no career, no YTD, AND no BvP evidence.")
        lines.append("")
        lines.append("| Batter | Team | Pitcher | p_hit | p_hr | p_xbh |")
        lines.append("|---|---|---|---:|---:|---:|")
        for r in summary["pure_default_rows"][:25]:
            lines.append(
                f"| {r['batter_name']} | {r['batter_team']} | {r['pitcher_name']} | "
                f"{(r.get('p_hit') or 0):.4f} | {(r.get('p_hr') or 0):.4f} | {(r.get('p_xbh') or 0):.4f} |"
            )
    else:
        lines.append("- PASS: no pure-default rows.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--post", action="store_true", help="Run post-prediction audit instead of pre-run check")
    p.add_argument("--predictions-json", default=str(REPORTS / "todays_matchup_predictions.json"))
    p.add_argument("--slate-date", default=None)
    args = p.parse_args()

    if args.post:
        ok, _ = run_post_checks(Path(args.predictions_json), args.slate_date, verbose=True)
    else:
        ok, _ = run_pre_checks(verbose=True)

    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
