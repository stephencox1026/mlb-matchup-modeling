#!/usr/bin/env python3
"""
Per-conf-bucket calibration report — the primary trust report for model outputs.

Headline: per-(target, conf_bucket) realized rate with Wilson 95% CIs, lift vs
baseline, monotonicity check. Top-decile lift and Brier are demoted to a
secondary diagnostics section.

Usage:
  python3 src/analyze_prediction_tracking.py
  python3 src/analyze_prediction_tracking.py --high-conf
  python3 src/analyze_prediction_tracking.py --target hr
  python3 src/analyze_prediction_tracking.py --window-days 14
  python3 src/analyze_prediction_tracking.py --legacy   # also emit old-format file

Outputs:
  data/reports/conf_bucket_calibration.md            (primary; this report)
  data/reports/conf_bucket_calibration_high_conf.md  (--high-conf slice)
  data/reports/prediction_calibration_summary.md     (--legacy backward compat)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import REPORTS_DIR  # noqa: E402
from matchup_tracking import TRACKING_HIGH_CONF, TRACKING_MAIN  # noqa: E402

OUT_MD = REPORTS_DIR / "conf_bucket_calibration.md"
OUT_MD_HC = REPORTS_DIR / "conf_bucket_calibration_high_conf.md"
LEGACY_MD = REPORTS_DIR / "prediction_calibration_summary.md"
LEGACY_MD_HC = REPORTS_DIR / "prediction_calibration_summary_high_conf.md"

BUCKET_ORDER = ["Very Low", "Low", "Medium", "High"]

# Whole-game outcomes (back-compat). Picks up bullpen + late PAs as well.
TARGET_SPECS_WHOLEGAME = [
    ("HR", "p_hr", "adj_p_hr", "conf_hr", "conf_hr_label", "outcome_hr_flag"),
    ("Hit", "p_hit", "adj_p_hit", "conf_hit", "conf_hit_label", "outcome_hit_flag"),
    ("XBH", "p_xbh", "adj_p_xbh", "conf_xbh", "conf_xbh_label", "outcome_xbh_flag"),
]

# vs-SP outcomes (C3 cleanup): scoped to PAs vs the predicted starter only.
# This is the apples-to-apples evaluation set; whole-game is kept for back-compat.
TARGET_SPECS_VS_SP = [
    ("HR", "p_hr", "adj_p_hr", "conf_hr", "conf_hr_label", "outcome_hr_flag_vs_sp"),
    ("Hit", "p_hit", "adj_p_hit", "conf_hit", "conf_hit_label", "outcome_hit_flag_vs_sp"),
    ("XBH", "p_xbh", "adj_p_xbh", "conf_xbh", "conf_xbh_label", "outcome_xbh_flag_vs_sp"),
]

TARGET_SPECS = TARGET_SPECS_VS_SP  # default eval lens is vs-SP


def wilson_ci(hits: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    res = scistats.binomtest(int(hits), int(n)).proportion_ci(conf, method="wilson")
    return (float(res.low), float(res.high))


def _filter_window(d: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if days is None or days <= 0 or "slate_date" not in d.columns:
        return d
    if d.empty:
        return d
    max_date = pd.to_datetime(d["slate_date"]).max()
    cutoff = max_date - pd.Timedelta(days=days)
    return d[pd.to_datetime(d["slate_date"]) >= cutoff].copy()


def bucket_table(d: pd.DataFrame, conf_label_col: str, y_col: str,
                 score_col: str, raw_p_col: str, baseline: float) -> pd.DataFrame:
    """One row per bucket with n, realized, Wilson CI, mean predicted, lift, status.

    For vs-SP outcome columns, NaN means batter never faced the predicted starter and
    is excluded from the bucket count entirely (not counted as a 0 outcome).
    """
    if conf_label_col not in d.columns or y_col not in d.columns:
        return pd.DataFrame()
    rows = []
    for bucket in BUCKET_ORDER:
        g_all = d[d[conf_label_col] == bucket]
        # Exclude rows where outcome is NaN (e.g., batter never faced predicted SP for vs-SP cols).
        g = g_all[g_all[y_col].notna()]
        n = len(g)
        n_excluded = len(g_all) - n
        if n == 0:
            rows.append({
                "bucket": bucket, "n": 0, "hits": 0, "realized": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "mean_score": float("nan"), "mean_raw_p": float("nan"),
                "lift": float("nan"),
                "status": f"EMPTY ({n_excluded} excluded)" if n_excluded else "EMPTY",
            })
            continue
        hits = int(g[y_col].sum())
        realized = hits / n
        lo, hi = wilson_ci(hits, n)
        mean_score = float(g[score_col].mean()) if score_col in g.columns else float("nan")
        mean_raw_p = float(g[raw_p_col].mean()) if raw_p_col in g.columns else float("nan")
        lift = realized / baseline if baseline > 0 else float("nan")
        status = f"{n_excluded} excluded (no SP PA)" if n_excluded else ""
        rows.append({
            "bucket": bucket, "n": n, "hits": hits, "realized": realized,
            "ci_lo": lo, "ci_hi": hi,
            "mean_score": mean_score, "mean_raw_p": mean_raw_p,
            "lift": lift, "status": status,
        })
    return pd.DataFrame(rows)


def check_monotonicity(table: pd.DataFrame, tolerance_pp: float = 0.0) -> tuple[bool, str]:
    """True iff realized rate is non-decreasing across buckets (within tolerance)."""
    if table.empty:
        return False, "no buckets"
    rates = []
    for _, r in table.iterrows():
        if r["n"] > 0 and not np.isnan(r["realized"]):
            rates.append((r["bucket"], r["realized"]))
    if len(rates) < 2:
        return True, "too few populated buckets to test"
    violations = []
    for i in range(1, len(rates)):
        prev_b, prev_r = rates[i - 1]
        cur_b, cur_r = rates[i]
        if cur_r + tolerance_pp < prev_r:
            violations.append(f"{prev_b}({prev_r:.3f}) > {cur_b}({cur_r:.3f})")
    if violations:
        return False, "; ".join(violations)
    return True, "OK"


def render_bucket_section(name: str, table: pd.DataFrame, baseline: float) -> list[str]:
    lines = [f"### {name}", ""]
    if table.empty:
        lines.append("_no data_")
        lines.append("")
        return lines

    lines.append(
        "| Bucket | n | Hits | Realized | 95% CI | Mean Score | Mean Raw P | Lift | Status |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---|")
    for _, r in table.iterrows():
        if r["n"] == 0:
            ci_str = "—"
            realized_str = "—"
            mean_score_str = "—"
            mean_raw_p_str = "—"
            lift_str = "—"
        else:
            ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            realized_str = f"{r['realized']:.3f}"
            mean_score_str = f"{r['mean_score']:.4f}"
            mean_raw_p_str = f"{r['mean_raw_p']:.4f}"
            lift_str = f"{r['lift']:.2f}x"
        lines.append(
            f"| {r['bucket']} | {int(r['n'])} | {int(r['hits'])} | {realized_str} | "
            f"{ci_str} | {mean_score_str} | {mean_raw_p_str} | {lift_str} | {r['status']} |"
        )
    lines.append("")
    mono_ok, mono_msg = check_monotonicity(table)
    flag = "PASS" if mono_ok else "**FAIL**"
    lines.append(f"_Monotonicity (Very Low ≤ Low ≤ Medium ≤ High): {flag} — {mono_msg}_")
    lines.append(f"_Baseline (all picks): {baseline:.4f}_")
    lines.append("")
    return lines


def render_overall_metrics(d: pd.DataFrame) -> list[str]:
    """Brier / AUC / log loss per target (secondary diagnostic)."""
    lines = ["## Overall metrics (secondary)", ""]
    lines.append("| Target | n | Realized | Mean P (raw) | Mean P (score) | Brier (raw) | Brier (score) | AUC (score) | Log Loss (raw) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, raw_col, score_col, _, _, y_col in TARGET_SPECS:
        if y_col not in d.columns:
            continue
        sub = d[[raw_col, score_col, y_col]].dropna()  # drops NaN outcomes (no-SP-PA rows for vs-SP)
        if sub.empty:
            continue
        y = sub[y_col].astype(int).values
        p_raw = np.clip(sub[raw_col].astype(float).values, 1e-6, 1 - 1e-6)
        p_score = np.clip(sub[score_col].astype(float).values, 1e-6, 1 - 1e-6)
        n = len(y)
        if np.unique(y).size < 2:
            lines.append(f"| {label} | {n} | {y.mean():.3f} | {p_raw.mean():.3f} | {p_score.mean():.3f} | — | — | — | — |")
            continue
        try:
            auc = roc_auc_score(y, p_score)
        except Exception:
            auc = float("nan")
        lines.append(
            f"| {label} | {n} | {y.mean():.3f} | {p_raw.mean():.3f} | {p_score.mean():.3f} | "
            f"{brier_score_loss(y, p_raw):.4f} | {brier_score_loss(y, p_score):.4f} | "
            f"{auc:.3f} | {log_loss(y, p_raw):.4f} |"
        )
    lines.append("")
    return lines


def render_topdecile(d: pd.DataFrame) -> list[str]:
    """Top-decile lift per target (secondary diagnostic, kept for reference)."""
    lines = ["## Top-decile lift by score (secondary)", ""]
    lines.append("| Target | Top 10% n | Top 10% realized | Bottom 10% realized | Overall | Top-decile lift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, _, score_col, _, _, y_col in TARGET_SPECS:
        if y_col not in d.columns or score_col not in d.columns:
            continue
        sub = d[[score_col, y_col]].dropna()
        if len(sub) < 20:
            continue
        try:
            sub = sub.copy()
            sub["dec"] = pd.qcut(sub[score_col], 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        top = sub[sub["dec"] == sub["dec"].max()]
        bot = sub[sub["dec"] == 0]
        overall = float(sub[y_col].mean())
        top_rate = float(top[y_col].mean()) if len(top) else float("nan")
        bot_rate = float(bot[y_col].mean()) if len(bot) else float("nan")
        lift = top_rate / overall if overall > 0 else float("nan")
        lines.append(
            f"| {label} | {len(top)} | {top_rate:.3f} | {bot_rate:.3f} | {overall:.3f} | {lift:.2f}x |"
        )
    lines.append("")
    return lines


def render_markdown(df: pd.DataFrame, title: str, window_days: int | None,
                    target_filter: str | None, high_only: bool) -> str:
    filled = df["outcome_filled_at"].notna()
    d_all = df[filled].copy()
    lines = [
        f"# {title}",
        "",
    ]
    if d_all.empty:
        lines.append("_No filled outcomes yet. Run `python3 src/fill_matchup_prediction_outcomes.py` "
                     "after Statcast includes the slate calendar day._")
        lines.append("")
        return "\n".join(lines)

    d = _filter_window(d_all, window_days)
    if d.empty:
        lines.append(f"_No outcomes within last {window_days} days. Showing full window._")
        d = d_all

    date_min = pd.to_datetime(d["slate_date"]).min().date()
    date_max = pd.to_datetime(d["slate_date"]).max().date()
    n_total = len(d_all)
    n_window = len(d)
    lines.append(f"**Window:** {date_min} → {date_max} ({n_window:,} filled outcomes "
                 f"of {n_total:,} total tracked)")
    if window_days:
        lines.append(f"**Rolling window:** {window_days} days")
    lines.append("")

    targets_to_render = TARGET_SPECS
    if target_filter:
        tf = target_filter.lower()
        targets_to_render = [t for t in TARGET_SPECS if t[0].lower() == tf]
        if not targets_to_render:
            lines.append(f"_Unknown target {target_filter!r}. Valid: hr, hit, xbh._")
            return "\n".join(lines)

    # === HEADLINE: per-conf-bucket reliability (vs-SP) ===
    has_vs_sp = "outcome_hr_flag_vs_sp" in d.columns
    primary_specs = TARGET_SPECS_VS_SP if has_vs_sp else TARGET_SPECS_WHOLEGAME
    primary_label = "vs-SP" if has_vs_sp else "whole-game"

    lines.append(f"## HEADLINE: Per-conf-bucket realized rate ({primary_label} outcomes)")
    lines.append("")
    if has_vs_sp:
        lines.append("**vs-SP** = scoped to PAs where the batter actually faced the predicted "
                     "starter. This is the apples-to-apples eval (C3 cleanup). Whole-game "
                     "outcomes are kept as a secondary section for back-compat.")
    lines.append("The High-conf bucket is the betting deliverable; monotonic Realized "
                 "across (Very Low → High) is the goal.")
    lines.append("")

    if target_filter:
        tf = target_filter.lower()
        primary_specs = [t for t in primary_specs if t[0].lower() == tf]

    any_violation = False
    for label, raw_col, score_col, conf_col, conf_label_col, y_col in primary_specs:
        if y_col not in d.columns or conf_label_col not in d.columns:
            continue
        sub = d[d[y_col].notna()] if has_vs_sp else d
        baseline = float(sub[y_col].fillna(0).mean()) if len(sub) else float("nan")
        table = bucket_table(d, conf_label_col, y_col, score_col, raw_col, baseline)
        if high_only:
            table = table[table["bucket"] == "High"]
        lines.extend(render_bucket_section(label, table, baseline))
        ok, _ = check_monotonicity(table)
        if not ok:
            any_violation = True

    if any_violation:
        lines.append("> **WARNING**: at least one target has non-monotonic bucket realized rates. "
                     "This indicates a structural issue in the conf labeling system "
                     "(see [docs/conf_label_audit.md](../../docs/conf_label_audit.md)).")
        lines.append("")

    # === SECONDARY: whole-game outcomes (for context / back-compat) ===
    if has_vs_sp and not high_only:
        wg_specs = TARGET_SPECS_WHOLEGAME
        if target_filter:
            tf = target_filter.lower()
            wg_specs = [t for t in wg_specs if t[0].lower() == tf]
        lines.append("## Secondary: Per-conf-bucket realized rate (whole-game outcomes)")
        lines.append("")
        lines.append("Includes bullpen + late-game PAs we never predicted. Useful as a "
                     "cross-check; vs-SP above is the primary lens.")
        lines.append("")
        for label, raw_col, score_col, conf_col, conf_label_col, y_col in wg_specs:
            if y_col not in d.columns or conf_label_col not in d.columns:
                continue
            baseline = float(d[y_col].fillna(0).mean())
            table = bucket_table(d, conf_label_col, y_col, score_col, raw_col, baseline)
            lines.extend(render_bucket_section(label, table, baseline))

    if high_only:
        return "\n".join(lines)

    # === Drift check: compare last 7d vs 8-30d ===
    if window_days is None or window_days >= 14:
        lines.append("## Drift check: last 7d vs 8-30d (per High bucket)")
        lines.append("")
        d_recent = _filter_window(d_all, 7)
        d_older_full = _filter_window(d_all, 30)
        d_older = d_older_full[~d_older_full.index.isin(d_recent.index)]
        if len(d_recent) > 0 and len(d_older) > 0:
            lines.append("| Target | 7d High n | 7d High realized | 8-30d High n | 8-30d High realized | Δ |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for label, _, _, _, conf_label_col, y_col in TARGET_SPECS:
                if conf_label_col not in d_recent.columns:
                    continue
                rec_high = d_recent[d_recent[conf_label_col] == "High"]
                old_high = d_older[d_older[conf_label_col] == "High"]
                rec_rate = rec_high[y_col].mean() if len(rec_high) else float("nan")
                old_rate = old_high[y_col].mean() if len(old_high) else float("nan")
                delta = (rec_rate - old_rate) if not (np.isnan(rec_rate) or np.isnan(old_rate)) else float("nan")
                lines.append(
                    f"| {label} | {len(rec_high)} | {rec_rate:.3f} | "
                    f"{len(old_high)} | {old_rate:.3f} | {delta:+.3f} |"
                )
        else:
            lines.append("_Not enough rolling data for drift check yet._")
        lines.append("")

    # === Conf factor breakdown per bucket ===
    if "conf_factors_json" in d.columns:
        import json as _json
        cf_keys = [
            "pitcher_data",
            "bvp_hr",
            "bvp_hit",
            "staleness",
            "convergence_hr",
            "convergence_hit",
            "convergence_xbh",
        ]
        d_cf = d.copy()
        d_cf["__cf"] = d_cf["conf_factors_json"].apply(
            lambda s: _json.loads(s) if isinstance(s, str) and s else {}
        )
        for k in cf_keys:
            d_cf[f"cf_{k}"] = d_cf["__cf"].apply(lambda x, k=k: x.get(k, float("nan")))

        lines.append("## Conf factor means by HR bucket")
        lines.append("")
        lines.append("| Bucket | n | " + " | ".join(cf_keys) + " |")
        lines.append("|---|---:|" + "|".join("---:" for _ in cf_keys) + "|")
        for bucket in BUCKET_ORDER:
            g = d_cf[d_cf["conf_hr_label"] == bucket]
            if len(g) == 0:
                row_vals = " | ".join("—" for _ in cf_keys)
                lines.append(f"| {bucket} | 0 | {row_vals} |")
                continue
            row_vals = " | ".join(f"{g[f'cf_{k}'].mean():.3f}" for k in cf_keys)
            lines.append(f"| {bucket} | {len(g)} | {row_vals} |")
        lines.append("")

    # === Secondary: top-decile lift + overall metrics ===
    lines.extend(render_topdecile(d))
    lines.extend(render_overall_metrics(d))

    return "\n".join(lines)


def render_legacy_markdown(df: pd.DataFrame, title: str) -> str:
    """Old-format report for backward compatibility (--legacy flag)."""
    filled = df["outcome_filled_at"].notna()
    d = df[filled].copy()
    lines = [f"# {title}", "", f"Rows with outcomes: **{len(d):,}** (of {len(df):,} total)", ""]
    if len(d) == 0:
        lines.append("_No filled outcomes yet._")
        lines.append("")
        return "\n".join(lines)
    for label, pcol, _, _, _, ycol in TARGET_SPECS:
        if ycol not in d.columns or d[ycol].isna().all():
            continue
        y = d[ycol].fillna(0).astype(int).values
        p = np.clip(d[pcol].astype(float).fillna(0).values, 1e-6, 1 - 1e-6)
        if len(y) >= 5 and np.unique(y).size > 1:
            brier = brier_score_loss(y, p)
            try:
                auc = roc_auc_score(y, p)
                ap = average_precision_score(y, p)
            except Exception:
                auc = ap = float("nan")
            lines.append(f"## {label}")
            lines.append(f"- n = {len(y):,} | realized = {y.mean():.4f} | mean predicted = {p.mean():.4f}")
            lines.append(f"- Brier = {brier:.5f} | AUC = {auc:.4f} | AP = {ap:.4f}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-conf-bucket calibration report.")
    parser.add_argument("--high-conf", action="store_true",
                        help="Use the high-confidence tracking slice file only.")
    parser.add_argument("--target", choices=["hr", "hit", "xbh"], default=None,
                        help="Only render this target (default: all three).")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Rolling window in days (default: full history).")
    parser.add_argument("--high-only", action="store_true",
                        help="Show only the High bucket per target.")
    parser.add_argument("--legacy", action="store_true",
                        help="Also write the old-format prediction_calibration_summary.md.")
    args = parser.parse_args()

    path = TRACKING_HIGH_CONF if args.high_conf else TRACKING_MAIN
    if not path.exists():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)

    title = "Per-conf-bucket calibration"
    if args.high_conf:
        title += " (high-conf tracking file)"
    if args.target:
        title += f" — target: {args.target.upper()}"
    if args.window_days:
        title += f" — last {args.window_days}d"

    md = render_markdown(df, title, args.window_days, args.target, args.high_only)
    out = OUT_MD_HC if args.high_conf else OUT_MD
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")

    if args.legacy:
        legacy_title = ("Prediction calibration (high confidence)"
                        if args.high_conf else "Prediction calibration (all matchups)")
        legacy_md = render_legacy_markdown(df, legacy_title)
        legacy_out = LEGACY_MD_HC if args.high_conf else LEGACY_MD
        legacy_out.write_text(legacy_md, encoding="utf-8")
        print(f"Wrote {legacy_out} (legacy format)")


if __name__ == "__main__":
    main()
