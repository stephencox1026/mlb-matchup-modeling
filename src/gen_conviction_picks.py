#!/usr/bin/env python3
"""
CB3: Conviction Picks digest (Section 14 of the daily output).

Reads today's predictions JSON. For each target: **M1 Lock** rows require
**hand conf High or Medium** (dual gate); else fall back to legacy **High**.
Caps picks per target; joins trailing 14-day batter outcome streak; footer = M3 bucket health.

Output:
  data/reports/section_14_conviction_picks.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR, REPORTS_DIR  # noqa: E402

DEFAULT_PRED = REPORTS_DIR / "todays_matchup_predictions.json"
DEFAULT_OUT = REPORTS_DIR / "section_14_conviction_picks.md"
DEFAULT_DRIFT = REPORTS_DIR / "section_0_drift.md"
PA_LEVEL_PARQUET = RAW_DIR / "statcast_pa_level_league.parquet"

CB3_HR_LIFT_FLOOR = 1.30  # warn in footer if HR High bucket lift below this on rolling 14d
SEC14_MAX_PICKS_PER_TARGET = 15
SEC14_MIN_HAND_LABEL = frozenset({"High", "Medium"})


def _three_pa(p: float) -> float:
    return float((1 - (1 - p) ** 3) * 100)


def _score_col(row: dict, target: str) -> float:
    return float(row.get(f"score_{target}", row.get(f"adj_p_{target}", 0.0)))


def _cal_p_col(row: dict, target: str) -> float:
    return float(row.get(f"p_{target}_calibrated", row.get(f"p_{target}", 0.0)))


def _format_pick_row(rank: int, p: dict, target: str) -> str:
    bvp = p.get("bvp_text") or "—"
    if not bvp:
        bvp = "—"
    cf = p.get("conf_factors") or {}
    if target == "hr":
        conv_k = "convergence_hr"
        bvk = "bvp_hr"
    elif target == "hit":
        conv_k = "convergence_hit"
        bvk = "bvp_hit"
    else:
        conv_k = "convergence_xbh"
        bvk = "bvp_hit"
    factor_summary = (
        f"BvP{cf.get(bvk, 1.0):.2f}"
        f"·Conv{cf.get(conv_k, 1.0):.2f}"
        f"·Pitch{cf.get('pitcher_data', 1.0):.2f}"
    )
    return (
        f"| {rank} | {p['batter_name']} ({p.get('batter_team', '?')}) "
        f"vs {p['pitcher_name']} ({p.get('pitcher_team', '?')}) | "
        f"{_score_col(p, target)*100:.2f} | "
        f"{_cal_p_col(p, target)*100:.1f}% | "
        f"{p[f'p_{target}']*100:.1f}% | "
        f"{_three_pa(p[f'p_{target}']):.1f}% | "
        f"{factor_summary} | {bvp} |"
    )


def _batter_streak_table(pred_rows: list[dict], days_back: int = 14) -> dict[int, str]:
    """For each batter in the picks, fetch last-N-day H/HR/XBH counts (whole-game)."""
    bids = sorted({int(p["batter_mlbam_id"]) for p in pred_rows
                   if p.get("batter_mlbam_id") is not None})
    if not bids:
        return {}
    if not PA_LEVEL_PARQUET.exists():
        return {}
    pa = pd.read_parquet(PA_LEVEL_PARQUET, columns=["game_date", "batter", "events"])
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")
    pa = pa.dropna(subset=["game_date", "batter"])
    cutoff = pa["game_date"].max() - pd.Timedelta(days=days_back)
    pa = pa[(pa["game_date"] >= cutoff) & pa["batter"].astype(int).isin(bids)]
    if pa.empty:
        return {}
    ev = pa["events"].fillna("").str.lower()
    pa["is_h"] = ev.isin({"single", "double", "triple", "home_run"}).astype(int)
    pa["is_hr"] = (ev == "home_run").astype(int)
    pa["is_xbh"] = ev.isin({"double", "triple", "home_run"}).astype(int)
    g = pa.groupby("batter").agg(
        pa=("events", "count"), h=("is_h", "sum"),
        hr=("is_hr", "sum"), xbh=("is_xbh", "sum"),
    )
    return {int(bid): f"{int(r.h)}H/{int(r.hr)}HR/{int(r.xbh)}XBH in {int(r.pa)} PA"
            for bid, r in g.iterrows()}


def _pass_hand_conf_gate(p: dict, target: str) -> bool:
    return (p.get(f"conf_{target}_label") or "") in SEC14_MIN_HAND_LABEL


def render_section(preds: list[dict], target: str, days_back: int = 14) -> str:
    # Prefer M1 Lock rows that also pass hand-tuned conf tier (Medium+).
    # Fall back to hand-tuned High when no dual-gate M1 picks exist for this target.
    meta_label_col = f"score_label_{target}"
    legacy_label_col = f"conf_{target}_label"
    has_meta = any(p.get(meta_label_col) for p in preds)
    m1_lock = [p for p in preds if p.get(meta_label_col) == "Lock"] if has_meta else []
    m1_dual = [p for p in m1_lock if _pass_hand_conf_gate(p, target)]
    legacy_picks = [p for p in preds if (p.get(legacy_label_col) or "") == "High"]
    if m1_dual:
        high_picks = m1_dual
        label_used = "M1 Lock + hand conf ≥ Medium"
    elif legacy_picks:
        high_picks = legacy_picks
        if has_meta and m1_lock:
            label_used = "hand-tuned conf (High); M1 Lock failed dual gate for this target"
        else:
            label_used = "hand-tuned conf (High; M1 yielded 0 Lock for this target)"
    else:
        high_picks = []
        label_used = (
            "M1 Lock exists but none pass hand ≥ Medium"
            if has_meta and m1_lock
            else ("M1 meta-model (Lock)" if has_meta else "hand-tuned conf (High)")
        )
    high_picks.sort(key=lambda p: -_score_col(p, target))
    high_picks = high_picks[:SEC14_MAX_PICKS_PER_TARGET]

    target_disp = {"hr": "HR", "hit": "Hit", "xbh": "XBH"}[target]
    lines = [f"### {target_disp} Locks (n={len(high_picks)}) — _{label_used}_", ""]
    if not high_picks:
        lines.append("_No Lock picks for this target tonight._")
        lines.append("")
        return "\n".join(lines)

    # Streak table for these batters
    streaks = _batter_streak_table(high_picks, days_back=days_back)

    cap = "P(HR)" if target == "hr" else ("P(Hit)" if target == "hit" else "P(XBH)")
    lines.append(f"| # | Matchup | Score | Cal {cap} | Raw {cap} | 3-PA | Conf Factors | BvP |")
    lines.append("|---:|---|---:|---:|---:|---:|---|---|")
    for i, p in enumerate(high_picks, 1):
        lines.append(_format_pick_row(i, p, target))

    # Streak block
    lines.append("")
    lines.append("**Trailing 14d streaks for these batters:**")
    for p in high_picks:
        bid = int(p.get("batter_mlbam_id") or 0)
        s = streaks.get(bid, "—")
        lines.append(f"- {p['batter_name']}: {s}")
    lines.append("")
    return "\n".join(lines)


def render_bucket_footer(drift_md_path: Path) -> str:
    lines = ["### Bucket Health Footer (rolling 14d)", ""]
    if not drift_md_path.exists():
        lines.append("_drift monitor not yet run; "
                     "execute `python3 src/monitor_calibration_drift.py` first._")
        lines.append("")
        return "\n".join(lines)
    drift = drift_md_path.read_text()
    # Pull just the table out of section 0
    table_lines = []
    in_table = False
    for ln in drift.splitlines():
        if ln.startswith("|"):
            in_table = True
            table_lines.append(ln)
        elif in_table and not ln.strip().startswith("|"):
            break
    if table_lines:
        lines.extend(table_lines)
    else:
        lines.append("_no drift table found in section_0_drift.md_")
    lines.append("")
    return "\n".join(lines)


def render(preds: list[dict], drift_md_path: Path, days_back: int) -> str:
    slate_date = preds[0].get("slate_date", "?") if preds else "?"
    n_total = len(preds)
    high_n = {t: sum(1 for p in preds if p.get(f"conf_{t}_label") == "High")
              for t in ("hr", "hit", "xbh")}

    lines = [
        "## Section 14: Conviction Picks (High-conf only)",
        "",
        f"**Slate:** {slate_date} | **Total picks scanned:** {n_total}",
        "",
        f"High-conf bucket sizes: HR={high_n['hr']}, Hit={high_n['hit']}, XBH={high_n['xbh']}",
        "",
        "_Filter rule: M1 **`Lock`** only if hand-tuned **`conf_*_label` is High or Medium**; "
        "else **`conf_*_label == \"High\"`** fallback. Caps **"
        + str(SEC14_MAX_PICKS_PER_TARGET)
        + "** picks per target. Sort: Score (raw P × confidence)._",
        "",
    ]
    for tgt in ("hr", "hit", "xbh"):
        lines.append(render_section(preds, tgt, days_back=days_back))
    lines.append(render_bucket_footer(drift_md_path))
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()}_")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 14: Conviction Picks digest.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--drift-md", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--days-back", type=int, default=14,
                        help="Rolling window for the per-batter streak block.")
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"Missing predictions JSON: {args.predictions}", file=sys.stderr)
        sys.exit(1)
    with open(args.predictions) as f:
        preds = json.load(f)

    md = render(preds, args.drift_md, args.days_back)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
