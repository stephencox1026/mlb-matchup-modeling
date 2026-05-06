"""
Render Section 9 markdown from todays_starter_run_expectancies.json.

Fixes float display artifacts (e.g. 2.5100000000000002), TBD / null predictions,
and confidence labels for the daily-output table.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
DEFAULT_JSON = REPORTS / "todays_starter_run_expectancies.json"


def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def _r2(x) -> float:
    return round(float(x), 2)


def format_starter_side(name: str | None, pred, lo, hi, conf) -> tuple[str, str]:
    """
    Returns (starter_cell, confidence_cell) for one side of the game.
    Rule: TBD / no prediction -> name shown as TBD, no numeric prediction.
    """
    name = (name or "TBD").strip() or "TBD"
    if _is_missing(pred) or _is_missing(lo) or _is_missing(hi):
        return name, "—"
    p, a, b = _r2(pred), _r2(lo), _r2(hi)
    # En dash between interval bounds (matches daily-output example 1.8–5.1)
    cell = f"{name}: {p:.2f} ({a:.2f}–{b:.2f})"
    c = conf if conf and str(conf).strip() else "—"
    return cell, str(c)


def render_section9_markdown(rows: list[dict]) -> str:
    lines = [
        "## Section 9: Starting Pitcher Run Predictions",
        "",
        "| Matchup | Away Starter (pred runs) | Confidence | Home Starter (pred runs) | Confidence |",
        "|---|---:|---|---:|---|",
    ]
    for row in rows:
        mu = row.get("matchup", "")
        away_n = row.get("away_pitcher")
        home_n = row.get("home_pitcher")
        away_cell, away_conf = format_starter_side(
            away_n,
            row.get("away_predicted_runs"),
            row.get("away_interval_low"),
            row.get("away_interval_high"),
            row.get("away_confidence"),
        )
        home_cell, home_conf = format_starter_side(
            home_n,
            row.get("home_predicted_runs"),
            row.get("home_interval_low"),
            row.get("home_interval_high"),
            row.get("home_confidence"),
        )
        lines.append(
            f"| {mu} | {away_cell} | {away_conf} | {home_cell} | {home_conf} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    with open(path) as f:
        rows = json.load(f)
    text = render_section9_markdown(rows)
    print(text, end="")


if __name__ == "__main__":
    main()
