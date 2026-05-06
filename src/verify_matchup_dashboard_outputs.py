#!/usr/bin/env python3
"""Sanity-check matchup dashboard JSON + HTML after a daily run."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
RAW = ROOT / "data" / "raw"


def _fail(msg: str) -> None:
    print("FAIL:", msg)
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, help="Expected slate_date YYYY-MM-DD prefix")
    args = p.parse_args()
    sd = args.date[:10]

    html = REPORTS / "matchup_dashboard.html"
    if not html.is_file():
        _fail(f"missing {html}")

    paths = {
        "prod": REPORTS / "todays_matchup_predictions.json",
        "exp": REPORTS / "todays_matchup_predictions_exp.json",
        "recency": REPORTS / "todays_matchup_predictions_recency.json",
        "beast": REPORTS / "todays_matchup_predictions_beast.json",
        "starter": REPORTS / "todays_starter_run_expectancies.json",
        "qualifying": RAW / "qualifying_batters_2026.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            _fail(f"missing {path} ({label})")

    bodies: dict[str, list] = {}
    for label in ("prod", "exp", "recency", "beast"):
        data = json.loads(paths[label].read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            _fail(f"{label} predictions empty or not a list")
        bodies[label] = data

    n0 = len(bodies["prod"])
    for label in ("exp", "recency", "beast"):
        if len(bodies[label]) != n0:
            _fail(f"row count mismatch {label}={len(bodies[label])} vs prod={n0}")

    for i, row in enumerate(bodies["prod"][:3]):
        rsd = str(row.get("slate_date") or "")[:10]
        if rsd != sd:
            _fail(f"prod row {i} slate_date {rsd!r} != expected {sd!r}")

    # Model bundles: park_* should be absent after no-park retrain (warn only).
    for rel in (
        "data/master/models/feature_columns.json",
        "data/master/models/exp_bpt_xwoba/feature_columns.json",
        "data/master/models/exp_recency_l3l5/feature_columns.json",
        "data/master/models/exp_beast/feature_columns.json",
    ):
        fp = ROOT / rel
        if not fp.is_file():
            print("WARN: missing", rel)
            continue
        cols = json.loads(fp.read_text(encoding="utf-8"))
        park = [c for c in cols if str(c).startswith("park_")]
        if park:
            print(f"WARN: {rel} still lists {len(park)} park_* columns (retrain may be incomplete)")

    text = html.read_text(encoding="utf-8")
    required_tabs = (
        'data-panel="main"',
        'data-panel="recency"',
        'data-panel="beast"',
        'data-panel="residual"',
        'data-panel="postgame"',
        'data-panel="bucket-health"',
        'data-panel="conviction"',
        'data-panel="no-hr"',
        'data-panel="long"',
        'data-panel="long-exp"',
    )
    for frag in required_tabs:
        if frag not in text:
            _fail(f"matchup_dashboard.html missing tab fragment {frag!r}")

    required_panels = (
        'id="panel-main"',
        'id="panel-recency"',
        'id="panel-beast"',
        'id="panel-residual"',
        'id="panel-postgame"',
        'id="panel-bucket-health"',
        'id="panel-conviction"',
        'id="panel-no-hr"',
        'id="panel-long"',
        'id="panel-long-exp"',
    )
    for frag in required_panels:
        if frag not in text:
            _fail(f"matchup_dashboard.html missing panel {frag!r}")

    if not re.search(r"Residual longshots", text):
        _fail("HTML missing Residual longshots heading/copy")

    if not re.search(r"Results vs actuals|postgame", text, re.I):
        _fail("HTML missing postgame / results section markers")

    arch = REPORTS / "archive" / sd / "matchup_dashboard.html"
    if arch.is_file():
        if arch.stat().st_mtime < html.stat().st_mtime - 1:
            print("WARN: archive HTML older than root HTML (mtime)")
    else:
        print("WARN: no archived copy at", arch)

    print("OK: dashboard artifacts verified")
    print(f"  slate_date: {sd}")
    print(f"  matchup rows (all four JSON): {n0}")
    print(f"  HTML: {html}")
    print(
        "  tabs: main, recency, beast, residual, postgame, "
        "bucket-health, conviction, no-hr, long, long-exp"
    )


if __name__ == "__main__":
    main()
