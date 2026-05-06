#!/usr/bin/env python3
"""
Fetch slate date, refresh qualifying batters CSV (YTD for residual longshots),
run prod + exp (+ recency + beast when present) matchup predictions, starter run expectancies,
emit Sections 1–6 / 7 / 8 / 9 / 10 artifacts (7–10 per model as separate markdown),
and refresh `docs/matchup_betting_board.html` (+ mirror under `data/reports/`).

Usage:
  PYTHONPATH=src python3 src/run_dual_model_daily.py --date 2026-04-21
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
RAW = ROOT / "data" / "raw"

sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    BEAST_MODEL_DIR,
    BEAST_VAL_FEATURES,
    DOCS_DIR,
    EXPERIMENT_MODEL_DIR,
    EXPERIMENT_VAL_FEATURES,
    MASTER_DIR,
    RAW_DIR,
    RECENCY_MODEL_DIR,
    RECENCY_VAL_FEATURES,
    RECENT365_MODEL_DIR,
    RECENT365_VAL_FEATURES,
    REPORTS_DIR,
)
from matchup_tracking import (  # noqa: E402
    append_dual_model_predictions,
    append_matchup_tracking_rows,
    materialize_high_conf_tracking,
)
from narrative_engine import predict_matchups, write_matchup_predictions_semantics_artifact  # noqa: E402
from render_section9 import render_section9_markdown  # noqa: E402
from starter_run_expectancy import (  # noqa: E402
    load_qualifying_pa_map,
    predict_starter_runs,
)


def _run(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True, env={**os.environ, **(env or {})})


def _emit(script: str, predictions: Path, out: Path) -> None:
    env = {**os.environ, "MATCHUP_PREDICTIONS_JSON": str(predictions)}
    txt = subprocess.check_output(
        [sys.executable, str(ROOT / "src" / script)],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    out.write_text(txt, encoding="utf-8")
    print(f"  Wrote → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, help="Slate date YYYY-MM-DD (matchups + archive)")
    p.add_argument("--skip-fetch", action="store_true", help="Do not refresh todays_matchups.json")
    p.add_argument(
        "--skip-statcast-append",
        action="store_true",
        help="Do not try to append previous-day Statcast to league parquet",
    )
    p.add_argument(
        "--skip-fetch-qualifying-batters",
        action="store_true",
        help="Do not refresh qualifying_batters_2026.csv (YTD for residual longshots tab + maps)",
    )
    p.add_argument(
        "--skip-dq",
        action="store_true",
        help="Skip the runtime data-quality pre-check (break-glass only)",
    )
    p.add_argument(
        "--skip-lineup-refresh",
        action="store_true",
        help="Do not refresh lineup_slots_by_game / batter_lineup_spot_last10_vs_hand parquets",
    )
    args = p.parse_args()
    slate = args.date

    if not args.skip_fetch:
        _run([sys.executable, str(ROOT / "src" / "fetch_todays_games.py"), "--date", slate])

    matchups_path = RAW_DIR / "todays_matchups.json"
    if not matchups_path.is_file():
        raise SystemExit(f"Missing {matchups_path}")

    # Data-quality pre-run gate. Block dummy/stale source data from leaking into the daily slate.
    if not args.skip_dq:
        from data_quality_check import run_pre_checks
        ok, _ = run_pre_checks(verbose=True)
        if not ok:
            raise SystemExit(
                "Data-quality pre-check FAILED. Refresh the flagged sources or pass "
                "--skip-dq for a manual override."
            )

    if not args.skip_statcast_append:
        from datetime import date, timedelta

        try:
            d0 = date.fromisoformat(slate) - timedelta(days=1)
            _run(
                [
                    sys.executable,
                    str(ROOT / "src" / "append_statcast_day_to_league_pa.py"),
                    "--date",
                    d0.isoformat(),
                ]
            )
        except subprocess.CalledProcessError as ex:
            print("WARN: Statcast append failed (offline or no rows):", ex)

    if not args.skip_lineup_refresh:
        lineup_cmd = [
            sys.executable,
            str(ROOT / "src" / "refresh_batter_lineup_context.py"),
            "--asof",
            slate,
        ]
        layer1 = RAW_DIR / "lineup_slots_by_game.parquet"
        if not layer1.is_file():
            lineup_cmd.extend(["--bootstrap-days", "21"])
        try:
            _run(lineup_cmd)
        except subprocess.CalledProcessError as ex:
            print("WARN: refresh_batter_lineup_context failed:", ex)

    if not args.skip_fetch_qualifying_batters:
        try:
            _run([sys.executable, str(ROOT / "src" / "fetch_qualifying_batters.py")])
        except subprocess.CalledProcessError as ex:
            print("WARN: qualifying batters fetch failed (offline or API):", ex)

    # T1.1 (HR audit): production model is now BEAST (135 features w/ park + xptw + month).
    # Beast wins H2 on every primary target (HR Brier 0.02937 vs prod 0.02944, AUC 0.7200 vs 0.7006);
    # M1 conf meta-model thresholds were calibrated on beast's distribution. The legacy prod model
    # lives on as a secondary stream for back-compat with the dual-model dashboard tab.
    print("Predicting (production = beast)...")
    prod_groups = predict_matchups(
        matchups_json_path=matchups_path,
        model_dir=BEAST_MODEL_DIR,
        val_features_path=BEAST_VAL_FEATURES,
        model_source="prod",  # field name kept for tracking back-compat
    )
    prod_flat = [m for g in prod_groups for m in g["matchups"]]

    print("Predicting (experiment)...")
    exp_groups = predict_matchups(
        matchups_json_path=matchups_path,
        model_dir=EXPERIMENT_MODEL_DIR,
        val_features_path=EXPERIMENT_VAL_FEATURES,
        model_source="exp",
    )
    exp_flat = [m for g in exp_groups for m in g["matchups"]]

    # Legacy prod (82-feature) — kept for the dual-model "Matchups" tab so users can
    # compare the new production beast vs the old prod baseline. Skip if it would be redundant.
    legacy_prod_default_dir = MASTER_DIR / "models"
    print("Predicting (legacy prod baseline)...")
    legacy_prod_groups = predict_matchups(
        matchups_json_path=matchups_path,
        model_dir=legacy_prod_default_dir,
        model_source="prod_legacy",
    )
    legacy_prod_flat = [m for g in legacy_prod_groups for m in g["matchups"]]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_matchup_predictions_semantics_artifact(REPORTS_DIR)

    prod_json = REPORTS_DIR / "todays_matchup_predictions.json"  # now beast
    exp_json = REPORTS_DIR / "todays_matchup_predictions_exp.json"
    legacy_prod_json = REPORTS_DIR / "todays_matchup_predictions_prod_legacy.json"
    with open(prod_json, "w") as f:
        json.dump(prod_flat, f, indent=2, default=str)
    with open(exp_json, "w") as f:
        json.dump(exp_flat, f, indent=2, default=str)
    with open(legacy_prod_json, "w") as f:
        json.dump(legacy_prod_flat, f, indent=2, default=str)
    print(f"  {len(prod_flat)} matchups → {prod_json} (beast)")
    print(f"  {len(exp_flat)} matchups → {exp_json}")
    print(f"  {len(legacy_prod_flat)} matchups → {legacy_prod_json} (legacy prod)")

    if not args.skip_dq:
        from data_quality_check import run_post_checks
        ok_post, _summary = run_post_checks(prod_json, slate_date=slate, verbose=True)
        if not ok_post:
            print(
                "WARN: data-quality POST check flagged pure-default rows. See "
                "data/reports/data_quality_report.md before trusting the slate."
            )

    run_ts = datetime.now(timezone.utc).isoformat()
    n_trk = append_matchup_tracking_rows(prod_flat, run_ts)
    materialize_high_conf_tracking()
    print(f"  Tracking Parquet: {n_trk} row(s) appended → data/tracking/matchup_predictions_runs.parquet")
    dual_ok = (
        EXPERIMENT_MODEL_DIR.is_dir()
        and (EXPERIMENT_MODEL_DIR / "feature_columns.json").exists()
        and EXPERIMENT_VAL_FEATURES.is_file()
    )
    if dual_ok:
        n_dual = append_dual_model_predictions(
            prod_flat,
            exp_flat,
            run_ts,
            prod_model_dir=str(MASTER_DIR / "models"),
            exp_model_dir=str(EXPERIMENT_MODEL_DIR),
            prod_val_path=str(MASTER_DIR / "features_val_league.parquet"),
            exp_val_path=str(EXPERIMENT_VAL_FEATURES),
        )
        print(f"  Dual-model Parquet: {n_dual} row(s) → data/tracking/matchup_dual_model_predictions.parquet")
    else:
        print("  WARN: Skipped dual-model tracking Parquet (experiment paths missing)")

    # recent365 replaces recency (data-window diversity, same arch as beast).
    # H2 backtest shows recent365 wins Hit decisively over recency (Brier -0.0005,
    # AUC +0.0035) and is equivalent on HR/XBH overall.
    rec_ok = (
        RECENT365_MODEL_DIR.is_dir()
        and (RECENT365_MODEL_DIR / "feature_columns.json").exists()
        and RECENT365_VAL_FEATURES.is_file()
    )
    if rec_ok:
        print("Predicting (recent365 bundle)...")
        rec_groups = predict_matchups(
            matchups_json_path=matchups_path,
            model_dir=RECENT365_MODEL_DIR,
            val_features_path=RECENT365_VAL_FEATURES,
            model_source="recent365",
        )
        rec_flat = [m for g in rec_groups for m in g["matchups"]]
        # Write to the recency-named JSON for back-compat with the dashboard tab
        # which still says "Matchups · recency" on its label. Future cleanup: rename
        # the dashboard tab to "Matchups · recent365" once everyone's used to it.
        rec_json = REPORTS_DIR / "todays_matchup_predictions_recency.json"
        with open(rec_json, "w") as f:
            json.dump(rec_flat, f, indent=2, default=str)
        print(f"  {len(rec_flat)} matchups → {rec_json} (using recent365 model)")
    elif (
        RECENCY_MODEL_DIR.is_dir()
        and (RECENCY_MODEL_DIR / "feature_columns.json").exists()
        and RECENCY_VAL_FEATURES.is_file()
    ):
        # Fallback: legacy recency model still exists and is usable.
        print("Predicting (legacy recency L3/L5 bundle — recent365 not available)...")
        rec_groups = predict_matchups(
            matchups_json_path=matchups_path,
            model_dir=RECENCY_MODEL_DIR,
            val_features_path=RECENCY_VAL_FEATURES,
            model_source="recency",
        )
        rec_flat = [m for g in rec_groups for m in g["matchups"]]
        rec_json = REPORTS_DIR / "todays_matchup_predictions_recency.json"
        with open(rec_json, "w") as f:
            json.dump(rec_flat, f, indent=2, default=str)
        print(f"  {len(rec_flat)} matchups → {rec_json}")
    else:
        print("  WARN: Skipped recent365/recency model (missing bundle at "
              "data/master/models/recent365/ or data/master/models/exp_recency_l3l5/)")

    beast_ok = (
        BEAST_MODEL_DIR.is_dir()
        and (BEAST_MODEL_DIR / "feature_columns.json").exists()
        and BEAST_VAL_FEATURES.is_file()
    )
    if beast_ok:
        print("Predicting (beast)...")
        beast_groups = predict_matchups(
            matchups_json_path=matchups_path,
            model_dir=BEAST_MODEL_DIR,
            val_features_path=BEAST_VAL_FEATURES,
            model_source="beast",
        )
        beast_flat = [m for g in beast_groups for m in g["matchups"]]
        beast_json = REPORTS_DIR / "todays_matchup_predictions_beast.json"
        with open(beast_json, "w") as f:
            json.dump(beast_flat, f, indent=2, default=str)
        print(f"  {len(beast_flat)} matchups → {beast_json}")
    else:
        print("  WARN: Skipped beast model (missing bundle at data/master/models/exp_beast/ or val parquet)")

    pa_map = load_qualifying_pa_map()
    starter_rows = predict_starter_runs(prod_groups, pa_map)
    starter_path = REPORTS_DIR / "todays_starter_run_expectancies.json"
    with open(starter_path, "w") as f:
        json.dump(starter_rows, f, indent=2, default=str)
    print(f"  Starter runs → {starter_path}")

    section9_path = REPORTS_DIR / "section_9.md"
    section9_path.write_text(render_section9_markdown(starter_rows), encoding="utf-8")
    print(f"  Section 9 → {section9_path}")

    # Sections 1–6 (prod = canonical filename for legacy tools)
    _emit("gen_sections_1_6.py", prod_json, REPORTS_DIR / "sections_1_6.md")
    _emit("gen_sections_1_6.py", prod_json, REPORTS_DIR / "sections_1_6_prod.md")
    _emit("gen_sections_1_6.py", exp_json, REPORTS_DIR / "sections_1_6_exp.md")

    # Sections 7, 8, 10 — per model markdown
    for tag, jpath in (("prod", prod_json), ("exp", exp_json)):
        _emit("gen_section7.py", jpath, REPORTS_DIR / f"section_7_{tag}.md")
        _emit("gen_section8.py", jpath, REPORTS_DIR / f"section_8_{tag}.md")
        _emit("gen_section10.py", jpath, REPORTS_DIR / f"section_10_{tag}.md")

    # Legacy single filenames → production copy
    shutil.copyfile(REPORTS_DIR / "section_7_prod.md", REPORTS_DIR / "section_7.md")
    shutil.copyfile(REPORTS_DIR / "section_8_prod.md", REPORTS_DIR / "section_8.md")
    shutil.copyfile(REPORTS_DIR / "section_10_prod.md", REPORTS_DIR / "section_10.md")
    print("  Legacy section_7/8/10.md ← prod copies")

    # Section 0: drift monitor + per-bucket health (auto-refits isotonic when drift > 5%)
    try:
        _run([sys.executable, str(ROOT / "src" / "monitor_calibration_drift.py")])
        print(f"  Section 0 drift → {REPORTS_DIR / 'section_0_drift.md'}")
    except subprocess.CalledProcessError as ex:
        print("WARN: drift monitor failed:", ex)

    # Section 14: Conviction Picks digest (M1 Lock labels preferred over hand-tuned High)
    try:
        _run([sys.executable, str(ROOT / "src" / "gen_conviction_picks.py")])
        print(f"  Section 14 conviction → {REPORTS_DIR / 'section_14_conviction_picks.md'}")
    except subprocess.CalledProcessError as ex:
        print("WARN: conviction picks failed:", ex)

    # Section 15: No HR Model (P(0 HR) per game; NB-calibrated zero with park-aware base)
    try:
        _run([sys.executable, str(ROOT / "src" / "gen_zero_hr_predictions.py")])
        print(f"  Section 15 no-HR → {REPORTS_DIR / 'section_15_zero_hr.md'}")
    except subprocess.CalledProcessError as ex:
        print("WARN: no-HR model failed:", ex)

    # Refit per-conf-bucket calibration on the latest outcomes window (will pick up
    # any prior-day Statcast appended above). Defensive: failures are non-blocking.
    try:
        _run([sys.executable, str(ROOT / "src" / "calibrate_predictions.py")])
        print("  Per-bucket calibration refreshed")
    except subprocess.CalledProcessError as ex:
        print("WARN: calibrate_predictions failed:", ex)

    _run(
        [
            sys.executable,
            str(ROOT / "src" / "gen_matchup_dashboard_html.py"),
            "--date",
            slate,
        ]
    )

    try:
        _run([sys.executable, str(ROOT / "src" / "gen_matchup_betting_board_html.py")])
        brd_docs = DOCS_DIR / "matchup_betting_board.html"
        brd_reports = REPORTS_DIR / "matchup_betting_board.html"
        if brd_docs.is_file():
            shutil.copy2(brd_docs, brd_reports)
            print(f"  Matchup betting board → {brd_docs} (+ mirror {brd_reports})")
    except subprocess.CalledProcessError as ex:
        print("WARN: gen_matchup_betting_board_html failed:", ex)

    try:
        _run(
            [
                sys.executable,
                str(ROOT / "src" / "gen_sections_11_13.py"),
                "--recent-days",
                "7",
            ]
        )
    except subprocess.CalledProcessError as ex:
        print("WARN: gen_sections_11_13 failed:", ex)

    arch = REPORTS_DIR / "archive" / slate
    arch.mkdir(parents=True, exist_ok=True)
    for name in (
        "todays_matchup_predictions.json",
        "todays_matchup_predictions_exp.json",
        "todays_matchup_predictions_recency.json",
        "todays_matchup_predictions_beast.json",
        "todays_starter_run_expectancies.json",
        "section_0_drift.md",
        "sections_1_6.md",
        "sections_1_6_prod.md",
        "sections_1_6_exp.md",
        "section_7.md",
        "section_7_prod.md",
        "section_7_exp.md",
        "section_8.md",
        "section_8_prod.md",
        "section_8_exp.md",
        "section_9.md",
        "section_10.md",
        "section_10_prod.md",
        "section_10_exp.md",
        "matchup_dashboard.html",
        "matchup_betting_board.html",
        "section_11.md",
        "section_12.md",
        "section_13.md",
        "sections_11_13.md",
        "section_14_conviction_picks.md",
        "section_15_zero_hr.md",
        "todays_zero_hr_predictions.json",
        "conf_bucket_calibration.md",
    ):
        src = REPORTS_DIR / name
        if src.is_file():
            shutil.copy2(src, arch / name)
    shutil.copy2(RAW / "todays_matchups.json", arch / "todays_matchups.json")
    for extra_raw in ("lineup_slots_by_game.parquet", "batter_lineup_spot_last10_vs_hand.parquet"):
        src_e = RAW / extra_raw
        if src_e.is_file():
            shutil.copy2(src_e, arch / extra_raw)
    print(f"  Archived → {arch}/")

    try:
        _run(
            [
                sys.executable,
                str(ROOT / "src" / "verify_matchup_dashboard_outputs.py"),
                "--date",
                slate,
            ]
        )
    except subprocess.CalledProcessError as ex:
        print("WARN: verify_matchup_dashboard_outputs failed:", ex)


if __name__ == "__main__":
    main()
