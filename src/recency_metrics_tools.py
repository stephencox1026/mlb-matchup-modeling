#!/usr/bin/env python3
"""
Recency model evaluation helpers: compare val metrics CSVs, run baseline trains, grid half-life.

Examples:
  python3 src/recency_metrics_tools.py compare \\
    --baseline data/reports/multi_target_metrics.csv \\
    --label recency_noweight=data/reports/recency_ablation_unweighted.csv \\
    --label recency_w120=data/reports/recency_ablation_weighted_hl120.csv

  python3 src/recency_metrics_tools.py train-baseline

  python3 src/recency_metrics_tools.py tune-half-life --halves 60,120,180
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MASTER_DIR, REPORTS_DIR, PROJECT_ROOT

TRAIN_REC = MASTER_DIR / "features_train_league_recency.parquet"
VAL_REC = MASTER_DIR / "features_val_league_recency.parquet"
TRAIN_PROD = MASTER_DIR / "features_train_league.parquet"
VAL_PROD = MASTER_DIR / "features_val_league.parquet"
PROD_METRICS = REPORTS_DIR / "multi_target_metrics.csv"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_multi_target.py"

FOCUS_TARGETS = ("is_hit", "is_hr", "is_xbh")


def _calibrated_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["model"].str.endswith("_calibrated", na=False)].copy()
    out["target"] = out["model"].str.replace("_calibrated", "", regex=False)
    return out[out["target"].isin(FOCUS_TARGETS)]


def cmd_compare(args: argparse.Namespace) -> None:
    base_path = Path(args.baseline)
    if not base_path.is_file():
        print(f"ERROR: baseline metrics not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    base = _calibrated_rows(pd.read_csv(base_path)).set_index("target")[["brier", "roc_auc", "calibration_gap"]]

    rows = []
    for label, path_str in args.label:
        p = Path(path_str)
        if not p.is_file():
            print(f"WARN: skip {label} (missing {p})", file=sys.stderr)
            continue
        o = _calibrated_rows(pd.read_csv(p)).set_index("target")[["brier", "roc_auc", "calibration_gap"]]
        for tgt in FOCUS_TARGETS:
            if tgt not in base.index or tgt not in o.index:
                continue
            br_b = float(base.loc[tgt, "brier"])
            br_o = float(o.loc[tgt, "brier"])
            rows.append(
                {
                    "target": tgt,
                    "label": label,
                    "brier": br_o,
                    "delta_brier_vs_baseline": round(br_o - br_b, 6),
                    "roc_auc": float(o.loc[tgt, "roc_auc"]),
                    "calibration_gap": float(o.loc[tgt, "calibration_gap"]),
                }
            )

    if not rows:
        print("No rows to compare. Train recency models and pass --label paths.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(rows).sort_values(["target", "delta_brier_vs_baseline"])
    print(out_df.to_string(index=False))
    print("\nNegative delta_brier_vs_baseline = better than baseline.")


def _run_train(
    *,
    model_dir: Path,
    metrics_csv: Path,
    val_pred: Path,
    recency_weights: bool,
    half_life: float,
    train_path: Path,
    val_path: Path,
) -> int:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-path",
        str(train_path),
        "--val-path",
        str(val_path),
        "--model-dir",
        str(model_dir),
        "--metrics-csv",
        str(metrics_csv),
        "--val-pred-parquet",
        str(val_pred),
    ]
    if recency_weights:
        cmd += ["--recency-sample-weights", "--recency-weight-half-life-days", str(half_life)]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def cmd_train_baseline(args: argparse.Namespace) -> None:
    if not TRAIN_REC.is_file() or not VAL_REC.is_file():
        print(
            "ERROR: Recency feature parquet not found. Run:\n"
            "  python3 src/features.py --league --recency",
            file=sys.stderr,
        )
        sys.exit(1)

    u_dir = Path(args.unweighted_dir)
    w_dir = Path(args.weighted_dir)
    u_csv = Path(args.unweighted_metrics)
    w_csv = Path(args.weighted_metrics)

    rc = _run_train(
        model_dir=u_dir,
        metrics_csv=u_csv,
        val_pred=REPORTS_DIR / "val_predictions_recency_ablation_unweighted.parquet",
        recency_weights=False,
        half_life=120.0,
        train_path=TRAIN_REC,
        val_path=VAL_REC,
    )
    if rc != 0:
        sys.exit(rc)
    rc = _run_train(
        model_dir=w_dir,
        metrics_csv=w_csv,
        val_pred=REPORTS_DIR / "val_predictions_recency_ablation_weighted.parquet",
        recency_weights=True,
        half_life=float(args.half_life),
        train_path=TRAIN_REC,
        val_path=VAL_REC,
    )
    if rc != 0:
        sys.exit(rc)

    print("\n--- Compare to production (same 2025 val slice; prod trained on prod features) ---")
    print("NOTE: Production metrics below use features_val_league — compare Brier only as rough sanity;")
    print("      for strict apples-to-apples, re-train prod on recency feature matrix or compare two recency runs.\n")
    if PROD_METRICS.is_file():
        prod = _calibrated_rows(pd.read_csv(PROD_METRICS)).set_index("target")
        for tgt in FOCUS_TARGETS:
            if tgt not in prod.index:
                continue
            pb = float(prod.loc[tgt, "brier"])
            print(f"  prod {tgt}_calibrated Brier={pb:.5f}")

    cmd_compare(
        argparse.Namespace(
            baseline=str(PROD_METRICS) if PROD_METRICS.is_file() else str(u_csv),
            label=[
                ("recency_unweighted", str(u_csv)),
                ("recency_weighted", str(w_csv)),
            ],
        )
    )


def cmd_tune_half_life(args: argparse.Namespace) -> None:
    if not TRAIN_REC.is_file():
        print("ERROR: Missing", TRAIN_REC, file=sys.stderr)
        sys.exit(1)
    halves = [float(x.strip()) for x in args.halves.split(",") if x.strip()]
    results: list[tuple[float, str, float]] = []

    for hl in halves:
        mdir = MASTER_DIR / "models" / f"recency_tune_hl{int(hl)}"
        mcsv = REPORTS_DIR / f"recency_tune_hl{int(hl)}.csv"
        vp = REPORTS_DIR / f"val_predictions_recency_tune_hl{int(hl)}.parquet"
        rc = _run_train(
            model_dir=mdir,
            metrics_csv=mcsv,
            val_pred=vp,
            recency_weights=True,
            half_life=hl,
            train_path=TRAIN_REC,
            val_path=VAL_REC,
        )
        if rc != 0:
            print(f"WARN: train failed for hl={hl}", file=sys.stderr)
            continue
        df = pd.read_csv(mcsv)
        cal = _calibrated_rows(df)
        for _, r in cal.iterrows():
            tgt = str(r["model"]).replace("_calibrated", "")
            if tgt in FOCUS_TARGETS:
                results.append((hl, tgt, float(r["brier"])))

    if not results:
        sys.exit(1)
    res_df = pd.DataFrame(results, columns=["half_life_days", "target", "brier"])
    print(res_df.sort_values(["target", "brier"]).to_string(index=False))
    print("\nPer target, smallest Brier wins (among successful runs).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cmp = sub.add_parser("compare", help="Compare calibrated Brier for hit/hr/xbh across metrics CSVs")
    p_cmp.add_argument("--baseline", type=str, default=str(PROD_METRICS))
    p_cmp.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable: model label and metrics CSV path",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_bl = sub.add_parser("train-baseline", help="Train unweighted + weighted recency bundles and print compare")
    p_bl.add_argument("--unweighted-dir", type=str, default=str(MASTER_DIR / "models" / "exp_recency_ablation_unweighted"))
    p_bl.add_argument("--weighted-dir", type=str, default=str(MASTER_DIR / "models" / "exp_recency_ablation_weighted_hl120"))
    p_bl.add_argument(
        "--unweighted-metrics",
        type=str,
        default=str(REPORTS_DIR / "recency_ablation_unweighted.csv"),
    )
    p_bl.add_argument(
        "--weighted-metrics",
        type=str,
        default=str(REPORTS_DIR / "recency_ablation_weighted_hl120.csv"),
    )
    p_bl.add_argument("--half-life", type=float, default=120.0)
    p_bl.set_defaults(func=cmd_train_baseline)

    p_tune = sub.add_parser("tune-half-life", help="Train recency bundle at each half-life; report Brier grid")
    p_tune.add_argument("--halves", type=str, default="60,120,180")
    p_tune.set_defaults(func=cmd_tune_half_life)

    args = ap.parse_args()
    if args.cmd == "compare":
        pairs = []
        for raw in args.label:
            if "=" not in raw:
                print(f"ERROR: bad --label {raw!r}, expected NAME=PATH", file=sys.stderr)
                sys.exit(1)
            a, b = raw.split("=", 1)
            pairs.append((a.strip(), b.strip()))
        args.label = pairs
    args.func(args)


if __name__ == "__main__":
    main()
