#!/usr/bin/env python3
"""
Per-conf-bucket isotonic recalibration of model probabilities (C1).

Trains a separate isotonic map per (target, calibration bucket) on rolling vs-SP
outcomes from data/tracking/matchup_predictions_runs.parquet. Bucket assignment
prefers training-time quartiles of numeric ``conf_*`` (stable vs relabeling
heuristics); if the history is too sparse, falls back to the stored
``conf_*_label`` tiers. Falls back to a multiplicative constant when n < 100 in
a bucket. Persists to data/priors/calibration_isotonic.json.

At inference time, src/narrative_engine.py calls resolve_calibration_bucket()
then apply_calibration() to populate the p_*_calibrated fields.

Usage:
  python3 src/calibrate_predictions.py             # refit using full history
  python3 src/calibrate_predictions.py --window-days 30  # rolling 30-day window
  python3 src/calibrate_predictions.py --dry-run   # report only, do not write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR  # noqa: E402  (we just need ROOT, this resolves the path)
from matchup_tracking import TRACKING_MAIN  # noqa: E402

PRIORS_DIR = ROOT / "data" / "priors"
PRIORS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_JSON = PRIORS_DIR / "calibration_isotonic.json"

BUCKET_ORDER = ["Very Low", "Low", "Medium", "High"]

# Numeric composite confidence column in tracking (per target) for quantile buckets.
CONF_VALUE_COL_BY_TARGET = {
    "hr": "conf_hr",
    "hit": "conf_hit",
    "xbh": "conf_xbh",
}
MIN_ROWS_QUANTILE_CALIB_BUCKETS = 40

# (target, raw_prob_col, conf_label_col, vs_sp_outcome_col, vs_sp_pa_col)
TARGETS = [
    ("hr", "p_hr", "conf_hr_label", "outcome_hr_vs_sp", "outcome_pa_vs_sp"),
    ("hit", "p_hit", "conf_hit_label", "outcome_h_vs_sp", "outcome_pa_vs_sp"),
    ("xbh", "p_xbh", "conf_xbh_label", "outcome_xbh_vs_sp", "outcome_pa_vs_sp"),
]

MIN_N_ISOTONIC = 100  # below this, use multiplicative constant fallback
MIN_N_MULT_CONST = 5   # below this, use identity (no calibration)


def _load_filled() -> pd.DataFrame:
    df = pd.read_parquet(TRACKING_MAIN)
    return df[df["outcome_filled_at"].notna()].copy()


def _filter_window(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if not days or "slate_date" not in df.columns:
        return df
    max_d = pd.to_datetime(df["slate_date"]).max()
    cutoff = max_d - pd.Timedelta(days=days)
    return df[pd.to_datetime(df["slate_date"]) >= cutoff].copy()


def quantile_calibration_bucket(conf_val: float, edges: list[float]) -> str:
    """Map numeric conf to quartile-aligned labels (same names as BUCKET_ORDER)."""
    if not edges or len(edges) != 3 or not np.isfinite(conf_val):
        return "Medium"
    q25, q50, q75 = float(edges[0]), float(edges[1]), float(edges[2])
    if conf_val <= q25:
        return "Very Low"
    if conf_val <= q50:
        return "Low"
    if conf_val <= q75:
        return "Medium"
    return "High"


def resolve_calibration_bucket(conf_numeric: float, target: str,
                               cal: dict | None = None) -> str:
    """Bucket for C1 isotonic: quantile cuts on conf_* from training if available, else hand tiers."""
    try:
        cn = float(conf_numeric)
    except (TypeError, ValueError):
        return "Medium"
    if not np.isfinite(cn):
        return "Medium"
    cal = cal if cal is not None else _load_calibration_cached()
    tgt = (target or "hit").lower()
    edges = None
    if cal is not None:
        edges = (cal.get("targets", {}).get(tgt, {}) or {}).get(
            "confidence_bin_edges_quantile")
    if edges and len(edges) == 3:
        return quantile_calibration_bucket(cn, edges)
    from narrative_engine import confidence_label_for_target  # noqa: WPS433

    return confidence_label_for_target(cn, tgt)


def _per_pa_outcome(g: pd.DataFrame, outcome_col: str, pa_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-PA realized rate from the (outcome_count, pa_count) pair.

    For each row, the per-PA empirical probability is outcome_count / pa_count.
    A row contributes that rate weighted by its pa_count via sample weighting.

    Returns (rates, weights) ready for IsotonicRegression.fit(X, y, sample_weight).
    """
    sub = g[[outcome_col, pa_col]].dropna()
    sub = sub[sub[pa_col] > 0]
    if sub.empty:
        return np.array([]), np.array([])
    rates = (sub[outcome_col].astype(float) / sub[pa_col].astype(float)).values
    weights = sub[pa_col].astype(float).values
    return rates, weights


def _fit_bucket(p_raw: np.ndarray, y_rate: np.ndarray, weights: np.ndarray,
                bucket_name: str) -> dict:
    """Fit isotonic per bucket; fall back to multiplicative constant when sparse."""
    n_pa = float(weights.sum())
    if len(p_raw) < MIN_N_MULT_CONST or n_pa < MIN_N_MULT_CONST:
        return {
            "method": "identity",
            "n_rows": int(len(p_raw)),
            "n_pa": n_pa,
            "knots_x": [0.0, 1.0],
            "knots_y": [0.0, 1.0],
            "constant": 1.0,
            "note": "too sparse to fit; identity transform applied",
        }

    mean_pred = float(np.average(p_raw, weights=weights))
    mean_real = float(np.average(y_rate, weights=weights))

    if len(p_raw) < MIN_N_ISOTONIC:
        # Multiplicative constant fallback
        c = mean_real / mean_pred if mean_pred > 0 else 1.0
        c = float(np.clip(c, 0.25, 4.0))  # guard against runaway scaling on tiny samples
        return {
            "method": "mult_constant",
            "n_rows": int(len(p_raw)),
            "n_pa": n_pa,
            "constant": c,
            "mean_pred": mean_pred,
            "mean_realized": mean_real,
            "knots_x": [0.0, 1.0],
            "knots_y": [0.0 * c, 1.0 * c],
        }

    # Isotonic — guard against degenerate y (all 0 or all 1)
    if np.unique(y_rate).size < 2:
        c = mean_real / mean_pred if mean_pred > 0 else 1.0
        c = float(np.clip(c, 0.25, 4.0))
        return {
            "method": "mult_constant",
            "n_rows": int(len(p_raw)),
            "n_pa": n_pa,
            "constant": c,
            "mean_pred": mean_pred,
            "mean_realized": mean_real,
            "knots_x": [0.0, 1.0],
            "knots_y": [0.0 * c, 1.0 * c],
            "note": "isotonic skipped — degenerate target distribution",
        }

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw, y_rate, sample_weight=weights)

    # Persist as a sparse set of knots for cheap runtime application.
    # X_min_ / X_max_ define domain; X_thresholds_ + y_thresholds_ define the steps.
    knots_x = iso.X_thresholds_.tolist() if hasattr(iso, "X_thresholds_") else \
              [float(iso.X_min_), float(iso.X_max_)]
    knots_y = iso.y_thresholds_.tolist() if hasattr(iso, "y_thresholds_") else \
              [float(iso.predict([iso.X_min_])[0]), float(iso.predict([iso.X_max_])[0])]

    return {
        "method": "isotonic",
        "n_rows": int(len(p_raw)),
        "n_pa": n_pa,
        "knots_x": [float(x) for x in knots_x],
        "knots_y": [float(y) for y in knots_y],
        "x_min": float(iso.X_min_),
        "x_max": float(iso.X_max_),
        "mean_pred": mean_pred,
        "mean_realized": mean_real,
    }


def fit_calibration(window_days: int | None = None) -> dict:
    df = _load_filled()
    if window_days:
        df = _filter_window(df, window_days)

    if df.empty:
        return {
            "_meta": {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_rows_total": 0,
                "window_days": window_days,
                "note": "no filled outcomes available",
            },
            "targets": {},
        }

    out: dict = {
        "_meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_rows_total": int(len(df)),
            "window_days": window_days,
            "tracking_file": str(TRACKING_MAIN),
            "min_n_isotonic": MIN_N_ISOTONIC,
            "min_n_mult_constant": MIN_N_MULT_CONST,
            "calibration_bucket_policy": (
                "Train: prefer quartile bins of numeric conf_* per target when n≥"
                f"{MIN_ROWS_QUANTILE_CALIB_BUCKETS}; else use stored conf_*_label. "
                "Inference: resolve_calibration_bucket prefers saved quantile edges."
            ),
            "slate_date_range": [
                str(pd.to_datetime(df["slate_date"]).min().date()),
                str(pd.to_datetime(df["slate_date"]).max().date()),
            ],
        },
        "targets": {},
    }

    for tgt, raw_col, label_col, outcome_col, pa_col in TARGETS:
        conf_nm = CONF_VALUE_COL_BY_TARGET[tgt]
        edges: list[float] | None = None
        if conf_nm in df.columns:
            ser = pd.to_numeric(df[conf_nm], errors="coerce").dropna()
            if len(ser) >= MIN_ROWS_QUANTILE_CALIB_BUCKETS:
                qs = np.quantile(ser.astype(float).values, [0.25, 0.5, 0.75])
                edges = [float(qs[0]), float(qs[1]), float(qs[2])]
                if not all(np.isfinite(x) for x in edges):
                    edges = None

        tgt_bundle: dict = {"buckets": {}, "confidence_bin_edges_quantile": edges}
        if edges is None:
            bucket_series = df[label_col].astype(str)
        else:
            cv = pd.to_numeric(df[conf_nm], errors="coerce")
            fb = df[label_col].astype(str)
            assign: list[str] = []
            for i in df.index:
                v = cv.loc[i]
                if pd.notna(v):
                    assign.append(quantile_calibration_bucket(float(v), edges))
                else:
                    lv = fb.loc[i]
                    assign.append(str(lv) if pd.notna(lv) and str(lv) not in {"", "nan"} else "Medium")
            bucket_series = pd.Series(assign, index=df.index, dtype=str)

        for bucket in BUCKET_ORDER:
            g = df[bucket_series == bucket]
            rates, weights = _per_pa_outcome(g, outcome_col, pa_col)
            p_raw = g.loc[g[pa_col].notna() & (g[pa_col] > 0), raw_col].astype(float).values
            assert len(p_raw) == len(rates), "alignment bug"
            cell = _fit_bucket(p_raw, rates, weights, bucket)
            tgt_bundle["buckets"][bucket] = cell
        out["targets"][tgt] = tgt_bundle

    return out


def _piecewise_linear(x_pts: list[float], y_pts: list[float], val: float) -> float:
    """Piecewise-linear interpolation matching IsotonicRegression's runtime behavior."""
    if not x_pts:
        return val
    if val <= x_pts[0]:
        return float(y_pts[0])
    if val >= x_pts[-1]:
        return float(y_pts[-1])
    # binary-search-ish; small list so linear scan is fine
    for i in range(1, len(x_pts)):
        if val <= x_pts[i]:
            x0, x1 = x_pts[i - 1], x_pts[i]
            y0, y1 = y_pts[i - 1], y_pts[i]
            if x1 == x0:
                return float(y0)
            t = (val - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(y_pts[-1])


def _bucket_cell(cal: dict, target: str, bucket: str) -> dict | None:
    return (cal.get("targets", {}).get(target, {}).get("buckets", {}) or {}).get(bucket)


def apply_calibration(p_raw: float, target: str, bucket: str,
                      cal: dict | None = None) -> float:
    """Runtime helper: map raw P → calibrated P for a given (target, bucket).

    Loaded calibration is cached. Pass `cal` explicitly for testability.
    """
    cal = cal if cal is not None else _load_calibration_cached()
    if cal is None:
        return float(p_raw)
    cell = _bucket_cell(cal, target, bucket)
    if cell is None:
        return float(p_raw)
    method = cell.get("method", "identity")
    if method == "identity":
        return float(p_raw)
    if method == "mult_constant":
        return float(np.clip(p_raw * cell["constant"], 0.0, 1.0))
    if method == "isotonic":
        return float(np.clip(_piecewise_linear(cell["knots_x"], cell["knots_y"], p_raw), 0.0, 1.0))
    return float(p_raw)


_CAL_CACHE: dict | None = None
_CAL_CACHE_MTIME: float | None = None


def _load_calibration_cached() -> dict | None:
    global _CAL_CACHE, _CAL_CACHE_MTIME
    if not CALIBRATION_JSON.exists():
        return None
    mtime = CALIBRATION_JSON.stat().st_mtime
    if _CAL_CACHE is not None and _CAL_CACHE_MTIME == mtime:
        return _CAL_CACHE
    with open(CALIBRATION_JSON) as f:
        _CAL_CACHE = json.load(f)
    _CAL_CACHE_MTIME = mtime
    return _CAL_CACHE


def reset_cache() -> None:
    """Force re-load of the calibration JSON on next apply_calibration call."""
    global _CAL_CACHE, _CAL_CACHE_MTIME
    _CAL_CACHE = None
    _CAL_CACHE_MTIME = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-conf-bucket isotonic recalibration.")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Rolling window in days (default: full history).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the calibration table without writing JSON.")
    args = parser.parse_args()

    cal = fit_calibration(window_days=args.window_days)
    meta = cal.get("_meta", {})
    print(f"Trained on {meta.get('n_rows_total', 0)} rows, "
          f"window_days={meta.get('window_days')}, "
          f"slate range={meta.get('slate_date_range')}")
    print()

    print(f"{'Target':<6}{'Bucket':<10}{'Method':<14}{'n_rows':>8}{'n_pa':>10}"
          f"{'mean_pred':>12}{'mean_real':>12}{'mult/note':>14}")
    print("-" * 86)
    for tgt in ("hr", "hit", "xbh"):
        for bucket in BUCKET_ORDER:
            cell = cal["targets"].get(tgt, {}).get("buckets", {}).get(bucket, {})
            if not cell:
                continue
            method = cell.get("method", "—")
            n_rows = cell.get("n_rows", 0)
            n_pa = cell.get("n_pa", 0)
            mp = cell.get("mean_pred", float("nan"))
            mr = cell.get("mean_realized", float("nan"))
            extra = ""
            if method == "mult_constant":
                extra = f"x{cell.get('constant', 1.0):.3f}"
            elif method == "isotonic":
                extra = f"{len(cell.get('knots_x', []))} knots"
            elif method == "identity":
                extra = "identity"
            print(f"{tgt:<6}{bucket:<10}{method:<14}{n_rows:>8}{n_pa:>10.0f}"
                  f"{mp:>12.4f}{mr:>12.4f}{extra:>14}")

    if args.dry_run:
        print("\n--dry-run: not writing JSON.")
        return

    CALIBRATION_JSON.write_text(json.dumps(cal, indent=2))
    print(f"\nWrote {CALIBRATION_JSON}")


if __name__ == "__main__":
    main()
