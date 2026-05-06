"""
Train a HistGradientBoostingRegressor to predict actual runs allowed by a starter.

Train: 2015-2023 (~40K outings)
Val:   2024 (~4.8K) — used for metrics + conformal interval calibration
Test:  2025 (~4.8K) — held out for final evaluation

Outputs:
  data/master/models/starter_runs_model.pkl
  data/master/models/starter_runs_conformal.json   (quantiles for intervals)
  data/master/models/starter_runs_feature_cols.json
  data/reports/starter_runs_metrics.csv
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MASTER_DIR, REPORTS_DIR

MODEL_DIR = MASTER_DIR / "models"

ID_COLS = [
    "game_pk", "game_date", "game_year", "pitcher", "p_throws",
    "fielding_side", "n_pa_faced",
]
TARGET = "actual_runs"


def load_splits():
    train = pd.read_parquet(MASTER_DIR / "starter_runs_train.parquet")
    val = pd.read_parquet(MASTER_DIR / "starter_runs_val.parquet")
    test = pd.read_parquet(MASTER_DIR / "starter_runs_test.parquet")
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = set(ID_COLS) | {TARGET}
    return sorted(c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32"))


def train_model(X_train, y_train):
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=500,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    preds = np.clip(preds, 0, None)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mean_pred = preds.mean()
    mean_actual = y.mean()
    cal_gap = abs(mean_pred - mean_actual)

    n_bins = 10
    bins = pd.qcut(preds, n_bins, duplicates="drop")
    cal_df = pd.DataFrame({"pred": preds, "actual": y, "bin": bins})
    bin_stats = cal_df.groupby("bin", observed=True).agg(
        mean_pred=("pred", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count"),
    )
    max_bin_gap = (bin_stats["mean_pred"] - bin_stats["mean_actual"]).abs().max()

    print(f"\n  {label} Evaluation:")
    print(f"    MAE:            {mae:.3f}")
    print(f"    RMSE:           {rmse:.3f}")
    print(f"    Mean predicted: {mean_pred:.3f}")
    print(f"    Mean actual:    {mean_actual:.3f}")
    print(f"    Calibration gap:{cal_gap:.3f}")
    print(f"    Max bin gap:    {max_bin_gap:.3f}")
    print(f"\n    Calibration by decile:")
    for _, row in bin_stats.iterrows():
        print(f"      pred={row['mean_pred']:.2f}  actual={row['mean_actual']:.2f}  n={int(row['count'])}")

    return {
        "split": label,
        "mae": mae,
        "rmse": rmse,
        "mean_pred": mean_pred,
        "mean_actual": mean_actual,
        "cal_gap": cal_gap,
        "max_bin_gap": max_bin_gap,
        "n": len(y),
    }, preds


def compute_conformal_quantiles(y_true, preds, coverages=(0.50, 0.80, 0.90, 0.95)):
    """Compute conformal residual quantiles for prediction intervals."""
    residuals = y_true - preds
    result = {}
    for cov in coverages:
        alpha = 1 - cov
        q_lo = np.percentile(residuals, 100 * alpha / 2)
        q_hi = np.percentile(residuals, 100 * (1 - alpha / 2))
        result[f"{int(cov*100)}"] = {"q_lo": round(float(q_lo), 4), "q_hi": round(float(q_hi), 4)}
        actual_coverage = np.mean((preds + q_lo <= y_true) & (y_true <= preds + q_hi))
        print(f"    {int(cov*100)}% interval: [{q_lo:+.2f}, {q_hi:+.2f}]  "
              f"width={q_hi - q_lo:.2f}  coverage={actual_coverage:.3f}")
    return result


def main():
    print("=" * 60)
    print("  TRAIN STARTER RUNS-ALLOWED MODEL")
    print("=" * 60)

    train, val, test = load_splits()
    feat_cols = get_feature_cols(train)
    print(f"\n  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"  Features: {len(feat_cols)}")

    X_train = train[feat_cols].values
    y_train = train[TARGET].values
    X_val = val[feat_cols].values
    y_val = val[TARGET].values
    X_test = test[feat_cols].values
    y_test = test[TARGET].values

    print("\n  Training HistGradientBoostingRegressor (Poisson loss)...")
    model = train_model(X_train, y_train)

    hgb = model.named_steps["hgb"]
    if hasattr(hgb, "n_iter_"):
        print(f"  Converged at {hgb.n_iter_} iterations")

    train_metrics, train_preds = evaluate(model, X_train, y_train, "Train")
    val_metrics, val_preds = evaluate(model, X_val, y_val, "Val (2024)")
    test_metrics, test_preds = evaluate(model, X_test, y_test, "Test (2025)")

    print("\n  Computing conformal prediction intervals on Val set...")
    val_preds_clipped = np.clip(val_preds, 0, None)
    conformal = compute_conformal_quantiles(y_val, val_preds_clipped)

    print("\n  Verifying interval coverage on Test set...")
    test_preds_clipped = np.clip(test_preds, 0, None)
    for cov_key, qs in conformal.items():
        lo = test_preds_clipped + qs["q_lo"]
        hi = test_preds_clipped + qs["q_hi"]
        coverage = np.mean((lo <= y_test) & (y_test <= hi))
        width = (hi - lo).mean()
        print(f"    {cov_key}% target → test coverage={coverage:.3f}  mean_width={width:.2f}")

    # Feature importance (top 20)
    try:
        importances = hgb.feature_importances_
    except AttributeError:
        from sklearn.inspection import permutation_importance
        print("\n  Computing permutation importance (no native importances)...")
        perm = permutation_importance(model, X_val, y_val, n_repeats=5,
                                      scoring="neg_mean_absolute_error", random_state=42)
        importances = perm.importances_mean
    imp_idx = np.argsort(importances)[::-1][:20]
    print("\n  Top 20 features by importance:")
    for rank, idx in enumerate(imp_idx, 1):
        print(f"    {rank:2d}. {feat_cols[idx]:40s}  {importances[idx]:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "starter_runs_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model → {model_path}")

    conformal_path = MODEL_DIR / "starter_runs_conformal.json"
    with open(conformal_path, "w") as f:
        json.dump(conformal, f, indent=2)
    print(f"  Conformal quantiles → {conformal_path}")

    feat_path = MODEL_DIR / "starter_runs_feature_cols.json"
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print(f"  Feature columns → {feat_path}")

    metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    metrics_path = REPORTS_DIR / "starter_runs_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics → {metrics_path}")

    print(f"\n{'=' * 60}")
    print(f"  STARTER RUNS MODEL COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
