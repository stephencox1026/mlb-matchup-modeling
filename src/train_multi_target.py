"""
Multi-target model training with pitcher features.

Trains separate calibrated models for:
  - is_hit, is_hr, is_strikeout, is_walk, is_xbh

Uses the enhanced 79-feature set including pitcher profiles, BvP matchups,
and batter pitch-type performance.

Output: data/master/models/best_model_{target}.pkl
"""
import argparse
import json
import pickle
import warnings
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from config import MASTER_DIR, REPORTS_DIR, FIGURES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

TRAIN_PATH = MASTER_DIR / "features_train_league.parquet"
VAL_PATH = MASTER_DIR / "features_val_league.parquet"
MODEL_DIR = MASTER_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["is_hit", "is_hr", "is_strikeout", "is_walk", "is_xbh"]

FEATURE_PREFIXES = (
    "roll_", "cum_", "vs_lhp", "month", "day_of_week", "days_into_season",
    "pitch_count", "in_zone",
    "platoon_",
    "p_pct_", "p_velo_", "p_spin_", "p_pfx_", "p_k_rate", "p_bb_rate",
    "p_barrel_", "p_whiff_", "p_arm_", "p_extension", "p_pct_in_zone",
    "p_roll_", "p_g_roll_", "p_velo_overall",
    "g_roll_",
    "bpt_", "bvp_", "log_bvp_",
    "xptw_", "park_", "park_x_", "p_hr_",  # T2.4 + T2.2
    "times_thru", "pitcher_rest", "pitcher_age",
)


def get_feature_cols(df):
    return sorted([c for c in df.columns if c.startswith(FEATURE_PREFIXES)])


def build_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42,
        )),
    ])


def evaluate(name, model, X_val, y_val):
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "model": name,
        "brier": round(brier_score_loss(y_val, y_prob), 5),
        "roc_auc": round(roc_auc_score(y_val, y_prob), 4),
        "avg_precision": round(average_precision_score(y_val, y_prob), 4),
        "calibration_gap": round(abs(y_prob.mean() - y_val.mean()), 4),
        "mean_pred": round(y_prob.mean(), 4),
        "actual_rate": round(y_val.mean(), 4),
    }


def _recency_sample_weights(train_df: pd.DataFrame, half_life_days: float) -> np.ndarray | None:
    """Exponential decay: more weight to rows closer to the training-set max date."""
    if "game_date" not in train_df.columns:
        return None
    gd = pd.to_datetime(train_df["game_date"], errors="coerce")
    max_d = gd.max()
    if pd.isna(max_d):
        return None
    days = (max_d - gd).dt.days.clip(lower=0).astype(float)
    w = np.exp(-days / float(half_life_days))
    w = np.where(np.isfinite(w), w, 1.0)
    return w


def train_single_target(
    target,
    train_df,
    val_df,
    feat_cols,
    model_dir: Path,
    *,
    sample_weight: np.ndarray | None = None,
):
    print(f"\n{'='*50}")
    print(f"  TARGET: {target}")
    print(f"{'='*50}")

    if target not in train_df.columns:
        print(f"  SKIP: {target} not in data")
        return None

    X_train = train_df[feat_cols]
    y_train = train_df[target]
    X_val = val_df[feat_cols]
    y_val = val_df[target]

    valid_mask = y_train.notna()
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask].astype(int)
    sw_train = None
    if sample_weight is not None:
        sw_train = np.asarray(sample_weight, dtype=float)[valid_mask.to_numpy()]

    valid_mask_v = y_val.notna()
    X_val = X_val[valid_mask_v]
    y_val = y_val[valid_mask_v].astype(int)

    print(f"  Train: {len(X_train):,} PAs, rate={y_train.mean():.4f}")
    print(f"  Val:   {len(X_val):,} PAs, rate={y_val.mean():.4f}")

    model = build_model()
    if sw_train is not None:
        model.fit(X_train, y_train, hgb__sample_weight=sw_train)
    else:
        model.fit(X_train, y_train)
    raw_metrics = evaluate(f"{target}_raw", model, X_val, y_val)
    print(f"  Raw:   Brier={raw_metrics['brier']:.5f}  AUC={raw_metrics['roc_auc']}  "
          f"CalGap={raw_metrics['calibration_gap']}")

    cal_size = min(200_000, len(X_train) // 5)
    cal_model = CalibratedClassifierCV(model, method="isotonic", cv=2)
    if sw_train is not None:
        _, X_cal, _, y_cal, _, sw_cal = train_test_split(
            X_train,
            y_train,
            sw_train,
            test_size=cal_size,
            random_state=42,
            stratify=y_train,
        )
        cal_model.fit(X_cal, y_cal, sample_weight=sw_cal)
    else:
        _, X_cal, _, y_cal = train_test_split(
            X_train, y_train, test_size=cal_size, random_state=42, stratify=y_train
        )
        cal_model.fit(X_cal, y_cal)
    cal_metrics = evaluate(f"{target}_calibrated", cal_model, X_val, y_val)
    print(f"  Cal:   Brier={cal_metrics['brier']:.5f}  AUC={cal_metrics['roc_auc']}  "
          f"CalGap={cal_metrics['calibration_gap']}")

    model_path = model_dir / f"best_model_{target}.pkl"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(cal_model, f)
    print(f"  Saved → {model_path.name}")

    return {"raw": raw_metrics, "calibrated": cal_metrics}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH, help="Train feature parquet")
    parser.add_argument("--val-path", type=Path, default=VAL_PATH, help="Val feature parquet")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Output directory for pickles + feature_columns.json (e.g. models/exp_bpt_xwoba)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Override metrics CSV path (default: multi_target_metrics.csv or _exp if model-dir is not default)",
    )
    parser.add_argument(
        "--val-pred-parquet",
        type=Path,
        default=None,
        help="Override val predictions parquet path",
    )
    parser.add_argument(
        "--recency-sample-weights",
        action="store_true",
        help="Weight training rows by exponential decay from max(train game_date).",
    )
    parser.add_argument(
        "--recency-weight-half-life-days",
        type=float,
        default=120.0,
        help="Half-life in days for --recency-sample-weights (default 120).",
    )
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    is_default_dir = model_dir.resolve() == MODEL_DIR.resolve()
    name_l = model_dir.name.lower()
    is_recent365_dir = "recent365" in name_l
    is_recency_dir = "recency" in name_l and not is_recent365_dir
    is_beast_dir = "beast" in name_l

    def _route(default_name: str, recent365_name: str, recency_name: str,
               beast_name: str, exp_name: str) -> Path:
        if is_default_dir:
            return REPORTS_DIR / default_name
        if is_recent365_dir:
            return REPORTS_DIR / recent365_name
        if is_recency_dir:
            return REPORTS_DIR / recency_name
        if is_beast_dir:
            return REPORTS_DIR / beast_name
        return REPORTS_DIR / exp_name

    metrics_csv = args.metrics_csv or _route(
        "multi_target_metrics.csv",
        "multi_target_metrics_recent365.csv",
        "multi_target_metrics_recency.csv",
        "multi_target_metrics_beast.csv",
        "multi_target_metrics_exp.csv",
    )
    val_pred_out = args.val_pred_parquet or _route(
        "val_predictions_multi.parquet",
        "val_predictions_multi_recent365.parquet",
        "val_predictions_multi_recency.parquet",
        "val_predictions_multi_beast.parquet",
        "val_predictions_multi_exp.parquet",
    )

    print("=" * 60)
    print("  MULTI-TARGET MODEL TRAINING (Pitcher-Enhanced)")
    print("=" * 60)
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    print(f"  out:   {model_dir}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    feat_cols = get_feature_cols(train_df)

    print(f"\nFeatures: {len(feat_cols)}")
    print(f"Train: {len(train_df):,} PAs")
    print(f"Val:   {len(val_df):,} PAs")
    print(f"Targets: {TARGETS}")

    sw_all = None
    if args.recency_sample_weights:
        sw_all = _recency_sample_weights(train_df, args.recency_weight_half_life_days)
        if sw_all is not None:
            print(
                f"  Recency sample weights (half_life={args.recency_weight_half_life_days}d): "
                f"min={sw_all.min():.4f} max={sw_all.max():.4f} mean={sw_all.mean():.4f}"
            )
        else:
            print("  WARN: --recency-sample-weights skipped (no game_date on train)")

    all_metrics = []
    for target in TARGETS:
        result = train_single_target(
            target, train_df, val_df, feat_cols, model_dir, sample_weight=sw_all
        )
        if result:
            all_metrics.append(result["raw"])
            all_metrics.append(result["calibrated"])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — ALL TARGETS")
    print(f"{'='*60}")
    print(metrics_df[["model", "brier", "roc_auc", "calibration_gap", "actual_rate"]].to_string(index=False))

    feat_cols_list = feat_cols
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feat_cols_list, f)
    print(f"\nFeature column list saved → {model_dir / 'feature_columns.json'}")

    val_pred = val_df[["batter", "pitcher", "game_pk", "game_date",
                        "events", "player_name_clean"]].copy()
    for target in TARGETS:
        model_path = model_dir / f"best_model_{target}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            val_pred[f"p_{target}"] = model.predict_proba(val_df[feat_cols].fillna(0))[:, 1]

    val_pred.to_parquet(val_pred_out, index=False)
    print(f"Val predictions saved → {val_pred_out}")


if __name__ == "__main__":
    main()
