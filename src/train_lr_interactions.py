#!/usr/bin/env python3
"""
H5 part 1: train a logistic regression model with explicit interaction features
for real ensemble diversity vs the existing tree-based models (prod / exp / beast).

Different inductive bias from trees: linear in feature space (with explicit
crosses for the most-important pairs), heavy regularization. Should produce
predictions that are *correlated but different* from beast — exactly what an
ensemble needs.

Saves to data/master/models/lr_interactions/ following the same conventions as
train_multi_target.py.

Usage:
  python3 src/train_lr_interactions.py
  python3 src/train_lr_interactions.py --train-path ... --val-path ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MASTER_DIR, REPORTS_DIR  # noqa: E402

DEFAULT_TRAIN = MASTER_DIR / "features_train_league.parquet"
DEFAULT_VAL = MASTER_DIR / "features_val_league.parquet"
DEFAULT_DIR = MASTER_DIR / "models" / "lr_interactions"
TARGETS = ["is_hit", "is_hr", "is_strikeout", "is_walk", "is_xbh"]

# Use a curated, smaller feature set for the LR (linear models care about
# which features you give them; including all 100+ is slow + overfits with
# polynomial expansion). Pick the most-informative across batter / pitcher /
# matchup signal so the LR has a fighting chance vs the trees.
LR_BASE_FEATURES = [
    "roll_ba_30", "roll_ba_100", "roll_hr_rate_30", "roll_hr_rate_100",
    "roll_k_rate_30", "roll_ev_30",
    "p_k_rate", "p_bb_rate", "p_velo_overall",
    "p_pct_fastball", "p_pct_breaking", "p_pct_offspeed",
    "p_g_roll_hit_rate_allowed_30",
    "vs_lhp",
    "bvp_pa_count", "bvp_hr_count", "bvp_ba",
    "month",
]

# Pairs to explicitly cross (creates interaction features after polynomial expansion).
# Polynomial expansion would create n*(n-1)/2 interactions on the full set; we limit
# to a hand-picked few to keep the LR tractable.
INTERACTION_PAIRS = [
    ("roll_ba_30", "p_k_rate"),
    ("roll_hr_rate_30", "p_pct_fastball"),
    ("roll_hr_rate_30", "p_velo_overall"),
    ("roll_ev_30", "p_velo_overall"),
    ("vs_lhp", "roll_hr_rate_30"),
]


def _add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for a, b in INTERACTION_PAIRS:
        if a in out.columns and b in out.columns:
            out[f"x_{a}__x__{b}"] = out[a].fillna(0) * out[b].fillna(0)
    return out


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", n_jobs=1)),
    ])


def train_target(train: pd.DataFrame, val: pd.DataFrame, target: str,
                  feat_cols: list[str], model_dir: Path) -> dict:
    print(f"\n=== {target} ===")
    X_tr = _add_interactions(train[feat_cols])
    X_va = _add_interactions(val[feat_cols])
    y_tr = train[target].astype(int)
    y_va = val[target].astype(int)
    print(f"  features (after interactions): {X_tr.shape[1]}")
    print(f"  train n: {len(X_tr):,}  val n: {len(X_va):,}")

    pipe = _build_pipeline()
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_va)[:, 1]

    metrics = {
        "model": f"{target}_lr",
        "brier": float(brier_score_loss(y_va, np.clip(p, 1e-6, 1 - 1e-6))),
        "log_loss": float(log_loss(y_va, np.clip(p, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y_va, p)) if y_va.sum() > 0 else float("nan"),
        "ap": float(average_precision_score(y_va, p)) if y_va.sum() > 0 else float("nan"),
        "calibration_gap": float(abs(p.mean() - y_va.mean())),
        "mean_pred": float(p.mean()),
        "actual_rate": float(y_va.mean()),
    }
    print("  Brier={brier:.5f}  AUC={auc:.4f}  CalGap={calibration_gap:.4f}".format(**metrics))

    pkl = model_dir / f"best_model_{target}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"pipeline": pipe, "feature_cols": list(X_tr.columns)}, f)
    return {"metrics": metrics, "p_val": p, "feature_cols": list(X_tr.columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description="LR with interactions for ensemble diversity (H5).")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train_path)
    val = pd.read_parquet(args.val_path)
    feat_cols = [c for c in LR_BASE_FEATURES if c in train.columns]
    print(f"Using {len(feat_cols)} base features (will expand to "
          f"{len(feat_cols) + len(INTERACTION_PAIRS)} after interactions)")

    val_pred_cols = {"batter": val["batter"], "pitcher": val["pitcher"],
                     "game_pk": val["game_pk"], "game_date": val["game_date"]}
    metrics_rows = []
    for tgt in TARGETS:
        if tgt not in train.columns or tgt not in val.columns:
            print(f"WARN: skipping {tgt}; not in parquet")
            continue
        r = train_target(train, val, tgt, feat_cols, args.model_dir)
        metrics_rows.append(r["metrics"])
        val_pred_cols[f"p_{tgt}"] = r["p_val"]

    # Save val predictions and metrics file
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = REPORTS_DIR / "multi_target_metrics_lr_interactions.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    val_pred_df = pd.DataFrame(val_pred_cols)
    val_pred_path = REPORTS_DIR / "val_predictions_multi_lr_interactions.parquet"
    val_pred_df.to_parquet(val_pred_path, index=False)

    with open(args.model_dir / "feature_columns.json", "w") as f:
        json.dump(metrics_rows[0]["feature_cols"]
                  if metrics_rows and "feature_cols" in metrics_rows[0]
                  else feat_cols, f)

    print(f"\nWrote {metrics_csv}")
    print(f"Wrote {val_pred_path}")
    print(metrics_df[["model", "brier", "auc", "calibration_gap", "actual_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
