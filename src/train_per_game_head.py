#!/usr/bin/env python3
"""
M2: Per-game head model — predicts P(any HR / any Hit / any XBH) per
(batter, game) directly, instead of relying on the 1-(1-p)^N independence
approximation from per-PA predictions.

Aggregates the val parquet to per-(batter, game_pk) rows, building features:
  - mean / max / sum of per-PA predictions for that batter that game
  - count of PAs faced (proxy for opportunity)
  - the per-PA model's own predictions (from the base model, e.g. beast)

Target: per-game any-outcome flag (1 if batter had any HR / Hit / XBH that game).

Then bucket per-game prediction quartile and verify lift.

Usage:
  python3 src/train_per_game_head.py
  python3 src/train_per_game_head.py --base-model beast --target hr
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MASTER_DIR, REPORTS_DIR  # noqa: E402

VAL_PATH = MASTER_DIR / "features_val_league.parquet"
PER_GAME_DIR = MASTER_DIR / "models" / "per_game_head"
PER_GAME_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = REPORTS_DIR / "per_game_head_summary.md"

MODEL_PRED_PATHS = {
    "prod": REPORTS_DIR / "val_predictions_multi.parquet",
    "exp": REPORTS_DIR / "val_predictions_multi_exp.parquet",
    "beast": REPORTS_DIR / "val_predictions_multi_beast.parquet",
    "recency": REPORTS_DIR / "val_predictions_multi_recency.parquet",
}

TARGETS = [("hr", "is_hr", "HR"), ("hit", "is_hit", "Hit"), ("xbh", "is_xbh", "XBH")]


def _load_aggregated(base_model: str) -> pd.DataFrame:
    val = pd.read_parquet(VAL_PATH, columns=["batter", "pitcher", "game_pk", "game_date",
                                              "is_hr", "is_hit", "is_xbh"])
    preds = pd.read_parquet(MODEL_PRED_PATHS[base_model])
    val = val.copy().reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    for tgt in ("is_hr", "is_hit", "is_xbh"):
        val[f"p_{tgt}"] = preds[f"p_{tgt}"].values

    # Aggregate per (batter, game_pk)
    agg = val.groupby(["batter", "game_pk", "game_date"]).agg(
        n_pa=("is_hr", "size"),
        actual_hr=("is_hr", "max"),
        actual_hit=("is_hit", "max"),
        actual_xbh=("is_xbh", "max"),
        mean_p_hr=("p_is_hr", "mean"),
        max_p_hr=("p_is_hr", "max"),
        sum_p_hr=("p_is_hr", "sum"),
        mean_p_hit=("p_is_hit", "mean"),
        max_p_hit=("p_is_hit", "max"),
        sum_p_hit=("p_is_hit", "sum"),
        mean_p_xbh=("p_is_xbh", "mean"),
        max_p_xbh=("p_is_xbh", "max"),
        sum_p_xbh=("p_is_xbh", "sum"),
    ).reset_index()
    # Independence-baseline per-game probability for comparison
    # Approximation: 1-(1-mean_p)^n_pa. We'll use sum_p as a tighter approximation
    # since the per-PA p varies (sum_p ≈ E[count] for small p; P(0) ≈ exp(-sum_p))
    agg["baseline_any_hr"] = 1 - np.exp(-agg["sum_p_hr"])
    agg["baseline_any_hit"] = 1 - np.exp(-agg["sum_p_hit"])
    agg["baseline_any_xbh"] = 1 - np.exp(-agg["sum_p_xbh"])
    return agg


def train_one(agg: pd.DataFrame, target: str, outcome_col: str) -> dict:
    feat_cols = ["n_pa",
                 "mean_p_hr", "max_p_hr", "sum_p_hr",
                 "mean_p_hit", "max_p_hit", "sum_p_hit",
                 "mean_p_xbh", "max_p_xbh", "sum_p_xbh"]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(agg))
    cut = int(0.8 * len(agg))
    tr, te = idx[:cut], idx[cut:]
    X_tr = agg.iloc[tr][feat_cols].fillna(0)
    y_tr = agg.iloc[tr][outcome_col].astype(int)
    X_te = agg.iloc[te][feat_cols].fillna(0)
    y_te = agg.iloc[te][outcome_col].astype(int).values

    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.06,
        min_samples_leaf=80, random_state=42,
    )
    model.fit(X_tr, y_tr)
    pg_prob = model.predict_proba(X_te)[:, 1]

    # Baseline per-game from independence approximation
    base_col = f"baseline_any_{target}"
    base_te = agg.iloc[te][base_col].values

    base_brier = brier_score_loss(y_te, np.clip(base_te, 1e-6, 1 - 1e-6))
    pg_brier = brier_score_loss(y_te, np.clip(pg_prob, 1e-6, 1 - 1e-6))
    base_auc = roc_auc_score(y_te, base_te) if y_te.sum() > 0 else float("nan")
    pg_auc = roc_auc_score(y_te, pg_prob) if y_te.sum() > 0 else float("nan")

    # Decile lifts
    df_te = pd.DataFrame({"y": y_te, "pg": pg_prob, "base": base_te})
    df_te["pg_dec"] = pd.qcut(df_te["pg"], 10, labels=False, duplicates="drop")
    df_te["base_dec"] = pd.qcut(df_te["base"], 10, labels=False, duplicates="drop")
    pg_top = df_te[df_te["pg_dec"] == df_te["pg_dec"].max()]
    base_top = df_te[df_te["base_dec"] == df_te["base_dec"].max()]
    overall = float(df_te["y"].mean())

    pkl = PER_GAME_DIR / f"per_game_head_{target}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols}, f)

    return {
        "target": target,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "baseline_realized": overall,
        "base_brier": float(base_brier), "pg_brier": float(pg_brier),
        "brier_improvement": float(base_brier - pg_brier),
        "base_auc": float(base_auc), "pg_auc": float(pg_auc),
        "base_topdec_realized": float(base_top["y"].mean()) if len(base_top) else float("nan"),
        "pg_topdec_realized": float(pg_top["y"].mean()) if len(pg_top) else float("nan"),
        "base_topdec_lift": float(base_top["y"].mean() / overall) if len(base_top) and overall > 0 else float("nan"),
        "pg_topdec_lift": float(pg_top["y"].mean() / overall) if len(pg_top) and overall > 0 else float("nan"),
        "pkl_path": str(pkl),
    }


def render_md(results: list[dict], base_model: str) -> str:
    lines = ["# M2: Per-Game Head Summary", ""]
    lines.append(f"Stacked HGBR Classifier on per-(batter, game) aggregations of `{base_model}` "
                 "per-PA predictions. Compares the per-game model to the "
                 "`1-exp(-sum p_PA)` independence-approximation baseline.")
    lines.append("")
    lines.append("| Target | n_test | Baseline rate | Base Brier | PG Brier | Δ Brier | "
                 "Base AUC | PG AUC | Base Top-Dec | PG Top-Dec | Base Lift | PG Lift |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['target']} | {r['n_test']:,} | {r['baseline_realized']:.4f} | "
            f"{r['base_brier']:.5f} | {r['pg_brier']:.5f} | {r['brier_improvement']:+.5f} | "
            f"{r['base_auc']:.4f} | {r['pg_auc']:.4f} | "
            f"{r['base_topdec_realized']:.4f} | {r['pg_topdec_realized']:.4f} | "
            f"{r['base_topdec_lift']:.2f}x | {r['pg_topdec_lift']:.2f}x |"
        )
    lines.append("")
    lines.append("Δ Brier > 0 = per-game head improves over independence baseline.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-game head model.")
    parser.add_argument("--base-model", choices=list(MODEL_PRED_PATHS.keys()), default="beast")
    args = parser.parse_args()

    print("Aggregating per-(batter, game) ...")
    agg = _load_aggregated(args.base_model)
    print(f"Per-game rows: {len(agg):,}")
    print(f"  mean PAs per batter-game: {agg['n_pa'].mean():.2f}")

    results = []
    for tgt, outcome_col, _ in TARGETS:
        per_game_outcome_col = f"actual_{tgt}"
        print(f"Training per-game head for {tgt} ...")
        r = train_one(agg, tgt, per_game_outcome_col)
        results.append(r)

    md = render_md(results, args.base_model)
    OUT_MD.write_text(md)
    print(md)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
