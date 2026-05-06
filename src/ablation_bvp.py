"""
BvP ablation study: measure how much each feature block contributes to model AUC.

Trains is_hit model in multiple configurations dropping feature groups.

Output: data/reports/ablation_bvp_results.csv
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from config import MASTER_DIR, REPORTS_DIR

TRAIN_PATH = MASTER_DIR / "features_train_league.parquet"
VAL_PATH = MASTER_DIR / "features_val_league.parquet"

ABLATION_TARGETS = ["is_hit", "is_hr", "is_strikeout"]


def classify_feature(name):
    if name.startswith("roll_"):
        return "batter_rolling"
    if name.startswith("cum_"):
        return "batter_career_splits"
    if name.startswith("p_roll_"):
        return "pitcher_rolling"
    if name.startswith("p_"):
        return "pitcher_profile"
    if name.startswith("bpt_"):
        return "batter_pitch_type"
    if name.startswith("bvp_") or name.startswith("log_bvp_"):
        return "bvp_matchup"
    return "context"


def train_and_eval(X_train, y_train, X_val, y_val, label, feat_list):
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42,
        )),
    ])
    model.fit(X_train[feat_list], y_train)
    y_prob = model.predict_proba(X_val[feat_list])[:, 1]
    return {
        "config": label,
        "n_features": len(feat_list),
        "roc_auc": round(roc_auc_score(y_val, y_prob), 5),
        "brier": round(brier_score_loss(y_val, y_prob), 5),
    }


def main():
    print("=" * 60)
    print("  BvP ABLATION STUDY")
    print("=" * 60)

    with open(MASTER_DIR / "models" / "feature_columns.json") as f:
        all_feats = json.load(f)

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    feat_blocks = {}
    for f in all_feats:
        block = classify_feature(f)
        feat_blocks.setdefault(block, []).append(f)

    print(f"\nFeature blocks:")
    for block, feats in sorted(feat_blocks.items()):
        print(f"  {block:25s}: {len(feats)} features")

    all_results = []
    for target in ABLATION_TARGETS:
        print(f"\n{'='*60}")
        print(f"  TARGET: {target}")
        print(f"{'='*60}")

        y_train = train_df[target].fillna(0).astype(int)
        y_val = val_df[target].fillna(0).astype(int)
        X_train = train_df
        X_val = val_df

        configs = {
            "Full (all 79)": all_feats,
            "No BvP": [f for f in all_feats if classify_feature(f) != "bvp_matchup"],
            "No Pitcher Profile": [f for f in all_feats if classify_feature(f) not in ("pitcher_profile", "pitcher_rolling")],
            "No Pitcher (any)": [f for f in all_feats if classify_feature(f) not in ("pitcher_profile", "pitcher_rolling", "bvp_matchup")],
            "No Batter Pitch-Type": [f for f in all_feats if classify_feature(f) != "batter_pitch_type"],
            "BvP Only + Context": feat_blocks.get("bvp_matchup", []) + feat_blocks.get("context", []),
            "Pitcher Only + Context": (feat_blocks.get("pitcher_profile", []) +
                                        feat_blocks.get("pitcher_rolling", []) +
                                        feat_blocks.get("context", [])),
            "Batter Only": (feat_blocks.get("batter_rolling", []) +
                            feat_blocks.get("batter_career_splits", []) +
                            feat_blocks.get("batter_pitch_type", []) +
                            feat_blocks.get("context", [])),
        }

        for label, feats in configs.items():
            if not feats:
                continue
            print(f"  {label:30s} ({len(feats)} feats)...", end=" ", flush=True)
            result = train_and_eval(X_train, y_train, X_val, y_val, label, feats)
            result["target"] = target
            all_results.append(result)
            print(f"AUC={result['roc_auc']:.5f}  Brier={result['brier']:.5f}")

        target_results = [r for r in all_results if r["target"] == target]
        baseline_auc = next(r["roc_auc"] for r in target_results if r["config"] == "Full (all 79)")
        print(f"\n  {'Config':30s} {'AUC':>10s} {'Delta':>10s} {'Feats':>6s}")
        print(f"  {'-'*60}")
        for r in target_results:
            delta = r["roc_auc"] - baseline_auc
            sign = "+" if delta >= 0 else ""
            print(f"  {r['config']:30s} {r['roc_auc']:10.5f} {sign}{delta:9.5f} {r['n_features']:6d}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORTS_DIR / "ablation_bvp_results.csv", index=False)
    print(f"\nSaved → data/reports/ablation_bvp_results.csv")


if __name__ == "__main__":
    main()
