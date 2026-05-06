"""
Feature importance analysis for multi-target models.

Extracts HistGBM native importance, permutation importance, and
groups results by feature block (batter rolling, pitcher profile, BvP, etc.).

Output: data/reports/feature_importance_{target}.csv
        data/reports/feature_importance_summary.csv
"""
import json
import pickle
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from config import MASTER_DIR, REPORTS_DIR

MODEL_DIR = MASTER_DIR / "models"
VAL_PATH = MASTER_DIR / "features_val_league.parquet"
TARGETS = ["is_hit", "is_hr", "is_strikeout", "is_walk", "is_xbh"]

FEATURE_BLOCKS = {
    "batter_rolling": "roll_",
    "batter_career_splits": "cum_",
    "pitcher_profile": "p_pct_|p_velo_|p_spin_|p_pfx_|p_k_rate|p_bb_rate|p_barrel_|p_whiff_|p_arm_|p_extension|p_pct_in_zone",
    "pitcher_rolling": "p_roll_",
    "batter_pitch_type": "bpt_",
    "bvp_matchup": "bvp_",
    "context": "vs_lhp|month|day_of_week|pitch_count|in_zone|times_thru|pitcher_rest|pitcher_age",
}


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


def extract_hgb_importance(model, feat_cols):
    """Extract native feature_importances_ from the HistGBM inside calibrated pipeline."""
    cal = model
    if hasattr(cal, "calibrated_classifiers_"):
        base = cal.calibrated_classifiers_[0].estimator
    elif hasattr(cal, "estimator"):
        base = cal.estimator
    else:
        base = cal

    if hasattr(base, "named_steps"):
        hgb = base.named_steps.get("hgb", None)
        imputer = base.named_steps.get("impute", None)
    elif hasattr(base, "feature_importances_"):
        hgb = base
        imputer = None
    else:
        return None

    if hgb is None or not hasattr(hgb, "feature_importances_"):
        return None

    return dict(zip(feat_cols, hgb.feature_importances_))


def main():
    print("=" * 60)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    with open(MODEL_DIR / "feature_columns.json") as f:
        feat_cols = json.load(f)

    val_df = pd.read_parquet(VAL_PATH)
    print(f"Val: {len(val_df):,} PAs, {len(feat_cols)} features\n")

    X_val = val_df[feat_cols].fillna(0)

    all_importance = []
    block_summary_rows = []

    for target in TARGETS:
        model_path = MODEL_DIR / f"best_model_{target}.pkl"
        if not model_path.exists():
            print(f"  SKIP: {target} (no model)")
            continue

        y_val = val_df[target].fillna(0).astype(int)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"  {target}:")

        hgb_imp = extract_hgb_importance(model, feat_cols)
        if hgb_imp:
            print(f"    HGB importance extracted")
        else:
            print(f"    HGB importance NOT available")
            hgb_imp = {c: 0.0 for c in feat_cols}

        print(f"    Running permutation importance (5K sample)...", flush=True)
        sample_idx = np.random.RandomState(42).choice(len(X_val), min(5000, len(X_val)), replace=False)
        X_sample = X_val.iloc[sample_idx]
        y_sample = y_val.iloc[sample_idx]

        perm = permutation_importance(model, X_sample, y_sample,
                                       n_repeats=5, random_state=42,
                                       scoring="roc_auc", n_jobs=-1)
        print(f"    Done")

        rows = []
        for i, col in enumerate(feat_cols):
            rows.append({
                "target": target,
                "feature": col,
                "block": classify_feature(col),
                "hgb_importance": hgb_imp.get(col, 0.0),
                "perm_importance_mean": perm.importances_mean[i],
                "perm_importance_std": perm.importances_std[i],
            })

        df = pd.DataFrame(rows).sort_values("perm_importance_mean", ascending=False)
        df.to_csv(REPORTS_DIR / f"feature_importance_{target}.csv", index=False)
        all_importance.append(df)

        print(f"    Top 10 by permutation importance:")
        for _, r in df.head(10).iterrows():
            print(f"      {r['feature']:35s} [{r['block']:20s}]  "
                  f"perm={r['perm_importance_mean']:.5f}  hgb={r['hgb_importance']:.4f}")

        block_imp = df.groupby("block").agg(
            total_perm=("perm_importance_mean", "sum"),
            mean_perm=("perm_importance_mean", "mean"),
            max_perm=("perm_importance_mean", "max"),
            n_features=("feature", "count"),
        ).sort_values("total_perm", ascending=False).reset_index()
        block_imp["target"] = target
        block_summary_rows.append(block_imp)

        print(f"\n    Block-level importance:")
        for _, r in block_imp.iterrows():
            print(f"      {r['block']:25s} ({int(r['n_features']):2d} feats)  "
                  f"total_perm={r['total_perm']:.5f}  max={r['max_perm']:.5f}")
        print()

    if all_importance:
        combined = pd.concat(all_importance, ignore_index=True)
        combined.to_csv(REPORTS_DIR / "feature_importance_all_targets.csv", index=False)

    if block_summary_rows:
        blocks = pd.concat(block_summary_rows, ignore_index=True)
        blocks.to_csv(REPORTS_DIR / "feature_importance_by_block.csv", index=False)

        print("=" * 60)
        print("  BLOCK SUMMARY ACROSS ALL TARGETS")
        print("=" * 60)
        pivot = blocks.pivot_table(index="block", columns="target",
                                    values="total_perm", aggfunc="sum").fillna(0)
        pivot["avg_across_targets"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg_across_targets", ascending=False)
        print(pivot.to_string())


if __name__ == "__main__":
    main()
