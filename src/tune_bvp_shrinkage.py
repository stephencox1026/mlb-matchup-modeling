"""
Tune BvP Bayesian shrinkage prior_weight parameter.

Pre-computes raw BvP stats once, then applies different shrinkage weights
and evaluates impact on is_hit model AUC.

Output: data/reports/bvp_shrinkage_tuning.csv
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

LEAGUE_BA = 0.243
LEAGUE_K_RATE = 0.222
LEAGUE_BB_RATE = 0.084

PRIOR_WEIGHTS = [5, 10, 15, 20, 30, 50, 100]


def main():
    print("=" * 60)
    print("  BvP SHRINKAGE TUNING")
    print("=" * 60)

    with open(MASTER_DIR / "models" / "feature_columns.json") as f:
        all_feats = json.load(f)

    bvp_feats = [f for f in all_feats if f.startswith("bvp_")]
    print(f"BvP features to tune: {bvp_feats}")

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}\n")

    target = "is_hit"
    y_train = train_df[target].fillna(0).astype(int)
    y_val = val_df[target].fillna(0).astype(int)

    bvp_raw_path = MASTER_DIR.parent / "data" / "raw" / "bvp_matchup_features.parquet"
    if not bvp_raw_path.exists():
        from config import RAW_DIR
        bvp_raw_path = RAW_DIR / "bvp_matchup_features.parquet"
    bvp_all = pd.read_parquet(bvp_raw_path)
    print(f"Loaded BvP features: {len(bvp_all):,} rows")

    train_bvp_count = bvp_all.reindex(train_df.index)["bvp_pa_count"].fillna(0)
    val_bvp_count = bvp_all.reindex(val_df.index)["bvp_pa_count"].fillna(0)

    train_ba_orig = train_df["bvp_ba"].fillna(LEAGUE_BA)
    train_k_orig = train_df["bvp_k_rate"].fillna(LEAGUE_K_RATE)
    train_bb_orig = train_df["bvp_bb_rate"].fillna(LEAGUE_BB_RATE)
    val_ba_orig = val_df["bvp_ba"].fillna(LEAGUE_BA)
    val_k_orig = val_df["bvp_k_rate"].fillna(LEAGUE_K_RATE)
    val_bb_orig = val_df["bvp_bb_rate"].fillna(LEAGUE_BB_RATE)

    current_pw = 20
    train_n = train_bvp_count
    val_n = val_bvp_count

    def reverse_shrink(shrunk, n, prior, pw):
        """Recover the raw observed rate from a shrunk value."""
        denom = n + pw
        raw = np.where(denom > 0, (shrunk * denom - prior * pw) / np.maximum(n, 1e-9), prior)
        return np.clip(raw, 0, 1)

    def apply_shrink(raw, n, prior, pw):
        return (raw * n + prior * pw) / (n + pw)

    train_raw_ba = reverse_shrink(train_ba_orig.values, train_n.values, LEAGUE_BA, current_pw)
    train_raw_k = reverse_shrink(train_k_orig.values, train_n.values, LEAGUE_K_RATE, current_pw)
    train_raw_bb = reverse_shrink(train_bb_orig.values, train_n.values, LEAGUE_BB_RATE, current_pw)
    val_raw_ba = reverse_shrink(val_ba_orig.values, val_n.values, LEAGUE_BA, current_pw)
    val_raw_k = reverse_shrink(val_k_orig.values, val_n.values, LEAGUE_K_RATE, current_pw)
    val_raw_bb = reverse_shrink(val_bb_orig.values, val_n.values, LEAGUE_BB_RATE, current_pw)

    print(f"\nReverse-engineered raw BvP rates from pw=20 shrunk values")
    print(f"Training on {target}...\n")

    results = []
    for pw in PRIOR_WEIGHTS:
        print(f"  prior_weight={pw:3d}...", end=" ", flush=True)

        train_mod = train_df[all_feats].copy()
        train_mod["bvp_ba"] = apply_shrink(train_raw_ba, train_n.values, LEAGUE_BA, pw)
        train_mod["bvp_k_rate"] = apply_shrink(train_raw_k, train_n.values, LEAGUE_K_RATE, pw)
        train_mod["bvp_bb_rate"] = apply_shrink(train_raw_bb, train_n.values, LEAGUE_BB_RATE, pw)

        val_mod = val_df[all_feats].copy()
        val_mod["bvp_ba"] = apply_shrink(val_raw_ba, val_n.values, LEAGUE_BA, pw)
        val_mod["bvp_k_rate"] = apply_shrink(val_raw_k, val_n.values, LEAGUE_K_RATE, pw)
        val_mod["bvp_bb_rate"] = apply_shrink(val_raw_bb, val_n.values, LEAGUE_BB_RATE, pw)

        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.05,
                min_samples_leaf=50, l2_regularization=1.0, random_state=42,
            )),
        ])
        model.fit(train_mod, y_train)
        y_prob = model.predict_proba(val_mod)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)

        results.append({
            "prior_weight": pw,
            "roc_auc": round(auc, 5),
            "brier": round(brier, 5),
        })
        print(f"AUC={auc:.5f}  Brier={brier:.5f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "bvp_shrinkage_tuning.csv", index=False)

    best = results_df.loc[results_df["roc_auc"].idxmax()]
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nBest prior_weight: {int(best['prior_weight'])} "
          f"(AUC={best['roc_auc']:.5f}, Brier={best['brier']:.5f})")
    print(f"\nSaved → data/reports/bvp_shrinkage_tuning.csv")


if __name__ == "__main__":
    main()
